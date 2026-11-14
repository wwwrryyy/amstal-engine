"""
WebSocket Streaming Broadcaster with 100ms Tick Conflation (QARE-MM v1.3 PRD Section 2.1)
Dispatches live market quotes, volatility regimes, and QARE-MM v1.3 signal frames
matching the Target Output API Contract over ws://127.0.0.1:8194 (canonical Bloomberg port).
"""

import asyncio
from datetime import datetime, timezone
import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set


def build_qare_signal_frame(
    ticker: str,
    action: str = "BUY",
    conviction_tier: str = "HIGH",
    p_buy: float = 0.635,
    p_hold: float = 0.245,
    p_sell: float = 0.120,
    temperature: float = 1.34,
    unconstrained_kelly_pct: float = 6.40,
    l1_normalized_pct: float = 6.40,
    entry_price: float = 342.16,
    dynamic_take_profit: float = 365.40,
    dynamic_stop_loss: float = 330.54,
    reward_to_risk_ratio: float = 2.0,
    regime_state: str = "LOW_VOLATILITY",
    regime_index: int = 0,
    regime_probs: Optional[Dict[str, float]] = None,
    regime_kappa: Optional[float] = None,
    transition_matrix: Optional[List[List[float]]] = None,
    signal_breakdown: Optional[Dict[str, float]] = None,
    winning_engine: str = "LIGHTGBM_105",
    execution_mode: str = "LONG_ONLY",
    missing_modalities: Optional[List[str]] = None,
    intraday_mode: str = "HOURLY_FULL",
    point_in_time_shifted: bool = True,
    xbrl_filed_date_verified: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Constructs an institutional QARE-MM v1.3 Target Output API Contract frame.
    Never synthesizes fake factor scores or fabricated regime transitions when absent.
    """
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    msg_id = f"sig-{int(time.time() * 1000):x}"

    eff_probs = dict(regime_probs) if regime_probs else {}
    if regime_kappa is not None:
        eff_kappa: Optional[float] = round(float(regime_kappa), 2)
    elif eff_probs:
        p_med = float(eff_probs.get("medium_volatility", 0.0))
        p_high = float(eff_probs.get("high_volatility", 0.0))
        eff_kappa = round(0.85 * (1.0 - 0.80 * p_high) * (1.0 - 0.20 * p_med), 2)
    else:
        eff_kappa = None

    eff_transmat = transition_matrix if transition_matrix is not None else []
    eff_signals = dict(signal_breakdown) if signal_breakdown is not None else {}
    eff_missing = list(missing_modalities) if missing_modalities is not None else []
    is_valid_engine = winning_engine != "NONE" and not any(
        flag in eff_missing
        for flag in ("PRICE_HISTORY_UNAVAILABLE", "MODEL_WEIGHTS_OR_HISTORY_MISSING", "SIGNAL_COMPUTE_ERROR")
    )
    eff_xbrl_verified = (
        bool(xbrl_filed_date_verified)
        if xbrl_filed_date_verified is not None
        else is_valid_engine
    )

    return {
        "message_id": msg_id,
        "ticker": ticker.upper(),
        "timestamp_utc": now_iso,
        "horizon_days": 10,
        "winning_engine": winning_engine,
        "execution_mode": execution_mode,
        "data_integrity": {
            "point_in_time_shifted": bool(point_in_time_shifted),
            "xbrl_filed_date_verified": eff_xbrl_verified,
            "intraday_mode": intraday_mode,
            "missing_modalities": eff_missing,
        },
        "macro_regime": {
            "current_state": regime_state,
            "state_index": regime_index,
            "kappa": eff_kappa,
            "filtered_probabilities": eff_probs,
            "transition_matrix": eff_transmat,
        },
        "signal_breakdown": eff_signals,
        "calibration": {
            "temperature": round(temperature, 2),
            "calibrated_probabilities": {
                "outperform_buy": round(p_buy, 3),
                "neutral_hold": round(p_hold, 3),
                "underperform_sell": round(p_sell, 3),
            },
        },
        "execution_recommendation": {
            "action": action,
            "conviction_tier": conviction_tier,
            "unconstrained_kelly_pct": round(unconstrained_kelly_pct, 2),
            "l1_normalized_portfolio_allocation_pct": round(l1_normalized_pct, 2),
            "entry_price": round(entry_price, 2),
            "dynamic_take_profit": round(dynamic_take_profit, 2),
            "dynamic_stop_loss": round(dynamic_stop_loss, 2),
            "reward_to_risk_ratio": round(reward_to_risk_ratio, 2),
            "estimated_round_trip_cost_bps": 15.0,
        },
    }


class WebSocketBroadcaster:
    """
    Manages streaming client subscriptions and 100ms conflation batches over WebSocket.
    Port: 8194 (canonical Bloomberg port).
    """

    DEFAULT_PORT: int = 8194
    CONFLATION_INTERVAL_SEC: float = 0.100  # 100ms conflation window

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        conflation_interval: float = CONFLATION_INTERVAL_SEC,
        rpc_handler: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.conflation_interval = conflation_interval
        self.rpc_handler = rpc_handler
        self.pending_ticks: Dict[str, Dict[str, Any]] = {}
        self.pending_signals: Dict[str, Dict[str, Any]] = {}
        self.pending_regime: Optional[Dict[str, Any]] = None
        self.clients: Set[Any] = set()
        self._client_locks: Dict[Any, asyncio.Lock] = {}
        self._rpc_semaphore: Optional[asyncio.Semaphore] = None
        self.is_running: bool = False
        self._server = None
        self._conflation_task = None

    def queue_tick(self, ticker: str, data: Dict[str, Any]) -> None:
        """Conflates incoming tick updates in memory before broadcast."""
        t = ticker.upper()
        if t not in self.pending_ticks:
            self.pending_ticks[t] = {}
        self.pending_ticks[t].update(data)
        self.pending_ticks[t]["timestamp_ms"] = int(time.time() * 1000)

    def queue_signal(self, ticker: str, signal_frame: Dict[str, Any]) -> None:
        """Queues an updated QARE-MM v1.3 signal recommendation frame."""
        self.pending_signals[ticker.upper()] = signal_frame

    def queue_regime(self, regime_data: Dict[str, Any]) -> None:
        """Queues updated macro volatility regime state."""
        self.pending_regime = regime_data

    def has_pending_data(self) -> bool:
        return bool(self.pending_ticks or self.pending_signals or self.pending_regime)

    def flush_conflated_frame(self) -> Optional[str]:
        """Flushes aggregated state changes into a single conflated UTF-8 JSON payload."""
        if not self.has_pending_data():
            return None

        frame = {
            "type": "CONFLATED_STREAM_FRAME",
            "conflation_interval_ms": int(self.conflation_interval * 1000),
            "timestamp_ms": int(time.time() * 1000),
            "ticks": self.pending_ticks,
            "signals": self.pending_signals,
            "macro_regime": self.pending_regime,
        }

        # Clear buffer
        self.pending_ticks = {}
        self.pending_signals = {}
        self.pending_regime = None

        return json.dumps(frame)

    async def _safe_send(self, client: Any, payload: str) -> None:
        lock = self._client_locks.setdefault(client, asyncio.Lock())
        async with lock:
            await client.send(payload)

    async def broadcast_conflated_updates(self) -> None:
        """Flushes and sends conflated updates to all connected subscribers."""
        payload = self.flush_conflated_frame()
        if payload is not None and self.clients:
            disconnected = set()
            for client in list(self.clients):
                try:
                    await self._safe_send(client, payload)
                except Exception:
                    disconnected.add(client)
            self.clients.difference_update(disconnected)
            for c in disconnected:
                self._client_locks.pop(c, None)

    async def _conflation_loop(self) -> None:
        """Background loop executing conflation flushes."""
        while self.is_running:
            try:
                # If no clients or no pending data, back off to conserve CPU
                if not self.clients or not self.has_pending_data():
                    await asyncio.sleep(0.25)
                    continue
                await asyncio.sleep(self.conflation_interval)
                await self.broadcast_conflated_updates()
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def _dispatch_client_rpc(self, websocket: Any, message: str) -> None:
        if not self.rpc_handler:
            return
        if self._rpc_semaphore is None:
            self._rpc_semaphore = asyncio.Semaphore(8)
        try:
            async with self._rpc_semaphore:
                resp = await asyncio.to_thread(self.rpc_handler, message)
            if resp:
                await self._safe_send(websocket, resp)
        except Exception:
            pass

    async def register_client(self, websocket: Any) -> None:
        self.clients.add(websocket)
        self._client_locks[websocket] = asyncio.Lock()
        pending_tasks: Set[asyncio.Task[Any]] = set()
        try:
            # Send initial greeting
            welcome = json.dumps({
                "type": "CONNECTION_ESTABLISHED",
                "service": "amstal-engine-stream",
                "port": self.port,
                "timestamp_ms": int(time.time() * 1000),
            })
            await self._safe_send(websocket, welcome)
            async for message in websocket:
                if self.rpc_handler and isinstance(message, str):
                    task = asyncio.create_task(self._dispatch_client_rpc(websocket, message))
                    pending_tasks.add(task)
                    task.add_done_callback(pending_tasks.discard)
            if pending_tasks:
                await asyncio.gather(*list(pending_tasks), return_exceptions=True)
        except Exception:
            pass
        finally:
            for t in list(pending_tasks):
                t.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            self.clients.discard(websocket)
            self._client_locks.pop(websocket, None)

    async def start(self) -> None:
        """Starts WebSocket server and 100ms background conflation loop."""
        try:
            import websockets

            allowed_origins = [
                None,
                re.compile(r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"),
                re.compile(r"^tauri://localhost(:\d+)?$"),
                "https://tauri.localhost",
                "http://tauri.localhost",
            ]
            self.is_running = True
            self._server = await websockets.serve(
                self.register_client,
                self.host,
                self.port,
                origins=allowed_origins,
            )
            self._conflation_task = asyncio.create_task(self._conflation_loop())
        except ImportError:
            self.is_running = True
            self._conflation_task = asyncio.create_task(self._conflation_loop())

    async def stop(self) -> None:
        """Gracefully stops broadcaster and closes all client connections."""
        self.is_running = False
        if self._conflation_task:
            self._conflation_task.cancel()
            self._conflation_task = None
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self.clients.clear()
