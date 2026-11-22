"""
WebSocket Streaming Broadcaster with 100ms Tick Conflation (QARE-MM v1.3 PRD Section 2.1)
Dispatches live market quotes, volatility regimes, and QARE-MM v1.3 signal frames
matching the Target Output API Contract over ws://127.0.0.1:8194 (canonical Bloomberg port).
"""

import asyncio
from datetime import datetime, timezone
import json
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


def _safe_finite_float(val: Any) -> Optional[float]:
    if not isinstance(val, (int, float)) or isinstance(val, bool):
        return None
    try:
        f = float(val)
    except (OverflowError, ValueError, TypeError):
        return None
    return f if math.isfinite(f) else None


def _finite_float(val: Any, default: float = 0.0) -> float:
    f = _safe_finite_float(val)
    return f if f is not None else default


def _sanitize_json_obj(obj: Any) -> Any:
    if isinstance(obj, bool) or obj is None or isinstance(obj, (str, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _sanitize_json_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json_obj(v) for v in obj]
    return obj


def normalize_regime_probs(regime_probs: Any) -> Dict[str, float]:
    """Validates an HMM posterior distribution in [0, 1] summing to 1.0 +/- 0.02 and normalizes to sum 1."""
    if not isinstance(regime_probs, dict):
        return {}
    req_keys = ("low_volatility", "medium_volatility", "high_volatility")
    clean: Dict[str, float] = {}
    for pk, pv in regime_probs.items():
        fv = _safe_finite_float(pv)
        if fv is not None and 0.0 <= fv <= 1.0:
            clean[str(pk)] = fv
        else:
            return {}
    if all(k in clean for k in req_keys):
        total = sum(clean[k] for k in req_keys)
        if not (0.98 <= total <= 1.02) or total <= 0.0:
            return {}
        return {k: clean[k] / total for k in req_keys}
    if "medium_volatility" in clean and "high_volatility" in clean and len(clean) == 2:
        total_mh = clean["medium_volatility"] + clean["high_volatility"]
        if total_mh <= 1.02:
            tri = {
                "low_volatility": max(0.0, 1.0 - total_mh),
                "medium_volatility": clean["medium_volatility"],
                "high_volatility": clean["high_volatility"],
            }
            tot = sum(tri.values())
            if tot > 0.0:
                return {k: v / tot for k, v in tri.items()}
    return {}


def resolve_regime_kappa(raw_kappa: Any, regime_probs: Any) -> Optional[float]:
    """Resolves clamped macro multiplier kappa in [0.25, 0.50] from explicit kappa or validated probs."""
    fk = _safe_finite_float(raw_kappa)
    if fk is not None:
        return round(max(0.25, min(0.50, fk)), 2)
    valid_probs = normalize_regime_probs(regime_probs)
    if not valid_probs:
        return None
    p_low = valid_probs["low_volatility"]
    p_med = valid_probs["medium_volatility"]
    p_high = valid_probs["high_volatility"]
    p_mass = p_low + p_med + p_high
    return round(max(0.25, min(0.50, (0.50 * p_low + 0.35 * p_med + 0.25 * p_high) / p_mass)), 2)


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
    prior_probs: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    """
    Constructs an institutional QARE-MM v1.3 Target Output API Contract frame.
    Never synthesizes fake factor scores or fabricated regime transitions when absent.
    """
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    msg_id = f"sig-{int(time.time() * 1000):x}"

    eff_probs = normalize_regime_probs(regime_probs)
    eff_kappa = resolve_regime_kappa(regime_kappa, eff_probs)

    eff_transmat = _sanitize_json_obj(transition_matrix) if transition_matrix is not None else []
    eff_signals: Dict[str, float] = {}
    if isinstance(signal_breakdown, dict):
        for sk, sv in signal_breakdown.items():
            f_sv = _safe_finite_float(sv)
            if f_sv is not None:
                eff_signals[str(sk)] = f_sv

    p_f = [_safe_finite_float(v) for v in (p_buy, p_hold, p_sell)]
    probs_sum = sum(fv for fv in p_f if fv is not None) if all(fv is not None for fv in p_f) else -1.0
    probs_valid = (
        all(fv is not None and 0.0 <= fv <= 1.0 for fv in p_f)
        and 0.98 <= probs_sum <= 1.02
    )

    e_f = _safe_finite_float(entry_price)
    tp_f = _safe_finite_float(dynamic_take_profit)
    sl_f = _safe_finite_float(dynamic_stop_loss)
    e_r = round(e_f, 2) if e_f is not None else None
    tp_r = round(tp_f, 2) if tp_f is not None else None
    sl_r = round(sl_f, 2) if sl_f is not None else None
    prices_valid = True
    if action == "BUY":
        prices_valid = bool(
            e_r is not None and tp_r is not None and sl_r is not None and 0.0 < sl_r < e_r < tp_r
        )
    elif action == "SELL":
        mode_u = str(execution_mode or "").strip().upper()
        if e_r is not None and tp_r is not None and sl_r is not None:
            order_ok = (tp_r < e_r < sl_r) if mode_u == "LONG_SHORT" else (sl_r < e_r < tp_r)
        else:
            order_ok = False
        prices_valid = bool(
            e_r is not None
            and tp_r is not None
            and sl_r is not None
            and e_r > 0.0
            and tp_r > 0.0
            and sl_r > 0.0
            and order_ok
        )

    eff_missing = list(missing_modalities) if missing_modalities is not None else []
    if probs_valid and prices_valid and p_f[0] is not None and p_f[1] is not None and p_f[2] is not None:
        eff_p_buy, eff_p_hold, eff_p_sell = p_f[0], p_f[1], p_f[2]
        eff_action = action
        eff_conviction = conviction_tier
        eff_unconstrained = unconstrained_kelly_pct
        eff_l1 = l1_normalized_pct
        eff_winning_engine = winning_engine
        eff_execution_mode = execution_mode
    elif not probs_valid:
        eff_p_buy, eff_p_hold, eff_p_sell = 0.0, 1.0, 0.0
        eff_action = "HOLD"
        eff_conviction = "LOW"
        eff_unconstrained = 0.0
        eff_l1 = 0.0
        eff_winning_engine = "NONE"
        eff_execution_mode = "UNREADY"
        if "INVALID_CALIBRATED_PROBABILITIES" not in eff_missing:
            eff_missing.append("INVALID_CALIBRATED_PROBABILITIES")
    else:
        eff_p_buy = float(p_f[0]) if p_f[0] is not None else 0.0
        eff_p_hold = float(p_f[1]) if p_f[1] is not None else 1.0
        eff_p_sell = float(p_f[2]) if p_f[2] is not None else 0.0
        eff_action = "HOLD"
        eff_conviction = "LOW"
        eff_unconstrained = 0.0
        eff_l1 = 0.0
        eff_winning_engine = "NONE"
        eff_execution_mode = "UNREADY"
        if "INVALID_RECOMMENDATION_PRICES" not in eff_missing:
            eff_missing.append("INVALID_RECOMMENDATION_PRICES")

    is_valid_engine = eff_winning_engine != "NONE" and not any(
        flag in eff_missing
        for flag in (
            "PRICE_HISTORY_UNAVAILABLE",
            "MODEL_WEIGHTS_OR_HISTORY_MISSING",
            "SIGNAL_COMPUTE_ERROR",
            "INVALID_CALIBRATED_PROBABILITIES",
            "INVALID_RECOMMENDATION_PRICES",
        )
    )
    eff_xbrl_verified = (
        bool(xbrl_filed_date_verified)
        if xbrl_filed_date_verified is not None
        else is_valid_engine
    )

    calibration_block: Dict[str, Any] = {
        "temperature": round(_finite_float(temperature, 1.0), 2),
        "calibrated_probabilities": {
            "outperform_buy": round(eff_p_buy, 3),
            "neutral_hold": round(eff_p_hold, 3),
            "underperform_sell": round(eff_p_sell, 3),
        },
    }
    if isinstance(prior_probs, (tuple, list)) and len(prior_probs) == 3:
        prior_floats = [_safe_finite_float(v) for v in prior_probs]
        if all(fv is not None and fv >= 0.0 for fv in prior_floats):
            pb0, ph0, ps0 = (
                prior_floats[0] or 0.0,
                prior_floats[1] or 0.0,
                prior_floats[2] or 0.0,
            )
            max_w = max(pb0, ph0, ps0)
            if max_w > 0.0:
                sb, sh, ss = pb0 / max_w, ph0 / max_w, ps0 / max_w
                stot = sb + sh + ss
                calibration_block["prior_probabilities"] = {
                    "outperform_buy": round(sb / stot, 4),
                    "neutral_hold": round(sh / stot, 4),
                    "underperform_sell": round(ss / stot, 4),
                }

    return {
        "message_id": msg_id,
        "ticker": ticker.upper(),
        "timestamp_utc": now_iso,
        "horizon_days": 10,
        "winning_engine": eff_winning_engine,
        "execution_mode": eff_execution_mode,
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
        "calibration": calibration_block,
        "execution_recommendation": {
            "action": eff_action,
            "conviction_tier": eff_conviction,
            "unconstrained_kelly_pct": round(_finite_float(eff_unconstrained, 0.0), 2),
            "l1_normalized_portfolio_allocation_pct": round(_finite_float(eff_l1, 0.0), 2),
            "entry_price": e_r if e_r is not None else 0.0,
            "dynamic_take_profit": tp_r if tp_r is not None else 0.0,
            "dynamic_stop_loss": sl_r if sl_r is not None else 0.0,
            "reward_to_risk_ratio": round(_finite_float(reward_to_risk_ratio, 0.0), 2),
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

    @staticmethod
    def _is_valid_ts_ms(val: Any) -> bool:
        fv = _safe_finite_float(val)
        return fv is not None and fv > 0.0

    _TICK_MARKET_FIELDS: frozenset[str] = frozenset({
        "price",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "bid",
        "ask",
        "change",
        "change_pct",
        "high_24h",
        "low_24h",
        "prev_close",
        "date",
    })

    def queue_tick(self, ticker: str, data: Dict[str, Any]) -> None:
        """Conflates incoming tick updates in memory before broadcast."""
        t = ticker.upper()
        if isinstance(data, dict) and "price" in data:
            # Complete tick snapshot from get_latest_quote replaces prior pending fields while validating timestamps
            self.pending_ticks[t] = {
                k: v
                for k, v in data.items()
                if k not in ("timestamp_ms", "observation_timestamp_ms") or self._is_valid_ts_ms(v)
            }
        else:
            if t not in self.pending_ticks:
                self.pending_ticks[t] = {}
            if isinstance(data, dict):
                patch = {
                    k: v
                    for k, v in data.items()
                    if (k != "timestamp_ms" and k != "observation_timestamp_ms") or self._is_valid_ts_ms(v)
                }
                if "observation_timestamp_ms" not in patch and any(
                    k in self._TICK_MARKET_FIELDS and self.pending_ticks[t].get(k) != v
                    for k, v in patch.items()
                ):
                    self.pending_ticks[t].pop("observation_timestamp_ms", None)
                self.pending_ticks[t].update(patch)
        if not self._is_valid_ts_ms(self.pending_ticks[t].get("timestamp_ms")):
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

        return json.dumps(_sanitize_json_obj(frame), allow_nan=False)

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
