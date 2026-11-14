"""
Amstal Engine Daemon Entrypoint
Spawns the UDS JSON-RPC 2.0 listener (/tmp/amstal.sock) and WebSocket broadcaster (:8194).
"""

import asyncio
import datetime
import logging
import signal
import sys
import time
from typing import Optional
from zoneinfo import ZoneInfo

_NY_TZ = ZoneInfo("America/New_York")


def is_us_equity_market_open(now_utc: Optional[datetime.datetime] = None) -> bool:
    """Returns True during regular NYSE/NASDAQ cash trading hours (09:30–16:00 America/New_York, Mon–Fri)."""
    dt = now_utc or datetime.datetime.now(datetime.timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    et = dt.astimezone(_NY_TZ)
    if et.weekday() >= 5:
        return False
    et_minutes = et.hour * 60 + et.minute
    return 9 * 60 + 30 <= et_minutes <= 16 * 60

from src.ipc.live_providers import LiveInferenceProvider
from src.ipc.uds_server import JsonRpcUdsServer
from src.ipc.ws_broadcaster import WebSocketBroadcaster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("amstal.engine")


class AmstalEngineDaemon:
    """Coordinates lifecycle of Amstal IPC control plane and streaming servers."""

    def __init__(
        self,
        socket_path: str = "/tmp/amstal.sock",
        ws_host: str = "127.0.0.1",
        ws_port: int = 8194,
    ) -> None:
        self.live_provider = LiveInferenceProvider()
        logger.info("Initializing JsonRpcUdsServer with lakehouse & live model providers.")
        self.uds_server = JsonRpcUdsServer(
            socket_path=socket_path,
            signal_provider=self.live_provider.get_signal,
            portfolio_provider=self.live_provider.get_portfolio,
            regime_provider=self.live_provider.get_macro_regime,
            security_master_provider=self.live_provider.get_security_master,
            peer_graph_provider=self.live_provider.get_peer_graph,
            edgar_provider=self.live_provider.get_edgar_filings,
            news_provider=self.live_provider.get_news_feed,
            historical_prices_provider=self.live_provider.get_historical_prices,
            analyst_ratings_provider=self.live_provider.get_analyst_ratings,
            copilot_query_provider=self.live_provider.query_copilot,
            tick_provider=self.live_provider.get_latest_quote,
            financials_provider=self.live_provider.get_financials,
        )

        self.ws_broadcaster = WebSocketBroadcaster(
            host=ws_host,
            port=ws_port,
            rpc_handler=self.uds_server.process_message,
        )
        self._stop_event: Optional[asyncio.Event] = None
        self._stream_task: Optional[asyncio.Task] = None

    async def _stream_feed_loop(self) -> None:
        """Periodically refreshes live quotes, and queues conflated tick, signal, and regime frames."""
        watchlist = ["NVDA", "MSFT", "AAPL", "CRWD", "SPY", "TSLA"]
        step = 0
        last_closed_broadcast = 0.0
        last_refresh_ts = 0.0
        last_client_count = 0
        while self._stop_event and not self._stop_event.is_set():
            try:
                client_count = len(self.ws_broadcaster.clients)
                if client_count > 0:
                    # Regular NYSE/NASDAQ session in DST-aware America/New_York time
                    is_market_open = is_us_equity_market_open()
                    now_ts = time.time()
                    active_pool = list(dict.fromkeys(watchlist + list(self.uds_server.active_tickers)))

                    # Refresh stale lakehouse partitions / quote caches in worker thread every 60s
                    if now_ts - last_refresh_ts >= 60.0:
                        await asyncio.to_thread(self.live_provider.refresh_tickers_if_needed, active_pool)
                        last_refresh_ts = time.time()

                    new_client_joined = client_count > last_client_count
                    last_client_count = client_count

                    if not is_market_open:
                        # Off-market hours: push immediately on newly joined client or every 15s
                        if new_client_joined or (now_ts - last_closed_broadcast >= 15.0):
                            regime = await asyncio.to_thread(self.live_provider.get_macro_regime)
                            self.ws_broadcaster.queue_regime(regime)
                            for sym in active_pool:
                                sig = await asyncio.to_thread(self.live_provider.get_signal, sym, regime)
                                self.ws_broadcaster.queue_signal(sym, sig)
                                quote = await asyncio.to_thread(self.live_provider.get_latest_quote, sym)
                                if quote["price"] > 0:
                                    self.ws_broadcaster.queue_tick(sym, quote)
                            last_closed_broadcast = now_ts
                        await asyncio.sleep(2.0)
                        continue

                    # Active session: stream conflated updates across active symbols
                    regime = await asyncio.to_thread(self.live_provider.get_macro_regime)
                    self.ws_broadcaster.queue_regime(regime)
                    sym = active_pool[step % len(active_pool)]
                    sig = await asyncio.to_thread(self.live_provider.get_signal, sym, regime)
                    self.ws_broadcaster.queue_signal(sym, sig)
                    quote = await asyncio.to_thread(self.live_provider.get_latest_quote, sym)
                    if quote["price"] > 0:
                        self.ws_broadcaster.queue_tick(sym, quote)
                    step += 1
                    await asyncio.sleep(1.0)
                else:
                    last_client_count = 0
                    await asyncio.sleep(2.0)
            except Exception as e:
                logger.debug(f"Stream feed loop warning: {e}")
                await asyncio.sleep(2.0)

    async def start(self) -> None:
        logger.info("Initializing Amstal Quantitative Alpha & Risk Engine (QARE-MM v1.3)...")
        self._stop_event = asyncio.Event()

        # Start UDS Control Plane
        await self.uds_server.start()
        logger.info(f"UDS Control Plane listening on {self.uds_server.socket_path}")

        # Start WebSocket Broadcaster
        await self.ws_broadcaster.start()
        logger.info(
            f"WebSocket Broadcaster listening on ws://{self.ws_broadcaster.host}:{self.ws_broadcaster.port}"
        )

        self._stream_task = asyncio.create_task(self._stream_feed_loop())

        logger.info("Engine daemon fully active. Awaiting terminal connections...")
        await self._stop_event.wait()

    async def stop_async(self) -> None:
        """Asynchronously and cleanly shuts down UDS server and WebSocket broadcaster."""
        logger.info("Initiating graceful teardown of Amstal Engine daemon...")
        if self._stop_event:
            self._stop_event.set()
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        self.uds_server.stop()
        if self.ws_broadcaster:
            await self.ws_broadcaster.stop()
        logger.info("Amstal Engine daemon teardown complete.")

    def stop(self) -> None:
        if self._stop_event:
            self._stop_event.set()
        self.uds_server.stop()
        if self.ws_broadcaster:
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    loop.create_task(self.ws_broadcaster.stop())
            except RuntimeError:
                pass


async def _run_daemon() -> None:
    daemon = AmstalEngineDaemon()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, daemon.stop)
        except NotImplementedError:
            pass

    try:
        await daemon.start()
    except asyncio.CancelledError:
        pass
    finally:
        await daemon.stop_async()


def main() -> int:
    try:
        asyncio.run(_run_daemon())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

