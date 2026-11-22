"""
Amstal Live Model Inference Providers (QARE-MM v1.3)
Connects trained model weights from weights/ to IPC JSON-RPC requests,
enabling fast cached multi-modal inference, causal HMM regime filtering,
authentic SEC EDGAR / Yahoo Finance fundamentals, and Kelly sizing.
"""

from collections.abc import Mapping
from datetime import date, datetime, time as dt_time, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import pickle
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.data.fundamentals_client import FundamentalsClient
from src.data.market_loader import MarketDataLoader
from src.data.universe import UniverseSelector
from src.features.event_decay import EventDecayKernel
from src.features.graph_builder import GraphBuilder
from src.features.microstructure import MicrostructureFeatureExtractor
from src.features.residualizer import Residualizer
from src.ipc.ws_broadcaster import (
    _sanitize_json_obj,
    build_qare_signal_frame,
    normalize_regime_probs,
    resolve_regime_kappa,
)
from src.models.backbone_lgbm import LightGBMBackbone
from src.models.regime_hmm import RegimeHMM
from src.risk.calibration import TemperatureScaler
from src.risk.kelly_engine import MultiOutcomeKellySizer, PortfolioExposureNormalizer

logger = logging.getLogger("amstal.providers")


class LiveInferenceProvider:
    """Manages loaded production weights and provides cached, non-simulated IPC callbacks."""

    _QUOTE_TTL_SECONDS = 15.0
    _INTRADAY_TTL_SECONDS = 60.0
    _DAILY_REFRESH_COOLDOWN_SECONDS = 300.0
    _FAILED_TICKER_COOLDOWN_SECONDS = 900.0
    _REGIME_TTL_SECONDS = 30.0
    _SIGNAL_TTL_SECONDS = 20.0
    _PORTFOLIO_TTL_SECONDS = 15.0
    _PEER_GRAPH_TTL_SECONDS = 120.0
    _SEC_MASTER_TTL_SECONDS = 300.0
    _RATINGS_TTL_SECONDS = 300.0
    _EDGAR_TTL_SECONDS = 300.0
    _FINANCIALS_TTL_SECONDS = 600.0
    _NEWS_TTL_SECONDS = 120.0
    _TAIL_WARMUP_BARS = 320

    def __init__(
        self,
        weights_dir: Optional[Path] = None,
        lakehouse_dir: Optional[Path] = None,
        enable_live_fetch: Optional[bool] = None,
    ) -> None:
        engine_root = Path(__file__).resolve().parent.parent.parent
        self.weights_dir = weights_dir or (engine_root / "weights")
        self.lakehouse_dir = lakehouse_dir or (engine_root / "lakehouse")

        if enable_live_fetch is None:
            offline_flag = (
                bool(os.environ.get("PYTEST_CURRENT_TEST"))
                or os.environ.get("AMSTAL_OFFLINE_ONLY") == "1"
            )
            self.enable_live_fetch = not offline_flag
        else:
            self.enable_live_fetch = bool(enable_live_fetch)

        self.market_loader = MarketDataLoader(data_dir=self.lakehouse_dir)
        self.universe_selector = UniverseSelector()
        self.fundamentals_client = FundamentalsClient(lakehouse_dir=self.lakehouse_dir)
        self.graph_builder = GraphBuilder(universe_selector=self.universe_selector, lookback_corr=60)
        self.residualizer = Residualizer(window_size=60)
        self.micro_extractor = MicrostructureFeatureExtractor(lookback_window=126)
        self.kelly_sizer = MultiOutcomeKellySizer()
        self.portfolio_normalizer = PortfolioExposureNormalizer

        self._lock = threading.RLock()
        self._df_cache: Dict[str, Tuple[float, pd.DataFrame]] = {}
        self._live_quote_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._intraday_cache: Dict[Tuple[str, str], Tuple[float, List[Dict[str, Any]]]] = {}
        self._news_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
        self._daily_refresh_attempted: Dict[str, float] = {}
        self._failed_yf_until: Dict[str, float] = {}
        self._partial_bar_obs_ms: Dict[str, Tuple[str, int, int]] = {}

        # High-frequency inference & RPC caches
        self._regime_cache: Optional[Tuple[Tuple[str, float], float, Dict[str, Any]]] = None
        self._signal_cache: Dict[str, Tuple[Tuple[str, float, int, float], float, Dict[str, Any]]] = {}
        self._portfolio_cache: Optional[Tuple[float, Dict[str, Any]]] = None
        self._peer_graph_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._sec_master_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._ratings_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._edgar_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._financials_cache: Dict[Tuple[str, bool], Tuple[float, Dict[str, Any]]] = {}

        # Models
        self.regime_hmm: Optional[RegimeHMM] = None
        self.lgbm_backbone: Optional[LightGBMBackbone] = None
        self.temp_scaler: Optional[TemperatureScaler] = None
        self.manifest: Optional[Dict[str, Any]] = None
        self._class_priors: Optional[Tuple[float, float, float]] = None

        self._load_weights()

    def _load_weights(self) -> None:
        """Loads serialized weights if manifest exists."""
        manifest_path = self.weights_dir / "manifest.json"
        if not manifest_path.exists():
            logger.warning(f"No weights manifest found at {manifest_path}. Operating without model weights.")
            return

        try:
            with open(manifest_path, "r") as f:
                self.manifest = json.load(f)

            cd = self.manifest.get("class_distribution") if isinstance(self.manifest, dict) else None
            if isinstance(cd, dict):
                try:
                    raw_vals = [cd.get("0", 0.0), cd.get("1", 0.0), cd.get("2", 0.0)]
                    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw_vals):
                        c0, c1, c2 = float(raw_vals[0]), float(raw_vals[1]), float(raw_vals[2])
                        tot = c0 + c1 + c2
                        if all(math.isfinite(v) and v > 0.0 for v in (c0, c1, c2)) and math.isfinite(tot) and tot > 0.0:
                            self._class_priors = (c0 / tot, c1 / tot, c2 / tot)
                except (TypeError, ValueError, OverflowError):
                    self._class_priors = None

            hmm_path = self.weights_dir / "regime_hmm.pkl"
            if hmm_path.exists():
                with open(hmm_path, "rb") as f:
                    self.regime_hmm = pickle.load(f)

            lgbm_path = self.weights_dir / "lightgbm_backbone.pkl"
            if lgbm_path.exists():
                with open(lgbm_path, "rb") as f:
                    self.lgbm_backbone = pickle.load(f)

            scaler_path = self.weights_dir / "temperature_scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    self.temp_scaler = pickle.load(f)

            logger.info("Successfully loaded production weights into LiveInferenceProvider.")
        except Exception as e:
            logger.error(f"Failed to load production weights: {e}")

    def is_ready(self) -> bool:
        return (
            self.regime_hmm is not None
            and self.lgbm_backbone is not None
            and self.temp_scaler is not None
        )

    def _load_disk_daily(self, sym: str) -> Optional[pd.DataFrame]:
        """Reads daily Parquet for ticker with mtime-keyed cache (no network)."""
        pq_path = self.market_loader.daily_dir / f"{sym}.parquet"
        if not pq_path.exists():
            return None
        try:
            st = pq_path.stat()
            mtime = st.st_mtime
            mtime_ms = int(mtime * 1000)
            stat_sig = (st.st_dev, st.st_ino, st.st_mtime_ns, st.st_size)
            with self._lock:
                cached = self._df_cache.get(sym)
                if cached is not None and cached[0] == stat_sig:
                    return cached[1]
            df = self.market_loader.load_ticker_daily(sym)
            if df is not None and len(df) > 0:
                df_attrs = getattr(df, "attrs", {}) or {}
                gen_mtime_ms = df_attrs.get("disk_mtime_ms")
                eff_mtime_ms = int(gen_mtime_ms) if isinstance(gen_mtime_ms, (int, float)) else mtime_ms
                df.attrs["disk_mtime_ms"] = eff_mtime_ms
                last_dt_str = (
                    df.index[-1].strftime("%Y-%m-%d")
                    if hasattr(df.index[-1], "strftime")
                    else str(df.index[-1])[:10]
                )
                df.attrs["obs_bar_date"] = last_dt_str
                with self._lock:
                    if "observation_timestamp_ms" not in df.attrs:
                        part_rec = self._partial_bar_obs_ms.get(sym)
                        if part_rec is not None and part_rec[0] == last_dt_str and part_rec[1] == eff_mtime_ms:
                            df.attrs["observation_timestamp_ms"] = part_rec[2]
                    self._df_cache[sym] = (stat_sig, df)
            elif df is not None:
                with self._lock:
                    self._df_cache[sym] = (stat_sig, df)
            return df
        except Exception:
            return self.market_loader.load_ticker_daily(sym)

    def _maybe_refresh_daily(self, sym: str) -> None:
        """Refreshes stale or missing daily Parquet from yfinance with per-symbol cooldown."""
        if not self.enable_live_fetch:
            return
        now = time.monotonic()
        with self._lock:
            if now < self._failed_yf_until.get(sym, 0.0):
                return
            if now - self._daily_refresh_attempted.get(sym, 0.0) < self._DAILY_REFRESH_COOLDOWN_SECONDS:
                return

        attempt_started_ms = int(time.time() * 1000)
        expected_latest = self.market_loader._latest_expected_trading_date()
        pq_path_pre = self.market_loader.daily_dir / f"{sym}.parquet"
        prev_stat_gen: Optional[Tuple[int, int, int, int]] = None
        if pq_path_pre.exists():
            try:
                st_pre = pq_path_pre.stat()
                prev_stat_gen = (st_pre.st_dev, st_pre.st_ino, st_pre.st_mtime_ns, st_pre.st_size)
            except OSError:
                prev_stat_gen = None
        df_disk = self._load_disk_daily(sym)
        needs_post_close_refresh = False
        prev_mtime_ms: Optional[int] = None
        disk_sched_close: Optional[int] = None
        if df_disk is not None and len(df_disk) > 0:
            last_dt = pd.Timestamp(df_disk.index[-1]).normalize()
            _, disk_complete, disk_sched_close = self._daily_bar_status_ms(df_disk)
            prev_raw_mtime = (getattr(df_disk, "attrs", {}) or {}).get("disk_mtime_ms")
            if isinstance(prev_raw_mtime, (int, float)) and not isinstance(prev_raw_mtime, bool):
                prev_mtime_ms = int(prev_raw_mtime)
            needs_post_close_refresh = bool(
                not disk_complete
                and disk_sched_close is not None
                and attempt_started_ms >= disk_sched_close
            )
            if last_dt >= expected_latest and not needs_post_close_refresh:
                return

        # Reserve refresh cooldown atomically only after confirming a download is actually needed
        with self._lock:
            now_reserve = time.monotonic()
            if now_reserve < self._failed_yf_until.get(sym, 0.0):
                return
            if now_reserve - self._daily_refresh_attempted.get(sym, 0.0) < self._DAILY_REFRESH_COOLDOWN_SECONDS:
                return
            self._daily_refresh_attempted[sym] = now_reserve

        refreshed = self.market_loader.refresh_ticker_daily(
            sym, force_refresh_if_incomplete=needs_post_close_refresh
        )
        pq_path = self.market_loader.daily_dir / f"{sym}.parquet"
        cur_stat_gen: Optional[Tuple[int, int, int, int]] = None
        new_mtime_ms: Optional[int] = None
        if pq_path.exists():
            try:
                st_cur = pq_path.stat()
                cur_stat_gen = (st_cur.st_dev, st_cur.st_ino, st_cur.st_mtime_ns, st_cur.st_size)
                new_mtime_ms = int(st_cur.st_mtime * 1000)
            except OSError:
                cur_stat_gen = None

        refreshed_attrs = (getattr(refreshed, "attrs", {}) or {}) if refreshed is not None else {}
        has_outcome = "_refresh_wrote" in refreshed_attrs
        wrote_this_call = bool(refreshed_attrs.get("_refresh_wrote")) if has_outcome else False
        replaced_target = bool(refreshed_attrs.get("_target_row_replaced", True))
        outcome_gen = refreshed_attrs.get("_refresh_gen") if "_refresh_gen" in refreshed_attrs else prev_stat_gen
        disk_changed_vs_outcome = bool(
            cur_stat_gen is not None
            and (
                (has_outcome and cur_stat_gen != outcome_gen)
                or (not has_outcome and not replaced_target and prev_stat_gen is not None and cur_stat_gen != prev_stat_gen)
            )
        )

        # If another writer modified the partition vs our outcome/read, re-read disk status before scheduling cooldown
        recheck_disk: Optional[pd.DataFrame] = None
        if disk_changed_vs_outcome:
            with self._lock:
                self._df_cache.pop(sym, None)
                self._partial_bar_obs_ms.pop(sym, None)
            recheck_disk = self._load_disk_daily(sym)

        with self._lock:
            eff_frame = recheck_disk if disk_changed_vs_outcome else refreshed
            eff_attrs = (getattr(eff_frame, "attrs", {}) or {}) if eff_frame is not None else {}
            if has_outcome:
                did_write_new = bool(
                    refreshed is not None
                    and len(refreshed) > 0
                    and wrote_this_call
                    and replaced_target
                    and not disk_changed_vs_outcome
                )
            else:
                did_write_new = bool(
                    refreshed is not None
                    and len(refreshed) > 0
                    and replaced_target
                    and (prev_mtime_ms is None or new_mtime_ms is None or new_mtime_ms != prev_mtime_ms)
                )
            now_ms = int(time.time() * 1000)
            _, eff_complete, eff_bar_sched_close = (
                self._daily_bar_status_ms(eff_frame)
                if eff_frame is not None and len(eff_frame) > 0
                else (None, False, None)
            )
            eff_sched_close = (
                eff_bar_sched_close
                if ((did_write_new or disk_changed_vs_outcome) and eff_bar_sched_close is not None)
                else disk_sched_close
            )
            settle_ms = (
                eff_sched_close + MarketDataLoader._POST_CLOSE_SETTLEMENT_MS
                if eff_sched_close is not None
                else None
            )
            if did_write_new and refreshed is not None:
                self._df_cache.pop(sym, None)
                self._partial_bar_obs_ms.pop(sym, None)
                if refreshed_attrs.get("_bar_unsettled") and settle_ms is not None:
                    # Release 300s reservation when write remains unsettled so settlement triggers a prompt refresh
                    self._daily_refresh_attempted.pop(sym, None)
                    if now_ms < settle_ms:
                        delay_s = max(1.0, (settle_ms - now_ms) / 1000.0 + 0.5)
                        self._failed_yf_until[sym] = time.monotonic() + delay_s
                    else:
                        self._failed_yf_until.pop(sym, None)
            elif eff_frame is not None and len(eff_frame) > 0 and eff_complete and pd.Timestamp(eff_frame.index[-1]).normalize() >= expected_latest:
                self._df_cache.pop(sym, None)
                self._partial_bar_obs_ms.pop(sym, None)
                self._failed_yf_until.pop(sym, None)
            elif disk_changed_vs_outcome and eff_frame is not None and len(eff_frame) > 0:
                if settle_ms is not None and (eff_attrs.get("_bar_unsettled") or not eff_complete):
                    self._daily_refresh_attempted.pop(sym, None)
                    if now_ms < settle_ms:
                        delay_s = max(1.0, (settle_ms - now_ms) / 1000.0 + 0.5)
                        self._failed_yf_until[sym] = time.monotonic() + delay_s
                    else:
                        self._failed_yf_until.pop(sym, None)
                else:
                    self._failed_yf_until[sym] = time.monotonic() + self._FAILED_TICKER_COOLDOWN_SECONDS
            elif needs_post_close_refresh or not replaced_target or (has_outcome and not wrote_this_call) or refreshed is None or len(refreshed) == 0:
                pending_settlement = bool(
                    needs_post_close_refresh
                    and eff_frame is not None
                    and len(eff_frame) > 0
                    and settle_ms is not None
                    and attempt_started_ms < settle_ms
                )
                if pending_settlement and settle_ms is not None:
                    if now_ms < settle_ms:
                        # Wait only until post-close settlement window opens rather than full 15m error backoff
                        delay_s = max(1.0, (settle_ms - now_ms) / 1000.0 + 0.5)
                        self._failed_yf_until[sym] = time.monotonic() + delay_s
                    else:
                        # Pre-settlement fetch completed after settlement opened -> allow immediate retry
                        self._failed_yf_until.pop(sym, None)
                else:
                    self._failed_yf_until[sym] = time.monotonic() + self._FAILED_TICKER_COOLDOWN_SECONDS
                if needs_post_close_refresh:
                    # Keep post-close retry eligible cleanly once cooldown clears (#18, #21)
                    self._daily_refresh_attempted.pop(sym, None)

    def _fetch_live_quote_cached(self, sym: str) -> Optional[Dict[str, Any]]:
        """Fetches real-time quote from yfinance with short TTL cache and per-ticker backoff."""
        if not self.enable_live_fetch:
            return None
        now = time.monotonic()
        with self._lock:
            cached = self._live_quote_cache.get(sym)
            if cached is not None and (now - cached[0]) < self._QUOTE_TTL_SECONDS:
                return cached[1]
            if now < self._failed_yf_until.get(f"q:{sym}", 0.0):
                return cached[1] if cached is not None else None

        quote = self.market_loader.fetch_live_quote(sym)
        with self._lock:
            if (
                quote is not None
                and not quote.get("_future_observation_rejected")
                and quote.get("price", 0.0) > 0.0
            ):
                self._live_quote_cache[sym] = (now, quote)
                return quote
            self._failed_yf_until[f"q:{sym}"] = time.monotonic() + 60.0
            return cached[1] if cached is not None else None

    def refresh_tickers_if_needed(self, tickers: List[str]) -> int:
        """Proactively refreshes stale Parquet partitions and quote caches for active symbols."""
        if not self.enable_live_fetch:
            return 0
        updated = 0
        seen = set()
        for raw_sym in tickers:
            sym = raw_sym.upper().strip()
            if not sym or sym in seen:
                continue
            seen.add(sym)
            before_df = self._load_disk_daily(sym)
            before_len = len(before_df) if before_df is not None else 0
            before_last = str(before_df.index[-1]) if before_df is not None and len(before_df) > 0 else ""
            self._maybe_refresh_daily(sym)
            after_df = self._load_disk_daily(sym)
            if after_df is not None and (
                len(after_df) != before_len or (len(after_df) > 0 and str(after_df.index[-1]) != before_last)
            ):
                updated += 1
            self._fetch_live_quote_cached(sym)
        return updated

    def _load_cached_daily_with_quote_ts(
        self,
        ticker: str,
        allow_refresh: bool = True,
        require_quote_ts: bool = False,
    ) -> Tuple[Optional[pd.DataFrame], Optional[int]]:
        """Loads daily Parquet for ticker and returns (df, overlaid_live_quote_timestamp_ms)."""
        sym = ticker.upper()
        if allow_refresh and self.enable_live_fetch:
            self._maybe_refresh_daily(sym)

        df = self._load_disk_daily(sym)
        if df is None or len(df) == 0:
            return None, None

        if not self.enable_live_fetch or not allow_refresh:
            return df, None

        live_q = self._fetch_live_quote_cached(sym)
        if (
            live_q is None
            or live_q.get("_future_observation_rejected")
            or not live_q.get("date")
            or float(live_q.get("price", 0.0)) <= 0.0
        ):
            return df, None

        raw_obs_ts = live_q.get("observation_timestamp_ms")
        if raw_obs_ts is None:
            raw_obs_ts = live_q.get("regularMarketTime")
        q_ts_ms: Optional[int] = None
        if isinstance(raw_obs_ts, (int, float)) and not isinstance(raw_obs_ts, bool):
            f_obs = float(raw_obs_ts)
            if math.isfinite(f_obs) and f_obs > 0.0:
                q_ts_ms = int(f_obs if f_obs >= 1e12 else f_obs * 1000.0)
        elif raw_obs_ts:
            q_ts_ms = self._parse_iso_utc_ms(raw_obs_ts)
        now_ms = int(time.time() * 1000)
        if q_ts_ms is not None and q_ts_ms > now_ms:
            return df, None
        if require_quote_ts and q_ts_ms is None:
            return df, None

        try:
            q_ts = pd.Timestamp(live_q["date"]).normalize()
            last_ts = pd.Timestamp(df.index[-1]).normalize()
            price = float(live_q["price"])
            open_p = float(live_q.get("open") or price)
            high_p = max(float(live_q.get("high") or price), price, open_p)
            low_p = min(float(live_q.get("low") or price), price, open_p)
            vol_q = int(live_q.get("volume") or 0)

            df_live = df.copy()
            obs_ms_out = q_ts_ms
            if q_ts == last_ts:
                bar_obs_ms, bar_is_complete, _ = self._daily_bar_status_ms(df)
                if bar_obs_ms is not None:
                    if obs_ms_out is None or obs_ms_out < bar_obs_ms:
                        return df, None
                elif bar_is_complete and obs_ms_out is None:
                    return df, None
                idx_last = df_live.index[-1]
                if obs_ms_out is not None and bar_obs_ms is None and not bar_is_complete:
                    # Disk bar has no verified observation time: never merge unverified OHLCV into a dated quote
                    df_live.loc[idx_last, "open"] = open_p
                    df_live.loc[idx_last, "high"] = high_p
                    df_live.loc[idx_last, "low"] = low_p
                    df_live.loc[idx_last, "close"] = price
                    df_live.loc[idx_last, "volume"] = float(vol_q)
                    if "vwap" in df_live.columns:
                        df_live.loc[idx_last, "vwap"] = (high_p + low_p + price) / 3.0
                    df_live.attrs.pop("_bar_unsettled", None)
                else:
                    df_live.loc[idx_last, "close"] = price
                    df_live.loc[idx_last, "high"] = max(float(df_live.loc[idx_last, "high"]), high_p)
                    df_live.loc[idx_last, "low"] = min(float(df_live.loc[idx_last, "low"]), low_p)
                    if vol_q > 0:
                        df_live.loc[idx_last, "volume"] = max(float(df_live.loc[idx_last, "volume"]), float(vol_q))
                    if "vwap" in df_live.columns:
                        eff_hi = float(df_live.loc[idx_last, "high"])
                        eff_lo = float(df_live.loc[idx_last, "low"])
                        df_live.loc[idx_last, "vwap"] = (eff_hi + eff_lo + price) / 3.0
                if obs_ms_out is not None:
                    df_live.attrs["observation_timestamp_ms"] = obs_ms_out
                    df_live.attrs["obs_bar_date"] = last_ts.strftime("%Y-%m-%d")
                    df_live.attrs.pop("live_quote_undated", None)
                else:
                    df_live.attrs["obs_bar_date"] = last_ts.strftime("%Y-%m-%d")
                    df_live.attrs["live_quote_undated"] = True
                    df_live.attrs.pop("observation_timestamp_ms", None)
                return df_live, obs_ms_out
            elif q_ts > last_ts:
                orig_attrs = dict(getattr(df, "attrs", {}) or {})
                orig_last_dt = last_ts.strftime("%Y-%m-%d")
                row_prov = dict(orig_attrs.get("_row_provenance") or {})
                if orig_last_dt not in row_prov:
                    orig_matches = orig_attrs.get("obs_bar_date") is None or str(orig_attrs.get("obs_bar_date"))[:10] == orig_last_dt
                    row_prov[orig_last_dt] = {
                        "observation_timestamp_ms": orig_attrs.get("observation_timestamp_ms") if orig_matches else None,
                        "disk_mtime_ms": orig_attrs.get("disk_mtime_ms"),
                        "_bar_unsettled": bool(orig_attrs.get("_bar_unsettled")) if orig_matches else False,
                        "live_quote_undated": bool(orig_attrs.get("live_quote_undated")) if orig_matches else False,
                    }
                new_row_data: Dict[str, Any] = {
                    "open": open_p,
                    "high": high_p,
                    "low": low_p,
                    "close": price,
                    "volume": float(vol_q),
                }
                if "vwap" in df_live.columns:
                    new_row_data["vwap"] = (high_p + low_p + price) / 3.0
                new_row = pd.DataFrame(
                    [new_row_data],
                    index=pd.DatetimeIndex([q_ts], name=df_live.index.name or "timestamp"),
                )
                out_df = pd.concat([df_live, new_row]).sort_index()
                out_df.attrs["_row_provenance"] = row_prov
                out_df.attrs["obs_bar_date"] = pd.Timestamp(out_df.index[-1]).strftime("%Y-%m-%d")
                if obs_ms_out is not None:
                    out_df.attrs["observation_timestamp_ms"] = obs_ms_out
                    out_df.attrs.pop("live_quote_undated", None)
                else:
                    out_df.attrs["live_quote_undated"] = True
                    out_df.attrs.pop("observation_timestamp_ms", None)
                return out_df, obs_ms_out
        except Exception as e:
            logger.debug(f"Live quote overlay skipped: {e}")
        return df, None

    def _load_cached_daily(self, ticker: str, allow_refresh: bool = True) -> Optional[pd.DataFrame]:
        """Loads daily Parquet for ticker, optionally refreshing stale partitions and overlaying live quote."""
        df, _ = self._load_cached_daily_with_quote_ts(ticker, allow_refresh=allow_refresh)
        return df

    def _unavailable_regime_payload(self, with_hmm_meta: bool = False) -> Dict[str, Any]:
        if with_hmm_meta and self.regime_hmm is not None:
            return {
                "available": False,
                "current_state": "UNAVAILABLE",
                "state_index": -1,
                "kappa": None,
                "filtered_probabilities": {},
                "transition_matrix": self.regime_hmm.transmat_.tolist() if getattr(self.regime_hmm, "transmat_", None) is not None else [],
                "state_volatilities": self.regime_hmm.get_state_annualized_volatilities(),
                "stationary_distribution": self.regime_hmm.get_stationary_distribution(),
                "regime_history": [],
            }
        return {
            "available": False,
            "current_state": "UNAVAILABLE",
            "state_index": -1,
            "kappa": None,
            "filtered_probabilities": {},
            "transition_matrix": [],
            "state_volatilities": [],
            "stationary_distribution": [],
            "regime_history": [],
        }

    def _compute_macro_regime_from_df(self, df_spy: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Computes causal forward-filtered Gaussian HMM regime strictly from df_spy without lookahead."""
        if self.regime_hmm is None:
            return self._unavailable_regime_payload(with_hmm_meta=False)
        if df_spy is None or len(df_spy) < 30:
            return self._unavailable_regime_payload(with_hmm_meta=True)

        closes_spy = np.asarray(df_spy["close"].values, dtype=np.float64)
        highs_spy = np.asarray(df_spy["high"].values[1:], dtype=np.float64)
        lows_spy = np.asarray(df_spy["low"].values[1:], dtype=np.float64)
        if (
            not np.all(np.isfinite(closes_spy))
            or np.any(closes_spy <= 0.0)
            or not np.all(np.isfinite(highs_spy))
            or not np.all(np.isfinite(lows_spy))
        ):
            raise ValueError("Non-finite or non-positive SPY OHLC observation")

        spy_ret = np.diff(np.log(closes_spy))
        high_low = np.maximum(highs_spy, 1e-8) / np.maximum(lows_spy, 1e-8)
        spy_vol = (np.log(high_low) ** 2) / (4.0 * np.log(2.0))
        X_hmm = np.column_stack([spy_ret, spy_vol])

        filtered_probs = self.regime_hmm.filter_sequence(X_hmm)
        if filtered_probs is None or len(filtered_probs) == 0:
            raise ValueError("Empty HMM forward posteriors")
        curr_probs = np.asarray(filtered_probs[-1], dtype=np.float64)
        if (
            len(curr_probs) < 3
            or not np.all(np.isfinite(curr_probs[:3]))
            or float(np.sum(curr_probs[:3])) <= 0.0
            or bool(np.any(curr_probs[:3] < 0.0))
        ):
            raise ValueError("Non-finite or invalid HMM posterior distribution")

        curr_probs = curr_probs[:3] / float(np.sum(curr_probs[:3]))
        state_idx = int(np.argmax(curr_probs))
        state_label = self.regime_hmm.STATE_LABELS.get(state_idx, "LOW_VOLATILITY")
        p_low = float(curr_probs[0])
        p_med = float(curr_probs[1])
        p_high = float(curr_probs[2])
        kappa = round(self.regime_hmm.get_regime_multiplier(curr_probs), 2)
        if not math.isfinite(kappa):
            raise ValueError("Non-finite HMM multiplier")
        transmat = (
            self.regime_hmm.transmat_.tolist()
            if getattr(self.regime_hmm, "transmat_", None) is not None
            else []
        )

        hist_slice = filtered_probs[-60:]
        hist_dates = df_spy.index[1:][-60:]
        regime_history = []
        for dt_idx, row_p in zip(hist_dates, hist_slice):
            row_arr = np.asarray(row_p, dtype=np.float64)
            if len(row_arr) < 3 or not np.all(np.isfinite(row_arr[:3])):
                continue
            dt_str = dt_idx.strftime("%Y-%m-%d") if hasattr(dt_idx, "strftime") else str(dt_idx)[:10]
            s_k = int(np.argmax(row_arr[:3]))
            regime_history.append({
                "timestamp": dt_str,
                "state_index": s_k,
                "p_low": round(float(row_arr[0]), 4),
                "p_med": round(float(row_arr[1]), 4),
                "p_high": round(float(row_arr[2]), 4),
                "confidence": round(float(row_arr[s_k]), 4),
            })

        last_idx_r = df_spy.index[-1]
        as_of_dt_r = last_idx_r.strftime("%Y-%m-%d") if hasattr(last_idx_r, "strftime") else str(last_idx_r)[:10]
        spy_obs_ms_r, _, _ = self._daily_bar_status_ms(df_spy)
        return {
            "available": True,
            "current_state": state_label,
            "state_index": state_idx,
            "kappa": float(kappa),
            "as_of_date": as_of_dt_r,
            "observation_timestamp_ms": spy_obs_ms_r,
            "filtered_probabilities": {
                "low_volatility": p_low,
                "medium_volatility": p_med,
                "high_volatility": p_high,
            },
            "transition_matrix": transmat,
            "state_volatilities": self.regime_hmm.get_state_annualized_volatilities(),
            "stationary_distribution": self.regime_hmm.get_stationary_distribution(),
            "regime_history": regime_history,
        }

    @staticmethod
    def _df_tail_digest(d_opt: pd.DataFrame, n_bars: Optional[int] = 252) -> str:
        """Compact blake2b digest of trailing OHLCV timestamps and values."""
        cols = [c for c in ("open", "high", "low", "close", "volume") if c in d_opt.columns]
        tail_sub = d_opt.tail(n_bars) if n_bars is not None else d_opt
        idx_bytes = np.ascontiguousarray(pd.DatetimeIndex(tail_sub.index).asi8).tobytes()
        h = hashlib.blake2b(digest_size=16)
        h.update(idx_bytes)
        if cols:
            val_bytes = np.ascontiguousarray(tail_sub[cols].to_numpy(dtype=np.float64, copy=False)).tobytes()
            h.update(val_bytes)
        return h.hexdigest()

    def get_macro_regime(self) -> Dict[str, Any]:
        """Runs cached causal forward-filtered Gaussian HMM on SPY benchmark."""
        if self.regime_hmm is None:
            return self._unavailable_regime_payload(with_hmm_meta=False)

        try:
            df_spy = self._load_cached_daily("SPY")
            if df_spy is None or len(df_spy) < 30:
                return self._unavailable_regime_payload(with_hmm_meta=True)

            last_idx = df_spy.index[-1]
            last_dt = last_idx.strftime("%Y-%m-%d") if hasattr(last_idx, "strftime") else str(last_idx)[:10]
            tail_sig = self._df_tail_digest(df_spy, n_bars=None)
            spy_obs_ms, spy_complete, _ = self._daily_bar_status_ms(df_spy)
            spy_unsettled = bool((getattr(df_spy, "attrs", {}) or {}).get("_bar_unsettled"))
            cache_key = (len(df_spy), last_dt, tail_sig, spy_obs_ms, spy_complete, spy_unsettled)
            now_mono = time.monotonic()

            with self._lock:
                if (
                    self._regime_cache is not None
                    and self._regime_cache[0] == cache_key
                    and (now_mono - self._regime_cache[1]) < self._REGIME_TTL_SECONDS
                ):
                    return self._regime_cache[2]

            result = self._compute_macro_regime_from_df(df_spy)
            with self._lock:
                self._regime_cache = (cache_key, now_mono, result)
            return result
        except Exception as e:
            logger.warning(f"Error computing live macro regime: {e}")
            return self._unavailable_regime_payload(with_hmm_meta=False)

    @staticmethod
    def _ts_rank(series: pd.Series) -> float:
        """Computes trailing 126-day empirical rank in [-1.0, +1.0]."""
        vals = series.dropna().values[-126:]
        if len(vals) <= 1:
            return 0.0
        last = vals[-1]
        rank = float(np.sum(vals <= last))
        return round(float(np.clip(2.0 * (rank - 1.0) / max(len(vals) - 1.0, 1.0) - 1.0, -1.0, 1.0)), 2)

    def get_latest_quote(self, ticker: str) -> Dict[str, Any]:
        """Returns the latest live quote or last recorded bar from the lakehouse."""
        ticker = ticker.upper()
        if self.enable_live_fetch:
            self._maybe_refresh_daily(ticker)
            live_q = self._fetch_live_quote_cached(ticker)
            if live_q is not None and float(live_q.get("price", 0.0)) > 0.0:
                return live_q

        df = self._load_disk_daily(ticker)
        if df is None or len(df) == 0:
            return {
                "ticker": ticker,
                "price": 0.0,
                "change": 0.0,
                "change_pct": 0.0,
                "volume": 0,
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "timestamp_ms": int(time.time() * 1000),
            }
        close_p = float(df["close"].iloc[-1])
        prev_c = float(df["close"].iloc[-2]) if len(df) >= 2 else float(df["open"].iloc[-1])
        chg = round(close_p - prev_c, 2)
        chg_pct = round(((close_p - prev_c) / prev_c) * 100.0, 2) if prev_c > 0 else 0.0
        vol = int(df["volume"].iloc[-1])
        last_idx = df.index[-1]
        bar_date = last_idx.strftime("%Y-%m-%d") if hasattr(last_idx, "strftime") else str(last_idx)[:10]
        bar_obs_ms = self._market_bar_timestamp_ms(df, None)
        now_ms = int(time.time() * 1000)
        eff_obs_ms = bar_obs_ms if (bar_obs_ms is not None and bar_obs_ms <= now_ms) else None
        quote_fb: Dict[str, Any] = {
            "ticker": ticker,
            "price": round(close_p, 2),
            "change": chg,
            "change_pct": chg_pct,
            "volume": vol,
            "date": bar_date,
            "timestamp_ms": eff_obs_ms if eff_obs_ms is not None else now_ms,
        }
        if eff_obs_ms is not None:
            quote_fb["observation_timestamp_ms"] = eff_obs_ms
        return quote_fb

    def get_signal(self, ticker: str, regime: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extracts trailing microstructure/residual features, runs LightGBM inference, and computes Kelly sizing."""
        ticker = ticker.upper()
        caller_supplied_regime = regime is not None
        regime = regime or self.get_macro_regime()
        raw_kappa = regime.get("kappa")
        regime_state = str(regime.get("current_state", "UNAVAILABLE"))
        regime_idx = int(regime.get("state_index", -1))
        regime_probs = regime.get("filtered_probabilities") or {}
        kappa = resolve_regime_kappa(raw_kappa, regime_probs)
        transmat = regime.get("transition_matrix")

        try:
            sector_etf = self.universe_selector.get_sector_etf(ticker)
        except KeyError:
            sector_etf = "SPY"

        pre_spy_pair: Optional[Tuple[Optional[pd.DataFrame], Optional[int]]] = None
        pre_sec_pair: Optional[Tuple[Optional[pd.DataFrame], Optional[int]]] = None
        if self.enable_live_fetch and self.is_ready():
            try:
                # Refresh/quote mandatory SPY benchmark before target so active-session regularMarketTime stays <= target
                if ticker != "SPY":
                    pre_spy_pair = self._load_cached_daily_with_quote_ts("SPY")
            except Exception as e:
                logger.error(f"Error loading benchmark data for {ticker}: {e}")
                return build_qare_signal_frame(
                    ticker=ticker,
                    action="HOLD",
                    conviction_tier="LOW",
                    p_buy=0.0,
                    p_hold=1.0,
                    p_sell=0.0,
                    unconstrained_kelly_pct=0.0,
                    l1_normalized_pct=0.0,
                    entry_price=0.0,
                    dynamic_take_profit=0.0,
                    dynamic_stop_loss=0.0,
                    reward_to_risk_ratio=0.0,
                    regime_state=regime_state,
                    regime_index=regime_idx,
                    regime_probs=regime_probs,
                    regime_kappa=kappa,
                    transition_matrix=transmat,
                    signal_breakdown={},
                    winning_engine="NONE",
                    execution_mode="UNREADY",
                    missing_modalities=["LIVE_SIGNAL_EXCEPTION"],
                    intraday_mode="UNAVAILABLE",
                    xbrl_filed_date_verified=False,
                )
            if sector_etf not in (ticker, "SPY"):
                try:
                    pre_sec_pair = self._load_cached_daily_with_quote_ts(sector_etf)
                except Exception as sec_e:
                    logger.debug(f"Sector ETF {sector_etf} preload failed; falling back to SPY: {sec_e}")
                    pre_sec_pair = (None, None)

        df, overlaid_q_ms = self._load_cached_daily_with_quote_ts(ticker)
        if df is None or len(df) == 0:
            return build_qare_signal_frame(
                ticker=ticker,
                action="HOLD",
                conviction_tier="LOW",
                p_buy=0.0,
                p_hold=1.0,
                p_sell=0.0,
                unconstrained_kelly_pct=0.0,
                l1_normalized_pct=0.0,
                entry_price=0.0,
                dynamic_take_profit=0.0,
                dynamic_stop_loss=0.0,
                reward_to_risk_ratio=0.0,
                regime_state=regime_state,
                regime_index=regime_idx,
                regime_probs=regime_probs,
                regime_kappa=kappa,
                transition_matrix=transmat,
                signal_breakdown={},
                winning_engine="NONE",
                execution_mode="UNREADY",
                missing_modalities=["PRICE_HISTORY_UNAVAILABLE"],
                intraday_mode="UNAVAILABLE",
                xbrl_filed_date_verified=False,
            )

        last_idx = df.index[-1]
        last_dt = last_idx.strftime("%Y-%m-%d") if hasattr(last_idx, "strftime") else str(last_idx)[:10]
        raw_entry = float(df["close"].iloc[-1])
        entry_price = round(raw_entry, 2) if math.isfinite(raw_entry) and raw_entry > 0.0 else 0.0
        kappa_key = round(kappa, 3) if kappa is not None else -1.0
        now_mono = time.monotonic()

        tail_closes = np.asarray(df["close"].values[-21:], dtype=np.float64)
        valid_closes = (
            entry_price > 0.0
            and len(df) >= 22
            and len(tail_closes) >= 21
            and np.all(np.isfinite(tail_closes))
            and np.all(tail_closes > 0.0)
        )
        has_real_vol = bool(valid_closes)
        if has_real_vol:
            std_1d = float(np.std(np.diff(np.log(tail_closes))))
            vol_1d = max(std_1d, 0.005) if math.isfinite(std_1d) else 0.005
            sigma_fallback = max(float(math.sqrt(10.0) * vol_1d), 0.01)
            tp = round(entry_price * (1.0 + 2.0 * sigma_fallback), 2)
            sl = round(entry_price * (1.0 - 1.0 * sigma_fallback), 2)
            rr_val = 2.0
        else:
            vol_1d = None
            tp = 0.0
            sl = 0.0
            rr_val = 0.0

        min_history_bars = self.micro_extractor.lookback_window + 1
        if not self.is_ready() or len(df) < min_history_bars or not has_real_vol:
            return build_qare_signal_frame(
                ticker=ticker,
                action="HOLD",
                conviction_tier="LOW",
                p_buy=0.0,
                p_hold=1.0,
                p_sell=0.0,
                unconstrained_kelly_pct=0.0,
                l1_normalized_pct=0.0,
                entry_price=entry_price,
                dynamic_take_profit=tp,
                dynamic_stop_loss=sl,
                reward_to_risk_ratio=rr_val,
                regime_state=regime_state,
                regime_index=regime_idx,
                regime_probs=regime_probs,
                regime_kappa=kappa,
                transition_matrix=transmat,
                signal_breakdown={},
                winning_engine="NONE",
                execution_mode="UNREADY",
                missing_modalities=["MODEL_WEIGHTS_OR_HISTORY_MISSING"],
                intraday_mode="UNAVAILABLE",
                xbrl_filed_date_verified=False,
            )

        try:
            if ticker == "SPY":
                df_spy_raw, spy_q_ms = df, overlaid_q_ms
            elif pre_spy_pair is not None:
                df_spy_raw, spy_q_ms = pre_spy_pair
            else:
                df_spy_raw, spy_q_ms = self._load_cached_daily_with_quote_ts("SPY")

            if sector_etf == ticker:
                df_sec_raw, sec_q_ms = df, overlaid_q_ms
            elif sector_etf == "SPY":
                df_sec_raw, sec_q_ms = df_spy_raw, spy_q_ms
            elif pre_sec_pair is not None:
                df_sec_raw, sec_q_ms = pre_sec_pair
            else:
                df_sec_raw, sec_q_ms = self._load_cached_daily_with_quote_ts(sector_etf)

            def _df_obs_sig(d_opt: Optional[pd.DataFrame], q_ms: Optional[int]) -> Tuple[int, str, str, int, bool, int, bool, bool]:
                if d_opt is None or len(d_opt) == 0:
                    return (0, "", "", 0, False, 0, False, False)
                li = d_opt.index[-1]
                dt_s = li.strftime("%Y-%m-%d") if hasattr(li, "strftime") else str(li)[:10]
                tail_digest = self._df_tail_digest(d_opt, n_bars=None)
                bar_obs_ms, is_complete, _ = self._daily_bar_status_ms(d_opt)
                eff_obs = q_ms or bar_obs_ms or 0
                eff_complete = bool(is_complete or q_ms is not None)
                attrs = getattr(d_opt, "attrs", {}) or {}
                row_prov = attrs.get("_row_provenance")
                if isinstance(row_prov, dict) and row_prov:
                    h_prov = hashlib.blake2b(digest_size=16)
                    h_prov.update(tail_digest.encode("ascii"))
                    for pk in sorted(row_prov.keys()):
                        pv = row_prov[pk]
                        if isinstance(pv, dict):
                            p_obs = pv.get("observation_timestamp_ms")
                            p_dm = pv.get("disk_mtime_ms")
                            h_prov.update(
                                f"{pk}:{p_obs}:{p_dm}:{bool(pv.get('_bar_unsettled'))}:{bool(pv.get('live_quote_undated'))};".encode(
                                    "utf-8"
                                )
                            )
                    tail_digest = h_prov.hexdigest()
                raw_dm = attrs.get("disk_mtime_ms")
                disk_mtime_ms = int(raw_dm) if isinstance(raw_dm, (int, float)) and not isinstance(raw_dm, bool) else 0
                undated_q = bool(attrs.get("live_quote_undated"))
                unsettled_b = bool(attrs.get("_bar_unsettled"))
                return (len(d_opt), dt_s, tail_digest, eff_obs, eff_complete, disk_mtime_ms, undated_q, unsettled_b)

            norm_probs_sig = tuple(sorted(normalize_regime_probs(regime_probs).items()))
            eff_kappa_sig = resolve_regime_kappa(kappa, regime_probs)
            transmat_sig = repr(_sanitize_json_obj(transmat) if transmat is not None else [])
            target_sig = _df_obs_sig(df, overlaid_q_ms)
            reg_as_of_date = str(regime.get("as_of_date") or "")[:10]
            reg_obs_ms = regime.get("observation_timestamp_ms")
            reg_hist = regime.get("regime_history") or []
            reg_last_str = (
                str(reg_hist[-1].get("timestamp", ""))[:10]
                if reg_hist and isinstance(reg_hist[-1], dict)
                else ""
            )
            reg_obs_sig = (
                int(reg_obs_ms)
                if isinstance(reg_obs_ms, (int, float)) and not isinstance(reg_obs_ms, bool)
                else None
            )

            with self._lock:
                cached_news = self._news_cache.get(ticker)
                cached_edgar = self._edgar_cache.get(ticker)
                cached_pg = self._peer_graph_cache.get(ticker)
                alt_sig = (
                    cached_news[0] if cached_news is not None else 0.0,
                    cached_edgar[0] if cached_edgar is not None else 0.0,
                    cached_pg[0] if cached_pg is not None else 0.0,
                    target_sig,
                    _df_obs_sig(df_spy_raw, spy_q_ms),
                    _df_obs_sig(df_sec_raw, sec_q_ms),
                    regime_state,
                    norm_probs_sig,
                    eff_kappa_sig,
                    transmat_sig,
                    reg_as_of_date,
                    reg_obs_sig,
                    reg_last_str,
                    bool(self.enable_live_fetch),
                    bool(caller_supplied_regime),
                )
                cache_key = (last_dt, entry_price, regime_idx, kappa_key, alt_sig)
                cached = self._signal_cache.get(ticker)
                if cached is not None and cached[0] == cache_key and (now_mono - cached[1]) < self._SIGNAL_TTL_SECONDS:
                    return cached[2]

            target_bar_obs_ms, target_complete, target_sched_close = self._daily_bar_status_ms(df)
            target_obs_ms = overlaid_q_ms if (overlaid_q_ms is not None and overlaid_q_ms > 0) else target_bar_obs_ms
            target_eff_complete = bool(target_complete or (overlaid_q_ms is not None and overlaid_q_ms > 0))
            target_attrs = getattr(df, "attrs", {}) or {}
            target_last_str = pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d")
            target_attrs_match = (
                target_attrs.get("obs_bar_date") is None
                or str(target_attrs.get("obs_bar_date"))[:10] == target_last_str
            )
            has_explicit_target_obs = bool(
                (overlaid_q_ms is not None and overlaid_q_ms > 0)
                or (
                    target_attrs_match
                    and not target_attrs.get("live_quote_undated")
                    and target_attrs.get("observation_timestamp_ms") is not None
                )
            )
            target_event_obs_ms = (
                target_obs_ms
                if (has_explicit_target_obs or target_obs_ms is None or target_sched_close is None)
                else min(target_obs_ms, target_sched_close)
            )
            target_undated_quote = bool(target_attrs.get("live_quote_undated"))
            target_unsettled = bool(target_attrs.get("_bar_unsettled"))
            target_mtime_raw = target_attrs.get("disk_mtime_ms")
            target_mtime_ms = int(target_mtime_raw) if isinstance(target_mtime_raw, (int, float)) else None
            target_last_norm = pd.Timestamp(df.index[-1]).normalize()

            spy_aligned = False
            sec_aligned = False
            bench_timestamps_verified = False
            spy_obs_ms: Optional[int] = None
            sec_obs_ms: Optional[int] = None
            df_sub = df.tail(self._TAIL_WARMUP_BARS)
            spy_sub = df_sub
            sec_sub = df_sub

            if df_spy_raw is not None and len(df_spy_raw) > 0:
                common_spy = df.index.intersection(df_spy_raw.index)
                if len(common_spy) >= min_history_bars and pd.Timestamp(common_spy[-1]).normalize() == target_last_norm:
                    spy_tail_idx = common_spy[-self._TAIL_WARMUP_BARS:]
                    spy_cand = self._slice_aligned_frame(df_spy_raw, spy_tail_idx)
                    spy_tail_last = pd.Timestamp(spy_tail_idx[-1]).normalize()
                    spy_q = (
                        spy_q_ms
                        if (spy_q_ms is not None and spy_tail_last == pd.Timestamp(df_spy_raw.index[-1]).normalize())
                        else None
                    )
                    spy_bar_obs_ms, spy_complete, _ = self._daily_bar_status_ms(spy_cand)
                    spy_obs_ms = spy_q if (spy_q is not None and spy_q > 0) else spy_bar_obs_ms
                    spy_eff_complete = bool(spy_complete or (spy_q is not None and spy_q > 0))
                    spy_attrs = getattr(spy_cand, "attrs", {}) or {}
                    spy_undated_quote = bool(spy_attrs.get("live_quote_undated"))
                    spy_unsettled = bool(spy_attrs.get("_bar_unsettled"))
                    spy_mtime_raw = spy_attrs.get("disk_mtime_ms")
                    spy_mtime_ms = int(spy_mtime_raw) if isinstance(spy_mtime_raw, (int, float)) else None

                    ts_ok = target_obs_ms is not None and spy_obs_ms is not None and spy_obs_ms <= target_obs_ms
                    offline_batch_ok = (
                        not self.enable_live_fetch
                        and overlaid_q_ms is None
                        and spy_q is None
                        and not target_undated_quote
                        and not spy_undated_quote
                        and not target_unsettled
                        and not spy_unsettled
                        and target_obs_ms is None
                        and spy_obs_ms is None
                        and not target_eff_complete
                        and not spy_eff_complete
                        and target_mtime_ms is not None
                        and spy_mtime_ms is not None
                        and spy_mtime_ms <= target_mtime_ms
                    )
                    if ts_ok or offline_batch_ok:
                        spy_aligned = True
                        bench_timestamps_verified = bool(ts_ok)
                        df_sub = self._slice_aligned_frame(df, spy_tail_idx)
                        spy_sub = spy_cand

                        if sector_etf == ticker:
                            sec_aligned = True
                            sec_sub = df_sub
                        elif sector_etf == "SPY" or df_sec_raw is None or len(df_sec_raw) == 0:
                            sec_aligned = sector_etf == "SPY"
                            sec_sub = spy_sub
                        else:
                            common_all = common_spy.intersection(df_sec_raw.index)
                            if len(common_all) >= min_history_bars and pd.Timestamp(common_all[-1]).normalize() == target_last_norm:
                                all_tail_idx = common_all[-self._TAIL_WARMUP_BARS:]
                                sec_cand = self._slice_aligned_frame(df_sec_raw, all_tail_idx)
                                all_tail_last = pd.Timestamp(all_tail_idx[-1]).normalize()
                                sec_q = (
                                    sec_q_ms
                                    if (sec_q_ms is not None and all_tail_last == pd.Timestamp(df_sec_raw.index[-1]).normalize())
                                    else None
                                )
                                sec_bar_obs_ms, sec_complete, _ = self._daily_bar_status_ms(sec_cand)
                                sec_obs_ms = sec_q if (sec_q is not None and sec_q > 0) else sec_bar_obs_ms
                                sec_eff_complete = bool(sec_complete or (sec_q is not None and sec_q > 0))
                                sec_attrs = getattr(sec_cand, "attrs", {}) or {}
                                sec_undated_quote = bool(sec_attrs.get("live_quote_undated"))
                                sec_unsettled = bool(sec_attrs.get("_bar_unsettled"))
                                sec_mtime_raw = sec_attrs.get("disk_mtime_ms")
                                sec_mtime_ms = int(sec_mtime_raw) if isinstance(sec_mtime_raw, (int, float)) else None
                                sec_ts_ok = target_obs_ms is not None and sec_obs_ms is not None and sec_obs_ms <= target_obs_ms
                                sec_offline_ok = (
                                    offline_batch_ok
                                    and sec_q is None
                                    and not sec_undated_quote
                                    and not sec_unsettled
                                    and sec_obs_ms is None
                                    and not sec_eff_complete
                                    and sec_mtime_ms is not None
                                    and target_mtime_ms is not None
                                    and sec_mtime_ms <= target_mtime_ms
                                )
                                if sec_ts_ok or sec_offline_ok:
                                    sec_aligned = True
                                    bench_timestamps_verified = bench_timestamps_verified and bool(sec_ts_ok)
                                    df_sub = self._slice_aligned_frame(df, all_tail_idx)
                                    spy_sub = self._slice_aligned_frame(df_spy_raw, all_tail_idx)
                                    sec_sub = sec_cand
                            if not sec_aligned:
                                sec_sub = spy_sub
                                bench_timestamps_verified = False
                        if not sec_aligned and sector_etf not in (ticker, "SPY"):
                            bench_timestamps_verified = False

            has_regime_prov = bool(
                not caller_supplied_regime or reg_as_of_date or reg_obs_ms is not None or reg_last_str
            )
            if (
                has_regime_prov
                and self.regime_hmm is not None
                and df_spy_raw is not None
                and len(df_spy_raw) > 0
            ):
                spy_asof_norm = (
                    pd.Timestamp(spy_sub.index[-1]).normalize()
                    if (spy_aligned and len(spy_sub) > 0)
                    else target_last_norm
                )
                raw_spy_last_norm = pd.Timestamp(df_spy_raw.index[-1]).normalize()
                spy_asof_str = spy_asof_norm.strftime("%Y-%m-%d")
                prov_date_ahead = bool(
                    raw_spy_last_norm > spy_asof_norm
                    or (reg_as_of_date and reg_as_of_date > spy_asof_str)
                    or (reg_last_str and reg_last_str > spy_asof_str)
                )
                prov_obs_ahead = bool(
                    isinstance(reg_obs_ms, (int, float))
                    and not isinstance(reg_obs_ms, bool)
                    and (target_obs_ms is None or int(reg_obs_ms) > target_obs_ms)
                )
                reg_eff_date = reg_as_of_date or reg_last_str
                prov_same_day_unverified = bool(
                    target_obs_ms is not None
                    and reg_obs_sig is None
                    and (not caller_supplied_regime or not reg_eff_date or reg_eff_date >= spy_asof_str)
                )
                if prov_date_ahead or prov_obs_ahead or prov_same_day_unverified:
                    try:
                        if target_obs_ms is None and (prov_obs_ahead or self.enable_live_fetch or target_unsettled):
                            df_spy_asof = None
                        else:
                            cutoff_positions = np.flatnonzero(df_spy_raw.index.normalize() <= spy_asof_norm)
                            raw_last_ts_str = pd.Timestamp(df_spy_raw.index[-1]).strftime("%Y-%m-%d")
                            raw_spy_attrs = getattr(df_spy_raw, "attrs", {}) or {}
                            search_now_ms = int(time.time() * 1000)
                            selected_len = 0
                            for j in range(len(cutoff_positions) - 1, -1, -1):
                                cand_ts = df_spy_raw.index[int(cutoff_positions[j])]
                                cand_dt_str = pd.Timestamp(cand_ts).strftime("%Y-%m-%d")
                                cand_attrs = (
                                    raw_spy_attrs
                                    if cand_dt_str == raw_last_ts_str
                                    else self._sliced_endpoint_attrs(df_spy_raw, cand_dt_str)
                                )
                                ret_obs_ms, _, _ = self._bar_status_ms_for_end(
                                    cand_ts, cand_attrs, now_ms=search_now_ms
                                )
                                ret_unsettled = bool(cand_attrs.get("_bar_unsettled"))
                                ret_undated = bool(cand_attrs.get("live_quote_undated"))
                                ret_raw_dm = cand_attrs.get("disk_mtime_ms")
                                ret_disk_mtime = (
                                    int(ret_raw_dm)
                                    if isinstance(ret_raw_dm, (int, float)) and not isinstance(ret_raw_dm, bool)
                                    else None
                                )
                                ret_norm = pd.Timestamp(cand_ts).normalize()
                                if ret_unsettled or ret_undated:
                                    continue
                                if target_obs_ms is not None:
                                    if ret_obs_ms is not None and ret_obs_ms <= target_obs_ms:
                                        selected_len = j + 1
                                        break
                                    if (
                                        ret_obs_ms is None
                                        and ret_disk_mtime is not None
                                        and ret_disk_mtime <= target_obs_ms
                                        and ret_norm <= target_last_norm
                                    ):
                                        selected_len = j + 1
                                        break
                                elif (
                                    not prov_obs_ahead
                                    and not self.enable_live_fetch
                                    and not target_unsettled
                                    and ret_norm <= target_last_norm
                                    and (
                                        target_mtime_ms is None
                                        or (ret_disk_mtime is not None and ret_disk_mtime <= target_mtime_ms)
                                    )
                                ):
                                    selected_len = j + 1
                                    break
                            df_spy_asof = (
                                self._slice_aligned_frame(df_spy_raw, df_spy_raw.index[cutoff_positions[:selected_len]])
                                if selected_len > 0
                                else None
                            )
                        asof_reg = self._compute_macro_regime_from_df(df_spy_asof)
                        regime_state = str(asof_reg.get("current_state", "UNAVAILABLE"))
                        regime_idx = int(asof_reg.get("state_index", -1))
                        regime_probs = asof_reg.get("filtered_probabilities") or {}
                        kappa = resolve_regime_kappa(asof_reg.get("kappa"), regime_probs)
                        transmat = asof_reg.get("transition_matrix")
                    except Exception:
                        regime_state = "UNAVAILABLE"
                        regime_idx = -1
                        regime_probs = {}
                        kappa = None
                        transmat = []

            if not spy_aligned:
                unaligned_frame = build_qare_signal_frame(
                    ticker=ticker,
                    action="HOLD",
                    conviction_tier="LOW",
                    p_buy=0.0,
                    p_hold=1.0,
                    p_sell=0.0,
                    unconstrained_kelly_pct=0.0,
                    l1_normalized_pct=0.0,
                    entry_price=entry_price,
                    dynamic_take_profit=tp,
                    dynamic_stop_loss=sl,
                    reward_to_risk_ratio=rr_val,
                    regime_state=regime_state,
                    regime_index=regime_idx,
                    regime_probs=regime_probs,
                    regime_kappa=kappa,
                    transition_matrix=transmat,
                    signal_breakdown={},
                    winning_engine="NONE",
                    execution_mode="UNREADY",
                    missing_modalities=["BENCHMARK_OBSERVATIONS_UNALIGNED", "DIVERGENCE_OBSERVATIONS_UNALIGNED"],
                    intraday_mode="UNAVAILABLE",
                    xbrl_filed_date_verified=False,
                )
                with self._lock:
                    self._signal_cache[ticker] = (cache_key, now_mono, unaligned_frame)
                return unaligned_frame

            has_aligned_benchmarks = spy_aligned
            bench_not_ahead = bench_timestamps_verified

            r_stock = np.diff(np.log(df_sub["close"].values))
            r_spy = np.diff(np.log(spy_sub["close"].values))
            r_sec = np.diff(np.log(sec_sub["close"].values))

            roll_res = self.residualizer.compute_rolling_residuals(r_stock, r_spy, r_sec)
            residuals_series = pd.Series(roll_res.residuals, index=df_sub.index[1:])

            channels_df = self.micro_extractor.compute_raw_channels_df(df_sub.iloc[1:], residuals=residuals_series)
            lgbm_feats = self.micro_extractor.build_lgbm_features(channels_df)

            if len(lgbm_feats) == 0:
                raise ValueError(f"Empty microstructure feature matrix for {ticker}")

            raw_probs = np.asarray(self.lgbm_backbone.predict_proba(lgbm_feats[-1:])[0], dtype=np.float64)
            raw_sum = float(np.sum(raw_probs))
            if raw_sum <= 0 or not np.all(np.isfinite(raw_probs)):
                raise ValueError(f"Non-finite LightGBM probabilities for {ticker}")
            raw_probs = raw_probs / raw_sum

            cal_probs = np.asarray(self.temp_scaler.transform(raw_probs.reshape(1, -1))[0], dtype=np.float64)
            cal_sum = float(np.sum(cal_probs[:3])) if len(cal_probs) >= 3 else 0.0
            if len(cal_probs) < 3 or cal_sum <= 0.0 or not np.all(np.isfinite(cal_probs[:3])):
                raise ValueError(f"Non-finite calibrated probabilities for {ticker}")
            cal_probs = cal_probs[:3] / cal_sum

            p_buy, p_hold, p_sell = float(cal_probs[0]), float(cal_probs[1]), float(cal_probs[2])

            eff_kelly_kappa = kappa if kappa is not None else 0.25
            assert vol_1d is not None
            valid_res = roll_res.residuals[np.isfinite(roll_res.residuals)]
            if has_aligned_benchmarks and len(valid_res) >= 20:
                rms_20 = float(np.sqrt(np.mean(valid_res[-20:] ** 2)))
                sigma_10d = max(float(math.sqrt(10.0) * rms_20), 0.01)
            else:
                sigma_10d = max(float(math.sqrt(10.0) * vol_1d), 0.01)

            b_vert = 0.0
            if self._class_priors is not None and self._class_priors[1] > 0.0:
                p0_b, p0_h, p0_s = self._class_priors
                b_vert = (p0_s * sigma_10d - p0_b * (2.0 * sigma_10d)) / p0_h

            rec = self.kelly_sizer.compute_recommendation(
                p_buy=p_buy,
                p_hold=p_hold,
                p_sell=p_sell,
                b_upper=2.0 * sigma_10d,
                b_lower=1.0 * sigma_10d,
                b_vert=b_vert,
                entry_price=entry_price,
                kappa=eff_kelly_kappa,
                prior_probs=self._class_priors,
                execution_mode="LONG_ONLY",
            )

            res_mom_rank = self._ts_rank(channels_df["residual_momentum_20d"])
            on_ret_rank = self._ts_rank(channels_df["overnight_log_return"])
            intra_ret_rank = self._ts_rank(channels_df["intraday_log_return"])
            amihud_rank = self._ts_rank(channels_df["amihud_illiquidity_z"])
            cp_rank = self._ts_rank(channels_df["closing_power"])

            # Populate only authentic alt-data/peer values if already cached; mark missing honestly
            missing_modalities: List[str] = []
            if not sec_aligned and sector_etf not in (ticker, "SPY"):
                missing_modalities.append("SECTOR_BENCHMARK_UNALIGNED")
            if kappa is None:
                missing_modalities.append("REGIME_HMM_UNAVAILABLE")

            target_bar_date = pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d") if len(df) > 0 else ""

            def _is_news_item_pit_eligible(it: Dict[str, Any], obs_ms: Optional[int], bar_dt: str) -> bool:
                if not it.get("published_at"):
                    return False
                pub_raw = str(it["published_at"]).strip()
                if "T" in pub_raw or ":" in pub_raw:
                    if it.get("published_at_verified") is False:
                        return False
                    pub_us = self._parse_iso_utc_us(pub_raw)
                    return pub_us is not None and obs_ms is not None and pub_us <= (obs_ms * 1000)
                if len(pub_raw) != 10 or not bar_dt:
                    return False
                try:
                    date.fromisoformat(pub_raw)
                except ValueError:
                    return False
                if it.get("published_at_verified") is False:
                    return pub_raw < bar_dt
                return pub_raw <= bar_dt

            def _news_item_score_at_target(it: Dict[str, Any], obs_ms: Optional[int], bar_dt: str) -> Optional[float]:
                if not _is_news_item_pit_eligible(it, obs_ms, bar_dt):
                    return None
                raw_score = it.get("finbert_score")
                if (
                    isinstance(raw_score, (int, float))
                    and not isinstance(raw_score, bool)
                    and math.isfinite(float(raw_score))
                ):
                    pub_raw = str(it["published_at"]).strip()
                    if "T" in pub_raw or ":" in pub_raw:
                        pub_us = self._parse_iso_utc_us(pub_raw)
                        if pub_us is None or obs_ms is None:
                            return None
                        age_days = max(0.0, ((obs_ms * 1000) - pub_us) / 86_400_000_000.0)
                    else:
                        if obs_ms is not None:
                            d_pub_s = datetime.fromisoformat(pub_raw).replace(tzinfo=timezone.utc).timestamp()
                            age_days = max(0.0, ((obs_ms / 1000.0) - d_pub_s) / 86400.0)
                        else:
                            age_days = float(max(0, (date.fromisoformat(bar_dt) - date.fromisoformat(pub_raw)).days))
                    return round(
                        EventDecayKernel.compute_decay(float(raw_score), age_days, EventDecayKernel.TAU_NEWS),
                        2,
                    )
                dec_fallback = it.get("decayed_sentiment")
                if (
                    isinstance(dec_fallback, (int, float))
                    and not isinstance(dec_fallback, bool)
                    and math.isfinite(float(dec_fallback))
                ):
                    return float(dec_fallback)
                return None

            if cached_news is not None and cached_news[1]:
                news_scores = [
                    sc
                    for it in cached_news[1]
                    if isinstance(it, dict)
                    for sc in (_news_item_score_at_target(it, target_event_obs_ms, target_bar_date),)
                    if sc is not None
                ]
                if news_scores:
                    finbert_val = round(float(np.mean(news_scores)), 2)
                else:
                    finbert_val = 0.0
                    missing_modalities.append("FINBERT_NEWS_NOT_LOADED")
            else:
                finbert_val = 0.0
                missing_modalities.append("FINBERT_NEWS_NOT_LOADED")

            form4_val = 0.0
            legal_delta = 0.0
            mda_cat = 0.0
            xbrl_filing_verified = False
            if cached_edgar is not None and cached_edgar[1].get("available"):
                edg = cached_edgar[1]
                rep_acc_raw = edg.get("current_acceptance_datetime")
                rep_us = self._parse_iso_utc_us(rep_acc_raw) if rep_acc_raw else None
                report_pit_ok = False
                if rep_us is not None:
                    if target_event_obs_ms is not None and rep_us <= (target_event_obs_ms * 1000):
                        rep_ts = pd.Timestamp(rep_us, unit="us", tz="UTC")
                        rep_ny_d = rep_ts.tz_convert("America/New_York").date()
                        rep_cutoff = MarketDataLoader.nyse_session_close_time(rep_ny_d).strftime("%H:%M")
                        rep_eff_d = MarketDataLoader.apply_pit_cutoff(rep_ts, cutoff_time=rep_cutoff).date()
                        rep_ref_d = df.index[-1].date() if hasattr(df.index[-1], "date") else date.fromisoformat(last_dt)
                        report_pit_ok = rep_eff_d <= rep_ref_d
                else:
                    cur_fd = str(edg.get("current_filing_date") or "")[:10]
                    rep_ref_s = df.index[-1].strftime("%Y-%m-%d") if hasattr(df.index[-1], "strftime") else str(last_dt)[:10]
                    if cur_fd and rep_ref_s and cur_fd < rep_ref_s:
                        report_pit_ok = True
                xbrl_filing_verified = bool(rep_us is not None and report_pit_ok)

                if report_pit_ok and edg.get("item1a_sentiment_drift") is not None:
                    legal_delta = float(edg["item1a_sentiment_drift"])
                else:
                    missing_modalities.append("EDGAR_ITEM1A_DIFF")
                if report_pit_ok and edg.get("mda_catalyst_score") is not None:
                    mda_cat = float(edg["mda_catalyst_score"])
                txs = edg.get("form4_transactions") or []
                if txs:
                    ref_d = df.index[-1].date() if hasattr(df.index[-1], "date") else date.fromisoformat(last_dt)
                    signed_v = 0.0
                    scored_dated_tx = False
                    has_unverified_ps = False
                    has_ambiguous_f4 = False
                    eligible_f4_rows: List[Dict[str, Any]] = []
                    for tx in txs:
                        if not isinstance(tx, dict):
                            continue
                        code = (
                            FundamentalsClient.classify_insider_tx(str(tx["raw_text"]))
                            if tx.get("raw_text")
                            else str(tx.get("transaction_code") or "").upper()
                        )
                        disc_raw = tx.get("disclosed_at") or tx.get("acceptance_datetime")
                        disc_str = str(disc_raw or "").strip()
                        has_disc_time = bool(disc_str and ("T" in disc_str or ":" in disc_str))
                        disc_verified = bool(tx.get("disclosed_at_verified", has_disc_time)) and has_disc_time
                        disc_us = self._parse_iso_utc_us(disc_str) if disc_verified else None
                        if disc_us is None:
                            if code in ("P", "S"):
                                has_unverified_ps = True
                            continue
                        if target_event_obs_ms is None or disc_us > (target_event_obs_ms * 1000):
                            continue
                        disc_ts_us = pd.Timestamp(disc_us, unit="us", tz="UTC")
                        disc_ny_d = disc_ts_us.tz_convert("America/New_York").date()
                        eff_cutoff = MarketDataLoader.nyse_session_close_time(disc_ny_d).strftime("%H:%M")
                        tx_eff_d = MarketDataLoader.apply_pit_cutoff(
                            disc_ts_us,
                            cutoff_time=eff_cutoff,
                        ).date()
                        if tx_eff_d > ref_d:
                            continue
                        eligible_f4_rows.append({
                            **tx,
                            "_resolved_code": code,
                            "_tx_eff_d": tx_eff_d,
                        })

                    resolved_f4 = FundamentalsClient.resolve_pit_form4_transactions(eligible_f4_rows)
                    for tx in resolved_f4:
                        if tx.get("_amendment_ambiguous"):
                            has_ambiguous_f4 = True
                            continue
                        if (
                            tx.get("_amendment_unresolved")
                            or str(tx.get("amendment_action") or "").strip().upper() == "CANCEL"
                        ):
                            continue
                        code = str(tx.get("_resolved_code") or tx.get("transaction_code") or "").upper()
                        if code == "P":
                            dir_w = 1.0
                        elif code == "S":
                            dir_w = -0.5
                        else:
                            continue
                        tx_eff_d = tx["_tx_eff_d"]
                        scored_dated_tx = True
                        age_days = float((ref_d - tx_eff_d).days)
                        decay_w = EventDecayKernel.compute_decay(1.0, age_days, EventDecayKernel.TAU_INSIDER)
                        plan_w = 0.2 if tx.get("is_10b5_1") else 1.0
                        tx_val = self._clean_opt_float(tx.get("value"), digits=6) or 0.0
                        signed_v += dir_w * plan_w * decay_w * math.log1p(max(tx_val, 0.0))
                    if scored_dated_tx:
                        form4_val = round(math.tanh(signed_v / 25.0), 2)
                    elif has_unverified_ps:
                        missing_modalities.append("FORM4_FILING_DATES_MISSING")
                    if has_ambiguous_f4:
                        missing_modalities.append("FORM4_AMENDMENT_AMBIGUOUS")
            else:
                missing_modalities.append("EDGAR_FILINGS_NOT_LOADED")

            if cached_pg is not None and cached_pg[1].get("nodes"):
                pg_payload = cached_pg[1]
                top_w_dt = str(pg_payload.get("weight_bar_date") or "")[:10]
                top_w_obs_raw = pg_payload.get("weight_observation_timestamp_ms")
                top_w_obs = (
                    int(top_w_obs_raw)
                    if isinstance(top_w_obs_raw, (int, float)) and not isinstance(top_w_obs_raw, bool)
                    else None
                )
                top_w_uns = bool(pg_payload.get("_weight_unsettled"))
                top_w_lat_unstamped = bool(pg_payload.get("_weight_latest_unstamped"))
                top_w_mtime_raw = pg_payload.get("weight_disk_mtime_ms")
                top_w_mtime = (
                    int(top_w_mtime_raw)
                    if isinstance(top_w_mtime_raw, (int, float)) and not isinstance(top_w_mtime_raw, bool)
                    else None
                )
                eligible_spills: List[float] = []
                for n in pg_payload["nodes"]:
                    if n.get("role") == "TARGET":
                        continue
                    n_dt = str(n.get("bar_date") or "")[:10]
                    n_obs_raw = n.get("observation_timestamp_ms")
                    n_obs = (
                        int(n_obs_raw)
                        if isinstance(n_obs_raw, (int, float)) and not isinstance(n_obs_raw, bool)
                        else None
                    )
                    n_uns = bool(n.get("_bar_unsettled"))
                    n_mtime_raw = n.get("disk_mtime_ms")
                    n_mtime_ms = (
                        int(n_mtime_raw)
                        if isinstance(n_mtime_raw, (int, float)) and not isinstance(n_mtime_raw, bool)
                        else None
                    )
                    w_dt = str(n.get("weight_bar_date") or top_w_dt)[:10]
                    w_obs_raw = n["weight_observation_timestamp_ms"] if "weight_observation_timestamp_ms" in n else top_w_obs
                    w_obs = (
                        int(w_obs_raw)
                        if isinstance(w_obs_raw, (int, float)) and not isinstance(w_obs_raw, bool)
                        else None
                    )
                    w_uns = bool(n["_weight_unsettled"] if "_weight_unsettled" in n else top_w_uns)
                    w_lat_unstamped = bool(
                        n["_weight_latest_unstamped"] if "_weight_latest_unstamped" in n else top_w_lat_unstamped
                    )
                    w_mtime_raw = n["weight_disk_mtime_ms"] if "weight_disk_mtime_ms" in n else top_w_mtime
                    w_mtime_ms = (
                        int(w_mtime_raw)
                        if isinstance(w_mtime_raw, (int, float)) and not isinstance(w_mtime_raw, bool)
                        else None
                    )

                    if not n_dt or (target_bar_date and n_dt > target_bar_date):
                        continue
                    if w_dt and target_bar_date and w_dt > target_bar_date:
                        continue
                    if n_obs is not None and (target_obs_ms is None or n_obs > target_obs_ms):
                        continue
                    if w_obs is not None and (target_obs_ms is None or w_obs > target_obs_ms):
                        continue

                    sym_offline_peer_ok = (
                        not self.enable_live_fetch
                        and not target_unsettled
                        and target_obs_ms is None
                        and n_obs is None
                        and not n_uns
                        and w_obs is None
                        and not w_uns
                        and target_mtime_ms is not None
                        and n_mtime_ms is not None
                        and n_mtime_ms <= target_mtime_ms
                        and w_mtime_ms is not None
                        and w_mtime_ms <= target_mtime_ms
                    )
                    if (n_uns or w_uns) and not sym_offline_peer_ok:
                        continue
                    if target_bar_date and n_dt == target_bar_date and not sym_offline_peer_ok:
                        if n_obs is None or target_obs_ms is None or n_obs > target_obs_ms:
                            continue
                    if target_bar_date and w_dt == target_bar_date and not sym_offline_peer_ok:
                        if w_lat_unstamped or w_obs is None or target_obs_ms is None or w_obs > target_obs_ms:
                            continue

                    eligible_spills.append(float(n.get("spillover_contribution", 0.0)))
                if eligible_spills:
                    peer_spill = round(float(np.clip(sum(eligible_spills) / 10.0, -1.0, 1.0)), 2)
                else:
                    peer_spill = 0.0
                    missing_modalities.append("PEER_GAT_NOT_LOADED")
            else:
                peer_spill = 0.0
                missing_modalities.append("PEER_GAT_NOT_LOADED")

            raw_last_res = float(roll_res.residuals[-1]) if len(roll_res.residuals) > 0 else float("nan")
            last_res_valid = math.isfinite(raw_last_res)
            last_res = raw_last_res if last_res_valid else 0.0
            raw_last_vol = float(df_sub["volume"].iloc[-1]) if len(df_sub) > 0 else 0.0
            last_vol = raw_last_vol if math.isfinite(raw_last_vol) and raw_last_vol >= 0.0 else 0.0
            vols_tail = pd.to_numeric(df_sub["volume"].tail(20), errors="coerce").dropna()
            adv_20_raw = float(vols_tail.mean()) if len(vols_tail) > 0 else 1.0
            adv_20 = adv_20_raw if math.isfinite(adv_20_raw) and adv_20_raw > 0.0 else 1.0
            res_bar_date = pd.Timestamp(df_sub.index[-1]).strftime("%Y-%m-%d") if len(df_sub) > 0 else ""
            res_obs_ms = self._market_bar_timestamp_ms(
                df_sub,
                overlaid_q_ms
                if (
                    overlaid_q_ms is not None
                    and len(df_sub) > 0
                    and pd.Timestamp(df_sub.index[-1]).normalize() == pd.Timestamp(df.index[-1]).normalize()
                )
                else None,
            )
            _, _, res_sched_close = self._daily_bar_status_ms(df_sub)
            res_attrs = getattr(df_sub, "attrs", {}) or {}
            res_attrs_match = (
                res_attrs.get("obs_bar_date") is None
                or str(res_attrs.get("obs_bar_date"))[:10] == res_bar_date
            )
            res_has_explicit_obs = bool(
                (
                    overlaid_q_ms is not None
                    and len(df_sub) > 0
                    and pd.Timestamp(df_sub.index[-1]).normalize() == pd.Timestamp(df.index[-1]).normalize()
                )
                or (
                    res_attrs_match
                    and not res_attrs.get("live_quote_undated")
                    and res_attrs.get("observation_timestamp_ms") is not None
                )
            )
            res_event_obs_ms = (
                res_obs_ms
                if (res_has_explicit_obs or res_obs_ms is None or res_sched_close is None)
                else min(res_obs_ms, res_sched_close)
            )
            market_aligned = (
                has_aligned_benchmarks
                and bool(res_bar_date)
                and res_bar_date == target_bar_date
                and bench_not_ahead
            )
            news_aligned = True
            if cached_news is not None and cached_news[1] and res_bar_date:
                news_aligned = all(
                    _is_news_item_pit_eligible(it, res_event_obs_ms, res_bar_date)
                    for it in cached_news[1]
                    if isinstance(it, dict)
                )

            if (
                not last_res_valid
                or not market_aligned
                or (cached_news is not None and cached_news[1] and not news_aligned)
            ):
                d_plus, d_minus = 0.0, 0.0
                if "DIVERGENCE_OBSERVATIONS_UNALIGNED" not in missing_modalities:
                    missing_modalities.append("DIVERGENCE_OBSERVATIONS_UNALIGNED")
            else:
                d_plus, d_minus = EventDecayKernel.compute_smart_money_divergence(
                    sentiment=finbert_val,
                    residual_return=last_res,
                    volume=last_vol,
                    adv20=adv_20,
                )

            signal_breakdown = {
                "residual_momentum_20d_rank": res_mom_rank,
                "overnight_return_rank": on_ret_rank,
                "intraday_return_rank": intra_ret_rank,
                "amihud_illiquidity_rank": amihud_rank,
                "closing_power_rank": cp_rank,
                "decayed_finbert_sentiment": finbert_val,
                "decayed_form4_insider_cluster": form4_val,
                "decayed_pead_shock": 0.0,
                "decayed_10q_legal_risk_delta": legal_delta,
                "decayed_mda_catalyst_score": mda_cat,
                "peer_gat_spillover": peer_spill,
                "smart_money_divergence_pos": round(float(d_plus), 4),
                "smart_money_divergence_neg": round(float(d_minus), 4),
            }

            frame = build_qare_signal_frame(
                ticker=ticker,
                action=rec.action,
                conviction_tier=rec.conviction_tier,
                p_buy=p_buy,
                p_hold=p_hold,
                p_sell=p_sell,
                temperature=self.temp_scaler.temperature_,
                unconstrained_kelly_pct=round(rec.unconstrained_kelly_pct, 2),
                l1_normalized_pct=round(rec.l1_normalized_portfolio_allocation_pct, 2) if kappa is not None else 0.0,
                entry_price=entry_price,
                dynamic_take_profit=round(rec.dynamic_take_profit, 2),
                dynamic_stop_loss=round(rec.dynamic_stop_loss, 2),
                reward_to_risk_ratio=round(rec.reward_to_risk_ratio, 2),
                regime_state=regime_state,
                regime_index=regime_idx,
                regime_probs=regime_probs,
                regime_kappa=kappa,
                transition_matrix=transmat,
                signal_breakdown=signal_breakdown,
                missing_modalities=missing_modalities,
                intraday_mode="DAILY_FALLBACK_LGBM",
                xbrl_filed_date_verified=xbrl_filing_verified,
                prior_probs=self._class_priors,
            )
            with self._lock:
                self._signal_cache[ticker] = (cache_key, now_mono, frame)
            return frame
        except Exception as e:
            logger.error(f"Error computing live signal for {ticker}: {e}")
            return build_qare_signal_frame(
                ticker=ticker,
                action="HOLD",
                conviction_tier="LOW",
                p_buy=0.0,
                p_hold=1.0,
                p_sell=0.0,
                unconstrained_kelly_pct=0.0,
                l1_normalized_pct=0.0,
                entry_price=entry_price,
                dynamic_take_profit=tp,
                dynamic_stop_loss=sl,
                reward_to_risk_ratio=rr_val,
                regime_state=regime_state,
                regime_index=regime_idx,
                regime_probs=regime_probs,
                regime_kappa=kappa,
                transition_matrix=transmat,
                signal_breakdown={},
                winning_engine="NONE",
                execution_mode="UNREADY",
                missing_modalities=["SIGNAL_COMPUTE_ERROR"],
                intraday_mode="UNAVAILABLE",
                xbrl_filed_date_verified=False,
            )

    def get_portfolio(self) -> Dict[str, Any]:
        """Computes L1-normalized portfolio allocations across core institutional universe."""
        now_mono = time.monotonic()
        with self._lock:
            if self._portfolio_cache is not None and (now_mono - self._portfolio_cache[0]) < self._PORTFOLIO_TTL_SECONDS:
                return self._portfolio_cache[1]

        universe = ["NVDA", "MSFT", "AAPL", "CRWD"]
        raw_allocs: Dict[str, float] = {}
        regime = self.get_macro_regime()
        for sym in universe:
            sig = self.get_signal(sym, regime=regime)
            rec = sig.get("execution_recommendation", {})
            raw_allocs[sym] = float(rec.get("l1_normalized_portfolio_allocation_pct", 0.0)) / 100.0

        norm_weights = self.portfolio_normalizer.normalize_weights(raw_allocs, execution_mode="LONG_ONLY")
        allocs_pct = {k: round(v * 100.0, 2) for k, v in norm_weights.items()}
        gross = round(sum(abs(v) for v in norm_weights.values()), 4)
        res = {
            "allocations": allocs_pct,
            "gross_exposure": gross,
            "execution_mode": "LONG_ONLY",
        }
        with self._lock:
            self._portfolio_cache = (time.monotonic(), res)
        return res

    @staticmethod
    def _clean_opt_float(val: Any, scale: float = 1.0, digits: int = 2) -> Optional[float]:
        if val is None or isinstance(val, bool):
            return None
        try:
            f = float(val)
            if not math.isfinite(f):
                return None
            return round(f * scale, digits)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _compute_ols_beta_60d(
        df_target: Optional[pd.DataFrame],
        df_bench: Optional[pd.DataFrame],
    ) -> Optional[float]:
        """Computes trailing 60-return OLS beta of target close against benchmark close."""
        if df_target is None or df_bench is None or len(df_target) < 30 or len(df_bench) < 30:
            return None
        idx = df_target.index.intersection(df_bench.index)[-61:]
        if len(idx) < 25:
            return None
        c_t = np.asarray(df_target.loc[idx, "close"].values, dtype=np.float64)
        c_m = np.asarray(df_bench.loc[idx, "close"].values, dtype=np.float64)
        c_t = np.where(np.isfinite(c_t) & (c_t > 0.0), c_t, np.nan)
        c_m = np.where(np.isfinite(c_m) & (c_m > 0.0), c_m, np.nan)
        rs = np.diff(np.log(c_t))
        rm = np.diff(np.log(c_m))
        mask = np.isfinite(rs) & np.isfinite(rm)
        rs_clean = rs[mask]
        rm_clean = rm[mask]
        if len(rs_clean) < 25:
            return None
        var_m = float(np.var(rm_clean))
        if not math.isfinite(var_m) or var_m <= 1e-12:
            return None
        cov = float(np.mean((rs_clean - np.mean(rs_clean)) * (rm_clean - np.mean(rm_clean))))
        if not math.isfinite(cov):
            return None
        return round(float(np.clip(cov / var_m, -1.0, 4.0)), 2)

    def get_security_master(self, ticker: str) -> Dict[str, Any]:
        """Provides authentic Security Master metadata from lakehouse Parquet + live Yahoo/SEC."""
        ticker = ticker.upper()
        now_mono = time.monotonic()
        with self._lock:
            cached = self._sec_master_cache.get(ticker)
            if cached is not None and (now_mono - cached[0]) < self._SEC_MASTER_TTL_SECONDS:
                return cached[1]

        df = self._load_cached_daily(ticker)
        high_52w = round(float(df["high"].tail(252).max()), 2) if df is not None and len(df) > 0 else None
        low_52w = round(float(df["low"].tail(252).min()), 2) if df is not None and len(df) > 0 else None
        adv_30d = int(round(float(df["volume"].tail(30).mean()))) if df is not None and len(df) > 0 else None
        beta_mkt: Optional[float] = None
        beta_sec: Optional[float] = None

        try:
            rec = self.universe_selector.get_security_master(ticker)
            rec_sector = rec.gics_sector
            rec_sub = rec.gics_sub_industry
            rec_name = rec.name
            rec_cik = rec.cik
            rec_figi = rec.figi
            sector_etf = rec.sector_etf
        except KeyError:
            rec = None
            rec_sector = None
            rec_sub = None
            rec_name = None
            rec_cik = self.fundamentals_client.resolve_cik(ticker, enable_live_fetch=self.enable_live_fetch)
            rec_figi = f"BBG{hashlib.sha256(ticker.encode('utf-8')).hexdigest()[:6].upper()}"
            sector_etf = "SPY"

        if df is not None:
            try:
                df_spy = self._load_cached_daily("SPY")
                beta_mkt = self._compute_ols_beta_60d(df, df_spy)
                if len(df) >= 65 and df_spy is not None:
                    df_sec = self._load_cached_daily(sector_etf)
                    if df_sec is not None:
                        common_idx = df.index.intersection(df_spy.index).intersection(df_sec.index)
                        if len(common_idx) >= 65:
                            sub_idx = common_idx[-65:]
                            r_stock = np.diff(np.log(df.loc[sub_idx, "close"].values))
                            r_spy = np.diff(np.log(df_spy.loc[sub_idx, "close"].values))
                            r_sec = np.diff(np.log(df_sec.loc[sub_idx, "close"].values))
                            roll_res = self.residualizer.compute_rolling_residuals(r_stock, r_spy, r_sec)
                            if not np.isnan(roll_res.betas_sec[-1]):
                                beta_sec = round(float(roll_res.betas_sec[-1]), 2)
            except Exception as e:
                logger.debug(f"Beta calculation note for {ticker}: {e}")

        yf_meta = self.fundamentals_client.get_yf_meta(ticker, enable_live_fetch=self.enable_live_fetch)
        info = (yf_meta or {}).get("info", {})

        # Prefer authentic Yahoo company metadata when live/cached
        comp_name = info.get("longName") or info.get("shortName") or rec_name or ticker
        exch_raw = str(info.get("exchange") or info.get("fullExchangeName") or "").upper()
        if exch_raw in ("NMS", "NGM", "NCM", "NAS", "NASDAQGS", "NASDAQGM"):
            primary_exchange = "NASDAQ"
        elif exch_raw in ("NYQ", "NYSE", "NYS"):
            primary_exchange = "NYSE"
        elif exch_raw in ("PCX", "BTS", "ASE", "AMEX", "ARCX"):
            primary_exchange = "NYSE ARCA"
        elif exch_raw:
            primary_exchange = exch_raw
        elif rec_sector:
            primary_exchange = "NASDAQ" if "Tech" in rec_sector else "NYSE"
        else:
            primary_exchange = "US"

        market_cap_b = self._clean_opt_float(info.get("marketCap"), scale=1e-9, digits=2)
        pe_ltm = self._clean_opt_float(info.get("trailingPE"), digits=2)
        pe_fwd = self._clean_opt_float(info.get("forwardPE"), digits=2)

        div_yield = self._clean_opt_float(info.get("trailingAnnualDividendYield"), scale=100.0, digits=2)
        if div_yield is None:
            div_yield = self._clean_opt_float(info.get("dividendYield"), scale=1.0, digits=2)

        if high_52w is None:
            high_52w = self._clean_opt_float(info.get("fiftyTwoWeekHigh"), digits=2)
        if low_52w is None:
            low_52w = self._clean_opt_float(info.get("fiftyTwoWeekLow"), digits=2)

        officers = info.get("companyOfficers") or []
        ceo_name = None
        if isinstance(officers, list):
            for off in officers:
                if isinstance(off, dict):
                    title = str(off.get("title", "")).lower()
                    if "ceo" in title or "chief executive" in title:
                        ceo_name = off.get("name")
                        break
            if ceo_name is None and officers and isinstance(officers[0], dict):
                ceo_name = officers[0].get("name")

        city = info.get("city")
        state = info.get("state")
        country = info.get("country")
        hq_parts = [p for p in (city, state or country) if p]
        headquarters = ", ".join(hq_parts) if hq_parts else None

        employees_raw = info.get("fullTimeEmployees")
        employees = int(employees_raw) if isinstance(employees_raw, (int, float)) and employees_raw > 0 else None

        gics_sector = info.get("sector") or rec_sector or ("Exchange Traded Fund" if info.get("quoteType") == "ETF" else "N/A")
        gics_sub = info.get("industry") or rec_sub or ("Broad Market Index" if info.get("quoteType") == "ETF" else "N/A")
        summary = info.get("longBusinessSummary") or (
            f"{comp_name} ({gics_sector} / {gics_sub})."
            + (" Live Yahoo Finance fundamental feed unavailable offline." if not info else "")
        )

        res = {
            "ticker": ticker,
            "available_fundamentals": bool(info),
            "company_name": comp_name,
            "primary_exchange": primary_exchange,
            "cik": rec_cik or "N/A",
            "figi": rec_figi,
            "gics_sector": gics_sector,
            "gics_sub_industry": gics_sub,
            "beta_market_60d": beta_mkt,
            "beta_sector_60d": beta_sec,
            "adv_30d_shares": adv_30d,
            "market_cap_billions": market_cap_b,
            "pe_ratio_ltm": pe_ltm,
            "pe_forward": pe_fwd,
            "dividend_yield_pct": div_yield,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "business_summary": summary,
            "headquarters": headquarters,
            "ceo": ceo_name,
            "employees": employees,
        }
        with self._lock:
            self._sec_master_cache[ticker] = (now_mono, res)
        return res

    def get_peer_graph(self, ticker: str) -> Dict[str, Any]:
        """Computes rolling GICS & lead-lag peer subgraph and attention weights from local Parquet."""
        ticker = ticker.upper()
        now_mono = time.monotonic()
        with self._lock:
            cached = self._peer_graph_cache.get(ticker)
            if cached is not None and (now_mono - cached[0]) < self._PEER_GRAPH_TTL_SECONDS:
                return cached[1]

        try:
            sector_etf = self.universe_selector.get_sector_etf(ticker)
        except KeyError:
            sector_etf = "SPY"

        candidate_peers = self.universe_selector.get_peers(ticker, min_count=12)
        if not candidate_peers:
            candidate_peers = [t for t in self.universe_selector.get_all_registered_tickers() if t != ticker][:12]

        # Only allow network refresh on target + SPY; load peer candidates from local Parquet (<1ms each)
        df_spy = self._load_cached_daily("SPY", allow_refresh=True)
        spy_ret = (
            np.log(df_spy["close"]).diff().dropna()
            if df_spy is not None and len(df_spy) >= 30
            else None
        )

        loaded_dfs: Dict[str, pd.DataFrame] = {}
        if df_spy is not None:
            loaded_dfs["SPY"] = df_spy
        df_target = df_spy if ticker == "SPY" else self._load_cached_daily(ticker, allow_refresh=True)
        if df_target is not None:
            loaded_dfs[ticker] = df_target
        df_etf = loaded_dfs.get(sector_etf)
        if df_etf is None and sector_etf not in loaded_dfs:
            df_etf = self._load_cached_daily(sector_etf, allow_refresh=True)
            if df_etf is not None:
                loaded_dfs[sector_etf] = df_etf

        res_series_map: Dict[str, pd.Series] = {}
        for sym in [ticker] + candidate_peers[:12]:
            d = loaded_dfs.get(sym)
            if d is None:
                d = self._load_cached_daily(sym, allow_refresh=False)
                if d is not None:
                    loaded_dfs[sym] = d
            if d is None or len(d) < 30:
                continue
            r = np.log(d["close"]).diff().dropna()
            if spy_ret is not None:
                common = r.index.intersection(spy_ret.index)
                if len(common) >= 20:
                    res_series_map[sym] = (r.loc[common] - spy_ret.loc[common]).tail(90)
                    continue
            res_series_map[sym] = r.tail(90)

        res_df = pd.DataFrame(res_series_map) if len(res_series_map) >= 2 and ticker in res_series_map else None
        gics_peers, lead_lag_peers, corr_map = self.graph_builder.select_gics_and_lead_lag_peers(
            ticker, residual_returns=res_df
        )

        all_pool = [
            t for t in (candidate_peers + self.universe_selector.get_all_registered_tickers())
            if t not in (ticker, sector_etf) and t in loaded_dfs
        ]
        if not all_pool:
            all_pool = [t for t in candidate_peers if t not in (ticker, sector_etf)]

        used = {ticker, sector_etf}
        clean_gics: List[str] = []
        for p in gics_peers + all_pool:
            if p not in used:
                clean_gics.append(p)
                used.add(p)
            if len(clean_gics) == 4:
                break
        clean_ll: List[str] = []
        for p in lead_lag_peers + all_pool:
            if p not in used:
                clean_ll.append(p)
                used.add(p)
            if len(clean_ll) == 4:
                break

        def _get_df(sym: str) -> Optional[pd.DataFrame]:
            if sym not in loaded_dfs:
                d = self._load_cached_daily(sym, allow_refresh=(sym in (ticker, "SPY", sector_etf)))
                if d is not None:
                    loaded_dfs[sym] = d
            return loaded_dfs.get(sym)

        def _ret_5d(sym: str) -> float:
            d = _get_df(sym)
            if d is None or len(d) < 6:
                return 0.0
            c = d["close"].values
            return round(float(((c[-1] - c[-6]) / max(c[-6], 1e-6)) * 100.0), 2)

        def _beta_mkt(sym: str) -> Optional[float]:
            return self._compute_ols_beta_60d(_get_df(sym), df_spy)

        def _corr_5d(sym: str, target_sym: str) -> float:
            if sym in corr_map and np.isfinite(corr_map[sym]):
                return round(float(corr_map[sym]), 2)
            d1 = _get_df(sym)
            d2 = _get_df(target_sym)
            if d1 is None or d2 is None or len(d1) < 25 or len(d2) < 25:
                return 0.0
            idx = d1.index.intersection(d2.index)
            if len(idx) < 25:
                return 0.0
            r1 = np.diff(np.log(d1.loc[idx, "close"].values[-25:]))
            r2 = np.diff(np.log(d2.loc[idx, "close"].values[-25:]))
            corr = float(np.corrcoef(r1, r2)[0, 1])
            return round(corr if np.isfinite(corr) else 0.0, 2)

        ll_corrs = [_corr_5d(p, ticker) for p in clean_ll]
        raw_gics_w = [max(0.05, 0.14 - 0.02 * i) for i in range(len(clean_gics))]
        raw_ll_w = [max(0.05, 0.06 + 0.14 * abs(c)) for c in ll_corrs]
        raw_etf_w = 0.18
        total_w = sum(raw_gics_w) + sum(raw_ll_w) + raw_etf_w
        weights_gics = [round(w / total_w, 3) for w in raw_gics_w]
        weights_ll = [round(w / total_w, 3) for w in raw_ll_w]
        weight_etf = round(raw_etf_w / total_w, 3)

        def _frame_obs_meta(d_sym: Optional[pd.DataFrame]) -> Tuple[Optional[int], Optional[str], bool, Optional[int]]:
            if d_sym is None or len(d_sym) == 0:
                return None, None, False, None
            obs_sym, _, _ = self._daily_bar_status_ms(d_sym)
            li_sym = d_sym.index[-1]
            dt_sym = li_sym.strftime("%Y-%m-%d") if hasattr(li_sym, "strftime") else str(li_sym)[:10]
            attrs_sym = getattr(d_sym, "attrs", {}) or {}
            unsettled_sym = bool(attrs_sym.get("_bar_unsettled"))
            raw_dm_sym = attrs_sym.get("disk_mtime_ms")
            mtime_ms_sym = (
                int(raw_dm_sym)
                if isinstance(raw_dm_sym, (int, float)) and not isinstance(raw_dm_sym, bool)
                else None
            )
            return obs_sym, dt_sym, unsettled_sym, mtime_ms_sym

        def _node_obs_meta(sym: str) -> Tuple[Optional[int], Optional[str], bool, Optional[int]]:
            return _frame_obs_meta(_get_df(sym))

        weight_contrib_syms: List[str] = []
        for s in [ticker, "SPY", sector_etf] + list(res_series_map.keys()) + clean_gics + clean_ll:
            if s not in weight_contrib_syms:
                weight_contrib_syms.append(s)
        weight_dts: List[str] = []
        weight_obs_vals: List[int] = []
        weight_mtimes: List[int] = []
        weight_missing_mtime = False
        weight_unsettled = False
        weight_obs_rows: List[Tuple[str, Optional[int], bool]] = []
        for w_sym in weight_contrib_syms:
            d_w = loaded_dfs.get(w_sym) if w_sym != "SPY" else df_spy
            if d_w is None and w_sym != "SPY":
                d_w = _get_df(w_sym)
            w_obs_s, w_dt_s, w_uns_s, w_mtime_s = _frame_obs_meta(d_w)
            if w_dt_s:
                weight_dts.append(w_dt_s)
                weight_obs_rows.append((w_dt_s, w_obs_s, w_uns_s))
                if w_mtime_s is not None:
                    weight_mtimes.append(w_mtime_s)
                else:
                    weight_missing_mtime = True
            if w_obs_s is not None:
                weight_obs_vals.append(int(w_obs_s))
            if w_uns_s:
                weight_unsettled = True
        weight_bar_date = max(weight_dts) if weight_dts else None
        weight_obs_ms = max(weight_obs_vals) if weight_obs_vals else None
        weight_disk_mtime_ms = max(weight_mtimes) if (weight_mtimes and not weight_missing_mtime) else None
        weight_latest_unstamped = bool(
            weight_bar_date
            and any(
                dt_i == weight_bar_date and (obs_i is None or uns_i)
                for dt_i, obs_i, uns_i in weight_obs_rows
            )
        )
        weight_prov_fields: Dict[str, Any] = {
            "weight_observation_timestamp_ms": weight_obs_ms,
            "weight_bar_date": weight_bar_date,
            "weight_disk_mtime_ms": weight_disk_mtime_ms,
            "_weight_unsettled": weight_unsettled,
            "_weight_latest_unstamped": weight_latest_unstamped,
        }

        tgt_obs_pg, tgt_dt_pg, tgt_uns_pg, tgt_mtime_pg = _node_obs_meta(ticker)
        nodes: List[Dict[str, Any]] = [
            {
                "id": ticker,
                "role": "TARGET",
                "role_code": 0.0,
                "label": ticker,
                "beta_market": _beta_mkt(ticker),
                "return_5d": _ret_5d(ticker),
                "spillover_contribution": 0.0,
                "observation_timestamp_ms": tgt_obs_pg,
                "bar_date": tgt_dt_pg,
                "disk_mtime_ms": tgt_mtime_pg,
                "_bar_unsettled": tgt_uns_pg,
                **weight_prov_fields,
            }
        ]
        edges: List[Dict[str, Any]] = []
        for p, w in zip(clean_gics, weights_gics):
            r5 = _ret_5d(p)
            p_obs_pg, p_dt_pg, p_uns_pg, p_mtime_pg = _node_obs_meta(p)
            nodes.append({
                "id": p,
                "role": "GICS_PEER",
                "role_code": 0.33,
                "label": p,
                "beta_market": _beta_mkt(p),
                "return_5d": r5,
                "spillover_contribution": round(r5 * w * 0.25, 2),
                "observation_timestamp_ms": p_obs_pg,
                "bar_date": p_dt_pg,
                "disk_mtime_ms": p_mtime_pg,
                "_bar_unsettled": p_uns_pg,
                **weight_prov_fields,
            })
            edges.append({"source": ticker, "target": p, "attention_weight": w})

        best_ll_peer = clean_ll[0] if clean_ll else sector_etf
        best_ll_abs = -1.0
        for p, w, c5 in zip(clean_ll, weights_ll, ll_corrs):
            r5 = _ret_5d(p)
            p_obs_pg, p_dt_pg, p_uns_pg, p_mtime_pg = _node_obs_meta(p)
            spill = round(r5 * w * 0.30, 2)
            if abs(spill) > best_ll_abs:
                best_ll_abs = abs(spill)
                best_ll_peer = p
            nodes.append({
                "id": p,
                "role": "LEAD_LAG",
                "role_code": 0.66,
                "label": p,
                "cross_corr_5d": c5,
                "beta_market": _beta_mkt(p),
                "return_5d": r5,
                "spillover_contribution": spill,
                "observation_timestamp_ms": p_obs_pg,
                "bar_date": p_dt_pg,
                "disk_mtime_ms": p_mtime_pg,
                "_bar_unsettled": p_uns_pg,
                **weight_prov_fields,
            })
            edges.append({"source": p, "target": ticker, "attention_weight": w, "lead_days": 5})

        etf_r5 = _ret_5d(sector_etf)
        etf_obs_pg, etf_dt_pg, etf_uns_pg, etf_mtime_pg = _node_obs_meta(sector_etf)
        nodes.append({
            "id": sector_etf,
            "role": "SECTOR_ETF",
            "role_code": 1.0,
            "label": sector_etf,
            "beta_market": _beta_mkt(sector_etf),
            "return_5d": etf_r5,
            "spillover_contribution": round(-etf_r5 * 0.08, 2),
            "observation_timestamp_ms": etf_obs_pg,
            "bar_date": etf_dt_pg,
            "disk_mtime_ms": etf_mtime_pg,
            "_bar_unsettled": etf_uns_pg,
            **weight_prov_fields,
        })
        edges.append({"source": sector_etf, "target": ticker, "attention_weight": weight_etf})

        probs = [max(float(e["attention_weight"]), 1e-6) for e in edges]
        p_sum = sum(probs)
        entropy = round(-sum((p / p_sum) * float(np.log(p / p_sum)) for p in probs), 2) if p_sum > 0 else 0.0

        res = {
            "target_ticker": ticker,
            "nodes": nodes,
            "edges": edges,
            "gat_attention_entropy": entropy,
            "top_spillover_contributor": best_ll_peer,
            **weight_prov_fields,
        }
        with self._lock:
            self._peer_graph_cache[ticker] = (now_mono, res)
        return res

    def get_historical_prices(
        self, ticker: str, count: int = 30, interval: str = "1d"
    ) -> List[Dict[str, Any]]:
        """Loads latest daily or intraday OHLCV bars from live/lakehouse storage (most recent first)."""
        ticker = ticker.upper()
        iv = (interval or "1d").lower()
        if iv in ("1h", "4h") and self.enable_live_fetch and time.monotonic() >= self._failed_yf_until.get(f"ih:{ticker}", 0.0):
            cache_key = (ticker, iv)
            now = time.monotonic()
            with self._lock:
                cached = self._intraday_cache.get(cache_key)
                if cached is not None and (now - cached[0]) < self._INTRADAY_TTL_SECONDS:
                    return cached[1][:count]
            bars = self.market_loader.fetch_intraday_bars(ticker, interval=iv, count=count)
            if bars:
                with self._lock:
                    self._intraday_cache[cache_key] = (now, bars)
                return bars

        df, overlaid_q_ms = self._load_cached_daily_with_quote_ts(ticker)
        if df is None or len(df) == 0:
            return []

        df_slice = df.tail(count + 1)
        records: List[Dict[str, Any]] = []

        close_vals = df_slice["close"].values
        open_vals = df_slice["open"].values
        high_vals = df_slice["high"].values
        low_vals = df_slice["low"].values
        vol_vals = df_slice["volume"].values
        dates = df_slice.index

        start_idx = 1 if len(df_slice) > count else 0
        for i in range(start_idx, len(df_slice)):
            prev_c = close_vals[i - 1] if i > 0 else open_vals[i]
            tot_ret_pct = float(((close_vals[i] - prev_c) / prev_c) * 100.0) if prev_c > 0 else 0.0
            on_ret_pct = float(((open_vals[i] - prev_c) / prev_c) * 100.0) if prev_c > 0 else 0.0
            intra_ret_pct = float(((close_vals[i] - open_vals[i]) / open_vals[i]) * 100.0) if open_vals[i] > 0 else 0.0
            vwap_val = round(float((high_vals[i] + low_vals[i] + close_vals[i]) / 3.0), 2)

            dt_str = dates[i].strftime("%Y-%m-%d") if hasattr(dates[i], "strftime") else str(dates[i])[:10]
            records.append({
                "date": dt_str,
                "open": round(float(open_vals[i]), 2),
                "high": round(float(high_vals[i]), 2),
                "low": round(float(low_vals[i]), 2),
                "close": round(float(close_vals[i]), 2),
                "vwap": vwap_val,
                "volume": int(vol_vals[i]),
                "overnight_return": round(on_ret_pct, 2),
                "intraday_return": round(intra_ret_pct, 2),
                "total_return": round(tot_ret_pct, 2),
            })

        latest_obs_ms = self._market_bar_timestamp_ms(df, overlaid_q_ms)
        now_ms = int(time.time() * 1000)
        if records and latest_obs_ms is not None and latest_obs_ms <= now_ms:
            records[-1]["observation_timestamp_ms"] = latest_obs_ms

        return records[::-1]

    @staticmethod
    def _parse_iso_utc_us(raw_ts: Any) -> Optional[int]:
        if not raw_ts:
            return None
        try:
            s = str(raw_ts).strip()
            if not s or ("T" not in s and ":" not in s):
                return None
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            sec_i = int(dt.replace(microsecond=0).timestamp())
            return sec_i * 1_000_000 + dt.microsecond
        except Exception:
            return None

    @staticmethod
    def _parse_iso_utc_ms(raw_ts: Any) -> Optional[int]:
        us = LiveInferenceProvider._parse_iso_utc_us(raw_ts)
        return (us // 1000) if us is not None else None

    @staticmethod
    def _parse_yahoo_price_ts_ms(info: Dict[str, Any]) -> Optional[int]:
        raw_t = info.get("regularMarketTime")
        if raw_t is None or isinstance(raw_t, bool):
            return None
        if isinstance(raw_t, (int, float)):
            f = float(raw_t)
            if not math.isfinite(f) or f <= 0.0:
                return None
            return int(f if f >= 1e12 else f * 1000.0)
        return LiveInferenceProvider._parse_iso_utc_ms(raw_t)

    @staticmethod
    def _bar_status_ms_for_end(
        last_ts: Any,
        attrs: Mapping[str, Any],
        now_ms: Optional[int] = None,
    ) -> Tuple[Optional[int], bool, Optional[int]]:
        """Computes (known_observation_ms, is_known_complete, scheduled_close_ms) from endpoint ts & attrs."""
        bar_ts = pd.Timestamp(last_ts)
        try:
            close_t = MarketDataLoader.nyse_session_close_time(bar_ts.date())
            ny_close = datetime.combine(bar_ts.date(), close_t, tzinfo=ZoneInfo("America/New_York"))
            scheduled_close_ms = int(ny_close.timestamp() * 1000)
        except Exception:
            return None, False, None

        eff_now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
        bar_dt_str = bar_ts.strftime("%Y-%m-%d")
        obs_bar_date = attrs.get("obs_bar_date")
        attrs_match_row = obs_bar_date is None or str(obs_bar_date)[:10] == bar_dt_str
        if attrs_match_row and attrs.get("live_quote_undated"):
            return None, False, scheduled_close_ms
        rec_obs = attrs.get("observation_timestamp_ms") if attrs_match_row else None
        rec_obs_ms: Optional[int] = (
            int(rec_obs)
            if isinstance(rec_obs, (int, float))
            and not isinstance(rec_obs, bool)
            and math.isfinite(float(rec_obs))
            and float(rec_obs) > 0.0
            else None
        )
        if rec_obs_ms is not None and rec_obs_ms > eff_now_ms:
            return None, False, scheduled_close_ms
        # Preserve a matching pre-close partial observation even if file write finished post-close (#15)
        if rec_obs_ms is not None and rec_obs_ms < scheduled_close_ms:
            return rec_obs_ms, False, scheduled_close_ms

        if attrs_match_row and attrs.get("_bar_unsettled"):
            return (rec_obs_ms if rec_obs_ms is not None else None), False, scheduled_close_ms

        raw_disk_mtime = attrs.get("disk_mtime_ms") if attrs_match_row else None
        disk_mtime_ms = (
            int(raw_disk_mtime)
            if isinstance(raw_disk_mtime, (int, float))
            and not isinstance(raw_disk_mtime, bool)
            and math.isfinite(float(raw_disk_mtime))
            and float(raw_disk_mtime) > 0.0
            else None
        )
        if attrs.get("_sliced_unverified") and disk_mtime_ms is None:
            return None, False, scheduled_close_ms
        settle_close_ms = scheduled_close_ms + MarketDataLoader._POST_CLOSE_SETTLEMENT_MS
        req_complete_ms = scheduled_close_ms if rec_obs_ms is not None else settle_close_ms
        is_complete = eff_now_ms >= req_complete_ms and (
            disk_mtime_ms is None or disk_mtime_ms >= req_complete_ms
        )
        if is_complete:
            if attrs_match_row and attrs.get("_sliced_mtime_only") and rec_obs_ms is None:
                # Sliced partition lacks per-row observation provenance; only disk_mtime_ms of a later partition is known
                return None, True, scheduled_close_ms
            if rec_obs_ms is not None:
                return rec_obs_ms, True, scheduled_close_ms
            if disk_mtime_ms is not None:
                # Unprovenanced disk write at 16:25 only bounds existence at max(close, mtime), never backdated to 16:00 close
                return max(scheduled_close_ms, disk_mtime_ms), True, scheduled_close_ms
            return scheduled_close_ms, True, scheduled_close_ms

        if rec_obs_ms is not None:
            return rec_obs_ms, False, scheduled_close_ms
        return None, False, scheduled_close_ms

    @staticmethod
    def _daily_bar_status_ms(df: Optional[pd.DataFrame]) -> Tuple[Optional[int], bool, Optional[int]]:
        """Returns (known_observation_ms, is_known_complete, scheduled_close_ms) for the last daily bar."""
        if df is None or len(df) == 0:
            return None, False, None
        return LiveInferenceProvider._bar_status_ms_for_end(df.index[-1], getattr(df, "attrs", {}) or {})

    @staticmethod
    def _sliced_endpoint_attrs(df_raw: pd.DataFrame, sliced_last_dt: str) -> Dict[str, Any]:
        """Derives row-level provenance attrs for a prefix slice ending at sliced_last_dt without copying OHLCV."""
        raw_attrs = getattr(df_raw, "attrs", {}) or {}
        row_prov = raw_attrs.get("_row_provenance")
        restored = row_prov.get(sliced_last_dt) if isinstance(row_prov, dict) else None
        orig_disk_mtime = raw_attrs.get("disk_mtime_ms")
        out: Dict[str, Any] = {}
        if isinstance(restored, dict):
            out["obs_bar_date"] = sliced_last_dt
            if restored.get("observation_timestamp_ms") is not None:
                out["observation_timestamp_ms"] = restored["observation_timestamp_ms"]
            if restored.get("disk_mtime_ms") is not None:
                out["disk_mtime_ms"] = restored["disk_mtime_ms"]
            if restored.get("_bar_unsettled"):
                out["_bar_unsettled"] = True
            if restored.get("live_quote_undated"):
                out["live_quote_undated"] = True
        elif isinstance(orig_disk_mtime, (int, float)) and not isinstance(orig_disk_mtime, bool):
            out["obs_bar_date"] = sliced_last_dt
            out["disk_mtime_ms"] = orig_disk_mtime
            out["_sliced_mtime_only"] = True
        else:
            out["_sliced_unverified"] = True
        return out

    @staticmethod
    def _slice_aligned_frame(df_raw: pd.DataFrame, idx: pd.Index) -> pd.DataFrame:
        """Slices df_raw to idx while preserving row-level provenance or file write timestamp."""
        sliced = df_raw.loc[idx].copy()
        if len(sliced) > 0 and len(df_raw) > 0:
            raw_last = pd.Timestamp(df_raw.index[-1]).strftime("%Y-%m-%d")
            sliced_last = pd.Timestamp(sliced.index[-1]).strftime("%Y-%m-%d")
            if sliced_last != raw_last:
                for key in (
                    "observation_timestamp_ms",
                    "obs_bar_date",
                    "disk_mtime_ms",
                    "live_quote_undated",
                    "_bar_unsettled",
                ):
                    sliced.attrs.pop(key, None)
                sliced.attrs.pop("_sliced_unverified", None)
                sliced.attrs.pop("_sliced_mtime_only", None)
                sliced.attrs.update(LiveInferenceProvider._sliced_endpoint_attrs(df_raw, sliced_last))
        return sliced

    @staticmethod
    def _market_bar_timestamp_ms(df: Optional[pd.DataFrame], overlaid_q_ms: Optional[int]) -> Optional[int]:
        if overlaid_q_ms is not None and overlaid_q_ms > 0:
            return overlaid_q_ms
        obs_ms, _, _ = LiveInferenceProvider._daily_bar_status_ms(df)
        return obs_ms

    def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """Provides authentic Wall Street price targets & analyst consensus from Yahoo Finance."""
        ticker = ticker.upper()
        now_mono = time.monotonic()
        with self._lock:
            cached = self._ratings_cache.get(ticker)
            if cached is not None and (now_mono - cached[0]) < self._RATINGS_TTL_SECONDS:
                return cached[1]

        df, overlaid_q_ms = self._load_cached_daily_with_quote_ts(ticker, require_quote_ts=True)
        raw_bar = float(df["close"].iloc[-1]) if df is not None and len(df) > 0 else 0.0
        bar_p = round(raw_bar, 2) if math.isfinite(raw_bar) and raw_bar > 0.0 else 0.0
        current_p = bar_p

        yf_meta = self.fundamentals_client.get_yf_meta(ticker, enable_live_fetch=self.enable_live_fetch)
        info = (yf_meta or {}).get("info", {})
        rec_trend = (yf_meta or {}).get("rec_trend", [])

        mean_t = self._clean_opt_float(info.get("targetMeanPrice"), digits=2)
        median_t = self._clean_opt_float(info.get("targetMedianPrice"), digits=2)
        high_t = self._clean_opt_float(info.get("targetHighPrice"), digits=2)
        low_t = self._clean_opt_float(info.get("targetLowPrice"), digits=2)
        reg_p = self._clean_opt_float(info.get("regularMarketPrice"), digits=2)
        cur_p = self._clean_opt_float(info.get("currentPrice"), digits=2)
        if current_p <= 0:
            fallback_p = (
                reg_p
                if (reg_p is not None and reg_p > 0)
                else (cur_p if (cur_p is not None and cur_p > 0) else None)
            )
            if fallback_p is not None:
                current_p = fallback_p
        elif bar_p > 0 and reg_p and reg_p > 0:
            now_ms = int(time.time() * 1000)
            yahoo_price_ms = self._parse_yahoo_price_ts_ms(info)
            bar_ms = self._market_bar_timestamp_ms(df, overlaid_q_ms)
            if yahoo_price_ms is not None and yahoo_price_ms <= now_ms:
                if bar_ms is not None:
                    if bar_ms <= yahoo_price_ms:
                        current_p = reg_p
                elif df is not None and len(df) > 0:
                    y_ny_d = datetime.fromtimestamp(
                        yahoo_price_ms / 1000.0, tz=timezone.utc
                    ).astimezone(ZoneInfo("America/New_York")).date()
                    bar_d = pd.Timestamp(df.index[-1]).date()
                    if y_ny_d > bar_d:
                        current_p = reg_p

        num_analysts = int(info.get("numberOfAnalystOpinions") or 0)
        sb = buy = hold = underperf = sell = 0
        if rec_trend:
            latest_t = rec_trend[0]
            sb = int(latest_t.get("strong_buy", 0))
            buy = int(latest_t.get("buy", 0))
            hold = int(latest_t.get("hold", 0))
            underperf = int(latest_t.get("underperform", 0))
            sell = int(latest_t.get("sell", 0))
        trend_total = sb + buy + hold + underperf + sell
        total_analysts = trend_total if trend_total > 0 else num_analysts

        available = bool(mean_t is not None and mean_t > 0 and total_analysts > 0)
        upside = (
            round(((mean_t - current_p) / current_p) * 100.0, 2)
            if available and current_p > 0 and mean_t is not None
            else None
        )

        # Yahoo recommendationMean: 1.0 (Strong Buy) .. 5.0 (Sell) -> map to 5.0 .. 1.0 institutional scale
        yf_rec_mean = self._clean_opt_float(info.get("recommendationMean"), digits=2)
        if yf_rec_mean is not None and 1.0 <= yf_rec_mean <= 5.0:
            consensus_score = round(6.0 - yf_rec_mean, 2)
        elif total_analysts > 0 and trend_total > 0:
            consensus_score = round(
                (5.0 * sb + 4.0 * buy + 3.0 * hold + 2.0 * underperf + 1.0 * sell) / trend_total,
                2,
            )
        else:
            consensus_score = None

        rec_key = str(info.get("recommendationKey") or "").upper()
        key_map = {
            "STRONG_BUY": "STRONG_BUY",
            "BUY": "BUY",
            "HOLD": "HOLD",
            "UNDERPERFORM": "UNDERPERFORM",
            "SELL": "SELL",
        }
        consensus_rating = key_map.get(rec_key)
        if not consensus_rating:
            if consensus_score is None:
                consensus_rating = "UNAVAILABLE"
            elif consensus_score >= 4.25:
                consensus_rating = "STRONG_BUY"
            elif consensus_score >= 3.5:
                consensus_rating = "BUY"
            elif consensus_score >= 2.5:
                consensus_rating = "HOLD"
            elif consensus_score >= 1.75:
                consensus_rating = "UNDERPERFORM"
            else:
                consensus_rating = "SELL"

        period_labels = {"0m": "Current", "-1m": "1M Ago", "-2m": "2M Ago", "-3m": "3M Ago"}
        revisions: List[Dict[str, Any]] = []
        for row in rec_trend:
            if not isinstance(row, dict):
                continue
            sb_r = int(row.get("strong_buy") or 0)
            buy_r = int(row.get("buy") or 0)
            hold_r = int(row.get("hold") or 0)
            underperf_r = int(row.get("underperform") or 0)
            sell_r = int(row.get("sell") or 0)
            tot_r = sb_r + buy_r + hold_r + underperf_r + sell_r
            buy_pct = round(((sb_r + buy_r) / tot_r) * 100.0, 1) if tot_r > 0 else 0.0
            raw_period = str(row.get("period") or "").strip()
            revisions.append({
                "period": period_labels.get(raw_period, raw_period or "Unknown"),
                "mean_target": mean_t if raw_period == "0m" else None,
                "buy_pct": buy_pct,
                "total_analysts": tot_r,
            })

        is_stale = bool((yf_meta or {}).get("_cache_stale"))
        is_cache_hit = bool((yf_meta or {}).get("_cache_hit"))
        snapshot_at = (yf_meta or {}).get("fetched_at")
        if not available:
            rating_source = "UNAVAILABLE"
        elif is_stale:
            rating_source = "YFINANCE_CACHED_STALE"
        elif is_cache_hit or not self.enable_live_fetch:
            rating_source = "YFINANCE_CACHED"
        else:
            rating_source = "YFINANCE_LIVE"

        res = {
            "ticker": ticker,
            "available": available,
            "source": rating_source,
            "snapshot_at": snapshot_at,
            "current_price": current_p,
            "target_mean": mean_t,
            "target_median": median_t,
            "target_high": high_t,
            "target_low": low_t,
            "upside_pct": upside,
            "consensus_rating": consensus_rating,
            "consensus_score": consensus_score,
            "total_analysts": total_analysts,
            "ratings_breakdown": {
                "strong_buy": sb,
                "buy": buy,
                "hold": hold,
                "underperform": underperf,
                "sell": sell,
            },
            "consensus_price_target": {
                "mean": mean_t,
                "median": median_t,
                "high": high_t,
                "low": low_t,
                "current_price": current_p,
                "upside_pct": upside,
            },
            "recommendation_distribution": {
                "strong_buy": sb,
                "buy": buy,
                "hold": hold,
                "underperform": underperf,
                "sell": sell,
                "total_analysts": total_analysts,
                "consensus_score": consensus_score,
                "consensus_rating": consensus_rating,
            },
            "historical_revisions": revisions,
        }
        with self._lock:
            self._ratings_cache[ticker] = (now_mono, res)
        return res

    def get_edgar_filings(self, ticker: str) -> Dict[str, Any]:
        """Retrieves authentic SEC EDGAR submissions, PIT dates, Item 1A diffs, and insider transactions."""
        ticker = ticker.upper()
        now_mono = time.monotonic()
        with self._lock:
            cached = self._edgar_cache.get(ticker)
            if cached is not None and (now_mono - cached[0]) < self._EDGAR_TTL_SECONDS:
                return cached[1]

        known_cik = None
        try:
            known_cik = self.universe_selector.get_security_master(ticker).cik
        except KeyError:
            pass

        res = self.fundamentals_client.build_edgar_filings_payload(
            ticker=ticker,
            known_cik=known_cik,
            enable_live_fetch=self.enable_live_fetch,
        )
        with self._lock:
            self._edgar_cache[ticker] = (now_mono, res)
        return res

    def get_financials(self, ticker: str, pit_strict: bool = True) -> Dict[str, Any]:
        """Retrieves real 10-year US-GAAP financial statements from SEC EDGAR XBRL CompanyFacts."""
        ticker = ticker.upper()
        key = (ticker, bool(pit_strict))
        now_mono = time.monotonic()
        with self._lock:
            cached = self._financials_cache.get(key)
            if cached is not None and (now_mono - cached[0]) < self._FINANCIALS_TTL_SECONDS:
                return cached[1]

        known_cik = None
        try:
            known_cik = self.universe_selector.get_security_master(ticker).cik
        except KeyError:
            pass

        res = self.fundamentals_client.build_financials_payload(
            ticker=ticker,
            known_cik=known_cik,
            pit_strict=bool(pit_strict),
            enable_live_fetch=self.enable_live_fetch,
        )
        with self._lock:
            self._financials_cache[key] = (now_mono, res)
        return res

    def get_news_feed(self, ticker: str) -> List[Dict[str, Any]]:
        """Fetches authentic headlines via yfinance with FinBERT-style lexical decay scoring."""
        ticker = ticker.upper()
        now_mono = time.monotonic()
        with self._lock:
            cached = self._news_cache.get(ticker)
            if cached is not None and (now_mono - cached[0]) < self._NEWS_TTL_SECONDS:
                return cached[1]

        if not self.enable_live_fetch or now_mono < self._failed_yf_until.get(f"news:{ticker}", 0.0):
            return []

        now_utc = datetime.now(timezone.utc)
        pos_words = {
            "beat", "beats", "surge", "surges", "growth", "upgrade", "upgraded", "record",
            "expand", "expands", "strong", "outperform", "partnership", "approval", "gain", "rally",
        }
        neg_words = {
            "miss", "misses", "drop", "drops", "downgrade", "cut", "cuts", "lawsuit", "probe",
            "delay", "risk", "warning", "slump", "decline", "shortfall", "tariff", "recall",
        }

        def _score_headline(text: str) -> Tuple[float, str]:
            tokens = {w.strip(".,:;!?()[]{}\"'").lower() for w in text.split()}
            pos_hits = len(tokens & pos_words)
            neg_hits = len(tokens & neg_words)
            raw = float(np.clip((pos_hits - neg_hits) * 0.28, -0.95, 0.95))
            label = "BULLISH" if raw > 0.15 else ("BEARISH" if raw < -0.15 else "NEUTRAL")
            return round(raw, 2), label

        items: List[Dict[str, Any]] = []
        try:
            import yfinance as yf

            raw_news = yf.Ticker(ticker).news or []
            for idx, entry in enumerate(raw_news[:10]):
                content = entry.get("content") if isinstance(entry.get("content"), dict) else entry
                title = str(content.get("title") or entry.get("title") or "").strip()
                if not title:
                    continue
                pub = (
                    content.get("provider", {}).get("displayName")
                    if isinstance(content.get("provider"), dict)
                    else entry.get("publisher", "Market Wire")
                )
                pub_ts = entry.get("providerPublishTime")
                pub_iso = content.get("pubDate")
                pub_iso_str = str(pub_iso).strip() if pub_iso else ""
                has_pub_time = bool(pub_iso_str and ("T" in pub_iso_str or ":" in pub_iso_str or " " in pub_iso_str))
                pub_verified = False
                pub_date_only = False
                dt_pub: Optional[datetime] = None
                pub_out = ""
                age_days = 0.0

                if pub_ts is not None and pub_ts != "" and not isinstance(pub_ts, bool):
                    try:
                        cand_dt = datetime.fromtimestamp(float(pub_ts), tz=timezone.utc)
                        if cand_dt.timestamp() > 0.0:
                            dt_pub = cand_dt
                            pub_out = (
                                dt_pub.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                                if dt_pub.microsecond
                                else dt_pub.strftime("%Y-%m-%dT%H:%M:%SZ")
                            )
                            age_days = max(0.0, (now_utc - dt_pub).total_seconds() / 86400.0)
                            pub_verified = True
                    except (ValueError, TypeError, OverflowError, OSError):
                        dt_pub = None

                if dt_pub is None and has_pub_time:
                    try:
                        cand_dt = datetime.fromisoformat(pub_iso_str.replace("Z", "+00:00"))
                        if cand_dt.tzinfo is None:
                            cand_dt = cand_dt.replace(tzinfo=timezone.utc)
                        else:
                            cand_dt = cand_dt.astimezone(timezone.utc)
                        dt_pub = cand_dt
                        pub_out = (
                            dt_pub.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                            if dt_pub.microsecond
                            else dt_pub.strftime("%Y-%m-%dT%H:%M:%SZ")
                        )
                        age_days = max(0.0, (now_utc - dt_pub).total_seconds() / 86400.0)
                        pub_verified = True
                    except (ValueError, TypeError, OverflowError):
                        dt_pub = None

                if dt_pub is None and not has_pub_time and len(pub_iso_str) == 10:
                    try:
                        # Date-only pubDate cannot establish intraday order -> remain unverified date-only string
                        date.fromisoformat(pub_iso_str)
                        pub_out = pub_iso_str
                        d_pub = datetime.fromisoformat(pub_out).replace(tzinfo=timezone.utc)
                        age_days = max(0.0, (now_utc - d_pub).total_seconds() / 86400.0)
                        pub_date_only = True
                        dt_pub = d_pub
                    except (ValueError, TypeError, OverflowError):
                        dt_pub = None

                if dt_pub is None:
                    dt_pub = now_utc - pd.Timedelta(hours=idx * 6)
                    pub_out = dt_pub.strftime("%Y-%m-%dT%H:%M:%SZ")
                    age_days = max(0.0, (now_utc - dt_pub).total_seconds() / 86400.0)
                    pub_verified = False
                    pub_date_only = False
                score, label = _score_headline(title)
                decayed = round(score * float(0.5 ** (age_days / 3.0)), 2)
                items.append({
                    "id": f"yf-{ticker}-{idx + 1}",
                    "ticker": ticker,
                    "headline": title,
                    "source": str(pub or "Market Wire"),
                    "published_at": pub_out,
                    "published_at_verified": pub_verified,
                    "published_date_only": pub_date_only,
                    "finbert_score": score,
                    "sentiment_label": label,
                    "decay_half_life_days": 3.0,
                    "decayed_sentiment": decayed,
                    "category": "LIVE_WIRE",
                })
        except Exception:
            with self._lock:
                self._failed_yf_until[f"news:{ticker}"] = now_mono + 180.0
            return []

        with self._lock:
            self._news_cache[ticker] = (now_mono, items)
        return items

    def query_copilot(self, query: str, ticker: Optional[str] = None) -> Dict[str, Any]:
        """Deterministic quant copilot router backed by live model inference and market data."""
        eff_ticker = (ticker or "SPY").upper()
        q_lower = query.lower()

        if any(k in q_lower for k in ("kelly", "size", "sizing", "allocation", "conviction", "signal", "buy", "sell")):
            tool_routed = "kelly_sizer"
            sig = self.get_signal(eff_ticker)
            rec = sig.get("execution_recommendation", sig)
            cal = sig.get("calibration", {}).get("calibrated_probabilities", {})
            action = rec.get("action", "HOLD")
            conviction = rec.get("conviction_tier", "LOW")
            kelly_pct = float(rec.get("unconstrained_kelly_pct", 0.0))
            alloc_pct = float(rec.get("l1_normalized_portfolio_allocation_pct", 0.0))
            entry = float(rec.get("entry_price", 0.0))
            tp = float(rec.get("dynamic_take_profit", 0.0))
            sl = float(rec.get("dynamic_stop_loss", 0.0))
            rr = float(rec.get("reward_to_risk_ratio", 2.0))
            p_b = float(cal.get("outperform_buy", 0.33)) * 100.0
            p_h = float(cal.get("neutral_hold", 0.34)) * 100.0
            p_s = float(cal.get("underperform_sell", 0.33)) * 100.0
            resp_text = (
                f"QARE-MM v1.3 Live Signal for {eff_ticker} (@ ${entry:.2f}): "
                f"Action: {action} ({conviction} conviction). "
                f"Calibrated Posteriors: Buy {p_b:.1f}% / Hold {p_h:.1f}% / Sell {p_s:.1f}%. "
                f"Kelly Allocation: {alloc_pct:.2f}% (Unconstrained: {kelly_pct:.2f}%). "
                f"Dynamic ATR Take-Profit: ${tp:.2f}, Stop-Loss: ${sl:.2f} (R:R {rr:.2f}:1)."
            )
            routed_data = sig
        elif any(k in q_lower for k in ("regime", "volatility", "macro", "hmm")):
            tool_routed = "regime_hmm"
            reg = self.get_macro_regime()
            state = reg.get("current_state", "UNAVAILABLE")
            kappa_val = reg.get("kappa")
            kappa_str = (
                f"{float(kappa_val):.2f}"
                if isinstance(kappa_val, (int, float)) and not isinstance(kappa_val, bool) and math.isfinite(float(kappa_val))
                else "N/A"
            )
            state_idx = reg.get("state_index", -1)
            probs = reg.get("filtered_probabilities") or {}
            if reg.get("available") is False or not probs:
                resp_text = f"Macro Volatility Regime (SPY HMM): {state} (kappa={kappa_str}; probabilities unavailable)."
            else:
                resp_text = (
                    f"Live Macro Volatility Regime (SPY HMM): {state} (State {state_idx}, kappa={kappa_str}). "
                    f"Filtered posteriors: Low {probs.get('low_volatility', 0.0) * 100:.1f}%, "
                    f"Med {probs.get('medium_volatility', 0.0) * 100:.1f}%, "
                    f"High {probs.get('high_volatility', 0.0) * 100:.1f}%."
                )
            routed_data = reg
        elif any(k in q_lower for k in ("insider", "form 4", "cluster", "sec", "edgar", "10-k", "10-q")):
            tool_routed = "sec_edgar"
            edg = self.get_edgar_filings(eff_ticker)
            sim = edg.get("item1a_similarity")
            drift = edg.get("item1a_sentiment_drift")
            cat = edg.get("mda_catalyst_score")
            sim_str = f"{sim:.2f}" if sim is not None else "N/A"
            drift_str = f"{drift:+.2f}" if drift is not None else "N/A"
            cat_str = f"{cat:+.2f}" if cat is not None else "N/A"
            resp_text = (
                f"SEC EDGAR Intelligence for {eff_ticker} (latest filing {edg.get('current_filing_date') or 'Unavailable'}): "
                f"Item 1A Risk Factor cosine similarity is {sim_str} (sentiment drift {drift_str}), "
                f"MD&A catalyst score {cat_str}. "
                f"{len(edg.get('form4_transactions', []))} insider Form 4 disclosures tracked under T+1 PIT cutoff."
            )
            routed_data = edg
        elif any(k in q_lower for k in ("peer", "competitor", "supply chain", "splc", "graph")):
            tool_routed = "peer_gat"
            pg = self.get_peer_graph(eff_ticker)
            top_p = pg.get("top_spillover_contributor", "N/A")
            entropy = float(pg.get("gat_attention_entropy", 0.0))
            resp_text = (
                f"Dynamic Peer GAT Topology for {eff_ticker}: {len(pg.get('nodes', []))} nodes, "
                f"{len(pg.get('edges', []))} edges, attention entropy {entropy:.2f}. "
                f"Top 5-day lead-lag spillover contributor: {top_p}."
            )
            routed_data = pg
        elif any(k in q_lower for k in ("news", "headline", "sentiment")):
            tool_routed = "news_sentiment"
            feed = self.get_news_feed(eff_ticker)
            if feed:
                top_item = feed[0]
                resp_text = (
                    f"News Sentiment Feed ({len(feed)} items for {eff_ticker}): "
                    f"Latest [{top_item.get('sentiment_label', 'NEUTRAL')} {top_item.get('decayed_sentiment', 0.0):+.2f}] "
                    f"\"{top_item.get('headline', '')}\"."
                )
            else:
                resp_text = f"No live Yahoo Finance news headlines currently available for {eff_ticker}."
            routed_data = {"items": feed[:5]}
        elif any(k in q_lower for k in ("price", "quote", "history", "hp", "bar", "close")):
            tool_routed = "historical_prices"
            quote = self.get_latest_quote(eff_ticker)
            bars = self.get_historical_prices(eff_ticker, count=5, interval="1d")
            resp_text = (
                f"Live Quote for {eff_ticker}: ${quote.get('price', 0.0):.2f} "
                f"({quote.get('change', 0.0):+.2f} / {quote.get('change_pct', 0.0):+.2f}%), "
                f"as of {quote.get('date', 'N/A')}."
            )
            routed_data = {"quote": quote, "recent_bars": bars}
        else:
            tool_routed = "general_quant"
            sig = self.get_signal(eff_ticker)
            quote = self.get_latest_quote(eff_ticker)
            rec = sig.get("execution_recommendation", {})
            reg = sig.get("macro_regime", {})
            kappa_val = reg.get("kappa")
            kappa_str = (
                f"{float(kappa_val):.2f}"
                if isinstance(kappa_val, (int, float)) and not isinstance(kappa_val, bool) and math.isfinite(float(kappa_val))
                else "N/A"
            )
            resp_text = (
                f"Amstal QARE-MM Live Brief for {eff_ticker} (${quote.get('price', 0.0):.2f}, "
                f"{quote.get('date', 'N/A')}): Signal={rec.get('action', 'HOLD')} "
                f"({rec.get('conviction_tier', 'LOW')}), Kelly={rec.get('l1_normalized_portfolio_allocation_pct', 0.0):.2f}%, "
                f"Regime={reg.get('current_state', 'UNAVAILABLE')} (kappa={kappa_str})."
            )
            routed_data = {"quote": quote, "signal": sig}

        return {
            "query": query,
            "ticker": eff_ticker,
            "tool_routed": tool_routed,
            "answer": resp_text,
            "response": resp_text,
            "content": resp_text,
            "sources": [
                f"QARE-MM v1.3 {tool_routed.upper()} Pipeline",
                f"Amstal Live Lakehouse ({eff_ticker})",
                "Forward-Filtered 3-State HMM",
            ],
            "suggested_commands": [
                f"{eff_ticker} PORT <GO>",
                f"{eff_ticker} GP <GO>",
                f"{eff_ticker} HP <GO>",
                "HMM <GO>",
            ],
            "model": "QARE-MM-v1.3-Copilot",
            "data": routed_data,
        }
