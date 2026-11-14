"""
Amstal Live Model Inference Providers (QARE-MM v1.3)
Connects trained model weights from weights/ to IPC JSON-RPC requests,
enabling fast cached multi-modal inference, causal HMM regime filtering,
authentic SEC EDGAR / Yahoo Finance fundamentals, and Kelly sizing.
"""

from datetime import datetime, timezone
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

import numpy as np
import pandas as pd

from src.data.fundamentals_client import FundamentalsClient
from src.data.market_loader import MarketDataLoader
from src.data.universe import UniverseSelector
from src.features.graph_builder import GraphBuilder
from src.features.microstructure import MicrostructureFeatureExtractor
from src.features.residualizer import Residualizer
from src.ipc.ws_broadcaster import build_qare_signal_frame
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
            mtime = pq_path.stat().st_mtime
            with self._lock:
                cached = self._df_cache.get(sym)
                if cached is not None and cached[0] == mtime:
                    return cached[1]
            df = self.market_loader.load_ticker_daily(sym)
            if df is not None:
                with self._lock:
                    self._df_cache[sym] = (mtime, df)
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
            self._daily_refresh_attempted[sym] = now

        expected_latest = self.market_loader._latest_expected_trading_date()
        df_disk = self._load_disk_daily(sym)
        if df_disk is not None and len(df_disk) > 0:
            last_dt = pd.Timestamp(df_disk.index[-1]).normalize()
            if last_dt >= expected_latest:
                return

        refreshed = self.market_loader.refresh_ticker_daily(sym)
        with self._lock:
            if refreshed is not None and len(refreshed) > 0:
                self._df_cache.pop(sym, None)
            else:
                self._failed_yf_until[sym] = time.monotonic() + self._FAILED_TICKER_COOLDOWN_SECONDS

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
            if quote is not None and quote.get("price", 0.0) > 0.0:
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

    def _load_cached_daily(self, ticker: str, allow_refresh: bool = True) -> Optional[pd.DataFrame]:
        """Loads daily Parquet for ticker, optionally refreshing stale partitions and overlaying live quote."""
        sym = ticker.upper()
        if allow_refresh and self.enable_live_fetch:
            self._maybe_refresh_daily(sym)

        df = self._load_disk_daily(sym)
        if df is None or len(df) == 0:
            return None

        if not self.enable_live_fetch or not allow_refresh:
            return df

        live_q = self._fetch_live_quote_cached(sym)
        if live_q is None or not live_q.get("date") or float(live_q.get("price", 0.0)) <= 0.0:
            return df

        try:
            q_ts = pd.Timestamp(live_q["date"]).normalize()
            last_ts = pd.Timestamp(df.index[-1]).normalize()
            price = float(live_q["price"])
            open_p = float(live_q.get("open") or price)
            high_p = max(float(live_q.get("high") or price), price, open_p)
            low_p = min(float(live_q.get("low") or price), price, open_p)
            vol_q = int(live_q.get("volume") or 0)

            df_live = df.copy()
            if q_ts == last_ts:
                idx_last = df_live.index[-1]
                df_live.loc[idx_last, "close"] = price
                df_live.loc[idx_last, "high"] = max(float(df_live.loc[idx_last, "high"]), high_p)
                df_live.loc[idx_last, "low"] = min(float(df_live.loc[idx_last, "low"]), low_p)
                if vol_q > 0:
                    df_live.loc[idx_last, "volume"] = max(float(df_live.loc[idx_last, "volume"]), float(vol_q))
                return df_live
            elif q_ts > last_ts:
                new_row = pd.DataFrame(
                    [{
                        "open": open_p,
                        "high": high_p,
                        "low": low_p,
                        "close": price,
                        "volume": float(vol_q),
                    }],
                    index=pd.DatetimeIndex([q_ts], name=df_live.index.name or "timestamp"),
                )
                return pd.concat([df_live, new_row]).sort_index()
        except Exception:
            pass
        return df

    def get_macro_regime(self) -> Dict[str, Any]:
        """Runs cached causal forward-filtered Gaussian HMM on SPY benchmark."""
        if self.regime_hmm is None:
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

        try:
            df_spy = self._load_cached_daily("SPY")
            if df_spy is None or len(df_spy) < 30:
                return {
                    "available": False,
                    "current_state": "UNAVAILABLE",
                    "state_index": -1,
                    "kappa": None,
                    "filtered_probabilities": {},
                    "transition_matrix": self.regime_hmm.transmat_.tolist() if self.regime_hmm.transmat_ is not None else [],
                    "state_volatilities": self.regime_hmm.get_state_annualized_volatilities(),
                    "stationary_distribution": self.regime_hmm.get_stationary_distribution(),
                    "regime_history": [],
                }

            last_idx = df_spy.index[-1]
            last_dt = last_idx.strftime("%Y-%m-%d") if hasattr(last_idx, "strftime") else str(last_idx)[:10]
            cache_key = (last_dt, round(float(df_spy["close"].iloc[-1]), 4))
            now_mono = time.monotonic()

            with self._lock:
                if (
                    self._regime_cache is not None
                    and self._regime_cache[0] == cache_key
                    and (now_mono - self._regime_cache[1]) < self._REGIME_TTL_SECONDS
                ):
                    return self._regime_cache[2]

            spy_ret = np.diff(np.log(df_spy["close"].values))
            high_low = df_spy["high"].values[1:] / np.maximum(df_spy["low"].values[1:], 1e-8)
            spy_vol = (np.log(high_low) ** 2) / (4.0 * np.log(2.0))
            X_hmm = np.column_stack([spy_ret, spy_vol])

            filtered_probs = self.regime_hmm.filter_sequence(X_hmm)
            curr_probs = filtered_probs[-1]
            state_idx = int(np.argmax(curr_probs))
            state_label = self.regime_hmm.STATE_LABELS.get(state_idx, "LOW_VOLATILITY")
            p_low = float(curr_probs[0])
            p_med = float(curr_probs[1])
            p_high = float(curr_probs[2])
            kappa = round(0.85 * (1.0 - 0.80 * p_high) * (1.0 - 0.20 * p_med), 2)
            transmat = (
                self.regime_hmm.transmat_.tolist()
                if getattr(self.regime_hmm, "transmat_", None) is not None
                else default_transmat
            )

            # Build real trailing 60-day causal forward-filtered trajectory for RegimePanel
            hist_slice = filtered_probs[-60:]
            hist_dates = df_spy.index[1:][-60:]
            regime_history = []
            for dt_idx, row_p in zip(hist_dates, hist_slice):
                dt_str = dt_idx.strftime("%Y-%m-%d") if hasattr(dt_idx, "strftime") else str(dt_idx)[:10]
                s_k = int(np.argmax(row_p))
                regime_history.append({
                    "timestamp": dt_str,
                    "state_index": s_k,
                    "p_low": round(float(row_p[0]), 4),
                    "p_med": round(float(row_p[1]), 4),
                    "p_high": round(float(row_p[2]), 4),
                    "confidence": round(float(row_p[s_k]), 4),
                })

            result = {
                "available": True,
                "current_state": state_label,
                "state_index": state_idx,
                "kappa": float(kappa),
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
            with self._lock:
                self._regime_cache = (cache_key, now_mono, result)
            return result
        except Exception as e:
            logger.warning(f"Error computing live macro regime: {e}")
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
        return {
            "ticker": ticker,
            "price": round(close_p, 2),
            "change": chg,
            "change_pct": chg_pct,
            "volume": vol,
            "date": bar_date,
            "timestamp_ms": int(time.time() * 1000),
        }

    def get_signal(self, ticker: str, regime: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extracts trailing microstructure/residual features, runs LightGBM inference, and computes Kelly sizing."""
        ticker = ticker.upper()
        regime = regime or self.get_macro_regime()
        raw_kappa = regime.get("kappa")
        kappa: Optional[float] = float(raw_kappa) if isinstance(raw_kappa, (int, float)) else None
        regime_state = str(regime.get("current_state", "UNAVAILABLE"))
        regime_idx = int(regime.get("state_index", -1))
        regime_probs = regime.get("filtered_probabilities") or {}
        transmat = regime.get("transition_matrix")

        df = self._load_cached_daily(ticker)
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
        entry_price = round(float(df["close"].iloc[-1]), 2)
        kappa_key = round(kappa, 3) if kappa is not None else -1.0
        cache_key = (last_dt, entry_price, regime_idx, kappa_key)
        now_mono = time.monotonic()

        with self._lock:
            cached = self._signal_cache.get(ticker)
            if cached is not None and cached[0] == cache_key and (now_mono - cached[1]) < self._SIGNAL_TTL_SECONDS:
                return cached[2]

        has_real_vol = len(df) >= 22
        if has_real_vol:
            vol_20d = max(float(np.std(np.diff(np.log(df["close"].values[-21:])))), 0.005)
            tp = round(entry_price * (1.0 + 2.0 * vol_20d), 2)
            sl = round(entry_price * (1.0 - 1.0 * vol_20d), 2)
            rr_val = 2.0
        else:
            vol_20d = None
            tp = 0.0
            sl = 0.0
            rr_val = 0.0

        if not self.is_ready() or len(df) < 80 or not has_real_vol:
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
            try:
                sector_etf = self.universe_selector.get_sector_etf(ticker)
            except KeyError:
                sector_etf = "SPY"

            df_sec = self._load_cached_daily(sector_etf, allow_refresh=(sector_etf == "SPY"))
            df_spy = self._load_cached_daily("SPY")

            if df_sec is None:
                df_sec = df
            if df_spy is None:
                df_spy = df

            common_idx = df.index.intersection(df_spy.index).intersection(df_sec.index)
            # Trailing 320 bars is strictly >= 60d OLS + 126d rank warmup and avoids processing 6y of history per call
            if len(common_idx) >= 126:
                tail_idx = common_idx[-self._TAIL_WARMUP_BARS:]
                df_sub = df.loc[tail_idx]
                spy_sub = df_spy.loc[tail_idx]
                sec_sub = df_sec.loc[tail_idx]
            else:
                df_sub = df.tail(self._TAIL_WARMUP_BARS)
                spy_sub = df_sub
                sec_sub = df_sub

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

            cal_probs = self.temp_scaler.transform(raw_probs.reshape(1, -1))[0]
            cal_probs = np.clip(cal_probs, 0.015, 0.97)
            cal_probs /= np.sum(cal_probs)

            p_buy, p_hold, p_sell = float(cal_probs[0]), float(cal_probs[1]), float(cal_probs[2])

            eff_kelly_kappa = kappa if kappa is not None else 0.0
            assert vol_20d is not None
            rec = self.kelly_sizer.compute_recommendation(
                p_buy=p_buy,
                p_hold=p_hold,
                p_sell=p_sell,
                b_upper=2.0 * vol_20d,
                b_lower=1.0 * vol_20d,
                entry_price=entry_price,
                kappa=eff_kelly_kappa,
                execution_mode="LONG_ONLY",
            )

            res_mom_rank = self._ts_rank(channels_df["residual_momentum_20d"])
            on_ret_rank = self._ts_rank(channels_df["overnight_log_return"])
            intra_ret_rank = self._ts_rank(channels_df["intraday_log_return"])
            amihud_rank = self._ts_rank(channels_df["amihud_illiquidity_z"])
            cp_rank = self._ts_rank(channels_df["closing_power"])

            # Populate only authentic alt-data/peer values if already cached; mark missing honestly
            missing_modalities: List[str] = []
            if kappa is None:
                missing_modalities.append("REGIME_HMM_UNAVAILABLE")
            with self._lock:
                cached_news = self._news_cache.get(ticker)
                cached_edgar = self._edgar_cache.get(ticker)
                cached_pg = self._peer_graph_cache.get(ticker)

            if cached_news is not None and cached_news[1]:
                news_scores = [float(it.get("decayed_sentiment", 0.0)) for it in cached_news[1]]
                finbert_val = round(float(np.mean(news_scores)), 2)
            else:
                finbert_val = 0.0
                missing_modalities.append("FINBERT_NEWS_NOT_LOADED")

            form4_val = 0.0
            legal_delta = 0.0
            mda_cat = 0.0
            if cached_edgar is not None and cached_edgar[1].get("available"):
                edg = cached_edgar[1]
                if edg.get("item1a_sentiment_drift") is not None:
                    legal_delta = float(edg["item1a_sentiment_drift"])
                else:
                    missing_modalities.append("EDGAR_ITEM1A_DIFF")
                if edg.get("mda_catalyst_score") is not None:
                    mda_cat = float(edg["mda_catalyst_score"])
                txs = edg.get("form4_transactions") or []
                if txs:
                    signed_v = sum(
                        (1.0 if tx.get("transaction_code") == "P" else -0.5)
                        * (0.2 if tx.get("is_10b5_1") else 1.0)
                        * math.log1p(max(float(tx.get("value", 0.0)), 0.0))
                        for tx in txs
                    )
                    form4_val = round(math.tanh(signed_v / 25.0), 2)
            else:
                missing_modalities.append("EDGAR_FILINGS_NOT_LOADED")

            if cached_pg is not None and cached_pg[1].get("nodes"):
                spill_vals = [float(n.get("spillover_contribution", 0.0)) for n in cached_pg[1]["nodes"] if n.get("role") != "TARGET"]
                peer_spill = round(float(np.clip(sum(spill_vals) / 10.0, -1.0, 1.0)), 2)
            else:
                peer_spill = 0.0
                missing_modalities.append("PEER_GAT_NOT_LOADED")

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
                "smart_money_divergence_pos": round(max(0.0, float(cp_rank - intra_ret_rank) * 0.5), 2),
                "smart_money_divergence_neg": round(max(0.0, float(intra_ret_rank - cp_rank) * 0.5), 2),
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
                l1_normalized_pct=round(rec.l1_normalized_portfolio_allocation_pct, 2),
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

        if df is not None and len(df) >= 65:
            try:
                df_sec = self._load_cached_daily(sector_etf, allow_refresh=(sector_etf == "SPY"))
                df_spy = self._load_cached_daily("SPY")
                if df_sec is None:
                    df_sec = df
                if df_spy is None:
                    df_spy = df
                common_idx = df.index.intersection(df_spy.index).intersection(df_sec.index)
                if len(common_idx) >= 65:
                    sub_idx = common_idx[-65:]
                    r_stock = np.diff(np.log(df.loc[sub_idx, "close"].values))
                    r_spy = np.diff(np.log(df_spy.loc[sub_idx, "close"].values))
                    r_sec = np.diff(np.log(df_sec.loc[sub_idx, "close"].values))
                    roll_res = self.residualizer.compute_rolling_residuals(r_stock, r_spy, r_sec)
                    if not np.isnan(roll_res.betas_mkt[-1]):
                        beta_mkt = round(float(roll_res.betas_mkt[-1]), 2)
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
            raw_dy = self._clean_opt_float(info.get("dividendYield"), scale=1.0, digits=2)
            if raw_dy is not None:
                div_yield = round(raw_dy * 100.0, 2) if raw_dy < 0.25 else raw_dy

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
        df_target = self._load_cached_daily(ticker, allow_refresh=True)
        if df_target is not None:
            loaded_dfs[ticker] = df_target
        df_etf = self._load_cached_daily(sector_etf, allow_refresh=False)
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
                d = self._load_cached_daily(sym, allow_refresh=False)
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
            if sym == "SPY":
                return 1.0
            d = _get_df(sym)
            if d is None or df_spy is None or len(d) < 30 or len(df_spy) < 30:
                return None
            idx = d.index.intersection(df_spy.index)[-61:]
            if len(idx) < 25:
                return None
            rs = np.diff(np.log(d.loc[idx, "close"].values))
            rm = np.diff(np.log(df_spy.loc[idx, "close"].values))
            var_m = float(np.var(rm))
            if var_m <= 1e-12:
                return None
            cov = float(np.mean((rs - np.mean(rs)) * (rm - np.mean(rm))))
            return round(float(np.clip(cov / var_m, -1.0, 4.0)), 2)

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

        nodes: List[Dict[str, Any]] = [
            {
                "id": ticker,
                "role": "TARGET",
                "role_code": 0.0,
                "label": ticker,
                "beta_market": _beta_mkt(ticker),
                "return_5d": _ret_5d(ticker),
                "spillover_contribution": 0.0,
            }
        ]
        edges: List[Dict[str, Any]] = []
        for p, w in zip(clean_gics, weights_gics):
            r5 = _ret_5d(p)
            nodes.append({
                "id": p,
                "role": "GICS_PEER",
                "role_code": 0.33,
                "label": p,
                "beta_market": _beta_mkt(p),
                "return_5d": r5,
                "spillover_contribution": round(r5 * w * 0.25, 2),
            })
            edges.append({"source": ticker, "target": p, "attention_weight": w})

        best_ll_peer = clean_ll[0] if clean_ll else sector_etf
        best_ll_abs = -1.0
        for p, w, c5 in zip(clean_ll, weights_ll, ll_corrs):
            r5 = _ret_5d(p)
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
            })
            edges.append({"source": p, "target": ticker, "attention_weight": w, "lead_days": 5})

        etf_r5 = _ret_5d(sector_etf)
        nodes.append({
            "id": sector_etf,
            "role": "SECTOR_ETF",
            "role_code": 1.0,
            "label": sector_etf,
            "beta_market": _beta_mkt(sector_etf),
            "return_5d": etf_r5,
            "spillover_contribution": round(-etf_r5 * 0.08, 2),
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

        df = self._load_cached_daily(ticker)
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

        return records[::-1]

    def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """Provides authentic Wall Street price targets & analyst consensus from Yahoo Finance."""
        ticker = ticker.upper()
        now_mono = time.monotonic()
        with self._lock:
            cached = self._ratings_cache.get(ticker)
            if cached is not None and (now_mono - cached[0]) < self._RATINGS_TTL_SECONDS:
                return cached[1]

        df = self._load_cached_daily(ticker)
        current_p = round(float(df["close"].iloc[-1]), 2) if df is not None and len(df) > 0 else 0.0

        yf_meta = self.fundamentals_client.get_yf_meta(ticker, enable_live_fetch=self.enable_live_fetch)
        info = (yf_meta or {}).get("info", {})
        rec_trend = (yf_meta or {}).get("rec_trend", [])

        mean_t = self._clean_opt_float(info.get("targetMeanPrice"), digits=2)
        median_t = self._clean_opt_float(info.get("targetMedianPrice"), digits=2)
        high_t = self._clean_opt_float(info.get("targetHighPrice"), digits=2)
        low_t = self._clean_opt_float(info.get("targetLowPrice"), digits=2)
        live_p = self._clean_opt_float(info.get("currentPrice") or info.get("regularMarketPrice"), digits=2)
        if live_p and live_p > 0:
            current_p = live_p

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
            tot_r = row["strong_buy"] + row["buy"] + row["hold"] + row["underperform"] + row["sell"]
            buy_pct = round(((row["strong_buy"] + row["buy"]) / tot_r) * 100.0, 1) if tot_r > 0 else 0.0
            revisions.append({
                "period": period_labels.get(row["period"], row["period"]),
                "mean_target": mean_t if row["period"] == "0m" else None,
                "buy_pct": buy_pct,
                "total_analysts": tot_r,
            })

        res = {
            "ticker": ticker,
            "available": available,
            "source": "YFINANCE_LIVE" if available else "UNAVAILABLE",
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
            raw = float(np.clip((pos_hits - neg_hits) * 0.28 + 0.10, -0.95, 0.95))
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
                if pub_ts:
                    dt_pub = datetime.fromtimestamp(int(pub_ts), tz=timezone.utc)
                elif pub_iso:
                    dt_pub = datetime.fromisoformat(str(pub_iso).replace("Z", "+00:00"))
                else:
                    dt_pub = now_utc - pd.Timedelta(hours=idx * 6)
                age_days = max(0.0, (now_utc - dt_pub).total_seconds() / 86400.0)
                score, label = _score_headline(title)
                decayed = round(score * float(0.5 ** (age_days / 3.0)), 2)
                items.append({
                    "id": f"yf-{ticker}-{idx + 1}",
                    "ticker": ticker,
                    "headline": title,
                    "source": str(pub or "Market Wire"),
                    "published_at": dt_pub.strftime("%Y-%m-%dT%H:%M:%SZ"),
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
            kappa_str = f"{float(kappa_val):.2f}" if isinstance(kappa_val, (int, float)) else "N/A"
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
            kappa_str = f"{float(kappa_val):.2f}" if isinstance(kappa_val, (int, float)) else "N/A"
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
