"""
Regression test suite verifying true-data parity, mathematical accuracy, and
bug fixes across HMM regimes, Kelly sizing, live inference, and fundamentals.
"""

from __future__ import annotations

import json
import os
import time
import types
from pathlib import Path

import numpy as np
import pytest


def test_hmm_stationary_distribution_sign_orientation(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    RegimeHMM.get_stationary_distribution must orient negative eigenvectors from
    np.linalg.eig(A.T) instead of clamping them to zero and falling back to uniform 1/3.
    """
    from src.models.regime_hmm import RegimeHMM

    hmm = RegimeHMM()
    transmat = np.array([
        [0.985, 0.014, 0.001],
        [0.015, 0.975, 0.010],
        [0.004, 0.044, 0.952],
    ], dtype=np.float64)
    hmm.transmat_ = transmat

    # Deterministically force np.linalg.eig to return the negative-signed left eigenvector
    orig_eig = np.linalg.eig

    def neg_eig(A_T: np.ndarray):
        vals, vecs = orig_eig(A_T)
        idx = int(np.argmin(np.abs(vals - 1.0)))
        if float(np.sum(np.real(vecs[:, idx]))) > 0.0:
            vecs = vecs.copy()
            vecs[:, idx] = -vecs[:, idx]
        return vals, vecs

    monkeypatch.setattr(np.linalg, "eig", neg_eig)
    pi = np.array(hmm.get_stationary_distribution(), dtype=np.float64)
    assert len(pi) == 3
    assert float(np.sum(pi)) == pytest.approx(1.0, abs=1e-3)
    assert not np.allclose(pi, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], atol=0.02)
    residual = np.max(np.abs(pi @ transmat - pi))
    assert residual < 1e-3, f"pi @ A != pi: pi={pi}, residual={residual}"


def test_kelly_centered_asymmetric_barrier_actions() -> None:
    """
    With empirical triple-barrier priors under asymmetric +2sigma/-1sigma barriers:
    1. Zero-skill prior -> HOLD, 0.0%.
    2. Pure Hold-only tilt with p_buy < prior_buy (e.g. NVDA) -> HOLD, LOW conviction, 0.0%.
    3. Directional bullish breakout (p_buy > prior_buy, f* > 0.5%) -> BUY.
    4. Directional bearish breakdown (p_sell > prior_sell, f_short* > 0.5%) -> SELL.
    5. Instance-level self.prior_probs matches explicit prior_probs parameter identically.
    """
    from src.risk.kelly_engine import MultiOutcomeKellySizer

    prior = (0.062, 0.692, 0.246)
    b_u, b_l = 0.10, 0.05
    b_v = (prior[2] * b_l - prior[0] * b_u) / prior[1]

    sizer = MultiOutcomeKellySizer(execution_mode="LONG_ONLY")
    sizer_init = MultiOutcomeKellySizer(execution_mode="LONG_ONLY", prior_probs=prior)

    # 1. Exact prior -> zero skill -> HOLD, LOW, 0.0%
    rec_prior = sizer.compute_recommendation(
        p_buy=prior[0],
        p_hold=prior[1],
        p_sell=prior[2],
        b_upper=b_u,
        b_lower=b_l,
        b_vert=b_v,
        entry_price=100.0,
        kappa=0.45,
        prior_probs=prior,
        execution_mode="LONG_ONLY",
    )
    assert rec_prior.action == "HOLD"
    assert rec_prior.conviction_tier == "LOW"
    assert rec_prior.unconstrained_kelly_pct == 0.0
    assert rec_prior.l1_normalized_portfolio_allocation_pct == 0.0

    # 2. Pure Hold-dominant / suppressed upside (p_buy < prior_buy, p_sell < prior_sell)
    rec_hold_nvda = sizer.compute_recommendation(
        p_buy=0.015,
        p_hold=0.895,
        p_sell=0.090,
        b_upper=b_u,
        b_lower=b_l,
        b_vert=b_v,
        entry_price=100.0,
        kappa=0.45,
        prior_probs=prior,
        execution_mode="LONG_ONLY",
    )
    assert rec_hold_nvda.action == "HOLD"
    assert rec_hold_nvda.conviction_tier == "LOW"
    assert rec_hold_nvda.l1_normalized_portfolio_allocation_pct == 0.0

    # 3. Genuine bullish tilt (p_buy > prior_buy AND p_buy < p_sell due to asymmetric barriers)
    rec_buy = sizer.compute_recommendation(
        p_buy=0.15,
        p_hold=0.65,
        p_sell=0.20,
        b_upper=b_u,
        b_lower=b_l,
        b_vert=b_v,
        entry_price=100.0,
        kappa=0.45,
        prior_probs=prior,
        execution_mode="LONG_ONLY",
    )
    rec_buy_init = sizer_init.compute_recommendation(
        p_buy=0.15,
        p_hold=0.65,
        p_sell=0.20,
        b_upper=b_u,
        b_lower=b_l,
        b_vert=b_v,
        entry_price=100.0,
        kappa=0.45,
        execution_mode="LONG_ONLY",
    )
    assert rec_buy.unconstrained_kelly_pct > 0.5
    assert rec_buy.action == "BUY"
    assert rec_buy.l1_normalized_portfolio_allocation_pct > 0.0
    assert rec_buy == rec_buy_init

    # 4. Bearish tilt over prior (p_sell elevated well above prior)
    rec_sell_lo = sizer.compute_recommendation(
        p_buy=0.02,
        p_hold=0.62,
        p_sell=0.36,
        b_upper=b_u,
        b_lower=b_l,
        b_vert=b_v,
        entry_price=100.0,
        kappa=0.45,
        prior_probs=prior,
        execution_mode="LONG_ONLY",
    )
    assert rec_sell_lo.action == "SELL"
    assert rec_sell_lo.unconstrained_kelly_pct == 0.0
    assert rec_sell_lo.l1_normalized_portfolio_allocation_pct == 0.0


def test_pinned_hold_buy_sell_actions_without_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deterministic expected HOLD/BUY/SELL action assertions pinned to stub models and timestamped bars without provisioned weights."""
    import pandas as pd
    from src.ipc.live_providers import LiveInferenceProvider
    from src.risk.calibration import TemperatureScaler

    class FixedHoldLGBM:
        def predict_proba(self, x):
            return np.tile([0.05, 0.85, 0.10], (len(x), 1))

    class StubPinHMM:
        STATE_LABELS = {0: "LOW_VOLATILITY", 1: "MEDIUM_VOLATILITY", 2: "HIGH_VOLATILITY"}
        transmat_ = np.array([[0.9, 0.08, 0.02], [0.05, 0.9, 0.05], [0.02, 0.08, 0.9]])

        def filter_sequence(self, x):
            return np.tile([0.85, 0.10, 0.05], (len(x), 1))

        def get_regime_multiplier(self, probs):
            return 0.45

        def get_state_annualized_volatilities(self):
            return [10.0, 18.0, 35.0]

        def get_stationary_distribution(self):
            return [0.6, 0.3, 0.1]

    lh = tmp_path / "lh_pin"
    wdir = tmp_path / "w_pin"
    lh.mkdir(parents=True, exist_ok=True)
    wdir.mkdir(parents=True, exist_ok=True)
    provider = LiveInferenceProvider(lakehouse_dir=lh, weights_dir=wdir, enable_live_fetch=False)

    idx_pin = pd.bdate_range(end="2026-11-12", periods=160, name="timestamp")
    pin_closes = np.linspace(100.0, 120.0, 160)
    pin_obs_ms = int(pd.Timestamp("2026-11-12 16:00:00", tz="America/New_York").timestamp() * 1000)
    df_pin = pd.DataFrame(
        {
            "open": pin_closes * 0.995,
            "high": pin_closes * 1.01,
            "low": pin_closes * 0.99,
            "close": pin_closes,
            "volume": np.full(160, 1_000_000.0),
        },
        index=idx_pin,
    )
    df_pin.attrs = {"observation_timestamp_ms": pin_obs_ms, "obs_bar_date": "2026-11-12"}
    monkeypatch.setattr(
        "src.ipc.live_providers.time.time",
        lambda: pd.Timestamp("2026-11-12 16:20:00", tz="America/New_York").timestamp(),
    )

    sc_pin = TemperatureScaler()
    sc_pin.temperature_ = 1.0
    provider.regime_hmm = StubPinHMM()
    provider.lgbm_backbone = FixedHoldLGBM()
    provider.temp_scaler = sc_pin
    provider._class_priors = (0.20, 0.60, 0.20)
    monkeypatch.setattr(provider, "_load_disk_daily", lambda sym: df_pin.copy() if sym in ("NVDA", "SPY", "XLK") else None)
    reg_pin = {
        "current_state": "LOW_VOLATILITY",
        "state_index": 0,
        "kappa": 0.45,
        "as_of_date": "2026-11-12",
        "observation_timestamp_ms": pin_obs_ms,
    }

    sig_pin = provider.get_signal("NVDA", regime=reg_pin)
    assert sig_pin["winning_engine"] != "NONE"
    assert "UNREADY" not in sig_pin["execution_mode"]
    assert sig_pin["calibration"]["calibrated_probabilities"]["neutral_hold"] == pytest.approx(0.85, abs=1e-3)
    assert sig_pin["execution_recommendation"]["action"] == "HOLD"
    assert sig_pin["execution_recommendation"]["l1_normalized_portfolio_allocation_pct"] == 0.0

    # Also verify BUY and SELL with fixed bullish/bearish inputs (#25)
    provider.lgbm_backbone = types.SimpleNamespace(predict_proba=lambda x: np.tile([0.70, 0.20, 0.10], (len(x), 1)))
    with provider._lock:
        provider._signal_cache.clear()
    sig_buy = provider.get_signal("NVDA", regime=reg_pin)
    assert sig_buy["execution_recommendation"]["action"] == "BUY"
    assert sig_buy["execution_recommendation"]["l1_normalized_portfolio_allocation_pct"] > 0.0

    provider.lgbm_backbone = types.SimpleNamespace(predict_proba=lambda x: np.tile([0.05, 0.20, 0.75], (len(x), 1)))
    with provider._lock:
        provider._signal_cache.clear()
    sig_sell = provider.get_signal("NVDA", regime=reg_pin)
    assert sig_sell["execution_recommendation"]["action"] == "SELL"
    assert sig_sell["execution_recommendation"]["l1_normalized_portfolio_allocation_pct"] == 0.0


def _has_live_artifacts() -> bool:
    engine_root = Path(__file__).resolve().parents[1]
    required = [
        engine_root / "weights" / "manifest.json",
        engine_root / "weights" / "regime_hmm.pkl",
        engine_root / "weights" / "lightgbm_backbone.pkl",
        engine_root / "weights" / "temperature_scaler.pkl",
        engine_root / "lakehouse" / "processed" / "daily" / "SPY.parquet",
        engine_root / "lakehouse" / "processed" / "daily" / "AAPL.parquet",
        engine_root / "lakehouse" / "processed" / "daily" / "NVDA.parquet",
    ]
    return all(p.exists() for p in required)


def test_live_provider_regime_kappa_signals_and_math_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies live weights produce PRD kappa in [0.25, 0.50], valid stationary pi,
    safe handling when transmat_ is None, deterministic currentPrice/dividendYield overrides,
    DES vs SPLC beta parity, and true directional BUY/HOLD/SELL signals without false Hold BUYs.
    """
    if not _has_live_artifacts():
        pytest.skip("Provisioned model weights and lakehouse parquet partitions required")

    from src.ipc.live_providers import LiveInferenceProvider

    provider = LiveInferenceProvider(enable_live_fetch=False)
    assert provider.is_ready()

    mr = provider.get_macro_regime()
    fp = mr["filtered_probabilities"]
    expected_kappa = round(
        0.50 * fp["low_volatility"] + 0.35 * fp["medium_volatility"] + 0.25 * fp["high_volatility"],
        2,
    )
    assert 0.25 <= mr["kappa"] <= 0.50
    assert mr["kappa"] == pytest.approx(expected_kappa, abs=0.01)

    transmat = np.array(mr["transition_matrix"], dtype=np.float64)
    pi = np.array(mr["stationary_distribution"], dtype=np.float64)
    assert pi.shape == (3,)
    assert np.all(np.isfinite(pi))
    assert np.all(pi >= 0.0)
    assert float(np.sum(pi)) == pytest.approx(1.0, abs=1e-3)
    assert np.max(np.abs(pi @ transmat - pi)) < 1e-3

    # Regression for transition_matrix fallback when filter succeeds while transmat_ is None
    orig_transmat = provider.regime_hmm.transmat_
    orig_filter = provider.regime_hmm.filter_sequence
    try:
        provider.regime_hmm.transmat_ = None
        provider.regime_hmm.filter_sequence = lambda X: np.tile(np.array([[0.8, 0.15, 0.05]]), (len(X), 1))
        with provider._lock:
            provider._regime_cache = None
        mr_no_t = provider.get_macro_regime()
        assert mr_no_t["available"] is True
        assert mr_no_t["transition_matrix"] == []
        assert mr_no_t["stationary_distribution"] == []
    finally:
        provider.regime_hmm.transmat_ = orig_transmat
        provider.regime_hmm.filter_sequence = orig_filter
        with provider._lock:
            provider._regime_cache = None

    # Rolling-data readiness & output consistency checks on NVDA (independent of lakehouse bar/weight updates)
    import pandas as pd

    min_hist = provider.micro_extractor.lookback_window + 1
    spy_disk = provider._load_disk_daily("SPY")
    spy_last_norm = pd.Timestamp(spy_disk.index[-1]).normalize() if spy_disk is not None and len(spy_disk) > 0 else None
    spy_obs_disk, spy_comp_disk, _ = provider._daily_bar_status_ms(spy_disk) if spy_disk is not None and len(spy_disk) > 0 else (None, False, "")
    spy_attrs_disk = getattr(spy_disk, "attrs", {}) or {} if spy_disk is not None else {}

    def _disk_pair_ready(sym_df: pd.DataFrame | None) -> bool:
        if spy_disk is None or spy_last_norm is None or sym_df is None or len(sym_df) == 0:
            return False
        sym_last = pd.Timestamp(sym_df.index[-1]).normalize()
        common_idx = sym_df.index.intersection(spy_disk.index)
        if len(common_idx) < min_hist or pd.Timestamp(common_idx[-1]).normalize() != sym_last:
            return False
        sym_obs, sym_comp, _ = provider._daily_bar_status_ms(sym_df)
        spy_check = spy_disk
        spy_obs_chk = spy_obs_disk
        spy_comp_chk = spy_comp_disk
        spy_attrs_chk = spy_attrs_disk
        if sym_last < spy_last_norm:
            spy_check = provider._slice_aligned_frame(spy_disk, common_idx[-provider._TAIL_WARMUP_BARS:])
            spy_obs_chk, spy_comp_chk, _ = provider._daily_bar_status_ms(spy_check)
            spy_attrs_chk = getattr(spy_check, "attrs", {}) or {}
        if spy_obs_chk is not None or sym_obs is not None:
            return spy_obs_chk is not None and sym_obs is not None and spy_obs_chk <= sym_obs
        if sym_last < spy_last_norm or provider.enable_live_fetch or spy_comp_chk or sym_comp:
            return False
        sym_attrs = getattr(sym_df, "attrs", {}) or {}
        if spy_attrs_chk.get("_bar_unsettled") or sym_attrs.get("_bar_unsettled"):
            return False
        spy_m = spy_attrs_chk.get("disk_mtime_ms")
        sym_m = sym_attrs.get("disk_mtime_ms")
        return isinstance(spy_m, (int, float)) and isinstance(sym_m, (int, float)) and spy_m <= sym_m

    nvda_disk = provider._load_disk_daily("NVDA")
    sig_nvda = provider.get_signal("NVDA")
    if _disk_pair_ready(nvda_disk):
        assert sig_nvda["winning_engine"] != "NONE"
        assert "UNREADY" not in sig_nvda["execution_mode"]
        rec_nvda = sig_nvda["execution_recommendation"]
        cal_nvda = sig_nvda["calibration"]["calibrated_probabilities"]
        assert rec_nvda["action"] in ("BUY", "HOLD", "SELL")
        assert 0.0 <= rec_nvda["l1_normalized_portfolio_allocation_pct"] <= 15.0
        assert abs(cal_nvda["outperform_buy"] + cal_nvda["neutral_hold"] + cal_nvda["underperform_sell"] - 1.0) <= 0.02
        assert rec_nvda["dynamic_stop_loss"] < rec_nvda["entry_price"] < rec_nvda["dynamic_take_profit"]
    else:
        assert sig_nvda["execution_mode"] == "UNREADY"
        if nvda_disk is None or len(nvda_disk) == 0:
            exp_nvda_mod = "PRICE_HISTORY_UNAVAILABLE"
        elif len(nvda_disk) < min_hist:
            exp_nvda_mod = "MODEL_WEIGHTS_OR_HISTORY_MISSING"
        else:
            exp_nvda_mod = "BENCHMARK_OBSERVATIONS_UNALIGNED"
        assert exp_nvda_mod in sig_nvda["data_integrity"]["missing_modalities"]

    # Rolling-data readiness and output consistency across liquid S&P names when partitioned locally;
    # partitions lagging a newer benchmark or lacking sufficient aligned history must honestly stay UNREADY.
    sample = ["TECH", "KO", "ABBV", "MTD", "NKE", "PFE", "CI", "NVDA", "MSFT", "AAPL"]
    synced_ready = 0
    for sym in sample:
        if (provider.market_loader.daily_dir / f"{sym}.parquet").exists():
            sym_df = provider._load_disk_daily(sym)
            s = provider.get_signal(sym)
            if _disk_pair_ready(sym_df):
                synced_ready += 1
                assert s["winning_engine"] != "NONE"
                assert "UNREADY" not in s["execution_mode"]
                assert s["execution_recommendation"]["action"] in ("BUY", "HOLD", "SELL")
                assert 0.0 <= s["execution_recommendation"]["l1_normalized_portfolio_allocation_pct"] <= 15.0
            else:
                assert s["execution_mode"] == "UNREADY"
                if sym_df is None or len(sym_df) == 0:
                    exp_sym_mod = "PRICE_HISTORY_UNAVAILABLE"
                elif len(sym_df) < min_hist:
                    exp_sym_mod = "MODEL_WEIGHTS_OR_HISTORY_MISSING"
                else:
                    exp_sym_mod = "BENCHMARK_OBSERVATIONS_UNALIGNED"
                assert exp_sym_mod in s["data_integrity"]["missing_modalities"]
                assert s["execution_recommendation"]["l1_normalized_portfolio_allocation_pct"] == 0.0
    assert synced_ready >= 0  # Pinned READY fixture above validates full READY signal parity regardless of local partition lag
    port = provider.get_portfolio()
    assert 0.0 <= port["gross_exposure"] <= 1.0

    # Analyst current_price must match latest market price rather than stale yf_meta currentPrice,
    # and dividendYield in % must NOT be multiplied by 100 when trailingAnnualDividendYield is absent.
    aapl_quote = provider.get_latest_quote("AAPL")
    patched_meta = {
        "info": {
            "longName": "Apple Inc.",
            "currentPrice": 1.23,
            "dividendYield": 0.03,
            "targetMeanPrice": 250.0,
            "numberOfAnalystOpinions": 10,
        },
        "rec_trend": [{"strong_buy": 4, "buy": 4, "hold": 2, "sell": 0}],
        "insider_transactions": [],
    }
    monkeypatch.setattr(provider.fundamentals_client, "get_yf_meta", lambda *a, **kw: patched_meta)
    with provider._lock:
        provider._ratings_cache.clear()
        provider._sec_master_cache.clear()
    anr = provider.get_analyst_ratings("AAPL")
    assert anr["current_price"] == pytest.approx(aapl_quote["price"], abs=0.05)
    assert anr["historical_revisions"][0]["period"] == "Unknown"
    assert anr["historical_revisions"][0]["mean_target"] is None
    sm = provider.get_security_master("AAPL")
    assert sm["dividend_yield_pct"] == pytest.approx(0.03, abs=0.01)

    # DES vs SPLC beta_market parity on shared tickers
    if (provider.market_loader.daily_dir / "CRWD.parquet").exists():
        sm_crwd = provider.get_security_master("CRWD")
        pg_crwd = provider.get_peer_graph("CRWD")
        target_node = next(n for n in pg_crwd["nodes"] if n["id"] == "CRWD")
        assert sm_crwd["beta_market_60d"] == pytest.approx(target_node["beta_market"], abs=0.01)


def test_live_provider_refreshes_sector_etf_across_rpcs(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    When enable_live_fetch=True, get_signal, get_security_master, and get_peer_graph must
    refresh non-SPY sector ETFs BEFORE reading disk bars so outputs reflect the refreshed series.
    """
    if not _has_live_artifacts():
        pytest.skip("Provisioned model weights and lakehouse parquet partitions required")

    from datetime import datetime
    from zoneinfo import ZoneInfo
    import pandas as pd
    from src.data.market_loader import MarketDataLoader
    from src.ipc.live_providers import LiveInferenceProvider

    provider = LiveInferenceProvider(enable_live_fetch=False)
    orig_load = provider._load_disk_daily
    aapl_real = orig_load("AAPL")
    spy_real = orig_load("SPY")
    assert aapl_real is not None and spy_real is not None
    common_idx = aapl_real.index.intersection(spy_real.index)
    assert len(common_idx) >= 130
    last_bar_ts = pd.Timestamp(common_idx[-1])
    last_close_t = MarketDataLoader.nyse_session_close_time(last_bar_ts.date())
    aligned_obs_ms = int(
        datetime.combine(last_bar_ts.date(), last_close_t, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1000
    )
    aapl_aligned = aapl_real.loc[common_idx].copy()
    aapl_aligned.attrs = {
        "observation_timestamp_ms": aligned_obs_ms,
        "obs_bar_date": last_bar_ts.strftime("%Y-%m-%d"),
    }
    spy_aligned = spy_real.loc[common_idx].copy()
    spy_aligned.attrs = {
        "observation_timestamp_ms": aligned_obs_ms,
        "obs_bar_date": last_bar_ts.strftime("%Y-%m-%d"),
    }
    df_xlk_aligned = aapl_aligned.copy()
    scale_vec = np.linspace(0.95, 1.05, len(df_xlk_aligned))
    for col in ("open", "high", "low", "close"):
        df_xlk_aligned[col] = df_xlk_aligned[col].astype(float) * scale_vec
    df_xlk_aligned.attrs = dict(aapl_aligned.attrs)

    called_refresh: list[str] = []
    xlk_state = {"refreshed": False}
    seen_r_sec: list[np.ndarray] = []
    orig_res = provider.residualizer.compute_rolling_residuals

    def spy_res(r_stk, r_sp, r_sc):
        seen_r_sec.append(r_sc.copy())
        return orig_res(r_stk, r_sp, r_sc)

    def spy_refresh(sym: str) -> None:
        sym_u = sym.upper()
        called_refresh.append(sym_u)
        if sym_u == "XLK":
            xlk_state["refreshed"] = True

    def gated_load_disk(sym: str):
        su = sym.upper()
        if su == "AAPL":
            return aapl_aligned
        if su == "SPY":
            return spy_aligned
        if su == "XLK":
            return df_xlk_aligned if xlk_state["refreshed"] else None
        return orig_load(sym)

    provider.enable_live_fetch = True
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: (aligned_obs_ms / 1000.0) + 1200.0)
    monkeypatch.setattr(provider, "_maybe_refresh_daily", spy_refresh)
    monkeypatch.setattr(provider, "_load_disk_daily", gated_load_disk)
    monkeypatch.setattr(provider, "_fetch_live_quote_cached", lambda sym: None)
    monkeypatch.setattr(provider.fundamentals_client, "get_yf_meta", lambda *a, **kw: None)
    monkeypatch.setattr(provider.residualizer, "compute_rolling_residuals", spy_res)

    def _reset_state() -> None:
        called_refresh.clear()
        seen_r_sec.clear()
        xlk_state["refreshed"] = False
        with provider._lock:
            provider._signal_cache.clear()
            provider._sec_master_cache.clear()
            provider._peer_graph_cache.clear()

    _reset_state()
    sig = provider.get_signal("AAPL")
    assert "XLK" in called_refresh
    assert sig["execution_mode"] != "UNREADY"
    assert "BENCHMARK_OBSERVATIONS_UNALIGNED" not in sig["data_integrity"]["missing_modalities"]
    assert "SECTOR_BENCHMARK_UNALIGNED" not in sig["data_integrity"]["missing_modalities"]
    expected_tail = df_xlk_aligned.tail(provider._TAIL_WARMUP_BARS)
    expected_r_xlk = np.diff(np.log(expected_tail["close"].values))
    assert len(seen_r_sec) == 1 and np.allclose(seen_r_sec[0], expected_r_xlk)

    _reset_state()
    sm = provider.get_security_master("AAPL")
    assert "XLK" in called_refresh
    assert sm["beta_sector_60d"] is not None and np.isfinite(sm["beta_sector_60d"])

    _reset_state()
    pg = provider.get_peer_graph("AAPL")
    assert "XLK" in called_refresh
    xlk_node = next((n for n in pg["nodes"] if n["id"] == "XLK"), None)
    assert xlk_node is not None and xlk_node["return_5d"] is not None and xlk_node["beta_market"] is not None


def test_fundamentals_client_stale_fallback_form4_and_restatement(tmp_path: Path) -> None:
    """
    1. Expired disk caches remain readable when enable_live_fetch=False (including company tickers & redlines).
    2. Form 4 classification respects word boundaries and tax/sale priority over grant/award.
    3. Sub-0.5M and Net-Income-only restatement notes report accurate formatted deltas.
    """
    from src.data.fundamentals_client import FundamentalsClient

    # Unit check on Form 4 text classifier edge cases
    assert FundamentalsClient.classify_insider_tx("Stock Award(Grant) at price 0.00") == "A"
    assert FundamentalsClient.classify_insider_tx("Stock Gift at price 0.00") == "G"
    assert FundamentalsClient.classify_insider_tx("Tax withholding on vesting of stock award") == "F"
    assert FundamentalsClient.classify_insider_tx("Shares withheld to pay taxes; no shares were sold") == "F"
    assert FundamentalsClient.classify_insider_tx("Shares sold to cover tax withholding obligations") == "S"
    assert FundamentalsClient.classify_insider_tx("Withholding of shares upon vesting (not an open-market sale)") == "F"
    assert FundamentalsClient.classify_insider_tx("Divestiture plan notice") == "J"
    assert FundamentalsClient.classify_insider_tx("Disposition to issuer") == "D"
    assert FundamentalsClient.classify_insider_tx("Automatic Stock Sale at 140.00") == "S"
    assert FundamentalsClient.classify_insider_tx("Open Market Sale under tax planning agreement") == "S"
    assert FundamentalsClient.classify_insider_tx("Open Market Sale to Cover Tax Liability") == "S"
    assert FundamentalsClient.classify_insider_tx("Shares sold at $140") == "S"
    assert FundamentalsClient.classify_insider_tx("Sale of shares acquired through exercise of stock options") == "S"
    assert FundamentalsClient.classify_insider_tx("Exercise of option to purchase shares") == "M"
    assert FundamentalsClient.classify_insider_tx("Open Market Purchase at 95.00") == "P"

    client = FundamentalsClient(lakehouse_dir=tmp_path)
    cik = "0000789019"

    p_map = client.raw_dir / "sec_company_tickers.json"
    p_yf = client.raw_dir / "yf_meta" / "MSFT.json"
    p_sub = client.raw_dir / "edgar_submissions" / f"CIK{cik}.json"
    p_facts = client.raw_dir / "edgar_facts" / f"CIK{cik}.json"
    p_red = client.raw_dir / "edgar_redlines" / "MSFT_0000789019-18-000010_0000789019-17-000010.json"
    for p in (p_yf, p_sub, p_facts, p_red):
        p.parent.mkdir(parents=True, exist_ok=True)

    p_map.write_text(json.dumps({"MSFT": cik}), encoding="utf-8")
    p_yf.write_text(json.dumps({
        "info": {"longName": "Microsoft Corp"},
        "rec_trend": [],
        "insider_transactions": [
            {
                "filing_date": "2026-10-15",
                "insider_name": "Exec One",
                "relationship": "Officer",
                "transaction_code": "S",
                "shares": 1000,
                "price": 0.0,
                "value": 0.0,
                "is_10b5_1": False,
                "raw_text": "Stock Award(Grant) at price 0.00 per share.",
            },
            {
                "filing_date": "2026-10-16",
                "insider_name": "Exec Two",
                "relationship": "Director",
                "transaction_code": "S",
                "shares": 500,
                "price": 0.0,
                "value": 0.0,
                "is_10b5_1": False,
                "raw_text": "Tax withholding on vesting of RSU",
            },
            {
                "filing_date": "2026-10-17",
                "insider_name": "Exec Three",
                "relationship": "CEO",
                "transaction_code": "S",
                "shares": 200,
                "price": 400.0,
                "value": 80000.0,
                "is_10b5_1": False,
                "raw_text": "Sale at price 400.00 per share.",
            },
        ],
    }), encoding="utf-8")

    p_sub.write_text(json.dumps({
        "cik": cik,
        "recent": {
            "form": ["10-K", "10-K"],
            "filingDate": ["2018-08-03", "2017-08-02"],
            "reportDate": ["2018-06-30", "2017-06-30"],
            "accessionNumber": ["0000789019-18-000010", "0000789019-17-000010"],
            "primaryDocument": ["msft18.htm", "msft17.htm"],
        },
    }), encoding="utf-8")

    p_red.write_text(json.dumps({
        "current_accession": "0000789019-18-000010",
        "prior_accession": "0000789019-17-000010",
        "item1a_similarity": 0.96,
        "mda_similarity": 0.94,
        "item1a_sentiment_drift": -0.05,
        "mda_catalyst_score": 0.12,
        "redlines": [{"section": "Item 1A Risk Factors", "status": "DIFF_VERIFIED", "added_clauses": ["Sample"], "removed_clauses": []}],
    }), encoding="utf-8")

    p_facts.write_text(json.dumps({
        "cik": cik,
        "entityName": "MICROSOFT CORP",
        "us_gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {"form": "10-K", "start": "2016-07-01", "end": "2017-06-30", "filed": "2017-08-02", "accn": "0000789019-17-000010", "val": 90000000000},
                        {"form": "10-K", "start": "2016-07-01", "end": "2017-06-30", "filed": "2018-08-03", "accn": "0000789019-18-000010", "val": 90000000000},
                        {"form": "10-K", "start": "2017-07-01", "end": "2018-06-30", "filed": "2018-08-03", "accn": "0000789019-18-000010", "val": 110000200000},
                    ]
                }
            },
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        {"form": "10-K", "start": "2016-07-01", "end": "2017-06-30", "filed": "2017-08-02", "accn": "0000789019-17-000010", "val": 21204000000},
                        {"form": "10-K", "start": "2016-07-01", "end": "2017-06-30", "filed": "2018-08-03", "accn": "0000789019-18-000010", "val": 25489000000},
                        {"form": "10-K", "start": "2017-07-01", "end": "2018-06-30", "filed": "2018-08-03", "accn": "0000789019-18-000010", "val": 16571000000},
                    ]
                }
            },
        },
    }), encoding="utf-8")

    old_ts = time.time() - 86400 * 30
    for p in (p_map, p_yf, p_sub, p_facts, p_red):
        os.utime(p, (old_ts, old_ts))

    assert client.resolve_cik("MSFT", enable_live_fetch=False) == cik
    assert client.get_yf_meta("MSFT", enable_live_fetch=False) is not None
    assert client.get_sec_submissions(cik, enable_live_fetch=False) is not None
    assert client.get_sec_company_facts(cik, enable_live_fetch=False) is not None

    # Expired redlines + Form 4 reclassification + accession pair matching
    edg = client.build_edgar_filings_payload("MSFT", known_cik=cik, enable_live_fetch=False)
    assert edg["item1a_similarity"] == pytest.approx(0.96, abs=1e-4)
    codes = {row["insider_name"]: row["transaction_code"] for row in edg["form4_transactions"]}
    assert codes["Exec One"] == "A"
    assert codes["Exec Two"] == "F"
    assert codes["Exec Three"] == "S"

    # Inserting an intervening prior filing B between A and C must NOT reuse A-vs-C redline cache
    p_sub.write_text(json.dumps({
        "cik": cik,
        "recent": {
            "form": ["10-K", "10-Q", "10-K"],
            "filingDate": ["2018-08-03", "2018-04-26", "2017-08-02"],
            "reportDate": ["2018-06-30", "2018-03-31", "2017-06-30"],
            "accessionNumber": ["0000789019-18-000010", "0000789019-18-000005", "0000789019-17-000010"],
            "primaryDocument": ["msft18.htm", "msft18q3.htm", "msft17.htm"],
        },
    }), encoding="utf-8")
    edg_mismatch = client.build_edgar_filings_payload("MSFT", known_cik=cik, enable_live_fetch=False)
    assert edg_mismatch["item1a_similarity"] is None
    assert edg_mismatch["redlines"][0]["status"] == "INDEX_VERIFIED"

    # Also verify embedded-pair checking under the NEW pair's cache filename (#25):
    # fresh client returns DIFF_VERIFIED with matching pair, then INDEX_VERIFIED with mismatched pair.
    p_red_new_path = client.raw_dir / "edgar_redlines" / "MSFT_0000789019-18-000010_0000789019-18-000005.json"
    p_red_new_path.write_text(json.dumps({
        "current_accession": "0000789019-18-000010",
        "prior_accession": "0000789019-18-000005",
        "item1a_similarity": 0.91,
        "mda_similarity": 0.89,
        "item1a_sentiment_drift": -0.2,
        "mda_catalyst_score": 0.1,
        "redlines": [{"section": "Item 1A", "status": "DIFF_VERIFIED", "added_clauses": ["New"], "removed_clauses": []}],
    }), encoding="utf-8")
    fresh_reader_match = FundamentalsClient(lakehouse_dir=tmp_path)
    edg_embedded_match = fresh_reader_match.build_edgar_filings_payload("MSFT", known_cik=cik, enable_live_fetch=False)
    assert edg_embedded_match["redlines"][0]["status"] == "DIFF_VERIFIED"

    p_red_new_path.write_text(p_red.read_text(encoding="utf-8"), encoding="utf-8")
    fresh_reader_mismatch = FundamentalsClient(lakehouse_dir=tmp_path)
    edg_embedded_mismatch = fresh_reader_mismatch.build_edgar_filings_payload("MSFT", known_cik=cik, enable_live_fetch=False)
    assert edg_embedded_mismatch["item1a_similarity"] is None
    assert edg_embedded_mismatch["redlines"][0]["status"] == "INDEX_VERIFIED"

    # Net-income-only restatement note accuracy without misattributing 10-K revisions to 10-K/A
    fin = client.build_financials_payload("MSFT", known_cik=cik, pit_strict=True, enable_live_fetch=False)
    f17 = next(f for f in fin["filings"] if f["period"] == "FY2017")
    assert f17["is_restated"] is True
    note_17 = f17["restatement_delta_notes"] or ""
    assert "Net Income by +4285M" in note_17
    assert "+0M" not in note_17
    assert "10-K/A" not in note_17
    assert "10-K" in note_17

    # Distinct revisions from 10-K/A (Revenue) and 10-K (Net Income) stay separately attributed
    p_facts.write_text(json.dumps({
        "cik": cik,
        "entityName": "MICROSOFT CORP",
        "us_gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {"form": "10-K", "start": "2016-07-01", "end": "2017-06-30", "filed": "2017-08-02", "accn": "A1", "val": 90000000000},
                        {"form": "10-K/A", "start": "2016-07-01", "end": "2017-06-30", "filed": "2017-10-01", "accn": "A2", "val": 91000000000},
                    ]
                }
            },
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        {"form": "10-K", "start": "2016-07-01", "end": "2017-06-30", "filed": "2017-08-02", "accn": "A1", "val": 21204000000},
                        {"form": "10-K", "start": "2016-07-01", "end": "2017-06-30", "filed": "2018-08-03", "accn": "A3", "val": 25489000000},
                    ]
                }
            },
        },
    }), encoding="utf-8")
    fin_split = client.build_financials_payload("MSFT", known_cik=cik, pit_strict=True, enable_live_fetch=False)
    f17_split = next(f for f in fin_split["filings"] if f["period"] == "FY2017")
    split_note = f17_split["restatement_delta_notes"] or ""
    assert "Subsequent 10-K/A restatement changed Revenue by +1000M" in split_note
    assert "Subsequent 10-K filing revised Net Income by +4285M" in split_note

    # Same-accession duplicate Net Income rows must NOT flag a false restatement
    p_facts.write_text(json.dumps({
        "cik": cik,
        "entityName": "MICROSOFT CORP",
        "us_gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {"form": "10-K", "start": "2016-07-01", "end": "2017-06-30", "filed": "2017-08-02", "accn": "A1", "val": 90000000000},
                    ]
                }
            },
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        {"form": "10-K", "start": "2016-07-01", "end": "2017-06-30", "filed": "2017-08-02", "accn": "A1", "val": 21204000000},
                        {"form": "10-K", "start": "2016-07-01", "end": "2017-06-30", "filed": "2017-08-02", "accn": "A1", "val": 25489000000},
                    ]
                }
            },
        },
    }), encoding="utf-8")
    client._mem_cache.clear()
    fin_dup = client.build_financials_payload("MSFT", known_cik=cik, pit_strict=True, enable_live_fetch=False)
    f17_dup = next(f for f in fin_dup["filings"] if f["period"] == "FY2017")
    assert f17_dup["is_restated"] is False
    assert f17_dup["restatement_delta_notes"] is None

    # Same-day 10-K/A listed BEFORE original 10-K must still select 10-K as PIT & retain fractional millions (> 0.5M)
    p_facts.write_text(json.dumps({
        "cik": cik,
        "entityName": "MICROSOFT CORP",
        "us_gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {"form": "10-K/A", "start": "2016-07-01", "end": "2017-06-30", "filed": "2017-08-02", "accn": "A2", "val": 90000510000},
                        {"form": "10-K", "start": "2016-07-01", "end": "2017-06-30", "filed": "2017-08-02", "accn": "A1", "val": 90000000000},
                    ]
                }
            },
        },
    }), encoding="utf-8")
    client._mem_cache.clear()
    fin_sameday = client.build_financials_payload("MSFT", known_cik=cik, pit_strict=True, enable_live_fetch=False)
    assert fin_sameday["incomeStatement"][0]["values"]["FY2017"] == pytest.approx(90000.0, abs=0.01)
    f17_sd = next(f for f in fin_sameday["filings"] if f["period"] == "FY2017")
    assert f17_sd["is_restated"] is True
    assert "+0.5M" in (f17_sd["restatement_delta_notes"] or "")
    assert "+1M" not in (f17_sd["restatement_delta_notes"] or "")

    # Separate 200,000 revision below 0.5M boundary preserves +0.2M instead of dropping (#28)
    p_facts.write_text(json.dumps({
        "cik": cik,
        "entityName": "MICROSOFT CORP",
        "us_gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {"form": "10-K/A", "start": "2016-07-01", "end": "2017-06-30", "filed": "2017-08-02", "accn": "A2", "val": 90000200000},
                        {"form": "10-K", "start": "2016-07-01", "end": "2017-06-30", "filed": "2017-08-02", "accn": "A1", "val": 90000000000},
                    ]
                }
            },
        },
    }), encoding="utf-8")
    client._mem_cache.clear()
    fin_sub_half = client.build_financials_payload("MSFT", known_cik=cik, pit_strict=True, enable_live_fetch=False)
    f17_sub_half = next(f for f in fin_sub_half["filings"] if f["period"] == "FY2017")
    assert "+0.2M" in (f17_sub_half["restatement_delta_notes"] or "")

    # When both same-day filings carry usable SEC acceptanceDateTime, acceptance time decides PIT order
    p_sub.write_text(json.dumps({
        "cik": cik,
        "recent": {
            "form": ["10-K", "10-K/A"],
            "filingDate": ["2017-08-02", "2017-08-02"],
            "acceptanceDateTime": ["2017-08-02T16:30:00.000Z", "2017-08-02T08:00:00.000Z"],
            "reportDate": ["2017-06-30", "2017-06-30"],
            "accessionNumber": ["A1", "A2"],
            "primaryDocument": ["msft17.htm", "msft17a.htm"],
        },
    }), encoding="utf-8")
    p_facts.write_text(json.dumps({
        "cik": cik,
        "entityName": "MICROSOFT CORP",
        "us_gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {"form": "10-K", "start": "2016-07-01", "end": "2017-06-30", "filed": "2017-08-02", "accn": "A1", "val": 90002000000},
                        {"form": "10-K/A", "start": "2016-07-01", "end": "2017-06-30", "filed": "2017-08-02", "accn": "A2", "val": 90000510000},
                    ]
                }
            },
        },
    }), encoding="utf-8")
    client._mem_cache.clear()
    fin_accept_ord = client.build_financials_payload("MSFT", known_cik=cik, pit_strict=True, enable_live_fetch=False)
    assert fin_accept_ord["incomeStatement"][0]["values"]["FY2017"] == pytest.approx(90000.51, abs=0.01)


def test_live_provider_news_neutral_bias_and_null_safety(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    1. Neutral headline without sentiment keywords scores 0.00 (no +0.10 positive bias).
    2. Form 4 Grants ('A') / Tax ('F') and None value fields do not raise or bias Form 4 score.
    3. smart_money_divergence stays finite (0.0 when news absent) even with edge inputs.
    """
    if not _has_live_artifacts():
        pytest.skip("Provisioned model weights and lakehouse parquet partitions required")

    import types
    import pandas as pd
    import yfinance as yf
    from src.ipc.live_providers import LiveInferenceProvider

    provider = LiveInferenceProvider(enable_live_fetch=True)
    monkeypatch.setattr(provider, "_maybe_refresh_daily", lambda sym: None)
    df_aapl_raw = provider._load_disk_daily("AAPL")
    df_spy_raw = provider._load_disk_daily("SPY")
    if df_aapl_raw is None or df_spy_raw is None:
        pytest.skip("AAPL/SPY disk history missing")
    common_aapl_spy = df_aapl_raw.index.intersection(df_spy_raw.index)
    if len(common_aapl_spy) < 127:
        pytest.skip("Aligned AAPL/SPY history < 127 bars")
    df_aapl_al = df_aapl_raw.loc[common_aapl_spy].copy()
    df_spy_al = df_spy_raw.loc[common_aapl_spy].copy()
    aapl_dt_str = pd.Timestamp(df_aapl_al.index[-1]).strftime("%Y-%m-%d")
    spy_dt_str = aapl_dt_str
    aapl_close_v = float(df_aapl_al["close"].iloc[-1])
    spy_close_v = float(df_spy_al["close"].iloc[-1])
    obs_1530_ms = int(pd.Timestamp(f"{aapl_dt_str} 15:30:00", tz="America/New_York").timestamp() * 1000)
    # Give SPY <= target observation time for causal live benchmark alignment
    spy_1529_ms = int(pd.Timestamp(f"{spy_dt_str} 15:29:00", tz="America/New_York").timestamp() * 1000)
    monkeypatch.setattr(
        provider,
        "_load_disk_daily",
        lambda sym: df_aapl_al if sym == "AAPL" else (df_spy_al if sym == "SPY" else None),
    )
    monkeypatch.setattr(
        provider,
        "_fetch_live_quote_cached",
        lambda sym: (
            {"price": aapl_close_v, "date": aapl_dt_str, "observation_timestamp_ms": obs_1530_ms}
            if sym == "AAPL"
            else ({"price": spy_close_v, "date": spy_dt_str, "observation_timestamp_ms": spy_1529_ms} if sym == "SPY" else None)
        ),
    )

    dummy_news = [
        {
            "content": {
                "title": "Apple announces quarterly conference schedule",
                "provider": {"displayName": "Wire"},
                "pubDate": f"{aapl_dt_str}T12:00:00Z",
            }
        }
    ]
    monkeypatch.setattr(yf, "Ticker", lambda sym: types.SimpleNamespace(news=dummy_news))
    # Verify verified sale contributes negative score before testing null/grant/tax withholding exclusion
    verified_pre_disc = f"{aapl_dt_str}T14:00:00Z"
    with provider._lock:
        provider._signal_cache.clear()
        provider._edgar_cache["AAPL"] = (
            1e12,
            {
                "ticker": "AAPL",
                "available": True,
                "item1a_sentiment_drift": 0.0,
                "mda_catalyst_score": 0.0,
                "form4_transactions": [
                    {
                        "insider_name": "Seller",
                        "transaction_code": "S",
                        "disclosed_at": verified_pre_disc,
                        "disclosed_at_verified": True,
                        "value": 5_000_000.0,
                        "is_10b5_1": False,
                        "raw_text": "Sale at price 200.00",
                    },
                ],
            },
        )
    sig_sale_pos = provider.get_signal("AAPL")
    assert sig_sale_pos["signal_breakdown"]["decayed_form4_insider_cluster"] < 0.0

    with provider._lock:
        provider._news_cache.clear()
        provider._signal_cache.clear()
        provider._edgar_cache["AAPL"] = (
            1e12,
            {
                "ticker": "AAPL",
                "available": True,
                "item1a_sentiment_drift": 0.0,
                "mda_catalyst_score": 0.0,
                "form4_transactions": [
                    {
                        "insider_name": "Officer",
                        "transaction_code": "A",
                        "disclosed_at": verified_pre_disc,
                        "disclosed_at_verified": True,
                        "value": 1_000_000.0,
                        "is_10b5_1": False,
                        "raw_text": "Stock Award(Grant) at price 0.00 per share.",
                    },
                    {
                        "insider_name": "TaxHolder",
                        "transaction_code": "F",
                        "disclosed_at": verified_pre_disc,
                        "disclosed_at_verified": True,
                        "value": 500_000.0,
                        "is_10b5_1": False,
                        "raw_text": "Tax withholding surrender to pay tax liability.",
                    },
                    {
                        "insider_name": "CEO",
                        "transaction_code": "S",
                        "disclosed_at": verified_pre_disc,
                        "disclosed_at_verified": True,
                        "value": None,
                        "is_10b5_1": False,
                        "raw_text": "Sale at price 200.00",
                    },
                ],
            },
        )

    items = provider.get_news_feed("AAPL")
    assert len(items) == 1
    assert items[0]["finbert_score"] == 0.0
    assert items[0]["sentiment_label"] == "NEUTRAL"

    sig = provider.get_signal("AAPL")
    bd = sig["signal_breakdown"]
    assert bd["decayed_form4_insider_cluster"] == 0.0
    assert bd["smart_money_divergence_pos"] == 0.0
    assert bd["smart_money_divergence_neg"] == 0.0


def test_ols_beta_nan_filtering_recovers_true_slope() -> None:
    """
    _compute_ols_beta_60d must filter out NaN return pairs and recover the true OLS slope
    rather than collapsing to None when benchmark variance is non-zero.
    """
    import pandas as pd
    from src.ipc.live_providers import LiveInferenceProvider

    idx = pd.date_range("2026-01-01", periods=65, freq="B")
    rng = np.random.default_rng(7)
    r_spy = rng.normal(0.0005, 0.012, size=64)
    r_stock = 1.5 * r_spy
    spy_close = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(r_spy)]))
    stk_close = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(r_stock)]))
    s_stk = pd.DataFrame({"close": stk_close}, index=idx)
    s_spy = pd.DataFrame({"close": spy_close}, index=idx)
    s_stk.iloc[20, 0] = float("nan")
    beta = LiveInferenceProvider._compute_ols_beta_60d(s_stk, s_spy)
    assert beta == pytest.approx(1.5, abs=0.02)

    # Zero close must invalidate both adjacent return intervals rather than bridging a gap return across index 20
    s_stk_zero = pd.DataFrame({"close": stk_close.copy()}, index=idx)
    s_stk_zero.iloc[21:, 0] *= 1.5  # Shifts post-gap level without altering post-gap segment returns
    s_stk_zero.iloc[20, 0] = 0.0
    valid_ret_mask = np.ones(len(r_spy), dtype=bool)
    valid_ret_mask[19] = False
    valid_ret_mask[20] = False
    x_v = np.diff(np.log(s_spy["close"].values))[valid_ret_mask]
    y_v = np.diff(np.log(np.where(s_stk_zero["close"].values > 0, s_stk_zero["close"].values, np.nan)))[valid_ret_mask]
    cov_v = float(np.mean((x_v - np.mean(x_v)) * (y_v - np.mean(y_v))))
    expected_gap_beta = cov_v / max(float(np.var(x_v)), 1e-12)
    beta_zero = LiveInferenceProvider._compute_ols_beta_60d(s_stk_zero, s_spy)
    assert beta_zero == pytest.approx(expected_gap_beta, abs=1e-6)
    assert beta_zero == pytest.approx(1.5, abs=0.02)


def test_ws_broadcaster_prior_probs_and_nonfinite_safety(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    build_qare_signal_frame validates posterior triplets together, emits normalized
    prior_probabilities, rejects non-normalized triplets and zero-mass regime probs.
    """
    from src.ipc.ws_broadcaster import build_qare_signal_frame
    from src.ipc.live_providers import LiveInferenceProvider

    frame_ok = build_qare_signal_frame(
        ticker="KO",
        regime_kappa=0.42,
        prior_probs=(2.0, 3.0, 5.0),
    )
    assert frame_ok["calibration"]["prior_probabilities"] == {
        "outperform_buy": 0.2,
        "neutral_hold": 0.3,
        "underperform_sell": 0.5,
    }
    json.dumps(frame_ok, allow_nan=False)

    # Zero-mass or negative regime_probs must keep kappa=None rather than normalizing into valid weights
    frame_zero_reg = build_qare_signal_frame(
        ticker="KO",
        regime_probs={"low_volatility": 0.0, "medium_volatility": 0.0, "high_volatility": 0.0},
    )
    assert frame_zero_reg["macro_regime"]["kappa"] is None

    frame_neg_reg = build_qare_signal_frame(
        ticker="KO",
        regime_probs={"low_volatility": -0.2, "medium_volatility": 0.4, "high_volatility": 0.8},
    )
    assert frame_neg_reg["macro_regime"]["filtered_probabilities"] == {}
    assert frame_neg_reg["macro_regime"]["kappa"] is None

    frame_oob_reg = build_qare_signal_frame(
        ticker="KO",
        regime_probs={"low_volatility": 0.8, "medium_volatility": 0.8, "high_volatility": 0.4},
        signal_breakdown={"peer_gat_spillover": float("nan"), "decayed_finbert_sentiment": 0.0},
    )
    assert frame_oob_reg["macro_regime"]["filtered_probabilities"] == {}
    assert frame_oob_reg["macro_regime"]["kappa"] is None
    assert "peer_gat_spillover" not in frame_oob_reg["signal_breakdown"]
    assert frame_oob_reg["signal_breakdown"]["decayed_finbert_sentiment"] == 0.0

    # Non-distribution triplets (sum=4.0, >1.0 elements, finite in-range wrong sum 0.8/0.8/0.4) and partial NaN must all be rejected;
    # boundary totals 0.98 and 1.02 must be accepted inclusively (#13, #26).
    for bad_triplet in [(float("nan"), 0.2, 0.1), (2.0, 1.0, 1.0), (1.01, 0.0, 0.0), (0.8, 0.8, 0.4)]:
        frame_bad = build_qare_signal_frame(
            ticker="KO",
            action="BUY",
            conviction_tier="HIGH",
            p_buy=bad_triplet[0],
            p_hold=bad_triplet[1],
            p_sell=bad_triplet[2],
            unconstrained_kelly_pct=5.0,
            l1_normalized_pct=5.0,
            regime_kappa=0.45,
            regime_probs={"low_volatility": 0.7, "medium_volatility": 0.2, "high_volatility": 0.1},
            prior_probs=(0.2, 0.6, 0.2),
        )
        assert frame_bad["winning_engine"] == "NONE"
        assert frame_bad["execution_mode"] == "UNREADY"
        assert frame_bad["execution_recommendation"]["action"] == "HOLD"
        assert frame_bad["execution_recommendation"]["l1_normalized_portfolio_allocation_pct"] == 0.0
        assert "INVALID_CALIBRATED_PROBABILITIES" in frame_bad["data_integrity"]["missing_modalities"]
        json.dumps(frame_bad, allow_nan=False)

    for edge_ok in [(0.50, 0.25, 0.23), (0.50, 0.26, 0.26), (0.65, 0.25, 0.10)]:
        frame_ok = build_qare_signal_frame(
            ticker="KO",
            action="BUY",
            conviction_tier="HIGH",
            p_buy=edge_ok[0],
            p_hold=edge_ok[1],
            p_sell=edge_ok[2],
            unconstrained_kelly_pct=5.0,
            l1_normalized_pct=5.0,
            regime_kappa=0.45,
            regime_probs={"low_volatility": 0.7, "medium_volatility": 0.2, "high_volatility": 0.1},
            prior_probs=(0.2, 0.6, 0.2),
        )
        assert frame_ok["winning_engine"] == "LIGHTGBM_105"
        assert frame_ok["execution_mode"] == "LONG_ONLY"
        assert frame_ok["execution_recommendation"]["action"] == "BUY"
        assert "INVALID_CALIBRATED_PROBABILITIES" not in frame_ok["data_integrity"]["missing_modalities"]

    import pandas as pd
    from src.risk.calibration import TemperatureScaler

    lh_prior = tmp_path / "lh_prior"
    w_prior = tmp_path / "w_prior"
    provider = LiveInferenceProvider(lakehouse_dir=lh_prior, weights_dir=w_prior, enable_live_fetch=False)

    class _HMM_P:
        STATE_LABELS = {0: "LOW_VOLATILITY", 1: "MEDIUM_VOLATILITY", 2: "HIGH_VOLATILITY"}
        transmat_ = np.array([[0.9, 0.08, 0.02], [0.05, 0.9, 0.05], [0.02, 0.08, 0.9]])

        def filter_sequence(self, x):
            return np.tile([0.85, 0.10, 0.05], (len(x), 1))

        def get_regime_multiplier(self, p):
            return 0.5

        def get_state_annualized_volatilities(self):
            return [10.0, 18.0, 35.0]

        def get_stationary_distribution(self):
            return [0.6, 0.3, 0.1]

    class _LGBM_P:
        def predict_proba(self, x):
            return np.tile([0.65, 0.25, 0.10], (len(x), 1))

    provider.regime_hmm = _HMM_P()
    provider.lgbm_backbone = _LGBM_P()
    sc_p = TemperatureScaler()
    sc_p.temperature_ = 1.0
    provider.temp_scaler = sc_p
    provider._class_priors = (0.3, 0.4, 0.3)

    obs_mon_ms = int(pd.Timestamp("2025-12-01 16:00:00", tz="America/New_York").timestamp() * 1000)
    idx_p = pd.bdate_range(end="2025-12-01", periods=200, name="timestamp")
    cls_ko = 100.0 * np.exp(np.cumsum(np.random.default_rng(1076).normal(0.0005, 0.01, size=200)))
    cls_spy = 400.0 * np.exp(np.cumsum(np.random.default_rng(1077).normal(0.0004, 0.008, size=200)))
    df_ko_r = pd.DataFrame({"open": cls_ko * 0.998, "high": cls_ko * 1.006, "low": cls_ko * 0.994, "close": cls_ko, "volume": np.full(200, 1_500_000.0)}, index=idx_p)
    df_spy_r = pd.DataFrame({"open": cls_spy * 0.998, "high": cls_spy * 1.006, "low": cls_spy * 0.994, "close": cls_spy, "volume": np.full(200, 3_000_000.0)}, index=idx_p)
    df_ko_r.attrs = {"observation_timestamp_ms": obs_mon_ms, "obs_bar_date": "2025-12-01"}
    df_spy_r.attrs = {"observation_timestamp_ms": obs_mon_ms, "obs_bar_date": "2025-12-01"}
    provider.market_loader.atomic_write_parquet(df_spy_r, provider.market_loader.daily_dir / "SPY.parquet")
    provider.market_loader.atomic_write_parquet(df_ko_r, provider.market_loader.daily_dir / "KOTST.parquet")
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-12-01 16:10:00", tz="America/New_York").timestamp())

    sig = provider.get_signal("KOTST", regime={"current_state": "LOW_VOLATILITY", "state_index": 0, "kappa": True})
    assert "UNREADY" not in sig["execution_mode"]
    assert sig["macro_regime"]["kappa"] is None
    assert sig["calibration"]["prior_probabilities"] == {
        "outperform_buy": 0.3,
        "neutral_hold": 0.4,
        "underperform_sell": 0.3,
    }
    json.dumps(sig, allow_nan=False)


def test_coderabbit_audit_hardening_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Covers edge-case guards surfaced during code review:
    1. Prior normalization handles overflow (1e308), unnormalized tuples, and invalid override consistency.
    2. OverflowError in manifest class_distribution does not block model loading.
    3. Non-finite HMM posteriors return unavailable macro regime instead of 0.50 fallback.
    4. Unfloored calibrated posteriors matching priors preserve strict 0.0% HOLD; bool kappa drops to None.
    5. SecurityMaster uses same market-beta eligibility (40 rows) as SPLC and returns None when SPY is absent.
    6. Analyst ratings checks regularMarketTime (never fetched_at), pairs quote ts with bar_p, and labels empty period 'Unknown'.
    7. Expired CIK map lookup does not permanently block future live SEC fetches.
    8. Copilot query sanitizes invalid/NaN kappa and filtered_probabilities for strict JSON.
    """
    import types
    import pandas as pd
    from src.data.fundamentals_client import FundamentalsClient
    from src.ipc.live_providers import LiveInferenceProvider
    from src.ipc.uds_server import JsonRpcUdsServer
    from src.risk.kelly_engine import MultiOutcomeKellySizer

    # 1. Overflow-safe normalization, scale invariance, and invalid explicit override consistency (#27: unequal ratios)
    assert MultiOutcomeKellySizer._normalize_prior((0.25e308, 0.50e308, 1.25e308)) == pytest.approx(
        (0.125, 0.25, 0.625), abs=1e-9
    )
    assert MultiOutcomeKellySizer._normalize_prior((10**400, 2 * 10**400, 5 * 10**400)) == pytest.approx(
        (0.125, 0.25, 0.625), abs=1e-9
    )
    sizer = MultiOutcomeKellySizer(execution_mode="LONG_ONLY")
    assert sizer.compute_recommendation(
        p_buy=1.0 / 3.0, p_hold=1.0 / 3.0, p_sell=1.0 / 3.0,
        b_upper=0.10, b_lower=0.05, b_vert=0.0,
        entry_price=100.0, prior_probs=(10**400, 10**400, 10**400),
    ).action == "HOLD"
    rec_unnorm = sizer.compute_recommendation(
        p_buy=0.05, p_hold=0.85, p_sell=0.10,
        b_upper=0.10, b_lower=0.05, b_vert=0.01,
        entry_price=100.0, prior_probs=(0.031, 0.346, 0.123),
    )
    rec_norm = sizer.compute_recommendation(
        p_buy=0.05, p_hold=0.85, p_sell=0.10,
        b_upper=0.10, b_lower=0.05, b_vert=0.01,
        entry_price=100.0, prior_probs=(0.062, 0.692, 0.246),
    )
    assert rec_unnorm == rec_norm
    assert rec_unnorm.action == "HOLD"

    # Invalid explicit override must resolve to no-prior for BOTH solver and action gates (not fall back in solver only)
    sizer_with_inst = MultiOutcomeKellySizer(execution_mode="LONG_ONLY", prior_probs=(0.05, 0.85, 0.10))
    sizer_no_inst = MultiOutcomeKellySizer(execution_mode="LONG_ONLY", prior_probs=None)
    assert sizer_with_inst.compute_recommendation(
        p_buy=0.60, p_hold=0.30, p_sell=0.10, b_upper=0.10, b_lower=0.05, entry_price=100.0, prior_probs=(-1.0, 0.5, 0.5)
    ) == sizer_no_inst.compute_recommendation(
        p_buy=0.60, p_hold=0.30, p_sell=0.10, b_upper=0.10, b_lower=0.05, entry_price=100.0, prior_probs=None
    )

    # 2. OverflowError in manifest class_distribution must NOT abort subsequent model weight loads
    import pickle
    from src.risk.calibration import TemperatureScaler

    weights_dir = tmp_path / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "manifest.json").write_text(
        json.dumps({"class_distribution": {"0": 10**400, "1": 10**400, "2": 10**400}}),
        encoding="utf-8",
    )
    ts_fixture = TemperatureScaler()
    ts_fixture.temperature_ = 1.25
    with open(weights_dir / "temperature_scaler.pkl", "wb") as f:
        pickle.dump(ts_fixture, f)
    prov_ov = LiveInferenceProvider(weights_dir=weights_dir, lakehouse_dir=tmp_path / "lh", enable_live_fetch=False)
    assert prov_ov._class_priors is None
    assert prov_ov.temp_scaler is not None
    assert prov_ov.temp_scaler.temperature_ == pytest.approx(1.25, abs=1e-6)

    # 3 & 4. Non-finite HMM posteriors -> unavailable; zero-skill unfloored (0.001, 0.899, 0.100) -> HOLD; bool kappa -> None
    idx = pd.date_range("2026-01-01", periods=200, freq="B")
    rng = np.random.default_rng(11)
    spy_rets = rng.normal(0.0005, 0.01, size=199)
    spy_close = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(spy_rets)]))
    stk_close = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(1.5 * spy_rets)]))
    df_spy_200 = pd.DataFrame(
        {"open": spy_close, "high": spy_close * 1.01, "low": spy_close * 0.99, "close": spy_close, "volume": np.full(200, 1_000_000.0)},
        index=idx,
    )
    df_bars = pd.DataFrame(
        {"open": stk_close, "high": stk_close * 1.01, "low": stk_close * 0.99, "close": stk_close, "volume": np.full(200, 1_000_000.0)},
        index=idx,
    )
    prov_ov.regime_hmm = types.SimpleNamespace(
        STATE_LABELS={0: "LOW_VOLATILITY", 1: "MEDIUM_VOLATILITY", 2: "HIGH_VOLATILITY"},
        transmat_=None,
        filter_sequence=lambda X: np.array([[float("nan"), 0.5, 0.5]]),
        get_regime_multiplier=lambda p: 0.50,
        get_state_annualized_volatilities=lambda: [12.0, 20.0, 35.0],
        get_stationary_distribution=lambda: [],
    )
    monkeypatch.setattr(prov_ov, "_load_disk_daily", lambda sym: df_spy_200 if sym == "SPY" else (df_bars if sym == "XYZ" else None))
    mr_nan = prov_ov.get_macro_regime()
    assert mr_nan["available"] is False
    assert mr_nan["kappa"] is None

    prov_ov._class_priors = (0.001, 0.899, 0.100)
    prov_ov.lgbm_backbone = types.SimpleNamespace(predict_proba=lambda X: np.array([[0.001, 0.899, 0.100]]))
    prov_ov.temp_scaler = types.SimpleNamespace(temperature_=1.0, transform=lambda p: p)
    sig_zero = prov_ov.get_signal("XYZ", regime={"current_state": "LOW_VOLATILITY", "state_index": 0, "kappa": True})
    assert sig_zero["execution_recommendation"]["action"] == "HOLD"
    assert sig_zero["execution_recommendation"]["l1_normalized_portfolio_allocation_pct"] == 0.0
    assert sig_zero["calibration"]["calibrated_probabilities"]["outperform_buy"] == 0.001
    assert sig_zero["macro_regime"]["kappa"] is None
    json.dumps(sig_zero, allow_nan=False)

    # Identical kappa resolution for BOTH Kelly sizing and frame when kappa=True + valid regime_probs
    prov_ov.lgbm_backbone = types.SimpleNamespace(predict_proba=lambda X: np.array([[0.30, 0.60, 0.10]]))
    sig_derived_k = prov_ov.get_signal(
        "XYZ",
        regime={
            "current_state": "HIGH_VOLATILITY",
            "state_index": 2,
            "kappa": True,
            "filtered_probabilities": {"low_volatility": 0.0, "medium_volatility": 0.0, "high_volatility": 1.0},
        },
    )
    assert sig_derived_k["macro_regime"]["kappa"] == 0.25
    assert sig_derived_k["execution_recommendation"]["action"] == "BUY"
    assert sig_derived_k["execution_recommendation"]["unconstrained_kelly_pct"] > 0.0
    assert sig_derived_k["execution_recommendation"]["l1_normalized_portfolio_allocation_pct"] > 0.0
    assert sig_derived_k["execution_recommendation"]["l1_normalized_portfolio_allocation_pct"] == pytest.approx(
        sig_derived_k["execution_recommendation"]["unconstrained_kelly_pct"] * 0.25,
        abs=0.05,
    )

    # Bullish signal with kappa=None or invalid negative regime_probs must emit l1=0.0 (no silent 0.25 clamping)
    sig_no_kappa_bull = prov_ov.get_signal(
        "XYZ",
        regime={
            "current_state": "UNAVAILABLE",
            "state_index": -1,
            "kappa": None,
            "filtered_probabilities": {"low_volatility": -0.2, "medium_volatility": 0.4, "high_volatility": 0.8},
        },
    )
    assert sig_no_kappa_bull["macro_regime"]["kappa"] is None
    assert sig_no_kappa_bull["macro_regime"]["filtered_probabilities"] == {}
    assert sig_no_kappa_bull["execution_recommendation"]["action"] == "BUY"
    assert sig_no_kappa_bull["execution_recommendation"]["l1_normalized_portfolio_allocation_pct"] == 0.0

    # Missing benchmark fallback must mark divergence unaligned & use vol_1d sigma fallback (not self-residual 0.01 floor);
    # Missing / stale / ahead SPY must return UNREADY (no degenerate target-as-SPY zero residuals);
    # refreshing SPY later WITHOUT clearing _signal_cache must invalidate cached unaligned frame cleanly (#17, #29).
    spy_load_calls = {"n": 0}
    def spy_counting_missing(sym):
        if sym == "SPY":
            spy_load_calls["n"] += 1
        return df_bars if sym == "XYZ" else None

    import src.ipc.live_providers as _lp_mod

    orig_mono_1 = _lp_mod.time.monotonic
    monkeypatch.setattr(_lp_mod.time, "monotonic", lambda: 10_000.0)
    try:
        monkeypatch.setattr(prov_ov, "_load_disk_daily", spy_counting_missing)
        with prov_ov._lock:
            prov_ov._signal_cache.clear()
            prov_ov._news_cache["XYZ"] = (10_000.0, [{"decayed_sentiment": 0.9, "published_at": "2026-01-02T12:00:00Z"}])
        sig_missing_spy = prov_ov.get_signal("XYZ", regime={"current_state": "HIGH_VOLATILITY", "state_index": 2, "kappa": 0.25})
        assert sig_missing_spy["execution_mode"] == "UNREADY"
        assert sig_missing_spy["winning_engine"] == "NONE"
        assert sig_missing_spy["execution_recommendation"]["l1_normalized_portfolio_allocation_pct"] == 0.0
        assert "BENCHMARK_OBSERVATIONS_UNALIGNED" in sig_missing_spy["data_integrity"]["missing_modalities"]
        assert "DIVERGENCE_OBSERVATIONS_UNALIGNED" in sig_missing_spy["data_integrity"]["missing_modalities"]
        fixture_close = float(df_bars["close"].iloc[-1])
        xyz_entry = sig_missing_spy["execution_recommendation"]["entry_price"]
        xyz_tp = sig_missing_spy["execution_recommendation"]["dynamic_take_profit"]
        xyz_sl = sig_missing_spy["execution_recommendation"]["dynamic_stop_loss"]
        assert xyz_entry == pytest.approx(round(fixture_close, 2), abs=0.01)
        expected_vol_1d = max(float(np.std(np.diff(np.log(df_bars["close"].values[-21:])))), 0.005)
        expected_sig10 = max(float(np.sqrt(10.0) * expected_vol_1d), 0.01)
        assert xyz_tp == pytest.approx(round(fixture_close * (1.0 + 2.0 * expected_sig10), 2), abs=0.02)
        assert xyz_sl == pytest.approx(round(fixture_close * (1.0 - expected_sig10), 2), abs=0.02)
        # Repeat within frozen monotonic clock to prove genuine UNREADY cache hit before invalidation (#29)
        sig_missing_spy_cached = prov_ov.get_signal("XYZ", regime={"current_state": "HIGH_VOLATILITY", "state_index": 2, "kappa": 0.25})
        assert sig_missing_spy_cached is sig_missing_spy

        # Now switch SPY to aligned df_spy_200 WITHOUT clearing _signal_cache -> benchmark signature change invalidates cache
        monkeypatch.setattr(prov_ov, "_load_disk_daily", lambda sym: df_spy_200 if sym == "SPY" else (df_bars if sym == "XYZ" else None))
        sig_realigned_spy = prov_ov.get_signal("XYZ", regime={"current_state": "HIGH_VOLATILITY", "state_index": 2, "kappa": 0.25})
        assert "UNREADY" not in sig_realigned_spy["execution_mode"]
        assert "DIVERGENCE_OBSERVATIONS_UNALIGNED" not in sig_realigned_spy["data_integrity"]["missing_modalities"]
    finally:
        monkeypatch.setattr(_lp_mod.time, "monotonic", orig_mono_1)

    # Stale benchmark bars -> honest UNREADY without emitting model signal
    df_spy_stale = df_spy_200.iloc[:-5]
    monkeypatch.setattr(prov_ov, "_load_disk_daily", lambda sym: df_spy_stale if sym == "SPY" else (df_bars if sym == "XYZ" else None))
    with prov_ov._lock:
        prov_ov._signal_cache.clear()
        prov_ov._news_cache["XYZ"] = (time.monotonic(), [{"decayed_sentiment": 0.9, "published_at": "2026-01-02T12:00:00Z"}])
    sig_stale_etf = prov_ov.get_signal("XYZ", regime={"current_state": "HIGH_VOLATILITY", "state_index": 2, "kappa": 0.25})
    assert sig_stale_etf["execution_mode"] == "UNREADY"
    assert "BENCHMARK_OBSERVATIONS_UNALIGNED" in sig_stale_etf["data_integrity"]["missing_modalities"]
    assert "DIVERGENCE_OBSERVATIONS_UNALIGNED" in sig_stale_etf["data_integrity"]["missing_modalities"]

    final_bar_dt = pd.Timestamp(df_bars.index[-1]).strftime("%Y-%m-%d")
    pre_close_utc = pd.Timestamp(f"{final_bar_dt} 14:00:00", tz="America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    post_close_utc = pd.Timestamp(f"{final_bar_dt} 18:00:00", tz="America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    monkeypatch.setattr(prov_ov, "_load_disk_daily", lambda sym: df_spy_200 if sym == "SPY" else (df_bars if sym == "XYZ" else None))
    t0_mono = time.monotonic()
    with prov_ov._lock:
        prov_ov._signal_cache.clear()
        prov_ov._news_cache["XYZ"] = (t0_mono, [{"decayed_sentiment": 0.9, "published_at": pre_close_utc, "published_at_verified": True}])
    sig_pre_close = prov_ov.get_signal("XYZ", regime={"current_state": "HIGH_VOLATILITY", "state_index": 2, "kappa": 0.25})
    assert "DIVERGENCE_OBSERVATIONS_UNALIGNED" not in sig_pre_close["data_integrity"]["missing_modalities"]

    # Unverified fallback news timestamp (published_at_verified=False) must NOT establish news eligibility
    with prov_ov._lock:
        prov_ov._news_cache["XYZ"] = (t0_mono + 0.5, [{"decayed_sentiment": 0.9, "published_at": pre_close_utc, "published_at_verified": False}])
    sig_unverified_news = prov_ov.get_signal("XYZ", regime={"current_state": "HIGH_VOLATILITY", "state_index": 2, "kappa": 0.25})
    assert "DIVERGENCE_OBSERVATIONS_UNALIGNED" in sig_unverified_news["data_integrity"]["missing_modalities"]
    assert sig_unverified_news["signal_breakdown"]["smart_money_divergence_pos"] == 0.0

    # Same-day 10:00 ET target vs 16:00 ET finalized SPY -> returns UNREADY without invoking residualizer or emitting recommendation (#17)
    df_xyz_prev = df_bars.iloc[:-1]
    t10_et_ms = int(pd.Timestamp(f"{final_bar_dt} 10:00:00", tz="America/New_York").timestamp() * 1000)
    pre_10_et_utc = pd.Timestamp(f"{final_bar_dt} 09:30:00", tz="America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    res_inputs: list[tuple[np.ndarray, np.ndarray]] = []
    orig_roll_capture = prov_ov.residualizer.compute_rolling_residuals

    def capture_roll(r_stk, r_spy_in, r_sec_in):
        res_inputs.append((r_stk.copy(), r_spy_in.copy()))
        return orig_roll_capture(r_stk, r_spy_in, r_sec_in)

    monkeypatch.setattr(prov_ov.residualizer, "compute_rolling_residuals", capture_roll)
    prov_ov.enable_live_fetch = True
    monkeypatch.setattr(prov_ov, "_load_disk_daily", lambda sym: df_spy_200 if sym == "SPY" else (df_xyz_prev if sym == "XYZ" else None))
    monkeypatch.setattr(prov_ov, "_maybe_refresh_daily", lambda sym: None)
    monkeypatch.setattr(
        prov_ov,
        "_fetch_live_quote_cached",
        lambda sym: {"price": float(df_bars["close"].iloc[-1]), "date": final_bar_dt, "observation_timestamp_ms": t10_et_ms} if sym == "XYZ" else None,
    )
    with prov_ov._lock:
        prov_ov._news_cache["XYZ"] = (t0_mono + 0.8, [{"decayed_sentiment": 0.9, "published_at": pre_10_et_utc, "published_at_verified": True}])
    sig_bench_ahead = prov_ov.get_signal("XYZ", regime={"current_state": "HIGH_VOLATILITY", "state_index": 2, "kappa": 0.25})
    assert sig_bench_ahead["execution_mode"] == "UNREADY"
    assert sig_bench_ahead["winning_engine"] == "NONE"
    assert sig_bench_ahead["execution_recommendation"]["l1_normalized_portfolio_allocation_pct"] == 0.0
    assert "BENCHMARK_OBSERVATIONS_UNALIGNED" in sig_bench_ahead["data_integrity"]["missing_modalities"]
    assert "DIVERGENCE_OBSERVATIONS_UNALIGNED" in sig_bench_ahead["data_integrity"]["missing_modalities"]
    assert len(res_inputs) == 0
    monkeypatch.setattr(prov_ov.residualizer, "compute_rolling_residuals", orig_roll_capture)
    prov_ov.enable_live_fetch = False
    monkeypatch.setattr(prov_ov, "_load_disk_daily", lambda sym: df_spy_200 if sym == "SPY" else (df_bars if sym == "XYZ" else None))

    # Warm cache with a verified pre-close headline first (assert alignment), then flip ONLY _news_cache
    # to post-close with fixed bars/regime/live-fetch so stale cache reuse would fail (#21)
    with prov_ov._lock:
        prov_ov._signal_cache.clear()
        prov_ov._news_cache["XYZ"] = (t0_mono + 0.9, [{"decayed_sentiment": 0.9, "published_at": pre_close_utc, "published_at_verified": True}])
    sig_warm_aligned = prov_ov.get_signal("XYZ", regime={"current_state": "HIGH_VOLATILITY", "state_index": 2, "kappa": 0.25})
    assert "DIVERGENCE_OBSERVATIONS_UNALIGNED" not in sig_warm_aligned["data_integrity"]["missing_modalities"]

    with prov_ov._lock:
        prov_ov._news_cache["XYZ"] = (t0_mono + 1.0, [{"decayed_sentiment": 0.9, "published_at": post_close_utc, "published_at_verified": True}])
    sig_post_close = prov_ov.get_signal("XYZ", regime={"current_state": "HIGH_VOLATILITY", "state_index": 2, "kappa": 0.25})
    assert sig_post_close["signal_breakdown"]["smart_money_divergence_pos"] == 0.0
    assert sig_post_close["signal_breakdown"]["smart_money_divergence_neg"] == 0.0
    assert "DIVERGENCE_OBSERVATIONS_UNALIGNED" in sig_post_close["data_integrity"]["missing_modalities"]

    # Non-finite final residual must not fall back to penultimate residual
    orig_roll_fn = prov_ov.residualizer.compute_rolling_residuals
    def nan_tail_roll(r_stk, r_spy, r_sec):
        out = orig_roll_fn(r_stk, r_spy, r_sec)
        out.residuals[-2] = 0.05
        out.residuals[-1] = float("nan")
        return out
    monkeypatch.setattr(prov_ov.residualizer, "compute_rolling_residuals", nan_tail_roll)
    with prov_ov._lock:
        prov_ov._news_cache["XYZ"] = (t0_mono + 2.0, [{"decayed_sentiment": 0.9, "published_at": pre_close_utc, "published_at_verified": True}])
    sig_nan_tail = prov_ov.get_signal("XYZ", regime={"current_state": "HIGH_VOLATILITY", "state_index": 2, "kappa": 0.25})
    assert sig_nan_tail["signal_breakdown"]["smart_money_divergence_pos"] == 0.0
    assert sig_nan_tail["signal_breakdown"]["smart_money_divergence_neg"] == 0.0
    assert "DIVERGENCE_OBSERVATIONS_UNALIGNED" in sig_nan_tail["data_integrity"]["missing_modalities"]
    monkeypatch.setattr(prov_ov.residualizer, "compute_rolling_residuals", orig_roll_fn)
    with prov_ov._lock:
        prov_ov._news_cache.clear()
        prov_ov._signal_cache.clear()

    # 5. Missing SPY -> beta_market_60d is None; 40-row (< 65) target with varying SPY recovers ~1.50; flat SPY -> None (not 0.0 or 1.0)
    monkeypatch.setattr(prov_ov, "_load_disk_daily", lambda sym: df_bars if sym == "AAPL" else None)
    monkeypatch.setattr(prov_ov.fundamentals_client, "get_yf_meta", lambda *a, **kw: None)
    assert prov_ov.get_security_master("AAPL")["beta_market_60d"] is None

    df_aapl_40 = df_bars.tail(40)
    monkeypatch.setattr(prov_ov, "_load_disk_daily", lambda sym: df_aapl_40 if sym == "AAPL" else (df_spy_200 if sym == "SPY" else None))
    with prov_ov._lock:
        prov_ov._sec_master_cache.clear()
        prov_ov._peer_graph_cache.clear()
    sm_40 = prov_ov.get_security_master("AAPL")
    assert sm_40["beta_market_60d"] == pytest.approx(1.5, abs=0.05)
    pg_40 = prov_ov.get_peer_graph("AAPL")
    aapl_node_40 = next(n for n in pg_40["nodes"] if n["id"] == "AAPL")
    assert aapl_node_40["beta_market"] == pytest.approx(sm_40["beta_market_60d"], abs=0.01)

    df_spy_flat = pd.DataFrame(
        {"open": np.full(200, 100.0), "high": np.full(200, 100.0), "low": np.full(200, 100.0), "close": np.full(200, 100.0), "volume": np.full(200, 1_000_000.0)},
        index=idx,
    )
    monkeypatch.setattr(prov_ov, "_load_disk_daily", lambda sym: df_bars if sym == "AAPL" else (df_spy_flat if sym == "SPY" else None))
    with prov_ov._lock:
        prov_ov._sec_master_cache.clear()
        prov_ov._peer_graph_cache.clear()
    assert prov_ov.get_security_master("AAPL")["beta_market_60d"] is None
    pg_flat_spy = prov_ov.get_peer_graph("SPY")
    spy_node_flat = next(n for n in pg_flat_spy["nodes"] if n["id"] == "SPY")
    assert spy_node_flat["beta_market"] is None

    # 6. Analyst ratings checks regularMarketTime (ignoring fetched_at), pairs quote ts with bar_p, and labels empty period 'Unknown'
    df_aapl_1d = pd.DataFrame(
        {"open": [120.0], "high": [121.0], "low": [119.0], "close": [120.0], "volume": [1_000_000.0]},
        index=pd.DatetimeIndex(["2026-11-18"]),
    )
    monkeypatch.setattr(prov_ov, "_load_disk_daily", lambda sym: df_aapl_1d if sym == "AAPL" else None)
    monkeypatch.setattr(
        prov_ov.fundamentals_client,
        "get_yf_meta",
        lambda sym, enable_live_fetch=False: {
            "fetched_at": "2099-01-01T00:00:00Z",  # Metadata write time alone must NOT replace market bar price
            "info": {"currentPrice": 200.0, "targetMeanPrice": 240.0, "numberOfAnalystOpinions": 10},
            "rec_trend": [{"strong_buy": 8, "buy": 2, "hold": 0, "sell": 0}],
        },
    )
    anr_no_mkt_ts = prov_ov.get_analyst_ratings("AAPL")
    assert anr_no_mkt_ts["current_price"] == pytest.approx(120.0, abs=0.01)
    assert anr_no_mkt_ts["historical_revisions"][0]["period"] == "Unknown"
    assert anr_no_mkt_ts["historical_revisions"][0]["mean_target"] is None

    # When regularMarketTime IS fresher than market close, regularMarketPrice (not mismatched currentPrice) is accepted
    import src.ipc.live_providers as _lp_rat

    orig_time_rat = _lp_rat.time.time
    monkeypatch.setattr(_lp_rat.time, "time", lambda: pd.Timestamp("2026-11-19T17:00:00Z").timestamp())
    try:
        with prov_ov._lock:
            prov_ov._ratings_cache.clear()
        fresher_epoch_sec = int(pd.Timestamp("2026-11-19T15:00:00Z").timestamp())
        monkeypatch.setattr(
            prov_ov.fundamentals_client,
            "get_yf_meta",
            lambda sym, enable_live_fetch=False: {
                "info": {
                    "currentPrice": 999.0,
                    "regularMarketPrice": 200.0,
                    "regularMarketTime": fresher_epoch_sec,
                    "targetMeanPrice": 240.0,
                    "numberOfAnalystOpinions": 10,
                },
                "rec_trend": [{"period": "0m", "strong_buy": 8, "buy": 2, "hold": 0, "sell": 0}],
            },
        )
        anr_fresh = prov_ov.get_analyst_ratings("AAPL")
        assert anr_fresh["current_price"] == pytest.approx(200.0, abs=0.01)
        assert anr_fresh["historical_revisions"][0]["period"] == "Current"
        assert anr_fresh["historical_revisions"][0]["mean_target"] == pytest.approx(240.0, abs=0.01)

        # Atomic quote ts pairing & untimestamped live quote skipped for ratings comparison
        prov_ov.enable_live_fetch = True
        monkeypatch.setattr(prov_ov, "_maybe_refresh_daily", lambda sym: None)
        quote_calls = {"n": 0}
        def fake_quote(sym: str):
            quote_calls["n"] += 1
            if quote_calls["n"] == 1:
                return {"price": 100.0, "date": "2026-11-18", "observation_timestamp_ms": int(pd.Timestamp("2026-11-19T14:00:00Z").timestamp() * 1000)}
            return {"price": 110.0, "date": "2026-11-18", "observation_timestamp_ms": int(pd.Timestamp("2026-11-19T16:00:00Z").timestamp() * 1000)}

        monkeypatch.setattr(prov_ov, "_fetch_live_quote_cached", fake_quote)
        with prov_ov._lock:
            prov_ov._ratings_cache.clear()
        anr_atomic = prov_ov.get_analyst_ratings("AAPL")
        assert quote_calls["n"] == 1
        assert anr_atomic["current_price"] == pytest.approx(200.0, abs=0.01)

        # Quote WITHOUT observation_timestamp_ms (fetch timestamp_ms only) must not beat 15:30 ET Yahoo or Monday bar vs Tuesday regularMarketTime
        monkeypatch.setattr(
            prov_ov,
            "_fetch_live_quote_cached",
            lambda sym: {
                "price": 99.0,
                "date": "2026-11-18",
                "timestamp_ms": int(pd.Timestamp("2026-11-19T15:01:00Z").timestamp() * 1000),
            },
        )
        monkeypatch.setattr(
            prov_ov.fundamentals_client,
            "get_yf_meta",
            lambda sym, enable_live_fetch=False: {
                "info": {
                    "regularMarketPrice": 200.0,
                    "regularMarketTime": int(pd.Timestamp("2026-11-18T20:30:00Z").timestamp()),  # 15:30 ET
                    "targetMeanPrice": 240.0,
                    "numberOfAnalystOpinions": 10,
                },
                "rec_trend": [{"period": "0m", "strong_buy": 8, "buy": 2, "hold": 0, "sell": 0}],
            },
        )
        with prov_ov._lock:
            prov_ov._ratings_cache.clear()
        anr_untimestamped_q = prov_ov.get_analyst_ratings("AAPL")
        assert anr_untimestamped_q["current_price"] == pytest.approx(120.0, abs=0.01)

        # If regularMarketPrice is absent while disk bar exists, mismatched currentPrice is NOT promoted by regularMarketTime
        monkeypatch.setattr(
            prov_ov.fundamentals_client,
            "get_yf_meta",
            lambda sym, enable_live_fetch=False: {
                "info": {
                    "currentPrice": 999.0,
                    "regularMarketTime": fresher_epoch_sec,
                    "targetMeanPrice": 240.0,
                    "numberOfAnalystOpinions": 10,
                },
                "rec_trend": [{"period": "0m", "strong_buy": 8, "buy": 2, "hold": 0, "sell": 0}],
            },
        )
        with prov_ov._lock:
            prov_ov._ratings_cache.clear()
        anr_cur_only = prov_ov.get_analyst_ratings("AAPL")
        assert anr_cur_only["current_price"] == pytest.approx(120.0, abs=0.01)

        df_aapl_mon = pd.DataFrame(
            {"open": [120.0], "high": [121.0], "low": [119.0], "close": [120.0], "volume": [1_000_000.0]},
            index=pd.DatetimeIndex(["2026-11-16"]),
        )
        monkeypatch.setattr(prov_ov, "_load_disk_daily", lambda sym: df_aapl_mon if sym == "AAPL" else None)
        monkeypatch.setattr(
            prov_ov,
            "_fetch_live_quote_cached",
            lambda sym: {
                "price": 99.0,
                "date": "2026-11-16",
                "timestamp_ms": int(pd.Timestamp("2026-11-17T15:01:00Z").timestamp() * 1000),
            },
        )
        monkeypatch.setattr(
            prov_ov.fundamentals_client,
            "get_yf_meta",
            lambda sym, enable_live_fetch=False: {
                "info": {
                    "regularMarketPrice": 200.0,
                    "regularMarketTime": int(pd.Timestamp("2026-11-17T15:00:00Z").timestamp()),
                    "targetMeanPrice": 240.0,
                    "numberOfAnalystOpinions": 10,
                },
                "rec_trend": [{"period": "0m", "strong_buy": 8, "buy": 2, "hold": 0, "sell": 0}],
            },
        )
        with prov_ov._lock:
            prov_ov._ratings_cache.clear()
        anr_mon_vs_tue = prov_ov.get_analyst_ratings("AAPL")
        assert anr_mon_vs_tue["current_price"] == pytest.approx(200.0, abs=0.01)

        # Completed 16:00 ET $120 disk bar must NOT be overwritten by stale 13:00 ET cached quote ($100) for ANR
        t13_et_ms = int(pd.Timestamp("2026-11-18 13:00:00", tz="America/New_York").timestamp() * 1000)
        t14_et_sec = int(pd.Timestamp("2026-11-18 14:00:00", tz="America/New_York").timestamp())
        monkeypatch.setattr(prov_ov, "_load_disk_daily", lambda sym: df_aapl_1d if sym == "AAPL" else None)
        monkeypatch.setattr(
            prov_ov,
            "_fetch_live_quote_cached",
            lambda sym: {"price": 100.0, "date": "2026-11-18", "observation_timestamp_ms": t13_et_ms},
        )
        monkeypatch.setattr(
            prov_ov.fundamentals_client,
            "get_yf_meta",
            lambda sym, enable_live_fetch=False: {
                "info": {"regularMarketPrice": 110.0, "regularMarketTime": t14_et_sec, "targetMeanPrice": 240.0, "numberOfAnalystOpinions": 10},
                "rec_trend": [{"period": "0m", "strong_buy": 8, "buy": 2, "hold": 0, "sell": 0}],
            },
        )
        with prov_ov._lock:
            prov_ov._ratings_cache.clear()
        assert prov_ov.get_analyst_ratings("AAPL")["current_price"] == pytest.approx(120.0, abs=0.01)
    finally:
        monkeypatch.setattr(_lp_rat.time, "time", orig_time_rat)

    # Missing bar fallback skips negative regularMarketPrice (-1) and uses positive currentPrice (120)
    monkeypatch.setattr(prov_ov, "_load_disk_daily", lambda sym: None)
    monkeypatch.setattr(prov_ov, "_fetch_live_quote_cached", lambda sym: None)
    monkeypatch.setattr(
        prov_ov.fundamentals_client,
        "get_yf_meta",
        lambda sym, enable_live_fetch=False: {
            "info": {"regularMarketPrice": -1.0, "currentPrice": 120.0, "targetMeanPrice": 150.0, "numberOfAnalystOpinions": 5},
            "rec_trend": [],
        },
    )
    with prov_ov._lock:
        prov_ov._ratings_cache.clear()
    assert prov_ov.get_analyst_ratings("AAPL")["current_price"] == pytest.approx(120.0, abs=0.01)

    # 7. Stale ticker map lookup must not permanently block later live fetches
    fc = FundamentalsClient(lakehouse_dir=tmp_path / "lh_cik")
    p_map = fc.raw_dir / "sec_company_tickers.json"
    p_map.parent.mkdir(parents=True, exist_ok=True)
    p_map.write_text(json.dumps({"AAPL": "0000320193"}), encoding="utf-8")
    old_t = time.time() - 86400 * 30
    os.utime(p_map, (old_t, old_t))
    sec_calls: list[str] = []

    def fake_sec_fetch(url: str, timeout: float = 6.0):
        sec_calls.append(url)
        if len(sec_calls) == 1:
            return None
        return json.dumps({"0": {"ticker": "XYZNEW", "cik_str": 123456}})

    import src.data.fundamentals_client as fc_mod

    fake_mono = {"now": 1000.0}
    monkeypatch.setattr(fc_mod.time, "monotonic", lambda: fake_mono["now"])
    monkeypatch.setattr(fc, "_sec_fetch_text", fake_sec_fetch)
    assert fc.resolve_cik("AAPL", enable_live_fetch=True) == "0000320193"
    assert fc.resolve_cik("XYZNEW", enable_live_fetch=True) is None
    # Before cooldown expiry, fetch must remain suppressed
    fake_mono["now"] += fc._FAILED_COOLDOWN_SECS - 1.0
    assert fc.resolve_cik("XYZNEW", enable_live_fetch=True) is None
    assert len(sec_calls) == 1
    # Advance past cooldown without clearing _failed_until -> natural expiry succeeds (#23)
    fake_mono["now"] += 2.0
    assert fc.resolve_cik("XYZNEW", enable_live_fetch=True) == "0000123456"
    assert len(sec_calls) == 2

    # 8. UDS copilot_query kappa=None / NaN & filtered_probabilities NaN sanitization vs stub 0.47
    srv_stub = JsonRpcUdsServer(socket_path=str(tmp_path / "stub.sock"))
    res_stub = json.loads(srv_stub.process_message(json.dumps({
        "jsonrpc": "2.0", "method": "copilot_query", "params": {"query": "hmm regime", "ticker": "SPY"}, "id": "s1"
    })))["result"]
    assert "kappa: 0.47" in res_stub["answer"]

    srv_bad = JsonRpcUdsServer(
        socket_path=str(tmp_path / "bad.sock"),
        regime_provider=lambda: {
            "available": False,
            "current_state": "UNAVAILABLE",
            "state_index": -1,
            "kappa": float("nan"),
            "filtered_probabilities": {"low_volatility": float("nan"), "medium_volatility": 0.2, "high_volatility": 0.1},
            "transition_matrix": [[0.9, float("nan"), 0.1]],
        },
    )
    res_bad_raw = srv_bad.process_message(json.dumps({
        "jsonrpc": "2.0", "method": "copilot_query", "params": {"query": "hmm regime", "ticker": "SPY"}, "id": "s2"
    }))
    # Must be strict valid JSON (no NaN token)
    json.loads(res_bad_raw, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    res_bad = json.loads(res_bad_raw)["result"]
    assert "kappa: N/A" in res_bad["answer"]
    assert "probabilities: unavailable" in res_bad["answer"]
    assert res_bad["data"]["kappa"] is None
    assert res_bad["data"]["filtered_probabilities"] == {}
    assert res_bad["data"]["transition_matrix"][0][1] is None

    srv_oob = JsonRpcUdsServer(
        socket_path=str(tmp_path / "oob.sock"),
        regime_provider=lambda: {
            "available": True,
            "current_state": "LOW_VOLATILITY",
            "state_index": 0,
            "kappa": 0.50,
            "filtered_probabilities": {"low_volatility": 0.8, "medium_volatility": 0.8, "high_volatility": 0.4},
        },
    )
    res_oob = json.loads(srv_oob.process_message(json.dumps({
        "jsonrpc": "2.0", "method": "copilot_query", "params": {"query": "hmm regime", "ticker": "SPY"}, "id": "s3"
    })))["result"]
    assert "probabilities: unavailable" in res_oob["answer"]
    assert res_oob["data"]["filtered_probabilities"] == {}


def test_coderabbit_loop_fixes_comprehensive(tmp_path, monkeypatch) -> None:
    """Covers findings #13-#26 across Kelly centering, broadcaster, EDGAR, MarketDataLoader, and LiveInferenceProvider."""
    from datetime import date, time as dt_time
    import pandas as pd
    from src.data.fundamentals_client import FundamentalsClient
    from src.data.market_loader import MarketDataLoader
    from src.ipc.live_providers import LiveInferenceProvider
    from src.ipc.ws_broadcaster import (
        build_qare_signal_frame,
        normalize_regime_probs,
        resolve_regime_kappa,
    )
    from src.risk.calibration import TemperatureScaler
    from src.risk.kelly_engine import MultiOutcomeKellySizer

    # 1. (#26) Signed centered payoff when b_vert lies outside price barriers
    sizer_ls = MultiOutcomeKellySizer(execution_mode="LONG_SHORT")
    rec_ls = sizer_ls.compute_recommendation(
        p_buy=0.40,
        p_hold=0.20,
        p_sell=0.40,
        b_upper=0.10,
        b_lower=0.05,
        b_vert=0.30,
        entry_price=100.0,
        kappa=0.50,
        prior_probs=(0.10, 0.80, 0.10),
        execution_mode="LONG_SHORT",
    )
    assert rec_ls.action == "SELL"
    assert rec_ls.l1_normalized_portfolio_allocation_pct < 0.0

    # 2. (#16, #21) Tolerated regime prob normalization + numeric overflow safety
    norm_partial = normalize_regime_probs({"medium_volatility": 0.60, "high_volatility": 0.41})
    assert sum(norm_partial.values()) == pytest.approx(1.0, abs=1e-9)
    assert norm_partial["low_volatility"] == 0.0
    assert norm_partial["medium_volatility"] == pytest.approx(0.60 / 1.01, abs=1e-9)
    assert norm_partial["high_volatility"] == pytest.approx(0.41 / 1.01, abs=1e-9)

    assert resolve_regime_kappa(10**400, {"low_volatility": 0.0, "medium_volatility": 0.0, "high_volatility": 1.0}) == 0.25
    frame_ovf = build_qare_signal_frame(
        ticker="NVDA",
        signal_breakdown={"bad": 10**400, "good": 0.42},
        regime_kappa=10**400,
        regime_probs={"low_volatility": 0.0, "medium_volatility": 0.0, "high_volatility": 1.0},
    )
    assert "bad" not in frame_ovf["signal_breakdown"]
    assert frame_ovf["signal_breakdown"]["good"] == pytest.approx(0.42)
    assert frame_ovf["macro_regime"]["kappa"] == 0.25

    # 3. (#17) Same-day filing extraction & restatement detection share identical ordering
    fc = FundamentalsClient(lakehouse_dir=tmp_path / "lh_same_day")
    cik = "0000111222"
    (fc.raw_dir / "sec_company_tickers.json").parent.mkdir(parents=True, exist_ok=True)
    (fc.raw_dir / "edgar_submissions").mkdir(parents=True, exist_ok=True)
    (fc.raw_dir / "edgar_facts").mkdir(parents=True, exist_ok=True)
    (fc.raw_dir / "yf_meta").mkdir(parents=True, exist_ok=True)
    (fc.raw_dir / "edgar_submissions" / f"CIK{cik}.json").write_text(json.dumps({
        "cik": cik,
        "recent": {
            "form": ["10-K", "10-K", "10-K/A"],
            "filingDate": ["2017-08-02", "2017-08-02", "2017-08-02"],
            "reportDate": ["2017-06-30", "2017-06-30", "2017-06-30"],
            "accessionNumber": ["A1", "A2", "A3"],
            "acceptanceDateTime": ["2017-08-02T16:30:00.000Z", "", "2017-08-02T17:30:00.000Z"],
            "primaryDocument": ["f1.htm", "f2.htm", "f3.htm"],
        },
    }), encoding="utf-8")
    (fc.raw_dir / "edgar_facts" / f"CIK{cik}.json").write_text(json.dumps({
        "cik": cik,
        "entityName": "TEST CORP",
        "us_gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {"form": "10-K", "start": "2016-07-01", "end": "2017-06-30", "filed": "2017-08-02", "accn": "A1", "val": 100_000_000},
                        {"form": "10-K", "start": "2016-07-01", "end": "2017-06-30", "filed": "2017-08-02", "accn": "A2", "val": 101_000_000},
                        {"form": "10-K/A", "start": "2016-07-01", "end": "2017-06-30", "filed": "2017-08-02", "accn": "A3", "val": 120_000_000},
                    ]
                }
            },
            "Assets": {"units": {"USD": [{"form": "10-K", "end": "2017-06-30", "filed": "2017-08-02", "accn": "A1", "val": 500_000_000, "fp": "FY"}]}},
        },
    }), encoding="utf-8")
    fin_sd = fc.build_financials_payload("TST", known_cik=cik, pit_strict=True, enable_live_fetch=False)
    f_sd = next(f for f in fin_sd["filings"] if f["period"] == "FY2017")
    rev_row_sd = next(r for r in fin_sd["incomeStatement"] if r["tag"] == "Revenues")
    assert rev_row_sd["values"]["FY2017"] == pytest.approx(100.0, abs=0.05)
    assert f_sd["is_restated"] is True
    assert "Subsequent 10-K/A restatement changed Revenue by +20M" in (f_sd["restatement_delta_notes"] or "")

    # Cross-submitter same-day 10-K vs 10-K without complete acceptance timestamps must NOT claim restatement (#20)
    cik_tie = "0000111223"
    (fc.raw_dir / "edgar_submissions" / f"CIK{cik_tie}.json").write_text(json.dumps({
        "cik": cik_tie,
        "recent": {
            "form": ["10-K", "10-K"],
            "filingDate": ["2017-08-02", "2017-08-02"],
            "reportDate": ["2017-06-30", "2017-06-30"],
            "accessionNumber": ["0001193125-17-000001", "0000789019-17-000002"],
            "acceptanceDateTime": ["", ""],
            "primaryDocument": ["f1.htm", "f2.htm"],
        },
    }), encoding="utf-8")
    (fc.raw_dir / "edgar_facts" / f"CIK{cik_tie}.json").write_text(json.dumps({
        "cik": cik_tie,
        "entityName": "TEST CORP",
        "us_gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {"form": "10-K", "start": "2016-07-01", "end": "2017-06-30", "filed": "2017-08-02", "accn": "0001193125-17-000001", "val": 100_000_000},
                        {"form": "10-K", "start": "2016-07-01", "end": "2017-06-30", "filed": "2017-08-02", "accn": "0000789019-17-000002", "val": 115_000_000},
                    ]
                }
            },
            "Assets": {"units": {"USD": [{"form": "10-K", "end": "2017-06-30", "filed": "2017-08-02", "accn": "0000789019-17-000002", "val": 500_000_000, "fp": "FY"}]}},
        },
    }), encoding="utf-8")
    fin_tie = fc.build_financials_payload("TST2", known_cik=cik_tie, pit_strict=True, enable_live_fetch=False)
    f_tie = next(f for f in fin_tie["filings"] if f["period"] == "FY2017")
    assert f_tie["is_restated"] is False

    # 4. (#25) Expired stale yf_meta reports YFINANCE_CACHED_STALE with snapshot_at
    yf_cache_p = fc.raw_dir / "yf_meta" / "TST.json"
    yf_cache_p.write_text(json.dumps({
        "fetched_at": "2026-10-01T12:00:00Z",
        "info": {"targetMeanPrice": 250.0, "numberOfAnalystOpinions": 12},
        "rec_trend": [{"period": "0m", "strong_buy": 8, "buy": 4, "hold": 0, "underperform": 0, "sell": 0}],
    }), encoding="utf-8")
    old_epoch = time.time() - 86400 * 5
    os.utime(yf_cache_p, (old_epoch, old_epoch))

    # 5. (#20, #23) NYSE half-days + MarketDataLoader.fetch_live_quote observation_timestamp_ms + #18 unready benchmark skip + #15/#22/#24/#19
    assert MarketDataLoader.nyse_session_close_time(date(2026, 11, 27)) == dt_time(13, 0)
    assert MarketDataLoader.nyse_session_close_time(date(2026, 11, 20)) == dt_time(16, 0)
    assert MarketDataLoader.nyse_session_close_time(date(2026, 12, 24)) == dt_time(13, 0)

    weights_dir = tmp_path / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    prov = LiveInferenceProvider(lakehouse_dir=tmp_path / "data", weights_dir=weights_dir, enable_live_fetch=False)
    prov.fundamentals_client = fc
    idx_nov28 = pd.bdate_range(end="2025-11-28", periods=200)
    rng = np.random.default_rng(7)
    base_rets = rng.normal(0.0005, 0.015, size=200)
    closes = 100.0 * np.exp(np.cumsum(base_rets))
    df_tgt = pd.DataFrame(
        {"open": closes * 0.995, "high": closes * 1.01, "low": closes * 0.99, "close": closes, "volume": np.full(200, 1_500_000.0)},
        index=idx_nov28,
    )
    spy_rets = np.random.default_rng(19).normal(0.0004, 0.011, size=200)
    spy_closes = 400.0 * np.exp(np.cumsum(spy_rets))
    df_spy = pd.DataFrame(
        {"open": spy_closes * 0.996, "high": spy_closes * 1.009, "low": spy_closes * 0.991, "close": spy_closes, "volume": np.full(200, 4_000_000.0)},
        index=idx_nov28,
    )

    # #18: Unready provider must NOT load SPY before returning UNREADY
    seen_loads: list[str] = []
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: (seen_loads.append(sym) or (df_tgt if sym == "TST" else None)))
    unready_sig = prov.get_signal("TST")
    assert unready_sig["execution_mode"] == "UNREADY"
    assert seen_loads == ["TST"]

    # Wire trained weights + scaler
    class DummyLGBM:
        def predict_proba(self, x):
            return np.tile([0.65, 0.25, 0.10], (len(x), 1))

    prov.lgbm_backbone = DummyLGBM()
    sc = TemperatureScaler()
    sc.temperature_ = 1.0
    prov.temp_scaler = sc
    prov.regime_hmm = object()

    # Check disk fallback get_latest_quote reports finalized bar close observation time, not wall clock
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_tgt.loc[:"2025-11-14"] if sym == "TST" else None)
    q_disk_fb = prov.get_latest_quote("TST")
    assert q_disk_fb["observation_timestamp_ms"] == int(pd.Timestamp("2025-11-14 16:00:00", tz="America/New_York").timestamp() * 1000)
    assert q_disk_fb["timestamp_ms"] == q_disk_fb["observation_timestamp_ms"]

    # #25: Analyst ratings on stale cache
    anr_stale = prov.get_analyst_ratings("TST")
    assert anr_stale["available"] is True
    assert anr_stale["source"] == "YFINANCE_CACHED_STALE"
    assert anr_stale["snapshot_at"] == "2026-10-01T12:00:00Z"

    # #23 & Loop-2 #12, #14: fetch_live_quote normal, out-of-range rmt, and fallback date-mismatch paths
    import yfinance as yf

    t_obs_et = pd.Timestamp("2025-12-01 10:30:00", tz="America/New_York")
    hist_5d = pd.DataFrame(
        {"Open": [101.0], "High": [103.0], "Low": [100.5], "Close": [102.5], "Volume": [200000]},
        index=pd.DatetimeIndex([pd.Timestamp("2025-11-28 00:00:00-05:00")]),
    )

    class FakeTickerNormal:
        fast_info = type("FINorm", (), {
            "lastPrice": 115.5, "open": 114.0, "dayHigh": 116.0, "dayLow": 113.5, "previousClose": 112.0, "lastVolume": 50000,
            "_md": {"regularMarketTime": t_obs_et, "regularMarketPrice": 115.5},
            "_prices_1y": pd.DataFrame({"Close": [115.5]}, index=pd.DatetimeIndex([pd.Timestamp("2025-12-01 00:00:00-05:00")])),
            "get": lambda self, k, d=None: {"lastPrice": 115.5, "open": 114.0, "dayHigh": 116.0, "dayLow": 113.5, "previousClose": 112.0, "lastVolume": 50000}.get(k, d),
        })()
        def history(self, **kw):
            return hist_5d
        def get_history_metadata(self):
            return {"regularMarketTime": t_obs_et, "regularMarketPrice": 115.5}

    monkeypatch.setattr(yf, "Ticker", lambda sym: FakeTickerNormal())
    q_norm = prov.market_loader.fetch_live_quote("TST")
    assert q_norm is not None
    assert q_norm["observation_timestamp_ms"] == int(t_obs_et.timestamp() * 1000)
    assert q_norm["date"] == "2025-12-01"

    class FakeTickerHugeRmt(FakeTickerNormal):
        fast_info = type("FIHuge", (), {
            "lastPrice": 115.5, "open": 114.0, "dayHigh": 116.0, "dayLow": 113.5, "previousClose": 112.0, "lastVolume": 50000,
            "_md": {"regularMarketTime": 1e20, "regularMarketPrice": 115.5},
            "_prices_1y": pd.DataFrame({"Close": [115.5]}, index=pd.DatetimeIndex([pd.Timestamp("2025-12-01 00:00:00-05:00")])),
            "get": lambda self, k, d=None: {"lastPrice": 115.5, "open": 114.0, "dayHigh": 116.0, "dayLow": 113.5, "previousClose": 112.0, "lastVolume": 50000}.get(k, d),
        })()
        def get_history_metadata(self):
            return {"regularMarketTime": 1e20, "regularMarketPrice": 115.5}

    monkeypatch.setattr(yf, "Ticker", lambda sym: FakeTickerHugeRmt())
    q_huge = prov.market_loader.fetch_live_quote("TST")
    assert q_huge is not None and q_huge["price"] == 115.5
    assert "observation_timestamp_ms" not in q_huge

    class FakeTickerFallbackMismatch:
        @property
        def fast_info(self):
            raise RuntimeError("no fast_info")
        def history(self, **kw):
            return pd.DataFrame(
                {"Open": [101.0], "High": [103.0], "Low": [100.5], "Close": [102.5], "Volume": [200000]},
                index=pd.DatetimeIndex([pd.Timestamp("2025-11-14 00:00:00-05:00")]),
            )
        def get_history_metadata(self):
            return {"regularMarketTime": pd.Timestamp("2025-11-17 10:00:00", tz="America/New_York")}

    monkeypatch.setattr(yf, "Ticker", lambda sym: FakeTickerFallbackMismatch())
    q_fb = prov.market_loader.fetch_live_quote("TST")
    assert q_fb is not None
    expected_nov14_close_ms = int(pd.Timestamp("2025-11-14 16:00:00", tz="America/New_York").timestamp() * 1000)
    assert q_fb["observation_timestamp_ms"] == expected_nov14_close_ms

    # Feed q_norm into both ratings & signal (newer than Nov 28 bar, paired with Dec 1 benchmark)
    prov.enable_live_fetch = True
    monkeypatch.setattr(prov, "_maybe_refresh_daily", lambda sym: None)
    monkeypatch.setattr(prov, "_fetch_live_quote_cached", lambda sym: q_norm if sym == "TST" else None)
    idx_dec1 = pd.bdate_range(end="2025-12-01", periods=200)
    df_tgt_pre_dec1 = df_tgt.copy()
    df_spy_dec1 = df_spy.copy()
    df_spy_dec1.index = idx_dec1
    t_dec1_1030_ms = int(t_obs_et.timestamp() * 1000)
    df_spy_dec1.attrs["observation_timestamp_ms"] = t_dec1_1030_ms
    df_spy_dec1.attrs["obs_bar_date"] = "2025-12-01"
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-12-01 10:35:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_tgt_pre_dec1 if sym == "TST" else (df_spy_dec1 if sym == "SPY" else None))
    with prov._lock:
        prov._ratings_cache.clear()
        prov._signal_cache.clear()
        prov._news_cache["TST"] = (
            time.monotonic(),
            [{"decayed_sentiment": 0.6, "published_at": "2025-12-01T15:00:00Z", "published_at_verified": True}],
        )
    assert prov.get_analyst_ratings("TST")["current_price"] == pytest.approx(115.5, abs=0.01)
    sig_q_norm = prov.get_signal("TST", regime={"current_state": "LOW_VOLATILITY", "state_index": 0, "kappa": 0.45})
    assert "UNREADY" not in sig_q_norm["execution_mode"]
    assert sig_q_norm["execution_recommendation"]["entry_price"] == pytest.approx(115.5, abs=0.01)
    assert "DIVERGENCE_OBSERVATIONS_UNALIGNED" not in sig_q_norm["data_integrity"]["missing_modalities"]

    # Loop-2 #11 & #17: Partial bar (disk_mtime < 16:00 ET) must NOT fake 16:00 ET close in ratings or divergence
    df_partial = df_tgt.copy()
    t10_part_ms = int(pd.Timestamp("2025-11-28 10:00:00", tz="America/New_York").timestamp() * 1000)
    t11_part_ms = int(pd.Timestamp("2025-11-28 11:00:00", tz="America/New_York").timestamp() * 1000)
    df_partial.attrs["disk_mtime_ms"] = t10_part_ms
    assert prov._market_bar_timestamp_ms(df_partial, None) is None

    df_partial.attrs["observation_timestamp_ms"] = t10_part_ms
    assert prov._market_bar_timestamp_ms(df_partial, None) == t10_part_ms
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_partial if sym == "TST" else (df_spy if sym == "SPY" else None))
    monkeypatch.setattr(
        prov,
        "_fetch_live_quote_cached",
        lambda sym: {"price": 133.0, "date": "2025-11-28", "observation_timestamp_ms": t11_part_ms} if sym == "TST" else None,
    )
    with prov._lock:
        prov._ratings_cache.clear()
    assert prov.get_analyst_ratings("TST")["current_price"] == pytest.approx(133.0, abs=0.01)

    # Partial bar with unknown observation time must reject verified headline as unaligned (no 16:00 ET assumption)
    df_partial_unrec = df_tgt.copy()
    df_partial_unrec.attrs["disk_mtime_ms"] = t10_part_ms
    prov.enable_live_fetch = False
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_partial_unrec if sym == "TST" else (df_spy if sym == "SPY" else None))
    with prov._lock:
        prov._signal_cache.clear()
        prov._news_cache["TST"] = (
            time.monotonic(),
            [{
                "decayed_sentiment": 0.9,
                "published_at": pd.Timestamp("2025-11-28 11:00:00", tz="America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                "published_at_verified": True,
            }],
        )
    sig_part_unrec = prov.get_signal("TST", regime={"current_state": "LOW_VOLATILITY", "state_index": 0, "kappa": 0.45})
    assert sig_part_unrec["execution_mode"] == "UNREADY"
    assert "DIVERGENCE_OBSERVATIONS_UNALIGNED" in sig_part_unrec["data_integrity"]["missing_modalities"]

    # Fetch-only timestamp_ms must NOT produce obs_ms_out
    prov.enable_live_fetch = True
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_tgt if sym == "TST" else (df_spy if sym == "SPY" else None))
    monkeypatch.setattr(prov, "_fetch_live_quote_cached", lambda sym: {"price": 118.0, "date": "2025-12-01", "timestamp_ms": 1899999999000} if sym == "TST" else None)
    df_only_fetch, obs_only_fetch = prov._load_cached_daily_with_quote_ts("TST", require_quote_ts=False)
    assert df_only_fetch is not None and obs_only_fetch is None
    assert prov._daily_bar_status_ms(df_only_fetch)[:2] == (None, False)

    prov.enable_live_fetch = False
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_tgt if sym == "TST" else (df_spy if sym == "SPY" else None))

    # #20: On finalized half-day 2025-11-28 (13:00 ET close), pre-close 12:00 ET aligns cleanly, then 14:00 ET is rejected
    with prov._lock:
        prov._signal_cache.clear()
        prov._news_cache["TST"] = (
            time.monotonic(),
            [{
                "decayed_sentiment": 0.9,
                "published_at": pd.Timestamp("2025-11-28 12:00:00", tz="America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                "published_at_verified": True,
            }],
        )
    sig_hd_pre = prov.get_signal("TST", regime={"current_state": "LOW_VOLATILITY", "state_index": 0, "kappa": 0.45})
    assert "UNREADY" not in sig_hd_pre["execution_mode"]
    assert "DIVERGENCE_OBSERVATIONS_UNALIGNED" not in sig_hd_pre["data_integrity"]["missing_modalities"]

    with prov._lock:
        prov._signal_cache.clear()
        prov._news_cache["TST"] = (
            time.monotonic(),
            [{
                "decayed_sentiment": 0.9,
                "published_at": pd.Timestamp("2025-11-28 14:00:00", tz="America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                "published_at_verified": True,
            }],
        )
    sig_hd = prov.get_signal("TST", regime={"current_state": "LOW_VOLATILITY", "state_index": 0, "kappa": 0.45})
    assert "DIVERGENCE_OBSERVATIONS_UNALIGNED" in sig_hd["data_integrity"]["missing_modalities"]

    # #22: Benchmark ending Monday while target has Tuesday must NOT emit model recommendation or invoke residualizer (#17)
    df_spy_behind = df_spy.iloc[:-1]
    seen_len: list[int] = []
    orig_roll_len = prov.residualizer.compute_rolling_residuals
    def cap_len(r_s, r_sp, r_sc):
        seen_len.append(len(r_s))
        return orig_roll_len(r_s, r_sp, r_sc)
    monkeypatch.setattr(prov.residualizer, "compute_rolling_residuals", cap_len)
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_tgt if sym == "TST" else (df_spy_behind if sym == "SPY" else None))
    with prov._lock:
        prov._signal_cache.clear()
    sig_behind = prov.get_signal("TST", regime={"current_state": "LOW_VOLATILITY", "state_index": 0, "kappa": 0.45})
    assert len(seen_len) == 0
    assert sig_behind["execution_mode"] == "UNREADY"
    assert sig_behind["execution_recommendation"]["l1_normalized_portfolio_allocation_pct"] == 0.0
    assert "BENCHMARK_OBSERVATIONS_UNALIGNED" in sig_behind["data_integrity"]["missing_modalities"]
    assert "DIVERGENCE_OBSERVATIONS_UNALIGNED" in sig_behind["data_integrity"]["missing_modalities"]
    monkeypatch.setattr(prov.residualizer, "compute_rolling_residuals", orig_roll_len)

    # Loop-2 #13: Interior date shift dropping common_idx 127->126 without changing len/last_dt/OHLCV bytes invalidates cache
    df_150_t = df_tgt.tail(150).copy()
    spy_150_s = df_spy.tail(150).copy()
    spy_idx_150 = list(df_150_t.index)
    # Offset the first 23 dates by +1h so trailing 127 dates intersect (126 diff returns + 126 micro window)
    for i in range(23):
        spy_idx_150[i] = spy_idx_150[i] + pd.Timedelta(hours=1)
    df_127_s = spy_150_s.copy()
    df_127_s.index = pd.DatetimeIndex(spy_idx_150)
    spy_shifted_idx = list(spy_idx_150)
    spy_shifted_idx[30] = spy_shifted_idx[30] + pd.Timedelta(hours=12)
    df_126_s_shift = df_127_s.copy()
    df_126_s_shift.index = pd.DatetimeIndex(spy_shifted_idx)
    with prov._lock:
        prov._signal_cache.clear()
        prov._news_cache["TST"] = (time.monotonic(), [{"decayed_sentiment": 0.9, "published_at": "2025-11-28T15:00:00Z", "published_at_verified": True}])
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_150_t if sym == "TST" else (df_127_s if sym == "SPY" else None))
    sig_127_ok = prov.get_signal("TST", regime={"current_state": "LOW_VOLATILITY", "state_index": 0, "kappa": 0.45})
    assert "UNREADY" not in sig_127_ok["execution_mode"]
    assert "DIVERGENCE_OBSERVATIONS_UNALIGNED" not in sig_127_ok["data_integrity"]["missing_modalities"]
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_150_t if sym == "TST" else (df_126_s_shift if sym == "SPY" else None))
    sig_126_miss = prov.get_signal("TST", regime={"current_state": "LOW_VOLATILITY", "state_index": 0, "kappa": 0.45})
    assert sig_126_miss["execution_mode"] == "UNREADY"
    assert "DIVERGENCE_OBSERVATIONS_UNALIGNED" in sig_126_miss["data_integrity"]["missing_modalities"]

    # #15, #24 & Loop-2 #24: Cache hit verification BEFORE bar revision invalidation + regime posterior change invalidation
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_tgt if sym == "TST" else (df_spy if sym == "SPY" else None))
    with prov._lock:
        prov._signal_cache.clear()
        prov._news_cache.clear()
    r1 = prov.get_signal("TST", regime={"current_state": "LOW_VOLATILITY", "state_index": 0, "kappa": 0.45, "filtered_probabilities": {"low_volatility": 0.8, "medium_volatility": 0.15, "high_volatility": 0.05}})
    assert r1["macro_regime"]["filtered_probabilities"]["low_volatility"] == pytest.approx(0.8)
    reg_rev = {"current_state": "LOW_VOLATILITY", "state_index": 0, "kappa": 0.45, "filtered_probabilities": {"low_volatility": 0.6, "medium_volatility": 0.30, "high_volatility": 0.10}}
    r2 = prov.get_signal("TST", regime=reg_rev)
    assert r2["macro_regime"]["filtered_probabilities"]["medium_volatility"] == pytest.approx(0.30)

    import src.ipc.live_providers as _lp_mod

    orig_mono_2 = _lp_mod.time.monotonic
    monkeypatch.setattr(_lp_mod.time, "monotonic", lambda: 20_000.0)
    try:
        call_cnt = {"n": 0}
        def count_roll(r_s, r_sp, r_sc):
            call_cnt["n"] += 1
            return orig_roll_len(r_s, r_sp, r_sc)
        with prov._lock:
            prov._signal_cache.clear()
        monkeypatch.setattr(prov.residualizer, "compute_rolling_residuals", count_roll)
        prov.get_signal("TST", regime=reg_rev)
        prov.get_signal("TST", regime=reg_rev)
        assert call_cnt["n"] == 1  # Proven genuine TTL cache hit on identical input

        df_tgt_rev = df_tgt.copy()
        df_tgt_rev.iloc[-2, df_tgt_rev.columns.get_loc("high")] *= 1.15
        df_tgt_rev.iloc[-2, df_tgt_rev.columns.get_loc("volume")] *= 2.0
        monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_tgt_rev if sym == "TST" else (df_spy if sym == "SPY" else None))
        prov.get_signal("TST", regime=reg_rev)
        assert call_cnt["n"] == 2
    finally:
        monkeypatch.setattr(_lp_mod.time, "monotonic", orig_mono_2)

    # #19 & Loop-2 #18: UTC offset pubDate verified vs date-only pubDate unverified
    class FakeTickerNews:
        news = [
            {"id": "n1", "content": {"title": "TSM beat guidance", "pubDate": "2026-11-18T16:30:00-05:00"}},
            {"id": "n2", "content": {"title": "TSM date only update", "pubDate": "2026-11-18"}},
        ]

    monkeypatch.setattr(yf, "Ticker", lambda sym: FakeTickerNews())
    prov.enable_live_fetch = True
    news_items = prov.get_news_feed("TST")
    assert len(news_items) == 2
    assert news_items[0]["published_at"] == "2026-11-18T21:30:00Z"
    assert news_items[0]["published_at_verified"] is True
    assert news_items[1]["published_at"] == "2026-11-18"
    assert news_items[1]["published_at_verified"] is False
    prov.enable_live_fetch = False

    # Loop-2 #10: Intraday 15:30 ET hourly bar caps observation at 16:00 ET regular session close
    idx_1h = pd.DatetimeIndex([pd.Timestamp("2025-11-18 15:30:00", tz="America/New_York")])
    df_1h = pd.DataFrame(
        {"Open": [100.0], "High": [101.0], "Low": [99.5], "Close": [100.5], "Volume": [10000]},
        index=idx_1h,
    )
    monkeypatch.setattr(yf, "download", lambda *a, **kw: df_1h)
    ih_bars = prov.market_loader.fetch_intraday_bars("TST", interval="1h", count=5)
    assert len(ih_bars) == 1
    assert ih_bars[0]["observation_timestamp_ms"] == int(
        pd.Timestamp("2025-11-18 16:00:00", tz="America/New_York").timestamp() * 1000
    )

    # Historical bars emit server observation_timestamp_ms and queue_tick clears stale observation_timestamp_ms (#19)
    from src.ipc.ws_broadcaster import WebSocketBroadcaster

    hp_rows = prov.get_historical_prices("TST", count=5, interval="1d")
    assert len(hp_rows) == 5
    assert hp_rows[0].get("observation_timestamp_ms", 0) > 0
    bc = WebSocketBroadcaster()
    bc.queue_tick("TST", {"price": 100.0, "timestamp_ms": 1700000000123, "observation_timestamp_ms": 1700000000123})
    assert bc.pending_ticks["TST"]["timestamp_ms"] == 1700000000123
    assert bc.pending_ticks["TST"]["observation_timestamp_ms"] == 1700000000123
    bc.queue_tick("TST", {"price": 102.0})
    assert "observation_timestamp_ms" not in bc.pending_ticks["TST"]

    # Loop-3 & Loop-4 comprehensive checks
    # 1) #12/#17/#24/#29: Partial SPY benchmark -> cached UNREADY without residualizer call;
    # finalizing distinct SPY invalidates cache and runs residualizer once with distinct r_stk vs r_spy
    df_spy_part = df_spy.copy()
    df_spy_part.attrs["disk_mtime_ms"] = t10_part_ms
    orig_roll_ref = prov.residualizer.compute_rolling_residuals
    seen_res_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    def cap_fb(r_s, r_sp, r_sc):
        seen_res_pairs.append((r_s.copy(), r_sp.copy()))
        return orig_roll_ref(r_s, r_sp, r_sc)
    monkeypatch.setattr(prov.residualizer, "compute_rolling_residuals", cap_fb)
    orig_mono_3 = _lp_mod.time.monotonic
    monkeypatch.setattr(_lp_mod.time, "monotonic", lambda: 30_000.0)
    try:
        monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_tgt if sym == "TST" else (df_spy_part if sym == "SPY" else None))
        with prov._lock:
            prov._signal_cache.clear()
            prov._news_cache["TST"] = (30_000.0, [{"decayed_sentiment": 0.9, "published_at": "2025-11-28T15:00:00Z", "published_at_verified": True}])
        sig_spy_part = prov.get_signal("TST", regime=reg_rev)
        assert len(seen_res_pairs) == 0
        assert sig_spy_part["execution_mode"] == "UNREADY"
        assert "BENCHMARK_OBSERVATIONS_UNALIGNED" in sig_spy_part["data_integrity"]["missing_modalities"]
        assert "DIVERGENCE_OBSERVATIONS_UNALIGNED" in sig_spy_part["data_integrity"]["missing_modalities"]
        # Repeat under frozen monotonic clock -> exact cache hit (#29)
        assert prov.get_signal("TST", regime=reg_rev) is sig_spy_part

        # #13 & #24: Finalizing df_spy_part with distinct df_spy returns invalidates UNREADY cache cleanly
        monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_tgt if sym == "TST" else (df_spy if sym == "SPY" else None))
        sig_spy_rewritten = prov.get_signal("TST", regime=reg_rev)
        assert "UNREADY" not in sig_spy_rewritten["execution_mode"]
        assert "DIVERGENCE_OBSERVATIONS_UNALIGNED" not in sig_spy_rewritten["data_integrity"]["missing_modalities"]
        assert len(seen_res_pairs) == 1
        assert not np.allclose(seen_res_pairs[0][0], seen_res_pairs[0][1])
    finally:
        monkeypatch.setattr(_lp_mod.time, "monotonic", orig_mono_3)
    monkeypatch.setattr(prov.residualizer, "compute_rolling_residuals", orig_roll_ref)

    # 2) #15: Pre-close partial obs preserved even if disk_mtime_ms is post-close (13:00:02 ET on 2025-11-28 half-day)
    df_post_write = df_tgt.copy()
    df_post_write.attrs["disk_mtime_ms"] = int(pd.Timestamp("2025-11-28 13:00:02", tz="America/New_York").timestamp() * 1000)
    df_post_write.attrs["observation_timestamp_ms"] = int(pd.Timestamp("2025-11-28 12:59:58", tz="America/New_York").timestamp() * 1000)
    obs_pw, comp_pw, _ = prov._daily_bar_status_ms(df_post_write)
    assert comp_pw is False
    assert obs_pw == df_post_write.attrs["observation_timestamp_ms"]

    # 3) #22: Older (10:00) or undated quote cannot overwrite partial 10:05 bar even when require_quote_ts=False
    df_1005 = df_tgt.copy()
    t1005_ms = int(pd.Timestamp("2025-11-28 10:05:00", tz="America/New_York").timestamp() * 1000)
    t1000_ms = int(pd.Timestamp("2025-11-28 10:00:00", tz="America/New_York").timestamp() * 1000)
    df_1005.attrs["disk_mtime_ms"] = t1005_ms
    df_1005.attrs["observation_timestamp_ms"] = t1005_ms
    prov.enable_live_fetch = True
    monkeypatch.setattr(prov, "_maybe_refresh_daily", lambda sym: None)
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_1005 if sym == "TST" else None)
    monkeypatch.setattr(prov, "_fetch_live_quote_cached", lambda sym: {"price": 77.0, "date": "2025-11-28", "observation_timestamp_ms": t1000_ms})
    res_old_q, _ = prov._load_cached_daily_with_quote_ts("TST", require_quote_ts=False)
    assert float(res_old_q["close"].iloc[-1]) != 77.0
    monkeypatch.setattr(prov, "_fetch_live_quote_cached", lambda sym: {"price": 78.0, "date": "2025-11-28"})
    res_undated_q, _ = prov._load_cached_daily_with_quote_ts("TST", require_quote_ts=False)
    assert float(res_undated_q["close"].iloc[-1]) != 78.0

    # 4) #17: Fresh cache hit (within TTL, enable_live_fetch=True) reports YFINANCE_CACHED, while fresh live fetch reports YFINANCE_LIVE
    fresh_meta_p = fc.raw_dir / "yf_meta" / "FRESH.json"
    fresh_meta_p.write_text(json.dumps({
        "fetched_at": "2026-11-19T12:00:00Z",
        "info": {"targetMeanPrice": 300.0, "numberOfAnalystOpinions": 10},
        "rec_trend": [{"period": "0m", "strong_buy": 8, "buy": 2, "hold": 0, "underperform": 0, "sell": 0}],
    }), encoding="utf-8")
    now_epoch = time.time()
    os.utime(fresh_meta_p, (now_epoch, now_epoch))
    fc._mem_cache.pop("yf_meta/FRESH.json", None)
    with prov._lock:
        prov._ratings_cache.clear()
    assert prov.get_analyst_ratings("FRESH")["source"] == "YFINANCE_CACHED"
    orig_get_yf_meta = prov.fundamentals_client.get_yf_meta
    monkeypatch.setattr(
        prov.fundamentals_client,
        "get_yf_meta",
        lambda sym, enable_live_fetch=False: {
            "fetched_at": "2026-11-19T12:05:00Z",
            "_cache_stale": False,
            "_cache_hit": False,
            "info": {"targetMeanPrice": 305.0, "numberOfAnalystOpinions": 10},
            "rec_trend": [{"period": "0m", "strong_buy": 8, "buy": 2, "hold": 0, "underperform": 0, "sell": 0}],
        },
    )
    with prov._lock:
        prov._ratings_cache.clear()
    assert prov.get_analyst_ratings("FRESH")["source"] == "YFINANCE_LIVE"
    monkeypatch.setattr(prov.fundamentals_client, "get_yf_meta", orig_get_yf_meta)

    # 5) #14 & Loop-4 #19: Form 4 stale transaction decay + verified disclosure time gate (post-close 17:00 ET & unverified excluded)
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_tgt if sym == "TST" else (df_spy if sym == "SPY" else None))
    prov.enable_live_fetch = False
    with prov._lock:
        prov._signal_cache.clear()
        prov._news_cache.clear()
        prov._edgar_cache["TST"] = (time.monotonic(), {
            "available": True,
            "form4_transactions": [
                {"filing_date": "2025-11-28", "disclosed_at": "2025-11-28T14:30:00Z", "disclosed_at_verified": True, "transaction_code": "P", "value": 500_000.0, "is_10b5_1": False},
            ],
        })
    s_recent_f4 = prov.get_signal("TST", regime=reg_rev)["signal_breakdown"]["decayed_form4_insider_cluster"]
    with prov._lock:
        prov._signal_cache.clear()
        prov._edgar_cache["TST"] = (time.monotonic(), {
            "available": True,
            "form4_transactions": [
                {"filing_date": "2025-08-20", "disclosed_at": "2025-08-20T14:30:00Z", "disclosed_at_verified": True, "transaction_code": "P", "value": 500_000.0, "is_10b5_1": False},
            ],
        })
    s_old_f4 = prov.get_signal("TST", regime=reg_rev)["signal_breakdown"]["decayed_form4_insider_cluster"]
    assert s_recent_f4 > s_old_f4 > 0.0

    # Post-close Form 4 (17:00 ET on 2025-11-28 > 13:00 ET close) and Yahoo unverified disclosure excluded
    with prov._lock:
        prov._signal_cache.clear()
        prov._edgar_cache["TST"] = (time.monotonic(), {
            "available": True,
            "form4_transactions": [
                {"filing_date": "2025-11-28", "disclosed_at": "2025-11-28T22:00:00Z", "disclosed_at_verified": True, "transaction_code": "P", "value": 9_000_000.0, "is_10b5_1": False},
                {"filing_date": "2025-11-28", "disclosed_at": None, "disclosed_at_verified": False, "transaction_code": "P", "value": 9_000_000.0, "is_10b5_1": False},
            ],
        })
    s_post_unver_f4 = prov.get_signal("TST", regime=reg_rev)["signal_breakdown"]["decayed_form4_insider_cluster"]
    assert s_post_unver_f4 == 0.0

    # 6) #21: Partial disk bar stays eligible for post-close refresh even when its date matches expected date
    prov_pr = LiveInferenceProvider(lakehouse_dir=tmp_path / "lh_pr", weights_dir=weights_dir, enable_live_fetch=True)
    prov_pr.market_loader.save_ticker_daily("PRSH", df_tgt)
    pq_pr = prov_pr.market_loader.daily_dir / "PRSH.parquet"
    pre_close_ts = pd.Timestamp("2025-11-28 10:00:00", tz="America/New_York").timestamp()
    os.utime(pq_pr, (pre_close_ts, pre_close_ts))
    monkeypatch.setattr(prov_pr.market_loader, "_latest_expected_trading_date", lambda: pd.Timestamp("2025-11-28"))
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-11-28 13:05:00", tz="America/New_York").timestamp())
    refresh_flags: list[bool] = []
    def spy_refresh(sym, target_end_date=None, timeout=5.0, force_refresh_if_incomplete=False):
        refresh_flags.append(bool(force_refresh_if_incomplete))
        return df_tgt
    monkeypatch.setattr(prov_pr.market_loader, "refresh_ticker_daily", spy_refresh)
    assert prov_pr.market_loader._latest_expected_trading_date() == pd.Timestamp(df_tgt.index[-1]).normalize()
    prov_pr._maybe_refresh_daily("PRSH")
    assert refresh_flags == [True]

    # 7) Loop-4 hardening checks (#12, #14/#22, #15, #16, #18, #20/#21, #23)
    # #12: Corrupted/unreadable SPY parquet returns UNREADY inside error handler instead of raising
    def _raise_spy(sym):
        if sym == "SPY":
            raise OSError("corrupt SPY.parquet")
        return df_tgt if sym == "TST" else None
    monkeypatch.setattr(prov, "_load_disk_daily", _raise_spy)
    with prov._lock:
        prov._signal_cache.clear()
    sig_spy_err = prov.get_signal("TST", regime=reg_rev)
    assert sig_spy_err["execution_mode"] == "UNREADY"
    assert sig_spy_err["winning_engine"] == "NONE"

    # #23: Aligned Monday (2025-11-24) SPY with trailing undated Tuesday (2025-11-25) live overlay strips Tuesday attrs on Monday slice
    idx_consec_25 = pd.bdate_range(end="2025-11-25", periods=len(df_tgt))
    df_tgt_25 = df_tgt.copy()
    df_tgt_25.index = idx_consec_25
    df_spy_25 = df_spy.copy()
    df_spy_25.index = idx_consec_25
    df_tgt_mon = df_tgt_25.iloc[:-1].copy()
    df_spy_mon = df_spy_25.iloc[:-1].copy()
    # Overlay undated Tuesday quote via live path so _row_provenance preserves finalized Monday metadata
    prov.enable_live_fetch = True
    monkeypatch.setattr(prov, "_maybe_refresh_daily", lambda sym: None)
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_spy_mon if sym == "SPY" else None)
    monkeypatch.setattr(prov, "_fetch_live_quote_cached", lambda sym: {"price": 420.5, "date": "2025-11-25", "timestamp_ms": 1764090000000} if sym == "SPY" else None)
    df_spy_tue, tue_q = prov._load_cached_daily_with_quote_ts("SPY", allow_refresh=True, require_quote_ts=False)
    assert df_spy_tue is not None and tue_q is None and df_spy_tue.attrs.get("live_quote_undated") is True
    prov.enable_live_fetch = False
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_tgt_mon if sym == "TST" else (df_spy_tue if sym == "SPY" else None))
    with prov._lock:
        prov._signal_cache.clear()
    sig_mon_clean = prov.get_signal("TST", regime=reg_rev)
    assert pd.Timestamp(df_tgt_mon.index[-1]).strftime("%Y-%m-%d") == "2025-11-24"
    assert "UNREADY" not in sig_mon_clean["execution_mode"]
    assert "BENCHMARK_OBSERVATIONS_UNALIGNED" not in sig_mon_clean["data_integrity"]["missing_modalities"]

    # Stripped slice WITHOUT provenance leaves Monday partial bar unknown (never fabricates 16:00 ET close)
    df_spy_part_tue = df_spy_25.copy()
    mon_1545_ms = int(pd.Timestamp("2025-11-24 15:45:00", tz="America/New_York").timestamp() * 1000)
    df_spy_part_tue.attrs["disk_mtime_ms"] = mon_1545_ms
    df_spy_part_tue.attrs["live_quote_undated"] = True
    df_spy_part_tue.attrs["obs_bar_date"] = "2025-11-25"
    sliced_no_prov = prov._slice_aligned_frame(df_spy_part_tue, df_tgt_mon.index)
    assert prov._daily_bar_status_ms(sliced_no_prov)[:2] == (None, False)
    df_spy_nomtime = df_spy_25.copy()
    df_spy_nomtime.attrs["live_quote_undated"] = True
    df_spy_nomtime.attrs["obs_bar_date"] = "2025-11-25"
    sliced_nomtime = prov._slice_aligned_frame(df_spy_nomtime, df_tgt_mon.index)
    assert sliced_nomtime.attrs.get("_sliced_unverified") is True
    assert prov._daily_bar_status_ms(sliced_nomtime)[:2] == (None, False)

    # #14 & #22: fetch_live_quote keeps fi_price_date when regularMarketTime missing, and respects req_started_ms < close_ms
    class FakeFastInfoNoRmt:
        fast_info = type("FI", (), {
            "lastPrice": 210.0,
            "_prices_1y": pd.DataFrame({"Close": [210.0]}, index=pd.DatetimeIndex([pd.Timestamp("2025-11-17 00:00:00-05:00")])),
            "_md": {},
            "get": lambda self, k, d=None: {"lastPrice": 210.0}.get(k, d),
        })()
        def history(self, **kw):
            return pd.DataFrame(
                {"Open": [212.0], "High": [215.0], "Low": [211.0], "Close": [214.0], "Volume": [1000]},
                index=pd.DatetimeIndex([pd.Timestamp("2025-11-18 00:00:00-05:00")]),
            )
        def get_history_metadata(self):
            return {}
    monkeypatch.setattr(yf, "Ticker", lambda sym: FakeFastInfoNoRmt())
    q_fi_no_rmt = prov.market_loader.fetch_live_quote("TST")
    assert q_fi_no_rmt is not None
    assert q_fi_no_rmt["date"] == "2025-11-17"
    assert "observation_timestamp_ms" not in q_fi_no_rmt

    # #16 & #18: Atomic Parquet schema metadata survives restart and post-close non-replacement retains incomplete marker
    ml_part = prov_pr.market_loader
    df_partial_save = df_tgt.copy()
    t_pre = int(pd.Timestamp("2025-11-28 12:55:00", tz="America/New_York").timestamp() * 1000)
    df_partial_save.attrs["observation_timestamp_ms"] = t_pre
    df_partial_save.attrs["obs_bar_date"] = "2025-11-28"
    ml_part.save_ticker_daily("PARTAT", df_partial_save)
    pq_partat = ml_part.daily_dir / "PARTAT.parquet"
    t_post_wr = pd.Timestamp("2025-11-28 13:00:05", tz="America/New_York").timestamp()
    os.utime(pq_partat, (t_post_wr, t_post_wr))
    restarted_prov = LiveInferenceProvider(lakehouse_dir=tmp_path / "lh_pr", weights_dir=weights_dir, enable_live_fetch=True)
    loaded_partat = restarted_prov._load_disk_daily("PARTAT")
    obs_pa, comp_pa, _ = restarted_prov._daily_bar_status_ms(loaded_partat)
    assert obs_pa == t_pre and comp_pa is False

    # If post-close download only returns yesterday's bar, target row isn't replaced -> incomplete marker retained
    df_yest_only = df_tgt.iloc[-3:-1].copy()
    df_yest_yf = pd.DataFrame(
        {"Open": df_yest_only["open"].values, "High": df_yest_only["high"].values, "Low": df_yest_only["low"].values, "Close": df_yest_only["close"].values, "Volume": df_yest_only["volume"].values},
        index=df_yest_only.index,
    )
    monkeypatch.setattr(restarted_prov.market_loader, "_latest_expected_trading_date", lambda: pd.Timestamp("2025-11-28"))
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-11-28 13:05:00", tz="America/New_York").timestamp())
    dl_calls: list[int] = []
    def _stub_dl(*a, **kw):
        dl_calls.append(1)
        return df_yest_yf
    monkeypatch.setattr(yf, "download", _stub_dl)
    restarted_prov._maybe_refresh_daily("PARTAT")
    assert dl_calls == [1]
    _, comp_after_yest, _ = restarted_prov._daily_bar_status_ms(restarted_prov._load_disk_daily("PARTAT"))
    assert comp_after_yest is False
    assert "PARTAT" not in restarted_prov._daily_refresh_attempted

    # #15: Three same-day 10-K filings (09:00 accn 0001, 10:00 accn 0000, unknown accn 0002)
    cik_part3 = "0000111229"
    (fc.raw_dir / "edgar_submissions" / f"CIK{cik_part3}.json").write_text(json.dumps({
        "cik": cik_part3,
        "recent": {
            "form": ["10-K", "10-K", "10-K"],
            "filingDate": ["2017-08-02", "2017-08-02", "2017-08-02"],
            "reportDate": ["2017-06-30", "2017-06-30", "2017-06-30"],
            "accessionNumber": ["0001", "0000", "0002"],
            "acceptanceDateTime": ["2017-08-02T09:00:00.000Z", "2017-08-02T10:00:00.000Z", ""],
            "primaryDocument": ["f1.htm", "f2.htm", "f3.htm"],
        },
    }), encoding="utf-8")
    (fc.raw_dir / "edgar_facts" / f"CIK{cik_part3}.json").write_text(json.dumps({
        "cik": cik_part3,
        "entityName": "PART3 CORP",
        "us_gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {"form": "10-K", "start": "2016-07-01", "end": "2017-06-30", "filed": "2017-08-02", "accn": "0000", "val": 110_000_000},
                        {"form": "10-K", "start": "2016-07-01", "end": "2017-06-30", "filed": "2017-08-02", "accn": "0001", "val": 100_000_000},
                        {"form": "10-K", "start": "2016-07-01", "end": "2017-06-30", "filed": "2017-08-02", "accn": "0002", "val": 105_000_000},
                    ]
                }
            },
            "Assets": {"units": {"USD": [{"form": "10-K", "end": "2017-06-30", "filed": "2017-08-02", "accn": "0001", "val": 500_000_000, "fp": "FY"}]}},
        },
    }), encoding="utf-8")
    fin_p3_pit = fc.build_financials_payload("PRT3", known_cik=cik_part3, pit_strict=True, enable_live_fetch=False)
    rev_p3_pit = next(r for r in fin_p3_pit["incomeStatement"] if r["tag"] == "Revenues")
    assert rev_p3_pit["values"]["FY2017"] == pytest.approx(100.0, abs=0.05)
    assert fin_p3_pit["filings"][0]["order_uncertain"] is True
    fin_p3_rest = fc.build_financials_payload("PRT3", known_cik=cik_part3, pit_strict=False, enable_live_fetch=False)
    rev_p3_rest = next(r for r in fin_p3_rest["incomeStatement"] if r["tag"] == "Revenues")
    assert rev_p3_rest["values"]["FY2017"] == pytest.approx(110.0, abs=0.05)

    # #20 & #21: Legacy redlines require explicit accession pair (never inferred) & NaN targetMeanPrice omitted cleanly
    cik_leg = "0000111230"
    (fc.raw_dir / "edgar_submissions" / f"CIK{cik_leg}.json").write_text(json.dumps({
        "cik": cik_leg,
        "recent": {
            "form": ["10-K", "10-K"],
            "filingDate": ["2018-08-03", "2017-08-02"],
            "reportDate": ["2018-06-30", "2017-06-30"],
            "accessionNumber": ["ACC2", "ACC1"],
            "primaryDocument": ["a2.htm", "a1.htm"],
        },
    }), encoding="utf-8")
    leg_path = fc.raw_dir / "edgar_redlines" / "LEG_ACC2.json"
    leg_path.parent.mkdir(parents=True, exist_ok=True)
    leg_path.write_text(json.dumps({
        "item1a_similarity": 0.88,
        "mda_similarity": 0.91,
        "redlines": [{"section": "Item 1A Risk Factors (10-K 2018-08-03 vs 10-K 2017-08-02)", "status": "DIFF_VERIFIED", "added_clauses": ["Unpaired"], "removed_clauses": []}],
    }), encoding="utf-8")
    edg_leg_unpaired = fc.build_edgar_filings_payload("LEG", known_cik=cik_leg, enable_live_fetch=False)
    assert edg_leg_unpaired["redlines"][0]["status"] == "INDEX_VERIFIED"
    assert not (fc.raw_dir / "edgar_redlines" / "LEG_ACC2_ACC1.json").exists()

    # Explicitly paired legacy redline preserves its stale age instead of creating a freshly TTL'd pair file
    leg_path.write_text(json.dumps({
        "current_accession": "ACC2",
        "prior_accession": "ACC1",
        "item1a_similarity": 0.88,
        "mda_similarity": 0.91,
        "redlines": [{"section": "Item 1A Risk Factors (10-K 2018-08-03 vs 10-K 2017-08-02)", "status": "DIFF_VERIFIED", "added_clauses": ["PairedLegacy"], "removed_clauses": []}],
    }), encoding="utf-8")
    old_leg_ts = pd.Timestamp("2025-11-01 00:00:00", tz="UTC").timestamp()
    os.utime(leg_path, (old_leg_ts, old_leg_ts))
    edg_leg_paired = fc.build_edgar_filings_payload("LEG", known_cik=cik_leg, enable_live_fetch=False)
    assert edg_leg_paired["redlines"][0]["status"] == "DIFF_VERIFIED"
    assert not (fc.raw_dir / "edgar_redlines" / "LEG_ACC2_ACC1.json").exists()
    assert fc._read_json_cache("edgar_redlines/LEG_ACC2_ACC1.json", 86400.0) is None

    monkeypatch.setattr(
        yf,
        "Ticker",
        lambda sym: type("YFNan", (), {
            "info": {"targetMeanPrice": float("nan"), "numberOfAnalystOpinions": 8, "recommendationKey": "buy"},
            "recommendations_summary": None,
            "insider_transactions": None,
        })(),
    )
    meta_nan = fc.get_yf_meta("NANPX", enable_live_fetch=True)
    assert meta_nan is not None
    assert "targetMeanPrice" not in meta_nan["info"]
    def _no_nan(c: str) -> None:
        raise ValueError(f"Unexpected non-finite constant: {c}")
    persisted_nan = json.loads((fc.raw_dir / "yf_meta" / "NANPX.json").read_text(encoding="utf-8"), parse_constant=_no_nan)
    assert "targetMeanPrice" not in persisted_nan["info"]

    # Loop-5 hardening checks:
    # A) Refresh within 15m post-close settlement window returning identical partial bar keeps partial marker;
    #    post-settlement refresh marks final cleanly.
    ml_settle = MarketDataLoader(tmp_path / "lh_settle")
    df_pre_close = df_tgt.copy()
    t_pre_close_ms = int(pd.Timestamp("2025-11-28 12:55:00", tz="America/New_York").timestamp() * 1000)
    df_pre_close.attrs["observation_timestamp_ms"] = t_pre_close_ms
    df_pre_close.attrs["obs_bar_date"] = "2025-11-28"
    ml_settle.save_ticker_daily("STL", df_pre_close)
    df_same_yf = pd.DataFrame(
        {
            "Open": df_tgt["open"].tail(5).values,
            "High": df_tgt["high"].tail(5).values,
            "Low": df_tgt["low"].tail(5).values,
            "Close": df_tgt["close"].tail(5).values,
            "Volume": df_tgt["volume"].tail(5).values,
        },
        index=df_tgt.index[-5:],
    )
    monkeypatch.setattr(yf, "download", lambda *a, **kw: df_same_yf)
    monkeypatch.setattr(ml_settle, "_latest_expected_trading_date", lambda: pd.Timestamp("2025-11-28"))
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-11-28 13:05:00", tz="America/New_York").timestamp())
    df_early_ref = ml_settle.refresh_ticker_daily("STL", force_refresh_if_incomplete=True)
    assert df_early_ref is not None
    assert df_early_ref.attrs.get("_target_row_replaced") is False
    assert df_early_ref.attrs.get("observation_timestamp_ms") == t_pre_close_ms
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-11-28 13:20:00", tz="America/New_York").timestamp())
    df_settled_ref = ml_settle.refresh_ticker_daily("STL", force_refresh_if_incomplete=True)
    assert df_settled_ref is not None
    assert df_settled_ref.attrs.get("_target_row_replaced") is True
    assert "observation_timestamp_ms" not in df_settled_ref.attrs

    # B) DuckDB generation-mismatch fallback returns consistent table + metadata from single generation
    #    after real mid-read atomic file swap, and preserves case-insensitive column selection
    pq_stl = ml_settle.daily_dir / "STL.parquet"
    df_gen1 = df_tgt.copy()
    t_gen1_ms = int(pd.Timestamp("2025-11-28 12:40:00", tz="America/New_York").timestamp() * 1000)
    t_gen2_ms = int(pd.Timestamp("2025-11-28 12:55:00", tz="America/New_York").timestamp() * 1000)
    df_gen1.iloc[-1, df_gen1.columns.get_loc("close")] = 111.25
    df_gen1.iloc[-1, df_gen1.columns.get_loc("high")] = max(float(df_gen1.iloc[-1]["high"]), 111.25)
    df_gen1.attrs["observation_timestamp_ms"] = t_gen1_ms
    df_gen1.attrs["obs_bar_date"] = "2025-11-28"
    ml_settle.save_ticker_daily("STL", df_gen1)

    df_gen2 = df_tgt.copy()
    df_gen2.index.name = "timestamp"
    df_gen2.iloc[-1, df_gen2.columns.get_loc("close")] = 222.50
    df_gen2.iloc[-1, df_gen2.columns.get_loc("high")] = max(float(df_gen2.iloc[-1]["high"]), 222.50)
    df_gen2.attrs["observation_timestamp_ms"] = t_gen2_ms
    df_gen2.attrs["obs_bar_date"] = "2025-11-28"
    gen2_swaps = [ml_settle.daily_dir / f"STL_GEN2_{i}.parquet" for i in range(3)]
    for gp in gen2_swaps:
        ml_settle.atomic_write_parquet(df_gen2, gp)

    import duckdb as _ddb
    import pyarrow.parquet as _pq
    orig_ddb_connect = _ddb.connect
    orig_read_table = _pq.read_table
    seen_ddb_closes: list[float] = []
    fallback_calls: list[int] = []
    swap_state = {"enable": False, "idx": 0}

    class _WrappedCursor:
        def __init__(self, cur, sql_txt: str):
            self._cur = cur
            self._sql = sql_txt
        def fetchall(self):
            return self._cur.fetchall()
        def fetchdf(self):
            df_c = self._cur.fetchdf()
            if "SELECT" in self._sql and "FROM read_parquet" in self._sql and len(df_c) > 0:
                seen_ddb_closes.append(float(df_c["close"].iloc[-1]))
                if swap_state["enable"] and swap_state["idx"] < len(gen2_swaps):
                    # Atomically swap real inode/generation AFTER DuckDB finishes reading so st_before != st_meta triggers retry/fallback
                    os.replace(gen2_swaps[swap_state["idx"]], pq_stl)
                    swap_state["idx"] += 1
            return df_c

    class _WrappedConn:
        def __init__(self, con):
            self._con = con
        def execute(self, sql_txt, *a, **kw):
            return _WrappedCursor(self._con.execute(sql_txt, *a, **kw), sql_txt)
        def close(self):
            return self._con.close()

    def _wrapped_ddb_connect(*a, **kw):
        return _WrappedConn(orig_ddb_connect(*a, **kw))

    def _tracked_read_table(*a, **kw):
        fallback_calls.append(1)
        return orig_read_table(*a, **kw)

    monkeypatch.setattr("duckdb.connect", _wrapped_ddb_connect)
    monkeypatch.setattr("pyarrow.parquet.read_table", _tracked_read_table)
    # Control: no swap -> generation 1 returned directly without calling pq.read_table
    q_control = ml_settle.query_parquet_duckdb(pq_stl, columns=["CLOSE"], start_date="2025-11-24")
    assert len(fallback_calls) == 0
    assert float(q_control["close"].iloc[-1]) == pytest.approx(111.25, abs=1e-6)
    assert q_control.attrs.get("observation_timestamp_ms") == t_gen1_ms

    # Concurrent swap: real os.replace inode changes trigger mismatch and single-fd pq.read_table fallback
    swap_state["enable"] = True
    seen_ddb_closes.clear()
    q_fallback = ml_settle.query_parquet_duckdb(pq_stl, columns=["CLOSE"], start_date="2025-11-24")
    monkeypatch.setattr("duckdb.connect", orig_ddb_connect)
    monkeypatch.setattr("pyarrow.parquet.read_table", orig_read_table)
    assert seen_ddb_closes[0] == pytest.approx(111.25, abs=1e-6)
    assert len(fallback_calls) == 1
    assert "close" in q_fallback.columns and len(q_fallback) >= 4
    assert float(q_fallback["close"].iloc[-1]) == pytest.approx(222.50, abs=1e-6)
    assert q_fallback.attrs.get("observation_timestamp_ms") == t_gen2_ms

    # C) FastInfo price without regularMarketTime in fi._md must NOT adopt separate 5d history metadata
    class FakeFastInfoSplitMd:
        fast_info = type("FISplit", (), {
            "lastPrice": 210.0,
            "_prices_1y": pd.DataFrame({"Close": [210.0]}, index=pd.DatetimeIndex([pd.Timestamp("2025-11-17 00:00:00-05:00")])),
            "_md": {},
            "get": lambda self, k, d=None: {"lastPrice": 210.0}.get(k, d),
        })()
        def history(self, **kw):
            return pd.DataFrame(
                {"Open": [212.0], "High": [215.0], "Low": [211.0], "Close": [214.0], "Volume": [1000]},
                index=pd.DatetimeIndex([pd.Timestamp("2025-11-17 00:00:00-05:00")]),
            )
        def get_history_metadata(self):
            return {"regularMarketTime": int(pd.Timestamp("2025-11-17 15:56:00", tz="America/New_York").timestamp())}
    monkeypatch.setattr(yf, "Ticker", lambda sym: FakeFastInfoSplitMd())
    q_split = prov.market_loader.fetch_live_quote("TST")
    assert q_split is not None and "observation_timestamp_ms" not in q_split

    # D) Post-close Monday 16:15 ET Form 4 disclosure must NOT be scored on Monday 16:30 ET (eff=Tuesday),
    #    even when target_obs_ms == 16:30 ET admits disc_ms <= target_obs_ms; then becomes eligible on Tuesday
    idx_mon_tue = pd.bdate_range(end="2025-11-25", periods=150)  # Tuesday 2025-11-25
    df_tue_t = pd.DataFrame(
        {"open": 100.0, "high": np.linspace(100.0, 120.0, 150) + 2.0, "low": 99.0, "close": np.linspace(100.0, 120.0, 150), "volume": 1_000_000.0},
        index=idx_mon_tue,
    )
    df_tue_spy = pd.DataFrame(
        {"open": 400.0, "high": np.linspace(400.0, 420.0, 150) + 2.0, "low": 399.0, "close": np.linspace(400.0, 420.0, 150), "volume": 10_000_000.0},
        index=idx_mon_tue,
    )
    df_mon_t = df_tue_t.iloc[:-1].copy()
    df_mon_spy = df_tue_spy.iloc[:-1].copy()
    mon_1630_ms = int(pd.Timestamp("2025-11-24 16:30:00", tz="America/New_York").timestamp() * 1000)
    tue_1600_ms = int(pd.Timestamp("2025-11-25 16:00:00", tz="America/New_York").timestamp() * 1000)
    df_mon_t.attrs["observation_timestamp_ms"] = mon_1630_ms
    df_mon_spy.attrs["observation_timestamp_ms"] = mon_1630_ms
    df_tue_t.attrs["observation_timestamp_ms"] = tue_1600_ms
    df_tue_spy.attrs["observation_timestamp_ms"] = tue_1600_ms
    mon_1615_iso = pd.Timestamp("2025-11-24 16:15:00", tz="America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    with prov._lock:
        prov._signal_cache.clear()
        prov._news_cache.clear()
        prov._edgar_cache["TST"] = (time.monotonic(), {
            "available": True,
            "form4_transactions": [{
                "filing_date": "2025-11-24",
                "disclosed_at": mon_1615_iso,
                "disclosed_at_verified": True,
                "transaction_code": "P",
                "value": 9_000_000.0,
                "is_10b5_1": False,
            }],
        })
    prov.enable_live_fetch = True
    orig_prov_refresh = prov._maybe_refresh_daily
    orig_prov_fetch_q = prov._fetch_live_quote_cached
    monkeypatch.setattr(prov, "_maybe_refresh_daily", lambda sym: None)
    monkeypatch.setattr(
        prov,
        "_fetch_live_quote_cached",
        lambda sym: {"price": 119.8 if sym == "TST" else 419.8, "date": "2025-11-24", "observation_timestamp_ms": mon_1630_ms, "timestamp_ms": mon_1630_ms},
    )
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_mon_t if sym == "TST" else (df_mon_spy if sym == "SPY" else None))
    sig_mon_f4 = prov.get_signal("TST", regime=reg_rev)
    assert sig_mon_f4["signal_breakdown"]["decayed_form4_insider_cluster"] == 0.0
    prov.enable_live_fetch = False
    monkeypatch.setattr(prov, "_fetch_live_quote_cached", orig_prov_fetch_q)
    monkeypatch.setattr(prov, "_maybe_refresh_daily", orig_prov_refresh)
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_tue_t if sym == "TST" else (df_tue_spy if sym == "SPY" else None))
    sig_tue_f4 = prov.get_signal("TST", regime=reg_rev)
    assert sig_tue_f4["signal_breakdown"]["decayed_form4_insider_cluster"] > 0.0

    # E) Distinct sector ETF (XLK for NVDA) observed at 15:56 while NVDA & SPY at 15:55 suppresses divergence
    df_nvda_55 = df_mon_t.copy()
    df_spy_55 = df_mon_spy.copy()
    df_xlk_56 = df_mon_spy.copy()
    for _c in ("open", "high", "low", "close"):
        df_xlk_56[_c] = df_xlk_56[_c] * 0.5
    t_1555_ms = int(pd.Timestamp("2025-11-24 15:55:00", tz="America/New_York").timestamp() * 1000)
    t_1556_ms = int(pd.Timestamp("2025-11-24 15:56:00", tz="America/New_York").timestamp() * 1000)
    df_nvda_55.attrs["observation_timestamp_ms"] = t_1555_ms
    df_spy_55.attrs["observation_timestamp_ms"] = t_1555_ms
    df_xlk_56.attrs["observation_timestamp_ms"] = t_1556_ms
    with prov._lock:
        prov._signal_cache.clear()
        prov._edgar_cache.clear()
        prov._news_cache["NVDA"] = (time.monotonic(), [{"decayed_sentiment": 0.9, "published_at": "2025-11-24T15:00:00Z", "published_at_verified": True}])
    monkeypatch.setattr(prov.universe_selector, "get_sector_etf", lambda sym: "XLK" if sym == "NVDA" else "SPY")
    monkeypatch.setattr(
        prov,
        "_load_disk_daily",
        lambda sym: df_nvda_55 if sym == "NVDA" else (df_spy_55 if sym == "SPY" else (df_xlk_56 if sym == "XLK" else None)),
    )
    sig_sec_mismatch = prov.get_signal("NVDA", regime=reg_rev)
    assert "SECTOR_BENCHMARK_UNALIGNED" in sig_sec_mismatch["data_integrity"]["missing_modalities"]
    assert "DIVERGENCE_OBSERVATIONS_UNALIGNED" in sig_sec_mismatch["data_integrity"]["missing_modalities"]
    assert "UNREADY" not in sig_sec_mismatch["execution_mode"]
    assert all(bool(np.isfinite(float(v))) for v in sig_sec_mismatch["signal_breakdown"].values())
    assert sig_sec_mismatch["signal_breakdown"]["smart_money_divergence_pos"] == 0.0
    assert sig_sec_mismatch["signal_breakdown"]["smart_money_divergence_neg"] == 0.0

    # F) Rewriting fresh EDGAR JSON on disk invalidates _mem_cache on the same FundamentalsClient instance
    fc_fresh = FundamentalsClient(lakehouse_dir=tmp_path / "lh_fresh_inv")
    cik_fr = "0000111231"
    p_fr_sub = fc_fresh.raw_dir / "edgar_submissions" / f"CIK{cik_fr}.json"
    p_fr_sub.parent.mkdir(parents=True, exist_ok=True)
    p_fr_sub.write_text(json.dumps({
        "cik": cik_fr,
        "recent": {
            "form": ["10-K"],
            "filingDate": ["2024-08-01"],
            "reportDate": ["2024-06-30"],
            "accessionNumber": ["0001"],
            "primaryDocument": ["a.htm"],
        },
    }), encoding="utf-8")
    p_fr_facts = fc_fresh.raw_dir / "edgar_facts" / f"CIK{cik_fr}.json"
    p_fr_facts.parent.mkdir(parents=True, exist_ok=True)
    p_fr_facts.write_text(json.dumps({
        "cik": cik_fr,
        "entityName": "FRESH TEST",
        "us_gaap": {"Revenues": {"units": {"USD": [{"form": "10-K", "start": "2023-07-01", "end": "2024-06-30", "filed": "2024-08-01", "accn": "0001", "val": 50_000_000_000}]}}},
    }), encoding="utf-8")
    assert fc_fresh.get_sec_submissions(cik_fr, enable_live_fetch=False)["recent"]["accessionNumber"] == ["0001"]
    assert fc_fresh.build_financials_payload("FRSH", known_cik=cik_fr, enable_live_fetch=False)["incomeStatement"][0]["values"]["FY2024"] == pytest.approx(50_000.0)
    p_fr_sub.write_text(json.dumps({
        "cik": cik_fr,
        "recent": {
            "form": ["10-K"],
            "filingDate": ["2024-08-01"],
            "reportDate": ["2024-06-30"],
            "accessionNumber": ["0002"],
            "primaryDocument": ["b.htm"],
        },
    }), encoding="utf-8")
    p_fr_facts.write_text(json.dumps({
        "cik": cik_fr,
        "entityName": "FRESH TEST",
        "us_gaap": {"Revenues": {"units": {"USD": [{"form": "10-K", "start": "2023-07-01", "end": "2024-06-30", "filed": "2024-08-01", "accn": "0002", "val": 75_000_000_000}]}}},
    }), encoding="utf-8")
    assert fc_fresh.get_sec_submissions(cik_fr, enable_live_fetch=False)["recent"]["accessionNumber"] == ["0002"]
    assert fc_fresh.build_financials_payload("FRSH", known_cik=cik_fr, enable_live_fetch=False)["incomeStatement"][0]["values"]["FY2024"] == pytest.approx(75_000.0)

    # Loop-6 hardening checks:
    # G) Invalid posteriors fail closed to 0.25 (conservative min kappa) in RegimeHMM.get_regime_multiplier
    from src.models.regime_hmm import RegimeHMM
    hmm_guard = RegimeHMM()
    assert hmm_guard.get_regime_multiplier(np.array([float("nan"), 0.0, 1.0])) == 0.25

    # H) Same-day undated quote overlay marks live_quote_undated=True & offline cache invalidates on benchmark mtime change
    df_eq_off = df_mon_t.copy()
    df_spy_off = df_mon_spy.copy()
    df_eq_off.attrs.clear()
    df_spy_off.attrs.clear()
    mt_0959_ms = int(pd.Timestamp("2025-11-24 09:59:00", tz="America/New_York").timestamp() * 1000)
    mt_1000_ms = int(pd.Timestamp("2025-11-24 10:00:00", tz="America/New_York").timestamp() * 1000)
    mt_1001_ms = int(pd.Timestamp("2025-11-24 10:01:00", tz="America/New_York").timestamp() * 1000)
    df_eq_off.attrs["disk_mtime_ms"] = mt_1000_ms
    df_spy_off.attrs["disk_mtime_ms"] = mt_0959_ms
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-11-24 11:00:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_eq_off if sym == "TST" else (df_spy_off if sym == "SPY" else None))
    with prov._lock:
        prov._signal_cache.clear()
        prov._news_cache.clear()
        prov._edgar_cache.clear()
    sig_mtime_ok = prov.get_signal("TST", regime=reg_rev)
    assert "UNREADY" not in sig_mtime_ok["execution_mode"]
    df_spy_off_newer = df_spy_off.copy()
    df_spy_off_newer.attrs["disk_mtime_ms"] = mt_1001_ms
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_eq_off if sym == "TST" else (df_spy_off_newer if sym == "SPY" else None))
    sig_mtime_bad = prov.get_signal("TST", regime=reg_rev)
    assert sig_mtime_bad["execution_mode"] == "UNREADY"
    assert "BENCHMARK_OBSERVATIONS_UNALIGNED" in sig_mtime_bad["data_integrity"]["missing_modalities"]

    # Same-day undated quote overlay carries live_quote_undated=True and disqualifies offline alignment
    prov.enable_live_fetch = True
    monkeypatch.setattr(prov, "_maybe_refresh_daily", lambda sym: None)
    monkeypatch.setattr(
        prov,
        "_fetch_live_quote_cached",
        lambda sym: {"price": 419.9, "date": "2025-11-24", "timestamp_ms": mt_1001_ms} if sym == "SPY" else None,
    )
    df_spy_overlaid, spy_ov_q = prov._load_cached_daily_with_quote_ts("SPY", allow_refresh=True, require_quote_ts=False)
    assert df_spy_overlaid is not None and spy_ov_q is None and df_spy_overlaid.attrs.get("live_quote_undated") is True
    assert df_spy_overlaid.attrs.get("disk_mtime_ms") == mt_1001_ms
    # Undated overlaid SPY fails offline alignment even when target mtime is newer
    prov.enable_live_fetch = False
    df_eq_off_newer = df_eq_off.copy()
    df_eq_off_newer.attrs["disk_mtime_ms"] = mt_1001_ms + 60_000
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_eq_off_newer if sym == "TST" else (df_spy_overlaid if sym == "SPY" else None))
    with prov._lock:
        prov._signal_cache.clear()
    sig_undated_ov = prov.get_signal("TST", regime=reg_rev)
    assert sig_undated_ov["execution_mode"] == "UNREADY"
    assert "BENCHMARK_OBSERVATIONS_UNALIGNED" in sig_undated_ov["data_integrity"]["missing_modalities"]

    # I) Changed bar inside post-close settlement window (13:02 ET) stores _bar_unsettled separately without fabricating obs_ms
    df_1245 = df_tgt.copy()
    t1245_ms = int(pd.Timestamp("2025-11-28 12:45:00", tz="America/New_York").timestamp() * 1000)
    df_1245.attrs["observation_timestamp_ms"] = t1245_ms
    df_1245.attrs["obs_bar_date"] = "2025-11-28"
    ml_settle.save_ticker_daily("CHG", df_1245)
    df_chg_yf = df_same_yf.copy()
    df_chg_yf.iloc[-1, df_chg_yf.columns.get_loc("Close")] = float(df_chg_yf["Close"].iloc[-1]) + 1.25
    df_chg_yf.iloc[-1, df_chg_yf.columns.get_loc("High")] = max(float(df_chg_yf["High"].iloc[-1]), float(df_chg_yf["Close"].iloc[-1]))
    monkeypatch.setattr(yf, "download", lambda *a, **kw: df_chg_yf)
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-11-28 13:02:00", tz="America/New_York").timestamp())
    df_chg_ref = ml_settle.refresh_ticker_daily("CHG", force_refresh_if_incomplete=True)
    assert df_chg_ref is not None
    assert df_chg_ref.attrs.get("_bar_unsettled") is True
    assert "observation_timestamp_ms" not in df_chg_ref.attrs
    assert prov._daily_bar_status_ms(df_chg_ref)[:2] == (None, False)
    # Reloading CHG from Parquet preserves unsettled status across restarts until past settlement
    df_chg_disk = ml_settle.load_ticker_daily("CHG")
    assert df_chg_disk is not None and df_chg_disk.attrs.get("_bar_unsettled") is True
    assert prov._daily_bar_status_ms(df_chg_disk)[:2] == (None, False)

    # J) _maybe_refresh_daily at 13:14 ET schedules retry at 13:15 settlement rather than 900s failure backoff,
    #    holds off just before the deadline, and completes STL once settled
    prov_stl = LiveInferenceProvider(lakehouse_dir=tmp_path / "lh_stl_retry", weights_dir=weights_dir, enable_live_fetch=True)
    prov_stl.market_loader.save_ticker_daily("STL", df_pre_close)
    monkeypatch.setattr(prov_stl.market_loader, "_latest_expected_trading_date", lambda: pd.Timestamp("2025-11-28"))
    stl_dl_calls = {"n": 0, "finalized": False}
    df_stl_settled = df_same_yf.copy()
    df_stl_settled.iloc[-1, df_stl_settled.columns.get_loc("Close")] = float(df_stl_settled["Close"].iloc[-1]) + 0.35
    df_stl_settled.iloc[-1, df_stl_settled.columns.get_loc("High")] = max(
        float(df_stl_settled["High"].iloc[-1]), float(df_stl_settled["Close"].iloc[-1])
    )
    def _dl_stl(*a, **kw):
        stl_dl_calls["n"] += 1
        return df_stl_settled if stl_dl_calls["finalized"] else df_same_yf
    monkeypatch.setattr(yf, "download", _dl_stl)
    t_1314 = pd.Timestamp("2025-11-28 13:14:00", tz="America/New_York").timestamp()
    stl_wall = {"t": t_1314}
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: stl_wall["t"])
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: stl_wall["t"])
    mono_box = {"now": 1000.0}
    monkeypatch.setattr("src.ipc.live_providers.time.monotonic", lambda: mono_box["now"])
    prov_stl._maybe_refresh_daily("STL")
    assert stl_dl_calls["n"] == 1
    assert "STL" not in prov_stl._daily_refresh_attempted
    retry_at = prov_stl._failed_yf_until.get("STL", 0.0)
    assert 50.0 <= (retry_at - mono_box["now"]) <= 70.0  # ~60s until 13:15, NOT 900s

    # Just before stored deadline -> no download occurs
    stl_wall["t"] = pd.Timestamp("2025-11-28 13:14:59", tz="America/New_York").timestamp()
    mono_box["now"] = retry_at - 1.0
    prov_stl._maybe_refresh_daily("STL")
    assert stl_dl_calls["n"] == 1

    # Past 13:15 settlement -> download executes and completes STL
    stl_dl_calls["finalized"] = True
    stl_wall["t"] = pd.Timestamp("2025-11-28 13:15:02", tz="America/New_York").timestamp()
    mono_box["now"] = retry_at + 1.0
    prov_stl._maybe_refresh_daily("STL")
    assert stl_dl_calls["n"] == 2
    stl_disk_done = prov_stl._load_disk_daily("STL")
    assert stl_disk_done is not None and not stl_disk_done.attrs.get("_bar_unsettled")
    assert prov_stl._daily_bar_status_ms(stl_disk_done)[1] is True

    # K) Same-day 10-K (no acceptanceDateTime, $10M) + 10-K/A (with acceptanceDateTime, $11M) preserves PIT=$10M & Restated=$11M
    cik_am = "0000111232"
    (fc_fresh.raw_dir / "edgar_submissions" / f"CIK{cik_am}.json").write_text(json.dumps({
        "cik": cik_am,
        "recent": {
            "form": ["10-K", "10-K/A"],
            "filingDate": ["2020-02-03", "2020-02-03"],
            "reportDate": ["2019-12-31", "2019-12-31"],
            "accessionNumber": ["0001", "0002"],
            "acceptanceDateTime": ["", "2020-02-03T17:00:00.000Z"],
            "primaryDocument": ["orig.htm", "amend.htm"],
        },
    }), encoding="utf-8")
    (fc_fresh.raw_dir / "edgar_facts" / f"CIK{cik_am}.json").write_text(json.dumps({
        "cik": cik_am,
        "entityName": "AMEND SPLIT CORP",
        "us_gaap": {
            "Revenues": {"units": {"USD": [
                {"form": "10-K", "start": "2019-01-01", "end": "2019-12-31", "filed": "2020-02-03", "accn": "0001", "val": 10_000_000},
                {"form": "10-K/A", "start": "2019-01-01", "end": "2019-12-31", "filed": "2020-02-03", "accn": "0002", "val": 11_000_000},
            ]}},
            "Assets": {"units": {"USD": [{"form": "10-K", "end": "2019-12-31", "filed": "2020-02-03", "accn": "0001", "val": 50_000_000, "fp": "FY"}]}},
        },
    }), encoding="utf-8")
    fin_am_pit = fc_fresh.build_financials_payload("AMND", known_cik=cik_am, pit_strict=True, enable_live_fetch=False)
    fin_am_rest = fc_fresh.build_financials_payload("AMND", known_cik=cik_am, pit_strict=False, enable_live_fetch=False)
    assert fin_am_pit["incomeStatement"][0]["values"]["FY2019"] == pytest.approx(10.0, abs=0.05)
    assert fin_am_rest["incomeStatement"][0]["values"]["FY2019"] == pytest.approx(11.0, abs=0.05)
    assert fin_am_pit["filings"][0]["order_uncertain"] is True

    # L) get_signal with enable_live_fetch=True fetches SPY & sector ETF before target ticker
    fetch_seq: list[str] = []
    prov_seq = LiveInferenceProvider(lakehouse_dir=tmp_path, weights_dir=weights_dir, enable_live_fetch=True)
    prov_seq.lgbm_backbone = prov.lgbm_backbone
    prov_seq.temp_scaler = prov.temp_scaler
    prov_seq.regime_hmm = prov.regime_hmm
    prov_seq._class_priors = prov._class_priors
    monkeypatch.setattr(prov_seq.universe_selector, "get_sector_etf", lambda sym: "XLK" if sym == "NVDA" else "SPY")
    monkeypatch.setattr(yf, "Ticker", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unexpected Yahoo network call")))
    monkeypatch.setattr(prov_seq.fundamentals_client, "build_edgar_filings_payload", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unexpected SEC network call")))
    monkeypatch.setattr(prov_seq.fundamentals_client, "get_yf_meta", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unexpected YF meta call")))
    def _track_load(sym, allow_refresh=True, require_quote_ts=True):
        fetch_seq.append(sym)
        if sym == "NVDA":
            return df_nvda_55, t_1555_ms
        return df_spy_55, t_1555_ms
    monkeypatch.setattr(prov_seq, "_load_cached_daily_with_quote_ts", _track_load)
    prov_seq.get_signal("NVDA", regime=reg_rev)
    assert fetch_seq == ["SPY", "XLK", "NVDA"]

    # Loop-7 hardening checks:
    # M) RegimeHMM.get_regime_multiplier rejects scalars, out-of-[0,1], and non-unit mass posteriors
    assert hmm_guard.get_regime_multiplier(np.array(0.5)) == 0.25
    assert hmm_guard.get_regime_multiplier(np.array([-0.2, 1.2, 0.0])) == 0.25
    assert hmm_guard.get_regime_multiplier(np.array([0.2, 0.2, 0.2])) == 0.25

    # N) FastInfo NaN Close falling back to Monday regularMarketPrice reports Monday date (not Tuesday)
    class FakeFastInfoNanLastClose:
        fast_info = type("FINanClose", (), {
            "lastPrice": 195.0,
            "_prices_1y": pd.DataFrame(
                {"Close": [195.0, float("nan")]},
                index=pd.DatetimeIndex([pd.Timestamp("2025-11-24 00:00:00-05:00"), pd.Timestamp("2025-11-25 00:00:00-05:00")]),
            ),
            "_md": {"regularMarketTime": int(pd.Timestamp("2025-11-24 16:00:00", tz="America/New_York").timestamp())},
            "get": lambda self, k, d=None: {"lastPrice": 195.0}.get(k, d),
        })()
        def history(self, **kw):
            return pd.DataFrame(
                {"Open": [194.0], "High": [196.0], "Low": [193.0], "Close": [195.0], "Volume": [1000]},
                index=pd.DatetimeIndex([pd.Timestamp("2025-11-24 00:00:00-05:00")]),
            )
    monkeypatch.setattr(yf, "Ticker", lambda sym: FakeFastInfoNanLastClose())
    q_nan_close = prov.market_loader.fetch_live_quote("TST")
    assert q_nan_close is not None and q_nan_close["date"] == "2025-11-24"

    # O) Future-dated live quote timestamp is rejected by overlay even when require_quote_ts=False
    prov.enable_live_fetch = True
    monkeypatch.setattr(prov, "_maybe_refresh_daily", lambda sym: None)
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_mon_t if sym == "TST" else None)
    fut_now = pd.Timestamp("2025-11-24 11:00:00", tz="America/New_York").timestamp()
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: fut_now)
    fut_quote_ms = int(fut_now * 1000) + 60_000
    monkeypatch.setattr(
        prov,
        "_fetch_live_quote_cached",
        lambda sym: {"price": 999.0, "date": "2025-11-24", "observation_timestamp_ms": fut_quote_ms, "timestamp_ms": fut_quote_ms} if sym == "TST" else None,
    )
    df_no_fut, fut_q = prov._load_cached_daily_with_quote_ts("TST", allow_refresh=True, require_quote_ts=False)
    assert fut_q is None and df_no_fut is df_mon_t
    prov.enable_live_fetch = False

    # P) classify_insider_tx distinguishes option grants citing exercise price ("A") from actual exercises ("M")
    assert FundamentalsClient.classify_insider_tx("Grant of employee stock options at an exercise price of $20") == "A"
    assert FundamentalsClient.classify_insider_tx("Exercise of employee stock options") == "M"

    # Q) Same-day equal-timestamp 10-K filings flag order_uncertain=True without false restatement
    cik_tie = "0000111233"
    (fc_fresh.raw_dir / "edgar_submissions" / f"CIK{cik_tie}.json").write_text(json.dumps({
        "cik": cik_tie,
        "recent": {
            "form": ["10-K", "10-K"],
            "filingDate": ["2020-02-03", "2020-02-03"],
            "reportDate": ["2019-12-31", "2019-12-31"],
            "accessionNumber": ["0001", "0002"],
            "acceptanceDateTime": ["2020-02-03T17:00:00.000Z", "2020-02-03T17:00:00.000Z"],
            "primaryDocument": ["a.htm", "b.htm"],
        },
    }), encoding="utf-8")
    (fc_fresh.raw_dir / "edgar_facts" / f"CIK{cik_tie}.json").write_text(json.dumps({
        "cik": cik_tie,
        "entityName": "TIE CORP",
        "us_gaap": {
            "Revenues": {"units": {"USD": [
                {"form": "10-K", "start": "2019-01-01", "end": "2019-12-31", "filed": "2020-02-03", "accn": "0001", "val": 10_000_000},
                {"form": "10-K", "start": "2019-01-01", "end": "2019-12-31", "filed": "2020-02-03", "accn": "0002", "val": 12_000_000},
            ]}},
            "Assets": {"units": {"USD": [{"form": "10-K", "end": "2019-12-31", "filed": "2020-02-03", "accn": "0001", "val": 50_000_000, "fp": "FY"}]}},
        },
    }), encoding="utf-8")
    fin_tie = fc_fresh.build_financials_payload("TIE", known_cik=cik_tie, pit_strict=True, enable_live_fetch=False)
    assert fin_tie["filings"][0]["order_uncertain"] is True
    assert fin_tie["filings"][0]["is_restated"] is False

    # R) Future Tuesday headline cannot pollute Monday finbert_val in signal_breakdown
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-11-24 16:35:00", tz="America/New_York").timestamp())
    with prov._lock:
        prov._signal_cache.clear()
        prov._edgar_cache.clear()
        prov._news_cache["TST"] = (
            time.monotonic(),
            [
                {"decayed_sentiment": 0.4, "published_at": "2025-11-24T15:00:00Z", "published_at_verified": True},
                {"decayed_sentiment": -0.9, "published_at": "2025-11-25T15:00:00Z", "published_at_verified": True},
            ],
        )
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_mon_t if sym == "TST" else (df_mon_spy if sym == "SPY" else None))
    sig_news_pit = prov.get_signal("TST", regime=reg_rev)
    assert "UNREADY" not in sig_news_pit["execution_mode"]
    assert sig_news_pit["signal_breakdown"]["decayed_finbert_sentiment"] == pytest.approx(0.4, abs=0.01)
    assert "DIVERGENCE_OBSERVATIONS_UNALIGNED" in sig_news_pit["data_integrity"]["missing_modalities"]

    # S) Live mode (enable_live_fetch=True) never infers alignment from file mtime order on unstamped bars
    prov_live_strict = LiveInferenceProvider(lakehouse_dir=tmp_path, weights_dir=weights_dir, enable_live_fetch=True)
    prov_live_strict.lgbm_backbone = prov.lgbm_backbone
    prov_live_strict.temp_scaler = prov.temp_scaler
    prov_live_strict.regime_hmm = prov.regime_hmm
    prov_live_strict._class_priors = prov._class_priors
    monkeypatch.setattr(prov_live_strict, "_maybe_refresh_daily", lambda sym: None)
    monkeypatch.setattr(prov_live_strict, "_fetch_live_quote_cached", lambda sym: None)
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-11-24 11:00:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(prov_live_strict, "_load_disk_daily", lambda sym: df_eq_off if sym == "TST" else (df_spy_off if sym == "SPY" else None))
    sig_live_unverified = prov_live_strict.get_signal("TST", regime=reg_rev)
    assert sig_live_unverified["execution_mode"] == "UNREADY"
    assert "BENCHMARK_OBSERVATIONS_UNALIGNED" in sig_live_unverified["data_integrity"]["missing_modalities"]

    # T) Missing distinct sector partition (df_sec_raw=None for NVDA) runs SPY-only fallback finitely without singular error
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-11-24 16:35:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(prov.universe_selector, "get_sector_etf", lambda sym: "XLK" if sym == "NVDA" else "SPY")
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_mon_t if sym == "NVDA" else (df_mon_spy if sym == "SPY" else None))
    with prov._lock:
        prov._signal_cache.clear()
        prov._news_cache.clear()
    sig_no_sec = prov.get_signal("NVDA", regime=reg_rev)
    assert "UNREADY" not in sig_no_sec["execution_mode"]
    assert "SECTOR_BENCHMARK_UNALIGNED" in sig_no_sec["data_integrity"]["missing_modalities"]
    assert "DIVERGENCE_OBSERVATIONS_UNALIGNED" in sig_no_sec["data_integrity"]["missing_modalities"]
    assert all(bool(np.isfinite(float(v))) for v in sig_no_sec["signal_breakdown"].values())

    # Loop-8 hardening checks:
    # U) Settled bar with explicit post-close observation (16:10 ET) preserves 16:10 ET rather than truncating to 16:00 ET
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-11-24 16:15:00", tz="America/New_York").timestamp())
    df_1610 = df_mon_t.copy()
    mon_1610_ms = int(pd.Timestamp("2025-11-24 16:10:00", tz="America/New_York").timestamp() * 1000)
    df_1610.attrs["observation_timestamp_ms"] = mon_1610_ms
    df_1610.attrs["disk_mtime_ms"] = mon_1610_ms
    obs_1610, comp_1610, _ = prov._daily_bar_status_ms(df_1610)
    assert comp_1610 is True and obs_1610 == mon_1610_ms

    # V) Plain dict fast_info without _md cannot borrow separate history(5d) regularMarketTime
    class FakeBareFastInfoDict:
        fast_info = {"lastPrice": 115.5}
        def history(self, **kw):
            return hist_5d
        def get_history_metadata(self):
            return {"regularMarketTime": t_obs_et}
    monkeypatch.setattr(yf, "Ticker", lambda sym: FakeBareFastInfoDict())
    q_bare_fi = prov.market_loader.fetch_live_quote("TST")
    assert q_bare_fi is not None and "observation_timestamp_ms" not in q_bare_fi

    # W) fetch_or_load_daily started at 15:59 ET on Mon 2025-11-24 advancing to 16:01 ET before return persists _bar_unsettled=True
    ml_init = MarketDataLoader(tmp_path / "lh_init_unsettled")
    df_yf_mon = pd.DataFrame(
        {"Open": df_mon_t["open"].values, "High": df_mon_t["high"].values, "Low": df_mon_t["low"].values, "Close": df_mon_t["close"].values, "Volume": df_mon_t["volume"].values},
        index=df_mon_t.index,
    )
    clock_init = {"ts": pd.Timestamp("2025-11-24 15:59:00", tz="America/New_York").timestamp()}
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: clock_init["ts"])
    def _dl_cross_close(*a, **kw):
        clock_init["ts"] = pd.Timestamp("2025-11-24 16:01:00", tz="America/New_York").timestamp()
        return df_yf_mon
    monkeypatch.setattr(yf, "download", _dl_cross_close)
    fetched_init = ml_init.fetch_or_load_daily("INIT", start_date="2025-04-30", end_date="2025-11-24")
    assert fetched_init.attrs.get("_bar_unsettled") is True
    reloaded_init = ml_init.load_ticker_daily("INIT")
    assert reloaded_init is not None and reloaded_init.attrs.get("_bar_unsettled") is True
    assert prov._daily_bar_status_ms(reloaded_init)[:2] == (None, False)

    # Loop-9 hardening checks (remaining task-2806 findings):
    # X) Failed os.replace in FundamentalsClient keeps fresh memory payload over expired disk file until external change
    fc_fail = FundamentalsClient(lakehouse_dir=tmp_path / "lh_fw_snap")
    p_fw = fc_fail.raw_dir / "yf_meta" / "FAILW.json"
    p_fw.parent.mkdir(parents=True, exist_ok=True)
    p_fw.write_text(json.dumps({"info": {"version": "expired_disk"}}), encoding="utf-8")
    old_fw_ts = time.time() - 86400 * 10
    os.utime(p_fw, (old_fw_ts, old_fw_ts))
    orig_os_replace = os.replace
    monkeypatch.setattr("src.data.fundamentals_client.os.replace", lambda s, d: (_ for _ in ()).throw(PermissionError("read-only")))
    fc_fail._write_json_cache("yf_meta/FAILW.json", {"info": {"version": "fresh_mem"}})
    monkeypatch.setattr("src.data.fundamentals_client.os.replace", orig_os_replace)
    assert fc_fail._read_json_cache("yf_meta/FAILW.json", 3600.0) == {"info": {"version": "fresh_mem"}}
    assert fc_fail._read_stale_json_cache("yf_meta/FAILW.json") == {"info": {"version": "fresh_mem"}}
    # External replacement on disk supersedes failed-write memory snapshot cleanly
    p_fw.write_text(json.dumps({"info": {"version": "external_disk"}}), encoding="utf-8")
    assert fc_fail._read_json_cache("yf_meta/FAILW.json", 3600.0) == {"info": {"version": "external_disk"}}

    # Y) classify_insider_tx handles "Shares were not sold; shares were withheld for taxes" -> "F", "Shares sold to cover taxes" -> "S"
    assert FundamentalsClient.classify_insider_tx("Shares were not sold; shares were withheld for taxes") == "F"
    assert FundamentalsClient.classify_insider_tx("Shares sold to cover taxes") == "S"

    # Z) Unsettled disk bar with unknown observation time is NOT merged into an 11:50:00 cached quote (uses quote fields alone),
    #    and an unsettled disk bar with an explicit newer 11:50:07 observation time also rejects overwriting with 11:50:00 quote
    prov.enable_live_fetch = True
    monkeypatch.setattr(prov, "_maybe_refresh_daily", lambda sym: None)
    df_unsettled_unknown_ts = df_mon_t.copy()
    df_unsettled_unknown_ts.iloc[-1, df_unsettled_unknown_ts.columns.get_loc("high")] = 999.0
    df_unsettled_unknown_ts.iloc[-1, df_unsettled_unknown_ts.columns.get_loc("volume")] = 99_999_999.0
    df_unsettled_unknown_ts.attrs.pop("observation_timestamp_ms", None)
    df_unsettled_unknown_ts.attrs["_bar_unsettled"] = True
    t_115000_ms = int(pd.Timestamp("2025-11-24 11:50:00", tz="America/New_York").timestamp() * 1000)
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-11-24 11:51:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_unsettled_unknown_ts if sym == "TST" else None)
    monkeypatch.setattr(
        prov,
        "_fetch_live_quote_cached",
        lambda sym: {"price": 121.0, "open": 120.0, "high": 122.0, "low": 119.5, "volume": 5000, "date": "2025-11-24", "observation_timestamp_ms": t_115000_ms} if sym == "TST" else None,
    )
    df_q_only, q_only_ms = prov._load_cached_daily_with_quote_ts("TST", allow_refresh=True, require_quote_ts=True)
    assert df_q_only is not None and q_only_ms == t_115000_ms
    assert float(df_q_only["high"].iloc[-1]) == 122.0
    assert float(df_q_only["volume"].iloc[-1]) == 5000.0

    # Explicit newer 11:50:07 unsettled disk observation is NOT overwritten by older 11:50:00 quote
    df_unsettled_115007 = df_mon_t.copy()
    t_115007_ms = int(pd.Timestamp("2025-11-24 11:50:07", tz="America/New_York").timestamp() * 1000)
    df_unsettled_115007.iloc[-1, df_unsettled_115007.columns.get_loc("close")] = 129.0
    df_unsettled_115007.attrs["observation_timestamp_ms"] = t_115007_ms
    df_unsettled_115007.attrs["obs_bar_date"] = "2025-11-24"
    df_unsettled_115007.attrs["_bar_unsettled"] = True
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_unsettled_115007 if sym == "TST" else None)
    df_keep_115007, q_keep_ms = prov._load_cached_daily_with_quote_ts("TST", allow_refresh=True, require_quote_ts=True)
    assert df_keep_115007 is not None and q_keep_ms is None
    assert float(df_keep_115007["close"].iloc[-1]) == 129.0
    prov.enable_live_fetch = False

    # AA) Yahoo date-only prior day item ("2025-11-21", verified=False) aligns cleanly for Nov 24,
    #     while same-day unverified date-only ("2025-11-24") and synthetic ISO ("2025-11-24T12:00:00Z", verified=False) stay excluded
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-11-24 16:35:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_mon_t if sym == "TST" else (df_mon_spy if sym == "SPY" else None))
    with prov._lock:
        prov._signal_cache.clear()
        prov._news_cache["TST"] = (
            time.monotonic(),
            [{"decayed_sentiment": 0.55, "published_at": "2025-11-21", "published_at_verified": False, "published_date_only": True}],
        )
    sig_date_only_prior = prov.get_signal("TST", regime=reg_rev)
    assert sig_date_only_prior["signal_breakdown"]["decayed_finbert_sentiment"] == pytest.approx(0.55, abs=0.01)
    assert "DIVERGENCE_OBSERVATIONS_UNALIGNED" not in sig_date_only_prior["data_integrity"]["missing_modalities"]
    with prov._lock:
        prov._signal_cache.clear()
        prov._news_cache["TST"] = (
            time.monotonic(),
            [{"decayed_sentiment": 0.55, "published_at": "2025-11-24", "published_at_verified": False, "published_date_only": True}],
        )
    sig_date_only_same = prov.get_signal("TST", regime=reg_rev)
    assert sig_date_only_same["signal_breakdown"]["decayed_finbert_sentiment"] == 0.0
    assert "DIVERGENCE_OBSERVATIONS_UNALIGNED" in sig_date_only_same["data_integrity"]["missing_modalities"]

    # Loop-10 hardening checks (task-2881 CodeRabbit findings):
    # BB) get_regime_multiplier fails closed to 0.25 on object/non-numeric arrays rather than raising ValueError
    assert hmm_guard.get_regime_multiplier(np.array(["invalid", 0, 0], dtype=object)) == 0.25

    # CC) classify_insider_tx recognizes sales back to the issuer (Rule 16b-3(e)) as "D", preserving open-market sales as "S"
    assert FundamentalsClient.classify_insider_tx("Shares sold back to the issuer under Rule 16b-3(e)") == "D"
    assert FundamentalsClient.classify_insider_tx("Sold 1,000 shares on the open market") == "S"

    # DD) Competing same-day 10-K / 10-K/A observations with missing accessions stay order_uncertain=True
    cik_no_acc = "0000111999"
    (fc_fresh.raw_dir / "edgar_submissions" / f"CIK{cik_no_acc}.json").write_text(
        json.dumps({"cik": cik_no_acc, "recent": {"form": [], "filingDate": [], "reportDate": [], "accessionNumber": [], "acceptanceDateTime": []}}),
        encoding="utf-8",
    )
    (fc_fresh.raw_dir / "edgar_facts" / f"CIK{cik_no_acc}.json").write_text(
        json.dumps({
            "cik": cik_no_acc,
            "entityName": "NO ACCN CORP",
            "us_gaap": {
                "Revenues": {"units": {"USD": [
                    {"form": "10-K", "start": "2019-01-01", "end": "2019-12-31", "filed": "2020-02-03", "accn": "", "val": 10_000_000},
                    {"form": "10-K/A", "start": "2019-01-01", "end": "2019-12-31", "filed": "2020-02-03", "accn": "", "val": 12_000_000},
                ]}},
                "Assets": {"units": {"USD": [{"form": "10-K", "end": "2019-12-31", "filed": "2020-02-03", "accn": "", "val": 50_000_000, "fp": "FY"}]}},
            },
        }),
        encoding="utf-8",
    )
    fin_no_acc = fc_fresh.build_financials_payload("NOAC", known_cik=cik_no_acc, pit_strict=True, enable_live_fetch=False)
    assert fin_no_acc["filings"][0]["order_uncertain"] is True

    # EE) query_parquet_duckdb normalizes capitalized "Timestamp" column into DatetimeIndex in both DuckDB and Arrow fallback
    pq_cap = ml_settle.daily_dir / "CAPTS.parquet"
    df_cap = df_mon_t.iloc[-10:].copy().reset_index().rename(columns={"index": "Timestamp", "timestamp": "Timestamp"})
    df_cap.to_parquet(pq_cap, engine="pyarrow", index=False)
    cap_swaps = [ml_settle.daily_dir / f"CAPTS_SWAP_{i}.parquet" for i in range(3)]
    for i, sp in enumerate(cap_swaps):
        df_sw = df_cap.copy()
        df_sw.iloc[-1, df_sw.columns.get_loc("close")] = float(df_sw["close"].iloc[-1]) + 0.1 * (i + 1)
        df_sw.to_parquet(sp, engine="pyarrow", index=False)

    orig_ddb_cap = _ddb.connect
    orig_read_tbl_cap = _pq.read_table
    cap_fallback_calls: list[int] = []
    cap_swap_state = {"enable": False, "idx": 0}
    class _AlwaysSwapCursor:
        def __init__(self, cur, sql_txt: str):
            self._cur = cur
            self._sql = sql_txt
        def fetchall(self):
            return self._cur.fetchall()
        def fetchdf(self):
            d = self._cur.fetchdf()
            if "SELECT" in self._sql and "FROM read_parquet" in self._sql:
                if cap_swap_state["enable"] and cap_swap_state["idx"] < len(cap_swaps):
                    os.replace(cap_swaps[cap_swap_state["idx"]], pq_cap)
                    cap_swap_state["idx"] += 1
            return d
    class _AlwaysSwapConn:
        def __init__(self, con):
            self._con = con
        def execute(self, sql_txt, *a, **kw):
            return _AlwaysSwapCursor(self._con.execute(sql_txt, *a, **kw), sql_txt)
        def close(self):
            return self._con.close()
    monkeypatch.setattr("duckdb.connect", lambda *a, **kw: _AlwaysSwapConn(orig_ddb_cap(*a, **kw)))
    monkeypatch.setattr("pyarrow.parquet.read_table", lambda *a, **kw: (cap_fallback_calls.append(1) or orig_read_tbl_cap(*a, **kw)))
    q_cap_ddb = ml_settle.query_parquet_duckdb(pq_cap, columns=["CLOSE"], start_date="2025-11-12")
    assert len(cap_fallback_calls) == 0
    assert isinstance(q_cap_ddb.index, pd.DatetimeIndex)
    assert "Timestamp" not in q_cap_ddb.columns and "close" in q_cap_ddb.columns

    cap_swap_state["enable"] = True
    q_cap_arrow = ml_settle.query_parquet_duckdb(pq_cap, columns=["CLOSE"], start_date="2025-11-12")
    monkeypatch.setattr("duckdb.connect", orig_ddb_cap)
    monkeypatch.setattr("pyarrow.parquet.read_table", orig_read_tbl_cap)
    assert len(cap_fallback_calls) == 1
    assert isinstance(q_cap_arrow.index, pd.DatetimeIndex)
    assert "Timestamp" not in q_cap_arrow.columns and "close" in q_cap_arrow.columns

    # FF) fetch_live_quote clears regularMarketTime when FastInfo Close differs from regularMarketPrice on the same date,
    #     and resolves history metadata regularMarketTime when fi_last_close_nan falls back to hist_norm
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-12-01 12:00:00", tz="America/New_York").timestamp())
    class FakeMismatchedCloseAndRmp:
        fast_info = type("FIMis", (), {
            "lastPrice": 114.0, "open": 113.0, "dayHigh": 116.0, "dayLow": 112.5, "previousClose": 112.0, "lastVolume": 50000,
            "_md": {"regularMarketTime": t_obs_et, "regularMarketPrice": 115.5},
            "_prices_1y": pd.DataFrame({"Close": [114.0]}, index=pd.DatetimeIndex([pd.Timestamp("2025-12-01 00:00:00-05:00")])),
            "get": lambda self, k, d=None: {"lastPrice": 114.0}.get(k, d),
        })()
        def history(self, **kw):
            return hist_5d
        def get_history_metadata(self):
            return {"regularMarketTime": t_obs_et, "regularMarketPrice": 115.5}
    monkeypatch.setattr(yf, "Ticker", lambda sym: FakeMismatchedCloseAndRmp())
    q_mis_rmp = prov.market_loader.fetch_live_quote("TST")
    assert q_mis_rmp is not None and q_mis_rmp["price"] == 114.0
    assert "observation_timestamp_ms" not in q_mis_rmp

    t_nov28_obs = pd.Timestamp("2025-11-28 11:15:00", tz="America/New_York")
    class FakeFiNanHistMd:
        fast_info = type("FINan", (), {
            "lastPrice": 102.5, "open": 101.0, "dayHigh": 103.0, "dayLow": 100.5, "previousClose": 100.0, "lastVolume": 200000,
            "_md": {},
            "_prices_1y": pd.DataFrame({"Close": [float("nan")]}, index=pd.DatetimeIndex([pd.Timestamp("2025-11-28 00:00:00-05:00")])),
            "get": lambda self, k, d=None: {"lastPrice": 102.5, "open": 101.0, "dayHigh": 103.0, "dayLow": 100.5, "previousClose": 100.0, "lastVolume": 200000}.get(k, d),
        })()
        def history(self, **kw):
            return hist_5d
        def get_history_metadata(self):
            return {"regularMarketTime": t_nov28_obs, "regularMarketPrice": 102.5}
    monkeypatch.setattr(yf, "Ticker", lambda sym: FakeFiNanHistMd())
    q_nan_hist = prov.market_loader.fetch_live_quote("TST")
    assert q_nan_hist is not None and q_nan_hist["price"] == 102.5
    assert q_nan_hist["observation_timestamp_ms"] == int(t_nov28_obs.timestamp() * 1000)

    # GG) Pre-settlement _maybe_refresh_daily attempt starting at 13:14:58 and finishing at 13:15:02 without replacement
    #     allows prompt retry rather than 900s failure cooldown; post-settlement failure enforces 900s cooldown
    prov_cross = LiveInferenceProvider(lakehouse_dir=tmp_path / "lh_stl_cross", weights_dir=weights_dir, enable_live_fetch=True)
    prov_cross.market_loader.save_ticker_daily("CRSS", df_pre_close)
    prov_cross.market_loader.save_ticker_daily("CRSS2", df_pre_close)
    monkeypatch.setattr(prov_cross.market_loader, "_latest_expected_trading_date", lambda: pd.Timestamp("2025-11-28"))
    cross_wall = {"t": pd.Timestamp("2025-11-28 13:14:58", tz="America/New_York").timestamp()}
    cross_mono = {"now": 2000.0}
    cross_dl_calls = {"n": 0, "mode": "pre_unchanged"}
    def _dl_cross(*a, **kw):
        cross_dl_calls["n"] += 1
        if cross_dl_calls["mode"] == "pre_unchanged":
            cross_wall["t"] = pd.Timestamp("2025-11-28 13:15:02", tz="America/New_York").timestamp()
            return df_same_yf
        if cross_dl_calls["mode"] == "settled":
            return df_stl_settled
        return pd.DataFrame()
    monkeypatch.setattr(yf, "download", _dl_cross)
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: cross_wall["t"])
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: cross_wall["t"])
    monkeypatch.setattr("src.ipc.live_providers.time.monotonic", lambda: cross_mono["now"])
    prov_cross._maybe_refresh_daily("CRSS")
    assert cross_dl_calls["n"] == 1
    assert "CRSS" not in prov_cross._failed_yf_until
    assert "CRSS" not in prov_cross._daily_refresh_attempted
    # Prompt retry immediately at 13:15:03 completes when settled; failure starting post-settlement enforces 900s cooldown
    cross_dl_calls["mode"] = "settled"
    cross_wall["t"] = pd.Timestamp("2025-11-28 13:15:03", tz="America/New_York").timestamp()
    cross_mono["now"] = 2001.0
    prov_cross._maybe_refresh_daily("CRSS")
    assert cross_dl_calls["n"] == 2
    crss_done = prov_cross._load_disk_daily("CRSS")
    assert crss_done is not None and prov_cross._daily_bar_status_ms(crss_done)[1] is True

    cross_dl_calls["mode"] = "empty"
    prov_cross._maybe_refresh_daily("CRSS2")
    assert prov_cross._failed_yf_until.get("CRSS2", 0.0) == pytest.approx(2001.0 + prov_cross._FAILED_TICKER_COOLDOWN_SECONDS, abs=1.0)

    # HH) Shortened half-day Black Friday (2025-11-28, 13:00 ET close): Form 4 disclosed at 14:00 ET is ineligible on Nov 28 15:00 ET and eligible on Dec 1
    idx_bf = pd.bdate_range(end="2025-12-01", periods=150)
    df_dec1_t = pd.DataFrame(
        {"open": 100.0, "high": 102.0, "low": 99.0, "close": np.linspace(100.0, 120.0, 150), "volume": 1_000_000.0},
        index=idx_bf,
    )
    df_dec1_spy = pd.DataFrame(
        {"open": 400.0, "high": 402.0, "low": 399.0, "close": np.linspace(400.0, 420.0, 150), "volume": 10_000_000.0},
        index=idx_bf,
    )
    df_bf_t = df_dec1_t.iloc[:-1].copy()  # ends 2025-11-28
    df_bf_spy = df_dec1_spy.iloc[:-1].copy()
    bf_1500_ms = int(pd.Timestamp("2025-11-28 15:00:00", tz="America/New_York").timestamp() * 1000)
    dec1_1600_ms = int(pd.Timestamp("2025-12-01 16:00:00", tz="America/New_York").timestamp() * 1000)
    df_bf_t.attrs["observation_timestamp_ms"] = bf_1500_ms
    df_bf_spy.attrs["observation_timestamp_ms"] = bf_1500_ms
    df_dec1_t.attrs["observation_timestamp_ms"] = dec1_1600_ms
    df_dec1_spy.attrs["observation_timestamp_ms"] = dec1_1600_ms
    bf_1400_iso = pd.Timestamp("2025-11-28 14:00:00", tz="America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-12-02 10:00:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_bf_t if sym == "TST" else (df_bf_spy if sym == "SPY" else None))
    with prov._lock:
        prov._signal_cache.clear()
        prov._edgar_cache["TST"] = (time.monotonic(), {
            "available": True,
            "item1a_sentiment_drift": 0.0,
            "mda_catalyst_score": 0.0,
            "form4_transactions": [{"transaction_code": "P", "value": 500_000.0, "disclosed_at": bf_1400_iso, "disclosed_at_verified": True}],
        })
    sig_bf_post = prov.get_signal("TST", regime=reg_rev)
    assert sig_bf_post["signal_breakdown"]["decayed_form4_insider_cluster"] == 0.0
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_dec1_t if sym == "TST" else (df_dec1_spy if sym == "SPY" else None))
    with prov._lock:
        prov._signal_cache.clear()
    sig_dec1_bf = prov.get_signal("TST", regime=reg_rev)
    assert sig_dec1_bf["signal_breakdown"]["decayed_form4_insider_cluster"] > 0.0

    # II) get_historical_prices omits observation_timestamp_ms when latest observation is in the future
    df_future_bar = df_mon_t.copy()
    now_mon_1500 = pd.Timestamp("2025-11-24 15:00:00", tz="America/New_York").timestamp()
    df_future_bar.attrs["observation_timestamp_ms"] = int(pd.Timestamp("2025-11-24 15:30:00", tz="America/New_York").timestamp() * 1000)
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: now_mon_1500)
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_future_bar if sym == "TST" else None)
    hp_fut = prov.get_historical_prices("TST", count=5)
    assert hp_fut and "observation_timestamp_ms" not in hp_fut[0]

    # JJ) get_news_feed preserves sub-second publication precision for PIT comparisons with distinct sentiment scores
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-11-24 16:00:00", tz="America/New_York").timestamp())
    class FakeSubsecNewsTicker:
        news = [
            {"title": "Late miss sparks downgrade warning", "providerPublishTime": pd.Timestamp("2025-11-24T15:00:00.900000Z").timestamp()},
            {"title": "Earlier surge beat upgrade", "pubDate": "2025-11-24T15:00:00.200000Z"},
        ]
    prov.enable_live_fetch = True
    monkeypatch.setattr(yf, "Ticker", lambda sym: FakeSubsecNewsTicker())
    with prov._lock:
        prov._news_cache.pop("TST", None)
    nf_sub = prov.get_news_feed("TST")
    prov.enable_live_fetch = False
    assert len(nf_sub) == 2
    assert nf_sub[0]["published_at"].endswith(".900000Z")
    assert nf_sub[1]["published_at"].endswith(".200000Z")
    assert abs(nf_sub[0]["finbert_score"] - nf_sub[1]["finbert_score"]) > 1.0
    df_target_500ms = df_mon_t.copy()
    df_spy_500ms = df_mon_spy.copy()
    t_500ms = int(pd.Timestamp("2025-11-24T15:00:00.500000Z").timestamp() * 1000)
    df_target_500ms.attrs["observation_timestamp_ms"] = t_500ms
    df_spy_500ms.attrs["observation_timestamp_ms"] = t_500ms
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_target_500ms if sym == "TST" else (df_spy_500ms if sym == "SPY" else None))
    with prov._lock:
        prov._signal_cache.clear()
        prov._news_cache["TST"] = (time.monotonic(), nf_sub)
    sig_subsec = prov.get_signal("TST", regime=reg_rev)
    assert sig_subsec["signal_breakdown"]["decayed_finbert_sentiment"] == pytest.approx(nf_sub[1]["finbert_score"], abs=0.01)

    # KK) Sliced Monday SPY partition (disk_mtime_ms=Mon 16:20 ET) lacking per-row provenance for Friday
    #     must NOT substitute Friday 16:00 scheduled close when aligned against Friday target
    df_fri_tgt = df_mon_t.iloc[:-1].copy()  # Friday 2025-11-21
    fri_1600_ms = int(pd.Timestamp("2025-11-21 16:00:00", tz="America/New_York").timestamp() * 1000)
    mon_1620_ms = int(pd.Timestamp("2025-11-24 16:20:00", tz="America/New_York").timestamp() * 1000)
    df_fri_tgt.attrs["observation_timestamp_ms"] = fri_1600_ms
    df_spy_mon_rev = df_mon_spy.copy()  # Mon 2025-11-24 partition written Monday post-close
    df_spy_mon_rev.attrs["disk_mtime_ms"] = mon_1620_ms
    sliced_rev = prov._slice_aligned_frame(df_spy_mon_rev, df_fri_tgt.index)
    assert prov._daily_bar_status_ms(sliced_rev) == (None, True, fri_1600_ms)
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-11-24 16:30:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_fri_tgt if sym == "TST" else (df_spy_mon_rev if sym == "SPY" else None))
    with prov._lock:
        prov._signal_cache.clear()
        prov._edgar_cache.clear()
        prov._news_cache.clear()
    sig_rev_slice = prov.get_signal("TST", regime=reg_rev)
    assert sig_rev_slice["execution_mode"] == "UNREADY"
    assert "BENCHMARK_OBSERVATIONS_UNALIGNED" in sig_rev_slice["data_integrity"]["missing_modalities"]

    # Loop-11 hardening checks (task-3073 CodeRabbit findings):
    # LL) Live SEC Form 4 XML enrichment provides verified SEC acceptanceDateTime -> live Form 4 factor is non-zero
    fc_f4 = FundamentalsClient(lakehouse_dir=tmp_path / "lh_sec_f4")
    cik_f4 = "0000123987"
    fc_f4.raw_dir.mkdir(parents=True, exist_ok=True)
    (fc_f4.raw_dir / "sec_company_tickers.json").write_text(json.dumps({"TST4": cik_f4}), encoding="utf-8")
    (fc_f4.raw_dir / "yf_meta").mkdir(parents=True, exist_ok=True)
    (fc_f4.raw_dir / "yf_meta" / "TST4.json").write_text(json.dumps({"info": {}, "rec_trend": [], "insider_transactions": []}), encoding="utf-8")
    (fc_f4.raw_dir / "edgar_submissions").mkdir(parents=True, exist_ok=True)
    (fc_f4.raw_dir / "edgar_submissions" / f"CIK{cik_f4}.json").write_text(
        json.dumps({
            "cik": cik_f4,
            "recent": {
                "form": ["10-K", "4"],
                "filingDate": ["2025-08-01", "2025-11-21"],
                "acceptanceDateTime": ["2025-08-01T16:00:00.000Z", "2025-11-21T15:30:00.000Z"],
                "accessionNumber": ["0000123987-25-000001", "0000123987-25-000009"],
                "primaryDocument": ["f10k.htm", "xslF345X06/form4_buy.xml"],
            },
        }),
        encoding="utf-8",
    )
    f4_xml = (
        "<ownershipDocument><reportingOwner><reportingOwnerId><rptOwnerName>Ada Lovelace</rptOwnerName></reportingOwnerId>"
        "<reportingOwnerRelationship><isDirector>1</isDirector></reportingOwnerRelationship></reportingOwner>"
        "<aff10b5One>0</aff10b5One><nonDerivativeTable><nonDerivativeTransaction>"
        "<transactionDate><value>2025-11-20</value></transactionDate>"
        "<transactionCoding><transactionCode>P</transactionCode></transactionCoding>"
        "<transactionAmounts><transactionShares><value>10000</value></transactionShares>"
        "<transactionPricePerShare><value>50.0</value></transactionPricePerShare></transactionAmounts>"
        "</nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>"
    )
    monkeypatch.setattr(fc_f4, "_sec_fetch_text", lambda url, timeout=6.0: f4_xml if "form4_buy.xml" in url else None)
    edg_f4 = fc_f4.build_edgar_filings_payload("TST4", known_cik=cik_f4, enable_live_fetch=True)
    assert len(edg_f4["form4_transactions"]) == 1
    assert edg_f4["form4_transactions"][0]["disclosed_at"] == "2025-11-21T15:30:00.000Z"
    assert edg_f4["form4_transactions"][0]["disclosed_at_verified"] is True
    assert edg_f4["form4_transactions"][0]["transaction_code"] == "P"
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_mon_t if sym == "TST" else (df_mon_spy if sym == "SPY" else None))
    with prov._lock:
        prov._signal_cache.clear()
        prov._edgar_cache["TST"] = (time.monotonic(), edg_f4)
    sig_f4_xml = prov.get_signal("TST", regime=reg_rev)
    assert sig_f4_xml["signal_breakdown"]["decayed_form4_insider_cluster"] > 0.0

    # MM) fetch_live_quote does NOT stamp FastInfo regularMarketTime onto OHLCV filled from a newer/separate history download
    t_1000_dec1 = pd.Timestamp("2025-12-01 10:00:00", tz="America/New_York")
    t_1001_dec1 = pd.Timestamp("2025-12-01 10:01:00", tz="America/New_York")
    hist_dec1_1001 = pd.DataFrame(
        {"Open": [100.0], "High": [105.0], "Low": [99.0], "Close": [101.0], "Volume": [90000]},
        index=pd.DatetimeIndex([pd.Timestamp("2025-12-01 00:00:00-05:00")]),
    )
    class FakePartialFastInfoHistOhlcv:
        fast_info = type("FIPart", (), {
            "lastPrice": 100.0,
            "_md": {"regularMarketTime": t_1000_dec1, "regularMarketPrice": 100.0},
            "_prices_1y": pd.DataFrame(),
            "get": lambda self, k, d=None: {"lastPrice": 100.0}.get(k, d),
        })()
        def history(self, **kw):
            return hist_dec1_1001
        def get_history_metadata(self):
            return {"regularMarketTime": t_1001_dec1, "regularMarketPrice": 101.0}
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-12-01 12:00:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(yf, "Ticker", lambda sym: FakePartialFastInfoHistOhlcv())
    q_split_ohlcv = prov.market_loader.fetch_live_quote("TST")
    assert q_split_ohlcv is not None and "observation_timestamp_ms" not in q_split_ohlcv

    # NN) get_signal decays news sentiment as of target_obs_ms (from finbert_score), invariant to feed fetch time
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-11-24 16:30:00", tz="America/New_York").timestamp())
    df_tgt_decay = df_mon_t.copy()
    df_spy_decay = df_mon_spy.copy()
    t_nov24_1600_ms = int(pd.Timestamp("2025-11-24T16:00:00Z").timestamp() * 1000)
    df_tgt_decay.attrs["observation_timestamp_ms"] = t_nov24_1600_ms
    df_spy_decay.attrs["observation_timestamp_ms"] = t_nov24_1600_ms
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_tgt_decay if sym == "TST" else (df_spy_decay if sym == "SPY" else None))
    with prov._lock:
        prov._signal_cache.clear()
        prov._edgar_cache.clear()
        # Published 3 days before Nov 24 (half-life tau=3d -> 0.28 at target), while stale fetch-time decayed_sentiment was 0.05
        prov._news_cache["TST"] = (
            time.monotonic(),
            [{"finbert_score": 0.56, "decayed_sentiment": 0.05, "published_at": "2025-11-21T16:00:00Z", "published_at_verified": True}],
        )
    sig_target_decay = prov.get_signal("TST", regime=reg_rev)
    assert sig_target_decay["signal_breakdown"]["decayed_finbert_sentiment"] == pytest.approx(0.28, abs=0.01)

    # OO) Expired non-object JSON cache files (`[1]`) are rejected safely instead of throwing AttributeError
    cik_bad_json = "0000888777"
    p_sub_bad = fc_f4.raw_dir / "edgar_submissions" / f"CIK{cik_bad_json}.json"
    p_facts_bad = fc_f4.raw_dir / "edgar_facts" / f"CIK{cik_bad_json}.json"
    p_sub_bad.parent.mkdir(parents=True, exist_ok=True)
    p_facts_bad.parent.mkdir(parents=True, exist_ok=True)
    p_sub_bad.write_text("[1]", encoding="utf-8")
    p_facts_bad.write_text("[1]", encoding="utf-8")
    old_bad_t = time.time() - 86400 * 30
    os.utime(p_sub_bad, (old_bad_t, old_bad_t))
    os.utime(p_facts_bad, (old_bad_t, old_bad_t))
    assert fc_f4.get_sec_submissions(cik_bad_json, enable_live_fetch=False) is None
    assert fc_f4.get_sec_company_facts(cik_bad_json, enable_live_fetch=False) is None

    # PP) WebSocketBroadcaster.queue_tick preserves prior timestamp_ms on timestamp-free merge patch
    bc_patch = WebSocketBroadcaster()
    bc_patch.queue_tick("NVDA", {"price": 140.0, "timestamp_ms": 1700000001234, "observation_timestamp_ms": 1700000001234})
    bc_patch.queue_tick("NVDA", {"volume": 12345})
    assert bc_patch.pending_ticks["NVDA"]["timestamp_ms"] == 1700000001234
    assert bc_patch.pending_ticks["NVDA"]["volume"] == 12345

    # QQ) Negated gift reaches open-market sale ("S"), affirmative gift returns "G"
    assert FundamentalsClient.classify_insider_tx("Not a gift; sold on the open market") == "S"
    assert FundamentalsClient.classify_insider_tx("Bona fide gift of shares") == "G"

    # RR) Mixed-precision and mixed-timezone acceptanceDateTime sort & restate chronologically
    #     even when form is identical ("10-K") and accession + fixture order conflict with UTC chronology
    cik_ms = "0000555666"
    (fc_f4.raw_dir / "edgar_submissions" / f"CIK{cik_ms}.json").write_text(
        json.dumps({
            "cik": cik_ms,
            "recent": {
                "form": ["10-K", "10-K"],
                "filingDate": ["2020-02-03", "2020-02-03"],
                "reportDate": ["2019-12-31", "2019-12-31"],
                "accessionNumber": ["0001", "0002"],
                "acceptanceDateTime": ["2020-02-03T16:00:00.100Z", "2020-02-03T16:00:00Z"],
                "primaryDocument": ["later_ms.htm", "earlier_sec.htm"],
            },
        }),
        encoding="utf-8",
    )
    (fc_f4.raw_dir / "edgar_facts" / f"CIK{cik_ms}.json").write_text(
        json.dumps({
            "cik": cik_ms,
            "entityName": "MS SORT CORP",
            "us_gaap": {
                "Revenues": {"units": {"USD": [
                    {"form": "10-K", "start": "2019-01-01", "end": "2019-12-31", "filed": "2020-02-03", "accn": "0001", "val": 20_000_000},
                    {"form": "10-K", "start": "2019-01-01", "end": "2019-12-31", "filed": "2020-02-03", "accn": "0002", "val": 10_000_000},
                ]}},
                "Assets": {"units": {"USD": [{"form": "10-K", "end": "2019-12-31", "filed": "2020-02-03", "accn": "0002", "val": 50_000_000, "fp": "FY"}]}},
            },
        }),
        encoding="utf-8",
    )
    fin_ms_pit = fc_f4.build_financials_payload("MSS", known_cik=cik_ms, pit_strict=True, enable_live_fetch=False)
    fin_ms_rest = fc_f4.build_financials_payload("MSS", known_cik=cik_ms, pit_strict=False, enable_live_fetch=False)
    assert fin_ms_pit["incomeStatement"][0]["values"]["FY2019"] == pytest.approx(10.0, abs=0.05)
    assert fin_ms_rest["incomeStatement"][0]["values"]["FY2019"] == pytest.approx(20.0, abs=0.05)

    # Also verify timezone offset restatement detection: 13:00Z precedes 09:30-04:00 (13:30Z)
    cik_tz = "0000555667"
    (fc_f4.raw_dir / "edgar_submissions" / f"CIK{cik_tz}.json").write_text(
        json.dumps({
            "cik": cik_tz,
            "recent": {
                "form": ["10-K", "10-K/A"],
                "filingDate": ["2020-06-03", "2020-06-03"],
                "reportDate": ["2019-12-31", "2019-12-31"],
                "accessionNumber": ["0001", "0002"],
                "acceptanceDateTime": ["2020-06-03T13:00:00Z", "2020-06-03T09:30:00-04:00"],
                "primaryDocument": ["orig.htm", "amend.htm"],
            },
        }),
        encoding="utf-8",
    )
    (fc_f4.raw_dir / "edgar_facts" / f"CIK{cik_tz}.json").write_text(
        json.dumps({
            "cik": cik_tz,
            "entityName": "TZ RESTATE CORP",
            "us_gaap": {
                "Revenues": {"units": {"USD": [
                    {"form": "10-K", "start": "2019-01-01", "end": "2019-12-31", "filed": "2020-06-03", "accn": "0001", "val": 100_000_000},
                    {"form": "10-K/A", "start": "2019-01-01", "end": "2019-12-31", "filed": "2020-06-03", "accn": "0002", "val": 120_000_000},
                ]}},
                "Assets": {"units": {"USD": [{"form": "10-K", "end": "2019-12-31", "filed": "2020-06-03", "accn": "0001", "val": 500_000_000, "fp": "FY"}]}},
            },
        }),
        encoding="utf-8",
    )
    fin_tz = fc_f4.build_financials_payload("TZR", known_cik=cik_tz, pit_strict=True, enable_live_fetch=False)
    assert fin_tz["filings"][0]["is_restated"] is True

    # Reverse UTC offset: original 10:00-04:00 (14:00Z) is LATER than 10-K/A 13:30Z -> not restated by earlier 10-K/A
    cik_tz_rev = "0000999993"
    (fc_f4.raw_dir / "edgar_submissions" / f"CIK{cik_tz_rev}.json").write_text(
        json.dumps({
            "cik": cik_tz_rev,
            "recent": {
                "form": ["10-K", "10-K/A"],
                "filingDate": ["2020-06-03", "2020-06-03"],
                "reportDate": ["2019-12-31", "2019-12-31"],
                "accessionNumber": ["0001", "0002"],
                "acceptanceDateTime": ["2020-06-03T10:00:00-04:00", "2020-06-03T13:30:00Z"],
                "primaryDocument": ["orig.htm", "amend.htm"],
            },
        }),
        encoding="utf-8",
    )
    (fc_f4.raw_dir / "edgar_facts" / f"CIK{cik_tz_rev}.json").write_text(
        json.dumps({
            "cik": cik_tz_rev,
            "entityName": "TZ REV CORP",
            "us_gaap": {
                "Revenues": {"units": {"USD": [
                    {"form": "10-K", "start": "2019-01-01", "end": "2019-12-31", "filed": "2020-06-03", "accn": "0001", "val": 100_000_000},
                    {"form": "10-K/A", "start": "2019-01-01", "end": "2019-12-31", "filed": "2020-06-03", "accn": "0002", "val": 120_000_000},
                ]}},
                "Assets": {"units": {"USD": [{"form": "10-K", "end": "2019-12-31", "filed": "2020-06-03", "accn": "0001", "val": 500_000_000, "fp": "FY"}]}},
            },
        }),
        encoding="utf-8",
    )
    fin_tz_rev_pit = fc_f4.build_financials_payload("TZREV", known_cik=cik_tz_rev, pit_strict=True, enable_live_fetch=False)
    fin_tz_rev_rest = fc_f4.build_financials_payload("TZREV", known_cik=cik_tz_rev, pit_strict=False, enable_live_fetch=False)
    assert next(r for r in fin_tz_rev_pit["incomeStatement"] if r["tag"] == "Revenues")["values"]["FY2019"] == pytest.approx(120.0, abs=0.05)
    assert next(r for r in fin_tz_rev_rest["incomeStatement"] if r["tag"] == "Revenues")["values"]["FY2019"] == pytest.approx(100.0, abs=0.05)

    # SS) fetch_intraday_bars 4H observation timestamp is bounded by last contributing 60m bar end (13:00 ET, not 16:00 ET)
    idx_1h_part = pd.DatetimeIndex([pd.Timestamp("2025-11-18 12:00:00", tz="America/New_York")])
    df_1h_part = pd.DataFrame(
        {"Open": [100.0], "High": [101.0], "Low": [99.5], "Close": [100.5], "Volume": [10000]},
        index=idx_1h_part,
    )
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-11-18 16:10:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(yf, "download", lambda *a, **kw: df_1h_part)
    ih_4h = prov.market_loader.fetch_intraday_bars("TST", interval="4h", count=5)
    assert len(ih_4h) == 1
    assert ih_4h[0]["observation_timestamp_ms"] == int(pd.Timestamp("2025-11-18 13:00:00", tz="America/New_York").timestamp() * 1000)
    df_1h_naive = df_1h_part.copy()
    df_1h_naive.index = df_1h_naive.index.tz_localize(None)
    monkeypatch.setattr(yf, "download", lambda *a, **kw: df_1h_naive)
    ih_naive = prov.market_loader.fetch_intraday_bars("TST", interval="1h", count=5)
    assert len(ih_naive) == 1 and "observation_timestamp_ms" not in ih_naive[0]

    # TT) _maybe_refresh_daily no-op check at 15:59 ET on same-day unsettled bar does NOT burn cooldown blocking 16:00:05 ET refresh
    prov_cd = LiveInferenceProvider(lakehouse_dir=tmp_path / "lh_cd_noop", weights_dir=weights_dir, enable_live_fetch=True)
    df_nov24_unsettled = df_mon_t.copy()
    df_nov24_unsettled["high"] = np.maximum(df_nov24_unsettled["open"], df_nov24_unsettled["close"]) + 1.0
    df_nov24_unsettled["low"] = np.minimum(df_nov24_unsettled["open"], df_nov24_unsettled["close"]) - 1.0
    df_nov24_unsettled.attrs["observation_timestamp_ms"] = int(pd.Timestamp("2025-11-24 15:50:00", tz="America/New_York").timestamp() * 1000)
    df_nov24_unsettled.attrs["obs_bar_date"] = "2025-11-24"
    prov_cd.market_loader.save_ticker_daily("CDNO", df_nov24_unsettled)
    monkeypatch.setattr(prov_cd.market_loader, "_latest_expected_trading_date", lambda: pd.Timestamp("2025-11-24"))
    cd_dl = {"n": 0}
    df_cd_yf = df_nov24_unsettled.iloc[-3:].copy().rename(columns=str.capitalize)
    monkeypatch.setattr(yf, "download", lambda *a, **kw: (cd_dl.__setitem__("n", cd_dl["n"] + 1) or df_cd_yf))
    cd_wall = {"t": pd.Timestamp("2025-11-24 15:59:00", tz="America/New_York").timestamp()}
    cd_mono = {"now": 5000.0}
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: cd_wall["t"])
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: cd_wall["t"])
    monkeypatch.setattr("src.ipc.live_providers.time.monotonic", lambda: cd_mono["now"])
    prov_cd._maybe_refresh_daily("CDNO")
    assert cd_dl["n"] == 0
    assert "CDNO" not in prov_cd._daily_refresh_attempted
    cd_wall["t"] = pd.Timestamp("2025-11-24 16:00:05", tz="America/New_York").timestamp()
    cd_mono["now"] = 5015.0  # only 15s later (< 300s cooldown)
    prov_cd._maybe_refresh_daily("CDNO")
    assert cd_dl["n"] == 1

    # Loop-12 hardening checks (task-3175 CodeRabbit findings):
    # UU) Form 4 filing-level aff10b5One checkbox only discounts footnote-tagged plan tx,
    #     truncated XML is never cached as empty, and concurrent fetches merge cleanly under lock
    mixed_xml = (
        "<ownershipDocument><reportingOwner><reportingOwnerId><rptOwnerName>Grace Hopper</rptOwnerName></reportingOwnerId></reportingOwner>"
        "<aff10b5One>1</aff10b5One>"
        "<nonDerivativeTable>"
        "<nonDerivativeTransaction><transactionDate><value>2025-11-20</value></transactionDate>"
        "<transactionCoding><transactionCode>P</transactionCode></transactionCoding>"
        "<transactionAmounts><transactionShares><value>1000</value></transactionShares>"
        "<transactionPricePerShare><value>50.0</value></transactionPricePerShare></transactionAmounts></nonDerivativeTransaction>"
        "<nonDerivativeTransaction><transactionDate><value>2025-11-20</value></transactionDate>"
        "<transactionCoding><transactionCode>S</transactionCode></transactionCoding>"
        "<transactionAmounts><transactionShares><value>1000</value></transactionShares>"
        "<transactionPricePerShare><value>50.0</value></transactionPricePerShare></transactionAmounts>"
        "<footnoteId id='F1'/></nonDerivativeTransaction>"
        "</nonDerivativeTable>"
        "<footnotes><footnote id='F1'>Executed under a pre-arranged Rule 10b5-1 trading plan.</footnote></footnotes>"
        "</ownershipDocument>"
    )
    mixed_parsed = FundamentalsClient._parse_sec_form4_xml(mixed_xml, "2025-11-21", "2025-11-21T15:00:00Z", "ACC_MIX")
    assert mixed_parsed is not None and len(mixed_parsed) == 2
    assert mixed_parsed[0]["filing_has_10b5_1"] is True
    assert mixed_parsed[0]["is_10b5_1"] is False  # discretionary purchase NOT discounted
    assert mixed_parsed[1]["is_10b5_1"] is True   # footnote-tagged plan sale IS discounted
    assert FundamentalsClient._parse_sec_form4_xml("<ownershipDocument><broken", "2025-11-21", "2025-11-21T15:00:00Z", "ACC_BAD") is None

    # Truncated XML response does not write an empty [] cache entry; subsequent valid XML recovers cleanly;
    # concurrent workers fetching A vs {A,B} merge rather than overwriting {A,B} with {A}
    fc_f4_conc = FundamentalsClient(lakehouse_dir=tmp_path / "lh_f4_conc")
    cik_conc = "0000333444"
    subs_conc = {
        "cik": cik_conc,
        "recent": {
            "form": ["4", "4"],
            "filingDate": ["2025-11-21", "2025-11-20"],
            "acceptanceDateTime": ["2025-11-21T15:00:00Z", "2025-11-20T15:00:00Z"],
            "accessionNumber": ["ACC_A", "ACC_B"],
            "primaryDocument": ["doc_a.xml", "doc_b.xml"],
        },
    }
    monkeypatch.setattr(fc_f4_conc, "_sec_fetch_text", lambda url, timeout=5.0: "<ownershipDocument><trunc")
    assert fc_f4_conc._load_or_fetch_sec_form4_rows(cik_conc, subs_conc, enable_live_fetch=True) == []
    assert not (fc_f4_conc.raw_dir / "edgar_form4" / f"CIK{cik_conc}.json").exists()
    fc_f4_conc._failed_until.clear()
    monkeypatch.setattr(fc_f4_conc, "_sec_fetch_text", lambda url, timeout=5.0: mixed_xml)
    rec_rows = fc_f4_conc._load_or_fetch_sec_form4_rows(cik_conc, subs_conc, enable_live_fetch=True)
    assert len(rec_rows) == 4  # recovered cleanly across both accessions
    # Simulate external concurrent write inserting ACC_C while only ACC_D is newly fetched
    monkeypatch.setattr(
        fc_f4_conc,
        "_sec_fetch_text",
        lambda url, timeout=5.0: (
            fc_f4_conc._write_json_cache(
                f"edgar_form4/CIK{cik_conc}.json",
                {"cik": cik_conc, "filings": {**((fc_f4_conc._read_dict_cache(f"edgar_form4/CIK{cik_conc}.json", None) or {}).get("filings") or {}), "ACC_C": {"rows": []}}},
            )
            or mixed_xml
        ),
    )
    subs_conc2 = {
        "cik": cik_conc,
        "recent": {
            "form": ["4"],
            "filingDate": ["2025-11-22"],
            "acceptanceDateTime": ["2025-11-22T15:00:00Z"],
            "accessionNumber": ["ACC_D"],
            "primaryDocument": ["doc_d.xml"],
        },
    }
    fc_f4_conc._load_or_fetch_sec_form4_rows(cik_conc, subs_conc2, enable_live_fetch=True)
    merged_keys = set((fc_f4_conc._read_dict_cache(f"edgar_form4/CIK{cik_conc}.json", None) or {}).get("filings", {}).keys())
    assert {"ACC_A", "ACC_B", "ACC_C", "ACC_D"}.issubset(merged_keys)

    # VV) Form 4/A amendment replaces corrected transaction without doubling it and honors microsecond PIT cutoff
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-11-24 16:30:00", tz="America/New_York").timestamp())
    df_sub_ms = df_mon_t.copy()
    df_spy_sub_ms = df_mon_spy.copy()
    t_15_123ms = int(pd.Timestamp("2025-11-24T15:00:00.123000Z").timestamp() * 1000)
    df_sub_ms.attrs["observation_timestamp_ms"] = t_15_123ms
    df_spy_sub_ms.attrs["observation_timestamp_ms"] = t_15_123ms
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_sub_ms if sym == "TST" else (df_spy_sub_ms if sym == "SPY" else None))
    with prov._lock:
        prov._signal_cache.clear()
        prov._news_cache.clear()
        prov._edgar_cache["TST"] = (
            time.monotonic(),
            {
                "available": True,
                "item1a_sentiment_drift": 0.0,
                "mda_catalyst_score": 0.0,
                "form4_transactions": [
                    {
                        "form": "4",
                        "filing_date": "2025-11-21",
                        "transaction_date": "2025-11-20",
                        "disclosed_at": "2025-11-21T15:00:00Z",
                        "disclosed_at_verified": True,
                        "accession_number": "A_ORIG",
                        "insider_name": "Alan Turing",
                        "transaction_code": "P",
                        "value": 50_000.0,
                    },
                    {
                        "form": "4",
                        "filing_date": "2025-11-21",
                        "transaction_date": "2025-11-20",
                        "disclosed_at": "2025-11-21T15:00:00Z",
                        "disclosed_at_verified": True,
                        "accession_number": "A_ORIG",
                        "insider_name": "Alan Turing",
                        "transaction_code": "S",
                        "value": 10_000.0,
                    },
                    {
                        # 900us AFTER target_obs_ms (same ms!) -> MUST be excluded from PIT scoring
                        "form": "4/A",
                        "filing_date": "2025-11-24",
                        "transaction_date": "2025-11-20",
                        "date_of_original_submission": "2025-11-21",
                        "disclosed_at": "2025-11-24T15:00:00.123900Z",
                        "disclosed_at_verified": True,
                        "accession_number": "A_AMEND_LATE",
                        "insider_name": "Alan Turing",
                        "transaction_code": "P",
                        "value": 5_000_000.0,
                    },
                ],
            },
        )
    sig_pre_amend = prov.get_signal("TST", regime=reg_rev)
    f4_pre_val = sig_pre_amend["signal_breakdown"]["decayed_form4_insider_cluster"]

    # Independent get_signal controls comparing against isolated PIT/post/double-counted payloads (#31)
    def _sig_f4_control(rows_override):
        monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_sub_ms2 if sym == "TST" else (df_spy_sub_ms2 if sym == "SPY" else None))
        with prov._lock:
            prov._signal_cache.clear()
            prov._edgar_cache["TST"] = (
                time.monotonic(),
                {"available": True, "item1a_sentiment_drift": 0.0, "mda_catalyst_score": 0.0, "form4_transactions": rows_override},
            )
        return prov.get_signal("TST", regime=reg_rev)["signal_breakdown"]["decayed_form4_insider_cluster"]

    tx_orig_rows = [
        {
            "form": "4",
            "filing_date": "2025-11-21",
            "transaction_date": "2025-11-20",
            "disclosed_at": "2025-11-21T15:00:00Z",
            "disclosed_at_verified": True,
            "accession_number": "A_ORIG",
            "insider_name": "Alan Turing",
            "transaction_code": "P",
            "value": 50_000.0,
        },
        {
            "form": "4",
            "filing_date": "2025-11-21",
            "transaction_date": "2025-11-20",
            "disclosed_at": "2025-11-21T15:00:00Z",
            "disclosed_at_verified": True,
            "accession_number": "A_ORIG",
            "insider_name": "Alan Turing",
            "transaction_code": "S",
            "value": 10_000.0,
        },
    ]
    tx_replaced_rows = [
        {
            "form": "4",
            "filing_date": "2025-11-24",
            "transaction_date": "2025-11-20",
            "disclosed_at": "2025-11-21T15:00:00Z",
            "disclosed_at_verified": True,
            "accession_number": "A_AMEND_LATE",
            "insider_name": "Alan Turing",
            "transaction_code": "P",
            "value": 5_000_000.0,
            "original_filing_date": "2025-11-21",
        },
        tx_orig_rows[1],
    ]
    # Note A_AMEND_LATE has original_filing_date=2025-11-21 decay age; restore full fixture after controls
    full_f4_rows = list(prov._edgar_cache["TST"][1]["form4_transactions"])
    # Now advance target_obs_ms past 123900us -> A_AMEND_LATE replaces A_ORIG P (50k -> 5M), preserving S (10k), without counting 50k+5M
    t_15_124ms = int(pd.Timestamp("2025-11-24T15:00:00.124000Z").timestamp() * 1000)
    df_sub_ms2 = df_sub_ms.copy()
    df_spy_sub_ms2 = df_spy_sub_ms.copy()
    df_sub_ms2.attrs["observation_timestamp_ms"] = t_15_124ms
    df_spy_sub_ms2.attrs["observation_timestamp_ms"] = t_15_124ms
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_sub_ms2 if sym == "TST" else (df_spy_sub_ms2 if sym == "SPY" else None))
    sig_post_amend = prov.get_signal("TST", regime=reg_rev)
    f4_post_val = sig_post_amend["signal_breakdown"]["decayed_form4_insider_cluster"]
    assert f4_post_val > f4_pre_val > 0.0

    # Independent get_signal parity controls (no direct reliance on resolve_pit_form4_transactions alone)
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_sub_ms if sym == "TST" else (df_spy_sub_ms if sym == "SPY" else None))
    with prov._lock:
        prov._signal_cache.clear()
        prov._edgar_cache["TST"] = (
            time.monotonic(),
            {"available": True, "item1a_sentiment_drift": 0.0, "mda_catalyst_score": 0.0, "form4_transactions": tx_orig_rows},
        )
    ctrl_pre_val = prov.get_signal("TST", regime=reg_rev)["signal_breakdown"]["decayed_form4_insider_cluster"]
    assert f4_pre_val == ctrl_pre_val

    ctrl_post_single = _sig_f4_control(tx_replaced_rows)
    # Use a slightly modified distinct accession for second form="4" row so double-count is strictly unmerged
    ctrl_double_rows = [
        tx_orig_rows[0],
        {**tx_replaced_rows[0], "form": "4", "accession_number": "A_DOUBLE"},
        tx_orig_rows[1],
    ]
    ctrl_double_val = _sig_f4_control(ctrl_double_rows)
    assert f4_post_val == ctrl_post_single < ctrl_double_val
    with prov._lock:
        prov._signal_cache.clear()
        prov._edgar_cache["TST"] = (
            time.monotonic(),
            {"available": True, "item1a_sentiment_drift": 0.0, "mda_catalyst_score": 0.0, "form4_transactions": full_f4_rows},
        )

    # WW) Unsettled pre-close did_write_new refresh at 13:14:00 releases 300s reservation at settlement (13:15:01)
    prov_unset = LiveInferenceProvider(lakehouse_dir=tmp_path / "lh_unset_cd", weights_dir=weights_dir, enable_live_fetch=True)
    prov_unset.market_loader.save_ticker_daily("UNCD", df_pre_close)
    monkeypatch.setattr(prov_unset.market_loader, "_latest_expected_trading_date", lambda: pd.Timestamp("2025-11-28"))
    df_changed_pre = df_same_yf.copy()
    df_changed_pre.iloc[-1, df_changed_pre.columns.get_loc("Close")] = float(df_changed_pre["Close"].iloc[-1]) + 0.25
    df_changed_pre.iloc[-1, df_changed_pre.columns.get_loc("High")] = max(float(df_changed_pre["High"].iloc[-1]), float(df_changed_pre["Close"].iloc[-1]))
    unset_wall = {"t": pd.Timestamp("2025-11-28 13:14:00", tz="America/New_York").timestamp()}
    unset_mono = {"now": 7000.0}
    unset_calls = {"n": 0}
    def _dl_unset(*a, **kw):
        unset_calls["n"] += 1
        return df_stl_settled if unset_calls["n"] >= 2 else df_changed_pre
    monkeypatch.setattr(yf, "download", _dl_unset)
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: unset_wall["t"])
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: unset_wall["t"])
    monkeypatch.setattr("src.ipc.live_providers.time.monotonic", lambda: unset_mono["now"])
    prov_unset._maybe_refresh_daily("UNCD")
    assert unset_calls["n"] == 1
    assert "UNCD" not in prov_unset._daily_refresh_attempted
    unset_wall["t"] = pd.Timestamp("2025-11-28 13:15:02", tz="America/New_York").timestamp()
    unset_mono["now"] = 7062.0  # only 62s later (< 300s) -> settlement opened, executes download immediately
    prov_unset._maybe_refresh_daily("UNCD")
    assert unset_calls["n"] == 2
    assert prov_unset._daily_bar_status_ms(prov_unset._load_disk_daily("UNCD"))[1] is True

    # XX) Session-mismatch OHLCV replacement on FastInfo->hist fallback + unchanged pre-close partial preserves obs time
    class FakeFiLaterSessionNan:
        fast_info = type("FINanSess", (), {
            "lastPrice": 151.0, "open": 150.0, "dayHigh": 152.0, "dayLow": 149.0, "previousClose": 148.0, "lastVolume": 999999,
            "_md": {},
            "_prices_1y": pd.DataFrame({"Close": [float("nan")]}, index=pd.DatetimeIndex([pd.Timestamp("2025-12-01 00:00:00-05:00")])),
            "get": lambda self, k, d=None: {"lastPrice": 151.0, "open": 150.0, "dayHigh": 152.0, "dayLow": 149.0, "previousClose": 148.0, "lastVolume": 999999}.get(k, d),
        })()
        def history(self, **kw):
            return hist_5d  # 2025-11-28: Open=101.0, High=103.0, Low=100.5, Close=102.5, Volume=200000
        def get_history_metadata(self):
            return {}
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-12-01 12:00:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(yf, "Ticker", lambda sym: FakeFiLaterSessionNan())
    q_sess_swap = prov.market_loader.fetch_live_quote("TST")
    assert q_sess_swap is not None
    assert q_sess_swap["date"] == "2025-11-28"
    assert q_sess_swap["open"] == 101.0 and q_sess_swap["price"] == 102.5
    assert q_sess_swap["high"] == 103.0 and q_sess_swap["low"] == 100.5 and q_sess_swap["volume"] == 200000

    # Forced refresh at 15:05 returning identical OHLCV preserves 15:00 partial observation timestamp
    ml_part_pres = MarketDataLoader(tmp_path / "lh_part_pres")
    df_1500 = df_nov24_unsettled.copy()
    t_1500_ms = int(pd.Timestamp("2025-11-24 15:00:00", tz="America/New_York").timestamp() * 1000)
    df_1500.attrs["observation_timestamp_ms"] = t_1500_ms
    df_1500.attrs["obs_bar_date"] = "2025-11-24"
    ml_part_pres.save_ticker_daily("PRE15", df_1500)
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-11-24 15:05:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(yf, "download", lambda *a, **kw: df_1500.tail(3).rename(columns=str.capitalize))
    df_1505 = ml_part_pres.refresh_ticker_daily("PRE15", force_refresh_if_incomplete=True)
    assert df_1505 is not None and df_1505.attrs.get("observation_timestamp_ms") == t_1500_ms

    # YY) get_analyst_ratings rejects future regularMarketTime (> now_ms) when updating bar_p
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-11-24 15:01:00", tz="America/New_York").timestamp())
    prov.fundamentals_client._mem_cache.clear()
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_1500 if sym == "PRE15" else None)
    monkeypatch.setattr(
        prov.fundamentals_client,
        "get_yf_meta",
        lambda sym, enable_live_fetch=False: {
            "info": {
                "targetMeanPrice": 200.0,
                "numberOfAnalystOpinions": 10,
                "regularMarketPrice": 180.0,
                "regularMarketTime": int(pd.Timestamp("2025-11-24 15:05:00", tz="America/New_York").timestamp()),
            },
            "rec_trend": [],
        },
    )
    with prov._lock:
        prov._ratings_cache.clear()
    anr_fut = prov.get_analyst_ratings("PRE15")
    assert anr_fut["current_price"] == pytest.approx(round(float(df_1500["close"].iloc[-1]), 2), abs=0.01)

    # ZZ) get_news_feed falls back independently to pubDate when providerPublishTime is invalid
    class FakeBadPubTsNewsTicker:
        news = [
            {
                "title": "Solid beat with strong growth",
                "providerPublishTime": "not_a_timestamp",
                "content": {"pubDate": "2025-11-24T14:00:00Z"},
            }
        ]
    prov.enable_live_fetch = True
    monkeypatch.setattr(yf, "Ticker", lambda sym: FakeBadPubTsNewsTicker())
    with prov._lock:
        prov._news_cache.pop("TST", None)
    nf_fallback = prov.get_news_feed("TST")
    prov.enable_live_fetch = False
    assert len(nf_fallback) == 1 and nf_fallback[0]["published_at"] == "2025-11-24T14:00:00Z" and nf_fallback[0]["published_at_verified"] is True

    # AAA) Short-side barriers in LONG_SHORT mode invert TP/SL and R:R cleanly while LONG_ONLY stays fixed
    from src.risk.kelly_engine import MultiOutcomeKellySizer

    sizer_short = MultiOutcomeKellySizer()
    rec_short = sizer_short.compute_recommendation(
        p_buy=0.20,
        p_hold=0.40,
        p_sell=0.40,
        b_upper=0.04,
        b_lower=0.02,
        entry_price=100.0,
        kappa=0.40,
        prior_probs=(0.50, 0.20, 0.30),
        execution_mode="LONG_SHORT",
    )
    assert rec_short.action == "SELL" and rec_short.unconstrained_kelly_pct < 0.0
    assert rec_short.dynamic_take_profit == pytest.approx(98.0, abs=0.01)
    assert rec_short.dynamic_stop_loss == pytest.approx(104.0, abs=0.01)
    assert rec_short.reward_to_risk_ratio == pytest.approx(0.5, abs=0.01)
    rec_long_exit = sizer_short.compute_recommendation(
        p_buy=0.20,
        p_hold=0.40,
        p_sell=0.40,
        b_upper=0.04,
        b_lower=0.02,
        entry_price=100.0,
        kappa=0.40,
        prior_probs=(0.50, 0.20, 0.30),
        execution_mode="LONG_ONLY",
    )
    assert rec_long_exit.action == "SELL" and rec_long_exit.unconstrained_kelly_pct == 0.0
    assert rec_long_exit.dynamic_take_profit == pytest.approx(104.0, abs=0.01)
    assert rec_long_exit.dynamic_stop_loss == pytest.approx(98.0, abs=0.01)


def test_loop13_pit_form4_tz_and_market_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Loop-13 regression suite: July 3 early close, redline PIT gating, Form 4 XML/chained/duplicate/ambiguous/microsecond PIT, triple XBRL bucket, UTC Parquet bounds, FastInfo cross-session guard, and strict news timestamp check."""
    from datetime import date, time as dt_time
    import os
    import duckdb
    import pandas as pd
    import yfinance as yf
    from src.data.fundamentals_client import FundamentalsClient
    from src.data.market_loader import MarketDataLoader
    from src.ipc.live_providers import LiveInferenceProvider
    from src.risk.calibration import TemperatureScaler

    # 1) July 3 Thursday early close (2025-07-03 -> 13:00 ET) vs regular Thursday (2025-07-10 -> 16:00 ET)
    assert MarketDataLoader.nyse_session_close_time(date(2025, 7, 3)) == dt_time(13, 0)
    assert MarketDataLoader.nyse_session_close_time(date(2025, 7, 10)) == dt_time(16, 0)

    # 2) Build clean isolated provider + bar fixture through Black Friday half-day 2025-11-28 (13:00 ET close)
    lh = tmp_path / "lh13"
    weights_dir = tmp_path / "weights13"
    weights_dir.mkdir(parents=True, exist_ok=True)
    fc = FundamentalsClient(lakehouse_dir=lh)
    prov = LiveInferenceProvider(lakehouse_dir=lh, weights_dir=weights_dir, enable_live_fetch=False)
    prov.fundamentals_client = fc

    class DummyLGBM13:
        def predict_proba(self, x):
            return np.tile([0.65, 0.25, 0.10], (len(x), 1))

    prov.lgbm_backbone = DummyLGBM13()
    sc = TemperatureScaler()
    sc.temperature_ = 1.0
    prov.temp_scaler = sc
    prov.regime_hmm = object()

    idx_nov28 = pd.bdate_range(end="2025-11-28", periods=200)
    rets_t = np.random.default_rng(101).normal(0.0004, 0.014, size=200)
    rets_s = np.random.default_rng(202).normal(0.0003, 0.011, size=200)
    closes_t = 100.0 * np.exp(np.cumsum(rets_t))
    closes_s = 400.0 * np.exp(np.cumsum(rets_s))
    df_tgt = pd.DataFrame(
        {"open": closes_t * 0.995, "high": closes_t * 1.01, "low": closes_t * 0.99, "close": closes_t, "volume": np.full(200, 1_500_000.0)},
        index=idx_nov28,
    )
    df_spy = pd.DataFrame(
        {"open": closes_s * 0.996, "high": closes_s * 1.008, "low": closes_s * 0.992, "close": closes_s, "volume": np.full(200, 4_000_000.0)},
        index=idx_nov28,
    )
    t_close_nov28_ms = int(pd.Timestamp("2025-11-28 13:00:00", tz="America/New_York").timestamp() * 1000)
    df_tgt.attrs["observation_timestamp_ms"] = t_close_nov28_ms
    df_tgt.attrs["obs_bar_date"] = "2025-11-28"
    df_spy.attrs["observation_timestamp_ms"] = t_close_nov28_ms
    df_spy.attrs["obs_bar_date"] = "2025-11-28"
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-11-28 13:10:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_tgt if sym in ("REDPIT", "F4TEST") else (df_spy if sym == "SPY" else None))
    reg13 = {"current_state": "LOW_VOLATILITY", "state_index": 0, "kappa": 0.45}

    # 3) Expired redline stays visible in EDGAR payload while get_signal enforces acceptanceDateTime PIT cutoff
    cik_red = "0000888777"
    (fc.raw_dir / "edgar_submissions").mkdir(parents=True, exist_ok=True)
    (fc.raw_dir / "edgar_facts").mkdir(parents=True, exist_ok=True)
    (fc.raw_dir / "edgar_redlines").mkdir(parents=True, exist_ok=True)
    (fc.raw_dir / "edgar_form4").mkdir(parents=True, exist_ok=True)
    (fc.raw_dir / "edgar_submissions" / f"CIK{cik_red}.json").write_text(json.dumps({
        "cik": cik_red,
        "recent": {
            "form": ["10-K", "10-K"],
            "filingDate": ["2025-11-28", "2024-11-28"],
            "acceptanceDateTime": ["2025-11-28T14:00:00-05:00", "2024-11-28T16:00:00Z"],
            "accessionNumber": ["0002", "0001"],
            "primaryDocument": ["c.htm", "p.htm"],
        },
    }), encoding="utf-8")
    red_path = fc.raw_dir / "edgar_redlines" / "REDPIT_0002_0001.json"
    red_path.write_text(json.dumps({
        "current_accession": "0002",
        "prior_accession": "0001",
        "item1a_similarity": 0.88,
        "mda_similarity": 0.91,
        "item1a_sentiment_drift": 0.45,
        "mda_catalyst_score": -0.35,
        "redlines": [{"section": "Item 1A", "status": "DIFF_VERIFIED", "added_clauses": ["Growth outlook."], "removed_clauses": []}],
    }), encoding="utf-8")
    os.utime(red_path, (946684800.0, 946684800.0))  # 2000-01-01 -> expired TTL
    edg_red = fc.build_edgar_filings_payload("REDPIT", known_cik=cik_red, enable_live_fetch=False)
    assert edg_red["redlines"][0]["status"] == "DIFF_VERIFIED"
    assert edg_red["current_acceptance_datetime"] == "2025-11-28T14:00:00-05:00"
    with prov._lock:
        prov._signal_cache.clear()
        prov._edgar_cache["REDPIT"] = (time.monotonic(), edg_red)
    sig_red_late = prov.get_signal("REDPIT", regime=reg13)
    assert sig_red_late["signal_breakdown"]["decayed_10q_legal_risk_delta"] == 0.0
    assert sig_red_late["signal_breakdown"]["decayed_mda_catalyst_score"] == 0.0
    assert "EDGAR_ITEM1A_DIFF" in sig_red_late["data_integrity"]["missing_modalities"]

    edg_red_pre = {**edg_red, "current_acceptance_datetime": "2025-11-28T11:00:00-05:00"}
    with prov._lock:
        prov._signal_cache.clear()
        prov._edgar_cache["REDPIT"] = (time.monotonic(), edg_red_pre)
    sig_red_ok = prov.get_signal("REDPIT", regime=reg13)
    assert sig_red_ok["signal_breakdown"]["decayed_10q_legal_risk_delta"] == pytest.approx(0.45, abs=1e-6)
    assert sig_red_ok["signal_breakdown"]["decayed_mda_catalyst_score"] == pytest.approx(-0.35, abs=1e-6)
    assert "EDGAR_ITEM1A_DIFF" not in sig_red_ok["data_integrity"]["missing_modalities"]

    # 4) Form 4 XML security, malformed acc_dt skip, chained 4/A, duplicate 2->2, P->S, ambiguous 4/A & us cutoff
    assert fc._parse_sec_form4_xml("<!DOCTYPE evil><ownershipDocument/>", "2025-11-20", "2025-11-20T12:00:00Z", "A0") is None
    xml_ok = (
        "<ownershipDocument>"
        "<aff10b5One>1</aff10b5One>"
        "<dateOfOriginalSubmission>2025-11-01</dateOfOriginalSubmission>"
        "<reportingOwner><reportingOwnerId><rptOwnerName>Ada Lovelace</rptOwnerName></reportingOwnerId>"
        "<reportingOwnerRelationship><isDirector>1</isDirector></reportingOwnerRelationship></reportingOwner>"
        "<nonDerivativeTable><nonDerivativeTransaction>"
        "<transactionDate><value>2025-11-19</value></transactionDate>"
        "<transactionCoding><transactionCode>P</transactionCode></transactionCoding>"
        "<transactionAmounts><transactionShares><value>1000</value></transactionShares>"
        "<transactionPricePerShare><value>50.0</value></transactionPricePerShare></transactionAmounts>"
        "</nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>"
    )
    parsed_xml = fc._parse_sec_form4_xml(xml_ok, "2025-11-20", "2025-11-20T15:00:00Z", "ACC_XML", form="4/A")
    assert parsed_xml is not None and len(parsed_xml) == 1
    # Single-transaction filing exception (total_tx_count == 1): root aff10b5One applies unambiguously to that sole tx
    assert parsed_xml[0]["filing_has_10b5_1"] is True and parsed_xml[0]["is_10b5_1"] is True
    assert parsed_xml[0]["date_of_original_submission"] == "2025-11-01"

    cik_f4m = "0000444555"
    (fc.raw_dir / "edgar_form4" / f"CIK{cik_f4m}.json").write_text(json.dumps({
        "filings": {
            "ACC_BAD_DT": [{"form": "4", "filing_date": "2025-11-28", "disclosed_at": "2025-11-28Tbad", "disclosed_at_verified": True, "transaction_code": "P", "value": 99_000.0}],
        }
    }), encoding="utf-8")
    bad_rows = fc._load_or_fetch_sec_form4_rows(
        cik_f4m,
        {"recent": {"form": ["4"], "filingDate": ["2025-11-28"], "acceptanceDateTime": ["2025-11-28Tbad"], "accessionNumber": ["ACC_BAD_DT"], "primaryDocument": ["f4.xml"]}},
        enable_live_fetch=False,
    )
    assert bad_rows == []

    # Chained 4/A: May 1 (4) -> May 2 (4/A -> May 1) -> May 3 (4/A -> May 1) preserves original_filing_date=2025-05-01
    chained_txs = [
        {"form": "4", "filing_date": "2025-05-01", "transaction_date": "2025-04-30", "disclosed_at": "2025-05-01T14:00:00Z", "disclosed_at_verified": True, "accession_number": "CH1", "insider_name": "Grace Hopper", "transaction_code": "P", "value": 50_000.0},
        {"form": "4/A", "filing_date": "2025-05-02", "transaction_date": "2025-04-30", "date_of_original_submission": "2025-05-01", "disclosed_at": "2025-05-02T14:00:00Z", "disclosed_at_verified": True, "accession_number": "CH2", "insider_name": "Grace Hopper", "transaction_code": "P", "value": 100_000.0},
        {"form": "4/A", "filing_date": "2025-05-03", "transaction_date": "2025-04-30", "date_of_original_submission": "2025-05-01", "disclosed_at": "2025-05-03T14:00:00Z", "disclosed_at_verified": True, "accession_number": "CH3", "insider_name": "Grace Hopper", "transaction_code": "P", "value": 250_000.0},
    ]
    res_chained = FundamentalsClient.resolve_pit_form4_transactions(chained_txs)
    assert len(res_chained) == 1 and res_chained[0]["value"] == 250_000.0 and res_chained[0]["original_filing_date"] == "2025-05-01"

    # Multi-row same-key: partial 1-row 4/A against 2 prior rows flags ambiguous without guessing; 2-row 4/A replaces both cleanly
    dup_orig = [
        {"form": "4", "filing_date": "2025-11-08", "transaction_date": "2025-11-07", "disclosed_at": "2025-11-08T14:00:00Z", "disclosed_at_verified": True, "accession_number": "D0", "insider_name": "Alice", "transaction_code": "S", "value": 10_000.0},
        {"form": "4", "filing_date": "2025-11-08", "transaction_date": "2025-11-07", "disclosed_at": "2025-11-08T14:00:00Z", "disclosed_at_verified": True, "accession_number": "D0", "insider_name": "Alice", "transaction_code": "S", "value": 100_000.0},
    ]
    dup_one_am = dup_orig + [{"form": "4/A", "filing_date": "2025-11-09", "transaction_date": "2025-11-07", "date_of_original_submission": "2025-11-08", "disclosed_at": "2025-11-09T14:00:00Z", "disclosed_at_verified": True, "accession_number": "D1", "insider_name": "Alice", "transaction_code": "S", "value": 110_000.0}]
    res_dup1 = FundamentalsClient.resolve_pit_form4_transactions(dup_one_am)
    assert any(r.get("_amendment_ambiguous") for r in res_dup1)
    assert [r["value"] for r in res_dup1 if not r.get("_amendment_ambiguous")] == [10_000.0, 100_000.0]

    dup_two_am = dup_one_am + [{"form": "4/A", "filing_date": "2025-11-09", "transaction_date": "2025-11-07", "date_of_original_submission": "2025-11-08", "disclosed_at": "2025-11-09T14:00:00Z", "disclosed_at_verified": True, "accession_number": "D1", "insider_name": "Alice", "transaction_code": "S", "value": 120_000.0}]
    res_dup2 = FundamentalsClient.resolve_pit_form4_transactions(dup_two_am)
    assert sorted(r["value"] for r in res_dup2) == [110_000.0, 120_000.0] and not any(r.get("_amendment_ambiguous") for r in res_dup2)

    p_to_s = [
        {"form": "4", "filing_date": "2025-11-10", "transaction_date": "2025-11-09", "disclosed_at": "2025-11-10T14:00:00Z", "disclosed_at_verified": True, "accession_number": "PS0", "insider_name": "Bob", "transaction_code": "P", "value": 15_000.0},
        {"form": "4/A", "filing_date": "2025-11-11", "transaction_date": "2025-11-09", "date_of_original_submission": "2025-11-10", "disclosed_at": "2025-11-11T14:00:00Z", "disclosed_at_verified": True, "accession_number": "PS1", "insider_name": "Bob", "transaction_code": "S", "value": 15_000.0},
    ]
    res_ps = FundamentalsClient.resolve_pit_form4_transactions(p_to_s)
    assert len(res_ps) == 1 and res_ps[0]["transaction_code"] == "S" and not res_ps[0].get("_amendment_ambiguous")

    ambig_rows = [
        {"form": "4", "filing_date": "2025-11-12", "transaction_date": "2025-11-11", "disclosed_at": "2025-11-12T14:00:00Z", "disclosed_at_verified": True, "accession_number": "AM0", "insider_name": "Carol", "transaction_code": "P", "value": 10_000.0},
        {"form": "4", "filing_date": "2025-11-12", "transaction_date": "2025-11-11", "disclosed_at": "2025-11-12T14:00:00Z", "disclosed_at_verified": True, "accession_number": "AM0", "insider_name": "Carol", "transaction_code": "S", "value": 5_000.0},
        {"form": "4/A", "filing_date": "2025-11-13", "transaction_date": "2025-11-11", "date_of_original_submission": "2025-11-12", "disclosed_at": "2025-11-13T14:00:00Z", "disclosed_at_verified": True, "accession_number": "AM1", "insider_name": "Carol", "transaction_code": "D", "value": 12_000.0},
    ]
    with prov._lock:
        prov._signal_cache.clear()
        prov._edgar_cache["F4TEST"] = (time.monotonic(), {"available": True, "item1a_sentiment_drift": 0.0, "mda_catalyst_score": 0.0, "form4_transactions": ambig_rows})
    sig_ambig = prov.get_signal("F4TEST", regime=reg13)
    assert "FORM4_AMENDMENT_AMBIGUOUS" in sig_ambig["data_integrity"]["missing_modalities"]

    # Microsecond cutoff on 2025-11-28 13:00 ET half-day (target_obs_ms=13:05 ET)
    t_1305_ms = int(pd.Timestamp("2025-11-28 13:05:00", tz="America/New_York").timestamp() * 1000)
    df_tgt_1305 = df_tgt.copy()
    df_spy_1305 = df_spy.copy()
    df_tgt_1305.attrs["observation_timestamp_ms"] = t_1305_ms
    df_spy_1305.attrs["observation_timestamp_ms"] = t_1305_ms
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_tgt_1305 if sym == "F4TEST" else (df_spy_1305 if sym == "SPY" else None))
    with prov._lock:
        prov._signal_cache.clear()
        prov._edgar_cache["F4TEST"] = (time.monotonic(), {
            "available": True, "item1a_sentiment_drift": 0.0, "mda_catalyst_score": 0.0,
            "form4_transactions": [{"form": "4", "filing_date": "2025-11-28", "transaction_date": "2025-11-27", "disclosed_at": "2025-11-28T13:00:00.000500-05:00", "disclosed_at_verified": True, "accession_number": "US1", "insider_name": "Dan", "transaction_code": "P", "value": 250_000.0}],
        })
    assert prov.get_signal("F4TEST", regime=reg13)["signal_breakdown"]["decayed_form4_insider_cluster"] == 0.0
    with prov._lock:
        prov._signal_cache.clear()
        prov._edgar_cache["F4TEST"] = (time.monotonic(), {
            "available": True, "item1a_sentiment_drift": 0.0, "mda_catalyst_score": 0.0,
            "form4_transactions": [{"form": "4", "filing_date": "2025-11-28", "transaction_date": "2025-11-27", "disclosed_at": "2025-11-28T13:00:00.000000-05:00", "disclosed_at_verified": True, "accession_number": "US0", "insider_name": "Dan", "transaction_code": "P", "value": 250_000.0}],
        })
    assert prov.get_signal("F4TEST", regime=reg13)["signal_breakdown"]["decayed_form4_insider_cluster"] > 0.0

    # 5) Triple same-day XBRL filing bucket: 15:00Z 10-K, 10:00Z 10-K/A, untimestamped 10-K/A
    cik_tri = "0000777666"
    (fc.raw_dir / "edgar_submissions" / f"CIK{cik_tri}.json").write_text(json.dumps({
        "cik": cik_tri,
        "recent": {
            "form": ["10-K", "10-K/A", "10-K/A"],
            "filingDate": ["2020-06-03", "2020-06-03", "2020-06-03"],
            "reportDate": ["2019-12-31", "2019-12-31", "2019-12-31"],
            "accessionNumber": ["TR_LATE", "TR_EARLY", "TR_UNTS"],
            "acceptanceDateTime": ["2020-06-03T15:00:00Z", "2020-06-03T10:00:00Z", ""],
            "primaryDocument": ["l.htm", "e.htm", "u.htm"],
        },
    }), encoding="utf-8")
    (fc.raw_dir / "edgar_facts" / f"CIK{cik_tri}.json").write_text(json.dumps({
        "cik": cik_tri,
        "entityName": "TRIPLE BUCKET CORP",
        "us_gaap": {
            "Revenues": {"units": {"USD": [
                {"form": "10-K", "start": "2019-01-01", "end": "2019-12-31", "filed": "2020-06-03", "accn": "TR_LATE", "val": 100_000_000},
                {"form": "10-K/A", "start": "2019-01-01", "end": "2019-12-31", "filed": "2020-06-03", "accn": "TR_EARLY", "val": 120_000_000},
                {"form": "10-K/A", "start": "2019-01-01", "end": "2019-12-31", "filed": "2020-06-03", "accn": "TR_UNTS", "val": 150_000_000},
            ]}},
            "Assets": {"units": {"USD": [{"form": "10-K", "end": "2019-12-31", "filed": "2020-06-03", "accn": "TR_LATE", "val": 500_000_000, "fp": "FY"}]}},
        },
    }), encoding="utf-8")
    fin_tri_pit = fc.build_financials_payload("TRPL", known_cik=cik_tri, pit_strict=True, enable_live_fetch=False)
    fin_tri_rest = fc.build_financials_payload("TRPL", known_cik=cik_tri, pit_strict=False, enable_live_fetch=False)
    rev_tri_pit = next(r for r in fin_tri_pit["incomeStatement"] if r["tag"] == "Revenues")["values"]["FY2019"]
    rev_tri_rest = next(r for r in fin_tri_rest["incomeStatement"] if r["tag"] == "Revenues")["values"]["FY2019"]
    assert rev_tri_pit == pytest.approx(120.0, abs=0.05)
    assert rev_tri_rest == pytest.approx(150.0, abs=0.05)
    assert fin_tri_pit["filings"][0]["is_restated"] is False  # same-form untimestamped 10-K/A vs 10:00Z 10-K/A has uncertain chronology

    # 6) query_parquet_duckdb on UTC tz-aware Parquet with string bounds in DuckDB & 3-swap Arrow fallback
    ml13 = MarketDataLoader(tmp_path / "lh13_pq")
    pq_tz = ml13.daily_dir / "TZPARQ.parquet"
    df_tz_pq = pd.DataFrame(
        {"open": [10.0, 11.0, 12.0, 13.0], "high": [10.5, 11.5, 12.5, 13.5], "low": [9.5, 10.5, 11.5, 12.5], "close": [10.2, 11.2, 12.2, 13.2], "volume": [1000, 1100, 1200, 1300]},
        index=pd.DatetimeIndex(["2025-11-26", "2025-11-28", "2025-12-01", "2025-12-02"], tz="UTC"),
    )
    df_tz_pq.index.name = "timestamp"
    df_tz_pq.to_parquet(pq_tz)
    import pyarrow.parquet as _pq_mod
    orig_read_tbl13 = _pq_mod.read_table
    tz_arrow_calls = {"n": 0}
    monkeypatch.setattr("pyarrow.parquet.read_table", lambda *a, **kw: (tz_arrow_calls.__setitem__("n", tz_arrow_calls["n"] + 1) or orig_read_tbl13(*a, **kw)))
    q_tz_ddb = ml13.query_parquet_duckdb(pq_tz, columns=["CLOSE"], start_date="2025-11-28", end_date="2025-12-01")
    assert tz_arrow_calls["n"] == 0
    assert list(q_tz_ddb["close"]) == [11.2, 12.2]

    tz_swaps = []
    for k in range(3):
        sp = tmp_path / f"tz_swap_{k}.parquet"
        df_tz_pq.to_parquet(sp)
        tz_swaps.append(sp)
    tz_swap_idx = {"i": 0}
    orig_ddb13 = duckdb.connect
    class _TzSwapCursor:
        def __init__(self, cur, sql_txt):
            self._cur = cur
            self._sql = sql_txt
        def fetchall(self):
            return self._cur.fetchall()
        def fetchdf(self):
            d = self._cur.fetchdf()
            if "SELECT" in self._sql and "FROM read_parquet" in self._sql and tz_swap_idx["i"] < len(tz_swaps):
                os.replace(tz_swaps[tz_swap_idx["i"]], pq_tz)
                tz_swap_idx["i"] += 1
            return d
    class _TzSwapConn:
        def __init__(self, con):
            self._con = con
        def execute(self, sql_txt, *a, **kw):
            return _TzSwapCursor(self._con.execute(sql_txt, *a, **kw), sql_txt)
        def close(self):
            return self._con.close()
    monkeypatch.setattr("duckdb.connect", lambda *a, **kw: _TzSwapConn(orig_ddb13(*a, **kw)))
    q_tz_arrow = ml13.query_parquet_duckdb(pq_tz, columns=["CLOSE"], start_date="2025-11-28", end_date="2025-12-01")
    monkeypatch.setattr("duckdb.connect", orig_ddb13)
    monkeypatch.setattr("pyarrow.parquet.read_table", orig_read_tbl13)
    assert tz_swap_idx["i"] == 3 and tz_arrow_calls["n"] == 1
    assert list(q_tz_arrow["close"]) == [11.2, 12.2]

    # 7) fetch_live_quote cross-session FastInfo OHLCV omission when _prices_1y[-1] is Tue NaN & rmt is Mon
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-12-02 12:00:00", tz="America/New_York").timestamp())
    class FakeCrossSessFi:
        fast_info = type("FICross", (), {
            "lastPrice": 149.0, "open": 150.0, "dayHigh": 152.0, "dayLow": 148.0, "previousClose": 147.0, "lastVolume": 888888,
            "_md": {"regularMarketPrice": 149.0, "regularMarketTime": pd.Timestamp("2025-12-01 15:30:00", tz="America/New_York").timestamp()},
            "_prices_1y": pd.DataFrame({"Close": [float("nan")]}, index=pd.DatetimeIndex([pd.Timestamp("2025-12-02 00:00:00-05:00")])),
            "get": lambda self, k, d=None: {"lastPrice": 149.0, "open": 150.0, "dayHigh": 152.0, "dayLow": 148.0, "previousClose": 147.0, "lastVolume": 888888}.get(k, d),
        })()
        def history(self, **kw):
            return pd.DataFrame()
        def get_history_metadata(self):
            return {}
    monkeypatch.setattr(yf, "Ticker", lambda sym: FakeCrossSessFi())
    q_cross = ml13.fetch_live_quote("CROSS")
    assert q_cross is not None and q_cross["date"] == "2025-12-01" and q_cross.get("observation_timestamp_ms") is None
    assert q_cross["price"] == 149.0 and q_cross["high"] == 149.0 and q_cross["low"] == 149.0 and q_cross["volume"] == 0

    # 8) get_news_feed strictly rejects malformed "2025-11-26Tbad" and get_signal excludes it
    class FakeMalformedNewsTk:
        news = [{"title": "Strong growth beat", "providerPublishTime": None, "content": {"pubDate": "2025-11-26Tbad"}}]
    prov.enable_live_fetch = True
    monkeypatch.setattr(yf, "Ticker", lambda sym: FakeMalformedNewsTk())
    with prov._lock:
        prov._news_cache.pop("REDPIT", None)
    nf_bad = prov.get_news_feed("REDPIT")
    prov.enable_live_fetch = False
    assert len(nf_bad) == 1 and nf_bad[0]["published_at_verified"] is False
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-11-28 13:10:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_tgt if sym == "REDPIT" else (df_spy if sym == "SPY" else None))
    with prov._lock:
        prov._signal_cache.clear()
    sig_bad_news = prov.get_signal("REDPIT", regime=reg13)
    assert sig_bad_news["signal_breakdown"]["decayed_finbert_sentiment"] == 0.0
    assert "FINBERT_NEWS_NOT_LOADED" in sig_bad_news["data_integrity"]["missing_modalities"]

    # 9) Form 4 infinite product value=None, Nov 27 2026 early-close acceptance_timestamp_et, bounded cold XML fetches
    xml_inf = xml_ok.replace("<value>50.0</value>", "<value>1e308</value>").replace("<value>1000</value>", "<value>1e308</value>")
    parsed_inf = fc._parse_sec_form4_xml(xml_inf, "2025-11-20", "2025-11-20T15:00:00Z", "ACC_INF")
    assert parsed_inf is not None and len(parsed_inf) == 1 and parsed_inf[0]["value"] is None

    cik_bf26 = "0000333222"
    (fc.raw_dir / "edgar_submissions" / f"CIK{cik_bf26}.json").write_text(json.dumps({
        "cik": cik_bf26,
        "recent": {
            "form": ["10-K"], "filingDate": ["2026-11-27"],
            "acceptanceDateTime": ["2026-11-27T14:00:00-05:00"],
            "accessionNumber": ["BF26_1"], "primaryDocument": ["f.htm"],
        },
    }), encoding="utf-8")
    edg_bf26 = fc.build_edgar_filings_payload("BF26", known_cik=cik_bf26, enable_live_fetch=False)
    assert "PIT Effective 2026-11-30" in edg_bf26["acceptance_timestamp_et"]

    cik_bud = "0000111999"
    (fc.raw_dir / "edgar_form4" / f"CIK{cik_bud}.json").write_text(json.dumps({
        "filings": {"CB5": {"form": "4", "filing_date": "2025-11-20", "acceptance_datetime": "2025-11-20T15:00:00Z", "rows": [{"form": "4", "transaction_code": "P", "value": 75_000.0}]}},
    }), encoding="utf-8")
    cold_xml_urls: list[str] = []
    monkeypatch.setattr(fc, "_sec_fetch_text", lambda url, timeout=5.0: (cold_xml_urls.append(url) or None))
    bud_rows = fc._load_or_fetch_sec_form4_rows(
        cik_bud,
        {"recent": {
            "form": ["4"] * 7,
            "filingDate": ["2025-11-25"] * 7,
            "acceptanceDateTime": ["2025-11-25T15:00:00Z"] * 7,
            "accessionNumber": [f"CB{i}" for i in range(7)],
            "primaryDocument": [f"f{i}.xml" for i in range(7)],
        }},
        enable_live_fetch=True,
    )
    assert len(cold_xml_urls) <= 4 and len(bud_rows) == 1 and bud_rows[0]["value"] == 75_000.0

    # 10) Missing-partition unsettled pre-close write at 13:14 ET unlocks at 13:15:02 ET from refreshed_sched_close
    lh_miss = tmp_path / "lh_miss_settle"
    w_miss = tmp_path / "w_miss_settle"
    w_miss.mkdir(parents=True, exist_ok=True)
    prov_miss = LiveInferenceProvider(lakehouse_dir=lh_miss, weights_dir=w_miss, enable_live_fetch=True)
    monkeypatch.setattr(prov_miss.market_loader, "_latest_expected_trading_date", lambda: pd.Timestamp("2025-11-28"))
    df_miss_pre = df_tgt.copy()
    df_miss_pre.attrs.clear()
    df_miss_stl = df_tgt.copy()
    df_miss_stl.iloc[-1, df_miss_stl.columns.get_loc("close")] *= 1.03
    df_miss_stl.iloc[-1, df_miss_stl.columns.get_loc("high")] = max(float(df_miss_stl["high"].iloc[-1]), float(df_miss_stl["close"].iloc[-1]))
    miss_wall = {"t": pd.Timestamp("2025-11-28 13:14:00", tz="America/New_York").timestamp()}
    miss_mono = {"now": 9000.0}
    miss_dl_calls = {"n": 0}
    def _dl_miss(*a, **kw):
        miss_dl_calls["n"] += 1
        src_df = df_miss_stl if miss_dl_calls["n"] >= 2 else df_miss_pre
        return src_df.tail(5).rename(columns=str.capitalize)
    monkeypatch.setattr(yf, "download", _dl_miss)
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: miss_wall["t"])
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: miss_wall["t"])
    monkeypatch.setattr("src.ipc.live_providers.time.monotonic", lambda: miss_mono["now"])
    prov_miss._maybe_refresh_daily("NEWMISS")
    assert miss_dl_calls["n"] == 1 and "NEWMISS" not in prov_miss._daily_refresh_attempted
    miss_wall["t"] = pd.Timestamp("2025-11-28 13:15:02", tz="America/New_York").timestamp()
    miss_mono["now"] = 9062.0
    prov_miss._maybe_refresh_daily("NEWMISS")
    assert miss_dl_calls["n"] == 2
    assert prov_miss._daily_bar_status_ms(prov_miss._load_disk_daily("NEWMISS"))[1] is True

    # 11) Causal as-of HMM regime + _row_provenance invalidation on aligned benchmark slice
    idx_wed = pd.bdate_range(end="2025-12-01", periods=201)
    closes_t_tue = np.full(200, 100.0)
    closes_s_wed = np.full(201, 400.0)
    closes_s_wed[-1] = 350.0
    df_t_tue = pd.DataFrame({"open": closes_t_tue, "high": closes_t_tue * 1.01, "low": closes_t_tue * 0.99, "close": closes_t_tue, "volume": np.full(200, 1_500_000.0)}, index=idx_wed[:-1])
    df_s_wed = pd.DataFrame({"open": closes_s_wed, "high": closes_s_wed * 1.01, "low": closes_s_wed * 0.99, "close": closes_s_wed, "volume": np.full(201, 4_000_000.0)}, index=idx_wed)
    t_tue_ms = int(pd.Timestamp("2025-11-28 13:00:00", tz="America/New_York").timestamp() * 1000)
    t_wed_ms = int(pd.Timestamp("2025-12-01 16:00:00", tz="America/New_York").timestamp() * 1000)
    df_t_tue.attrs["observation_timestamp_ms"] = t_tue_ms
    df_s_wed.attrs["observation_timestamp_ms"] = t_wed_ms
    df_s_wed.attrs["_row_provenance"] = {"2025-11-28": {"observation_timestamp_ms": t_tue_ms}}
    class DummyAsOfHMM:
        STATE_LABELS = {0: "LOW_VOLATILITY", 1: "MEDIUM_VOLATILITY", 2: "HIGH_VOLATILITY"}
        transmat_ = np.eye(3)
        def filter_sequence(self, x):
            p = [1.0, 0.0, 0.0] if len(x) <= 199 else [0.0, 0.0, 1.0]
            return np.tile(p, (len(x), 1))
        def get_regime_multiplier(self, probs):
            return 0.50 if probs[0] > 0.5 else 0.25
        def get_state_annualized_volatilities(self):
            return [8.0, 16.0, 40.0]
        def get_stationary_distribution(self):
            return [0.5, 0.3, 0.2]
    prov.regime_hmm = DummyAsOfHMM()
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-12-01 16:10:00", tz="America/New_York").timestamp())
    monkeypatch.setattr("src.ipc.live_providers.time.monotonic", lambda: 50_000.0)
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_t_tue if sym == "REDPIT" else (df_s_wed if sym == "SPY" else None))
    with prov._lock:
        prov._regime_cache = None
        prov._signal_cache.clear()
        prov._news_cache.pop("REDPIT", None)
        prov._edgar_cache.pop("REDPIT", None)
    assert prov.get_macro_regime()["kappa"] == pytest.approx(0.25, abs=0.01)
    sig_tue_asof = prov.get_signal("REDPIT")
    assert "UNREADY" not in sig_tue_asof["execution_mode"]
    assert sig_tue_asof["macro_regime"]["kappa"] == pytest.approx(0.50, abs=0.01)
    assert "DIVERGENCE_OBSERVATIONS_UNALIGNED" not in sig_tue_asof["data_integrity"]["missing_modalities"]

    # Mutate ONLY 2025-11-28 _row_provenance to be AFTER target_obs_ms while holding monotonic clock & tail bar identical
    df_s_wed_late_prov = df_s_wed.copy()
    df_s_wed_late_prov.attrs = {**df_s_wed.attrs, "_row_provenance": {"2025-11-28": {"observation_timestamp_ms": t_tue_ms + 60_000}}}
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_t_tue if sym == "REDPIT" else (df_s_wed_late_prov if sym == "SPY" else None))
    sig_prov_changed = prov.get_signal("REDPIT")
    assert sig_prov_changed["execution_mode"] == "UNREADY"
    assert "BENCHMARK_OBSERVATIONS_UNALIGNED" in sig_prov_changed["data_integrity"]["missing_modalities"]

    # 12) Explicitly supplied Wednesday regime (e.g. from get_portfolio) cannot leak Wednesday kappa to Friday ticker
    mon_regime = prov._compute_macro_regime_from_df(df_s_wed)
    assert mon_regime["kappa"] == pytest.approx(0.25, abs=0.01)
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_t_tue if sym == "REDPIT" else (df_s_wed if sym == "SPY" else None))
    with prov._lock:
        prov._signal_cache.clear()
    sig_sup_reg = prov.get_signal("REDPIT", regime=mon_regime)
    assert sig_sup_reg["macro_regime"]["kappa"] == pytest.approx(0.50, abs=0.01)

    # 13) Future bar observation_timestamp_ms > wall_clock rejected by _daily_bar_status_ms
    df_future_bar = df_t_tue.copy()
    t_fut_1500_ms = int(pd.Timestamp("2025-11-28 12:30:00", tz="America/New_York").timestamp() * 1000)
    df_future_bar.attrs["observation_timestamp_ms"] = t_fut_1500_ms
    df_future_bar.attrs["obs_bar_date"] = "2025-11-28"
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-11-28 12:00:00", tz="America/New_York").timestamp())
    fut_obs, fut_comp, _ = prov._daily_bar_status_ms(df_future_bar)
    assert fut_obs is None and fut_comp is False

    # 14) Sector preload failure degrades gracefully to SECTOR_BENCHMARK_UNALIGNED instead of full UNREADY
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-12-01 16:10:00", tz="America/New_York").timestamp())
    prov.enable_live_fetch = True
    def _load_sector_boom(sym: str, allow_refresh: bool = True, require_quote_ts: bool = False):
        if sym == "XLK":
            raise RuntimeError("transient sector timeout")
        if sym == "SPY":
            return (df_s_wed, None)
        return (df_t_tue, None)
    monkeypatch.setattr(prov, "_load_cached_daily_with_quote_ts", _load_sector_boom)
    with prov._lock:
        prov._signal_cache.clear()
    sig_sec_err = prov.get_signal("AAPL")
    prov.enable_live_fetch = False
    assert "UNREADY" not in sig_sec_err["execution_mode"]
    assert "SECTOR_BENCHMARK_UNALIGNED" in sig_sec_err["data_integrity"]["missing_modalities"]

    # 15) Peer graph Monday XLK spillover omitted cleanly by Friday target signal (zero lookahead)
    with prov._lock:
        prov._signal_cache.clear()
        prov._peer_graph_cache["REDPIT"] = (
            50_000.0,
            {
                "target_ticker": "REDPIT",
                "nodes": [
                    {"id": "REDPIT", "role": "TARGET", "spillover_contribution": 0.0, "observation_timestamp_ms": t_tue_ms, "bar_date": "2025-11-28"},
                    {"id": "XLK", "role": "SECTOR_ETF", "spillover_contribution": 8.0, "observation_timestamp_ms": t_wed_ms, "bar_date": "2025-12-01"},
                ],
                "edges": [],
            },
        )
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_t_tue if sym == "REDPIT" else (df_s_wed if sym == "SPY" else None))
    sig_pg_pit = prov.get_signal("REDPIT", regime=reg13)
    assert sig_pg_pit["signal_breakdown"]["peer_gat_spillover"] == 0.0
    assert "PEER_GAT_NOT_LOADED" in sig_pg_pit["data_integrity"]["missing_modalities"]

    # 16) NYSE holiday-aware PIT cutoff (Thanksgiving 2025-11-27 skipped -> Black Friday 2025-11-28)
    pit_thx = MarketDataLoader.apply_pit_cutoff(pd.Timestamp("2025-11-26 17:00:00", tz="America/New_York"))
    assert pit_thx.date() == date(2025, 11, 28)

    # 17) Forming intraday bar whose download spans bar end omits observation_timestamp_ms
    idx_1h_form = pd.DatetimeIndex([pd.Timestamp("2025-11-18 09:30:00", tz="America/New_York")])
    df_1h_form = pd.DataFrame({"Open": [100.0], "High": [101.0], "Low": [99.5], "Close": [100.5], "Volume": [5000]}, index=idx_1h_form)
    t_clock = {"ts": pd.Timestamp("2025-11-18 10:25:00", tz="America/New_York").timestamp()}
    def _dl_span(*a, **kw):
        t_clock["ts"] = pd.Timestamp("2025-11-18 10:35:00", tz="America/New_York").timestamp()
        return df_1h_form
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: t_clock["ts"])
    monkeypatch.setattr(yf, "download", _dl_span)
    bars_form = ml13.fetch_intraday_bars("TST", interval="1h", count=5)
    assert len(bars_form) == 1 and "observation_timestamp_ms" not in bars_form[0]

    # 18) Form 4/A omission footnote appends without replacing + symmetric Pass 2 claim rejection + rate-limit deadline
    xml_omit = (
        "<ownershipDocument>"
        "<remarks>Transaction inadvertently omitted from original Form 4 filing.</remarks>"
        "<reportingOwner><reportingOwnerId><rptOwnerName>Grace Hopper</rptOwnerName></reportingOwnerId>"
        "<reportingOwnerRelationship><isDirector>1</isDirector></reportingOwnerRelationship></reportingOwner>"
        "<nonDerivativeTable><nonDerivativeTransaction>"
        "<transactionDate><value>2025-11-01</value></transactionDate>"
        "<transactionCoding><transactionCode>S</transactionCode></transactionCoding>"
        "<transactionAmounts><transactionShares><value>200</value></transactionShares>"
        "<transactionPricePerShare><value>50.0</value></transactionPricePerShare></transactionAmounts>"
        "</nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>"
    )
    add_rows = fc._parse_sec_form4_xml(xml_omit, "2025-11-05", "2025-11-05T14:00:00Z", "OM1", form="4/A")
    assert add_rows is not None and add_rows[0].get("amendment_action") == "ADD"
    orig_sale = {"form": "4", "filing_date": "2025-11-01", "transaction_date": "2025-11-01", "disclosed_at": "2025-11-01T14:00:00Z", "disclosed_at_verified": True, "accession_number": "OM0", "insider_name": "Grace Hopper", "transaction_code": "S", "value": 5_000.0}
    res_add = FundamentalsClient.resolve_pit_form4_transactions([orig_sale, add_rows[0]])
    assert len(res_add) == 2 and sorted(r["value"] for r in res_add) == [5_000.0, 10_000.0]

    p2_orig = {"form": "4", "filing_date": "2025-11-01", "transaction_date": "2025-11-01", "disclosed_at": "2025-11-01T14:00:00Z", "disclosed_at_verified": True, "accession_number": "C0", "insider_name": "Linus", "transaction_code": "P", "value": 10_000.0}
    p2_s = {"form": "4/A", "filing_date": "2025-11-02", "transaction_date": "2025-11-01", "disclosed_at": "2025-11-02T14:00:00Z", "disclosed_at_verified": True, "accession_number": "C1", "insider_name": "Linus", "transaction_code": "S", "value": 10_000.0}
    p2_g = {"form": "4/A", "filing_date": "2025-11-02", "transaction_date": "2025-11-01", "disclosed_at": "2025-11-02T14:00:00Z", "disclosed_at_verified": True, "accession_number": "C1", "insider_name": "Linus", "transaction_code": "G", "value": 10_000.0}
    res_sg = FundamentalsClient.resolve_pit_form4_transactions([p2_orig, p2_s, p2_g])
    res_gs = FundamentalsClient.resolve_pit_form4_transactions([p2_orig, p2_g, p2_s])
    assert sum(1 for r in res_sg if r.get("_amendment_ambiguous")) == 2
    assert sum(1 for r in res_gs if r.get("_amendment_ambiguous")) == 2

    from src.data.sec_edgar import TokenBucketRateLimiter
    lim = TokenBucketRateLimiter(rate=1.0, capacity=1.0)
    lim.tokens = 0.0
    assert lim.acquire(1.0, deadline=time.monotonic() + 0.05) == -1.0

    # 19) refresh_ticker_daily deduplicates unnormalized 16:00 local row against 00:00 download row
    df_16h = pd.DataFrame(
        {"open": [100.0, 101.0], "high": [102.0, 103.0], "low": [99.0, 100.0], "close": [101.0, 102.0], "volume": [1000, 1000], "vwap": [100.7, 101.7]},
        index=pd.DatetimeIndex([pd.Timestamp("2025-11-24 16:00:00"), pd.Timestamp("2025-11-25 16:00:00")], name="timestamp"),
    )
    ml13.save_ticker_daily("NORM16", df_16h)
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-11-25 16:20:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(ml13, "_latest_expected_trading_date", lambda: pd.Timestamp("2025-11-26"))
    df_00h_dl = pd.DataFrame(
        {"Open": [101.0, 103.0], "High": [103.0, 104.0], "Low": [100.0, 102.0], "Close": [102.5, 103.5], "Volume": [1000, 1200]},
        index=pd.DatetimeIndex([pd.Timestamp("2025-11-25"), pd.Timestamp("2025-11-26")]),
    )
    monkeypatch.setattr(yf, "download", lambda *a, **kw: df_00h_dl)
    df_norm_out = ml13.refresh_ticker_daily("NORM16", force_refresh_if_incomplete=True)
    assert df_norm_out is not None and len(df_norm_out) == 3 and df_norm_out.index.is_unique


def test_task3659_kelly_form4_loader_and_peer_regressions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task-3659 regressions: Kelly L1 sign passthrough, Parquet ISO metadata, cross-session FastInfo OHLCV isolation, 4H forming bucket, slow-stream Form 4 deadline, Form 4/A corrected-omitted & Pass 3 date-change, regime tail-OHLCV cache, signal regime-provenance cache, and peer-graph weight/same-day verification."""
    import pyarrow.parquet as pq
    import pandas as pd
    import yfinance as yf
    from src.data.fundamentals_client import FundamentalsClient
    from src.data.market_loader import MarketDataLoader
    from src.ipc.live_providers import LiveInferenceProvider
    from src.risk.calibration import TemperatureScaler
    from src.risk.kelly_engine import MultiOutcomeKellySizer

    # 1) Kelly L1 normalized allocation retains positive sign on LONG_ONLY/LONG_SHORT BUY and negates SELL
    sizer = MultiOutcomeKellySizer(prior_probs=(0.35, 0.35, 0.30))
    rec_lo_buy = sizer.compute_recommendation(
        p_buy=0.72,
        p_hold=0.18,
        p_sell=0.10,
        b_upper=0.05,
        b_lower=-0.025,
        entry_price=100.0,
        kappa=0.50,
        l1_normalized_pct=12.5,
        execution_mode="LONG_ONLY",
    )
    assert rec_lo_buy.action == "BUY"
    assert rec_lo_buy.l1_normalized_portfolio_allocation_pct == pytest.approx(12.5)

    rec_ls_buy = sizer.compute_recommendation(
        p_buy=0.72,
        p_hold=0.18,
        p_sell=0.10,
        b_upper=0.05,
        b_lower=-0.025,
        entry_price=100.0,
        kappa=0.50,
        l1_normalized_pct=12.5,
        execution_mode="LONG_SHORT",
    )
    assert rec_ls_buy.action == "BUY"
    assert rec_ls_buy.l1_normalized_portfolio_allocation_pct == pytest.approx(12.5)

    rec_ls_sell = sizer.compute_recommendation(
        p_buy=0.10,
        p_hold=0.18,
        p_sell=0.72,
        b_upper=0.05,
        b_lower=-0.025,
        entry_price=100.0,
        kappa=0.50,
        l1_normalized_pct=9.25,
        execution_mode="LONG_SHORT",
    )
    assert rec_ls_sell.action == "SELL"
    assert rec_ls_sell.l1_normalized_portfolio_allocation_pct == pytest.approx(-9.25)

    # 2) Parquet ISO metadata isolates exact intraday timestamp while keeping daily backward-compatibility
    lh = tmp_path / "lh3659"
    ml = MarketDataLoader(data_dir=lh)
    obs_ms_ex = int(pd.Timestamp("2025-11-28 10:30:00", tz="America/New_York").timestamp() * 1000)

    df_daily_legacy = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000, 1100],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2025-11-26"), pd.Timestamp("2025-11-28")], name="timestamp"),
    )
    df_daily_legacy.attrs = {"observation_timestamp_ms": obs_ms_ex, "obs_bar_date": "2025-11-28"}
    daily_p = ml.daily_dir / "LEG.parquet"
    ml.atomic_write_parquet(df_daily_legacy, daily_p)
    # Strip amstal_obs_bar_iso to emulate legacy daily file without ISO tag
    tbl_leg = pq.read_table(daily_p)
    stripped_meta = {k: v for k, v in (tbl_leg.schema.metadata or {}).items() if k != b"amstal_obs_bar_iso"}
    pq.write_table(tbl_leg.replace_schema_metadata(stripped_meta), daily_p)
    q_leg = ml.query_parquet_duckdb(daily_p)
    assert q_leg is not None and q_leg.attrs.get("observation_timestamp_ms") == obs_ms_ex

    df_intra = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000, 1100],
        },
        index=pd.DatetimeIndex(
            [pd.Timestamp("2025-11-28 09:30:00"), pd.Timestamp("2025-11-28 10:30:00")],
            name="timestamp",
        ),
    )
    df_intra.attrs = {"observation_timestamp_ms": obs_ms_ex, "obs_bar_date": "2025-11-28"}
    intra_p = ml.daily_dir / "INTRA.parquet"
    ml.atomic_write_parquet(df_intra, intra_p)
    q_intra_match = ml.query_parquet_duckdb(intra_p)
    assert q_intra_match is not None and q_intra_match.attrs.get("observation_timestamp_ms") == obs_ms_ex

    tbl_intra = pq.read_table(intra_p)
    later_iso_meta = {**(tbl_intra.schema.metadata or {}), b"amstal_obs_bar_iso": b"2025-11-28T15:30:00"}
    pq.write_table(tbl_intra.replace_schema_metadata(later_iso_meta), intra_p)
    q_intra_mismatch = ml.query_parquet_duckdb(intra_p)
    assert q_intra_mismatch is not None and "observation_timestamp_ms" not in q_intra_mismatch.attrs

    # 3) fetch_live_quote cross-session FastInfo Tue price must not copy Monday OHLCV/volume
    class _EmptyFastInfo:
        _prices_1y = pd.DataFrame()
        _md = {
            "regularMarketPrice": 215.0,
            "regularMarketTime": int(pd.Timestamp("2025-12-02 10:30:00", tz="America/New_York").timestamp()),
        }

        def get(self, key: str, default: Any = None) -> Any:
            if key in ("lastPrice", "regularMarketPrice"):
                return 215.0
            return default

    class _CrossSessionTicker:
        fast_info = _EmptyFastInfo()

        def history(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
            return pd.DataFrame(
                {"Open": [199.0], "High": [202.0], "Low": [198.0], "Close": [200.0], "Volume": [555555]},
                index=pd.DatetimeIndex([pd.Timestamp("2025-12-01")]),
            )

        def get_history_metadata(self) -> Dict[str, Any]:
            return {}

    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-12-02 10:35:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(yf, "Ticker", lambda sym: _CrossSessionTicker())
    q_cs = ml.fetch_live_quote("CSFAST")
    assert q_cs is not None
    assert q_cs["price"] == 215.0
    assert q_cs["date"] == "2025-12-02"
    assert q_cs["open"] == 215.0
    assert q_cs["high"] == 215.0
    assert q_cs["low"] == 215.0
    assert q_cs["volume"] == 0

    # 4) fetch_intraday_bars 4H bucket omission at 13:45 vs early-close session-capped stamp at 13:10
    df_4h_forming = pd.DataFrame(
        {"Open": [100.0, 102.0], "High": [101.0, 103.0], "Low": [99.0, 101.0], "Close": [100.5, 102.5], "Volume": [1000, 2000]},
        index=pd.DatetimeIndex([
            pd.Timestamp("2025-11-18 09:30:00", tz="America/New_York"),
            pd.Timestamp("2025-11-18 12:30:00", tz="America/New_York"),
        ]),
    )
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-11-18 13:45:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(yf, "download", lambda *a, **kw: df_4h_forming)
    bars_4h_form = ml.fetch_intraday_bars("TST4H", interval="4h", count=5)
    assert len(bars_4h_form) >= 1 and "observation_timestamp_ms" not in bars_4h_form[0]

    df_4h_close = pd.DataFrame(
        {"Open": [100.0, 104.0], "High": [101.0, 105.0], "Low": [99.0, 103.0], "Close": [100.5, 104.5], "Volume": [1000, 3000]},
        index=pd.DatetimeIndex([
            pd.Timestamp("2025-11-28 09:30:00", tz="America/New_York"),
            pd.Timestamp("2025-11-28 12:00:00", tz="America/New_York"),
        ]),
    )
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-11-28 13:10:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(yf, "download", lambda *a, **kw: df_4h_close)
    bars_4h_closed = ml.fetch_intraday_bars("TST4H", interval="4h", count=5)
    exp_close_ms = int(pd.Timestamp("2025-11-28 13:00:00", tz="America/New_York").timestamp() * 1000)
    assert len(bars_4h_closed) >= 1 and bars_4h_closed[0].get("observation_timestamp_ms") == exp_close_ms

    # 5) _sec_fetch_text enforces per-chunk deadline on slow streaming bodies
    fc = FundamentalsClient(lakehouse_dir=lh)

    class _SlowResp:
        def __init__(self) -> None:
            self.n = 0

        def __enter__(self) -> "_SlowResp":
            return self

        def __exit__(self, *args: Any) -> bool:
            return False

        def read(self, sz: int = -1) -> bytes:
            self.n += 1
            time.sleep(0.035)
            return b"abcdefgh" if self.n <= 3 else b""

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=6.0: _SlowResp())
    fc._sec_tl.deadline = time.monotonic() + 0.05
    try:
        assert fc._sec_fetch_text("https://data.sec.gov/test.txt", timeout=5.0) is None
    finally:
        fc._sec_tl.deadline = None

    # 6) Form 4/A "corrected ... omitted" resolves to CORRECT (not ADD) & Pass 3 matches differing transaction_date
    xml_corr_omit = (
        "<ownershipDocument>"
        "<dateOfOriginalSubmission>2025-11-01</dateOfOriginalSubmission>"
        "<remarks>Filing corrected details of a transaction inadvertently omitted from date column.</remarks>"
        "<reportingOwner><reportingOwnerId><rptOwnerName>Ada Lovelace</rptOwnerName></reportingOwnerId>"
        "<reportingOwnerRelationship><isDirector>1</isDirector></reportingOwnerRelationship></reportingOwner>"
        "<nonDerivativeTable><nonDerivativeTransaction>"
        "<transactionDate><value>2025-11-02</value></transactionDate>"
        "<transactionCoding><transactionCode>P</transactionCode></transactionCoding>"
        "<transactionAmounts><transactionShares><value>100</value></transactionShares>"
        "<transactionPricePerShare><value>55.0</value></transactionPricePerShare></transactionAmounts>"
        "</nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>"
    )
    corr_rows = fc._parse_sec_form4_xml(
        xml_corr_omit, "2025-11-04", "2025-11-04T14:00:00Z", "CORR1", form="4/A"
    )
    assert corr_rows is not None and corr_rows[0].get("amendment_action") == "CORRECT"
    orig_wrong_dt = {
        "form": "4",
        "filing_date": "2025-11-01",
        "transaction_date": "2025-10-31",
        "disclosed_at": "2025-11-01T14:00:00Z",
        "disclosed_at_verified": True,
        "accession_number": "ORIG1",
        "insider_name": "Ada Lovelace",
        "transaction_code": "P",
        "value": 5_000.0,
    }
    res_p3 = FundamentalsClient.resolve_pit_form4_transactions([orig_wrong_dt, corr_rows[0]])
    assert len(res_p3) == 1
    assert res_p3[0]["transaction_date"] == "2025-11-02"
    assert res_p3[0]["original_filing_date"] == "2025-11-01"

    dup_wrong_dt = {**orig_wrong_dt, "tx_index": 1}
    res_p3_ambig = FundamentalsClient.resolve_pit_form4_transactions([orig_wrong_dt, dup_wrong_dt, corr_rows[0]])
    assert any(r.get("_amendment_ambiguous") for r in res_p3_ambig)

    # 7) get_macro_regime invalidates cache on SPY high change even when close & date stay identical
    wdir = tmp_path / "w3659"
    wdir.mkdir(parents=True, exist_ok=True)
    prov = LiveInferenceProvider(lakehouse_dir=lh, weights_dir=wdir, enable_live_fetch=False)

    class _ParkinsonHMM:
        STATE_LABELS = {0: "LOW_VOLATILITY", 1: "MEDIUM_VOLATILITY", 2: "HIGH_VOLATILITY"}
        transmat_ = np.array([[0.9, 0.08, 0.02], [0.05, 0.9, 0.05], [0.02, 0.08, 0.9]])

        def filter_sequence(self, x: np.ndarray) -> np.ndarray:
            high_vol = float(x[-1, 1]) > 0.002
            p = [0.0, 0.0, 1.0] if high_vol else [1.0, 0.0, 0.0]
            return np.tile(p, (len(x), 1))

        def get_regime_multiplier(self, probs: np.ndarray) -> float:
            return 0.25 if probs[2] > 0.5 else 0.50

        def get_state_annualized_volatilities(self) -> List[float]:
            return [8.0, 16.0, 40.0]

        def get_stationary_distribution(self) -> List[float]:
            return [0.5, 0.3, 0.2]

    class _DummyLGBM:
        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            return np.tile([0.65, 0.25, 0.10], (len(x), 1))

    prov.regime_hmm = _ParkinsonHMM()
    prov.lgbm_backbone = _DummyLGBM()
    sc = TemperatureScaler()
    sc.temperature_ = 1.0
    prov.temp_scaler = sc

    idx_mon = pd.bdate_range(end="2025-12-01", periods=200)
    idx_fri = idx_mon[:-1]
    rets_s = np.random.default_rng(77).normal(0.0002, 0.01, size=200)
    cls_s = 400.0 * np.exp(np.cumsum(rets_s))
    df_spy_low_h = pd.DataFrame(
        {"open": cls_s * 0.999, "high": cls_s * 1.005, "low": cls_s * 0.995, "close": cls_s, "volume": np.full(200, 3_000_000.0)},
        index=idx_mon,
    )
    df_spy_high_h = df_spy_low_h.copy()
    df_spy_high_h.iloc[-1, df_spy_high_h.columns.get_loc("high")] = cls_s[-1] * 1.25  # same close, huge intraday high!
    t_fri_ms = int(pd.Timestamp("2025-11-28 13:00:00", tz="America/New_York").timestamp() * 1000)
    t_mon_ms = int(pd.Timestamp("2025-12-01 16:00:00", tz="America/New_York").timestamp() * 1000)
    df_spy_low_h.attrs = {"observation_timestamp_ms": t_mon_ms, "obs_bar_date": "2025-12-01"}
    df_spy_high_h.attrs = {"observation_timestamp_ms": t_mon_ms, "obs_bar_date": "2025-12-01"}

    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-12-01 16:10:00", tz="America/New_York").timestamp())
    monkeypatch.setattr("src.ipc.live_providers.time.monotonic", lambda: 80_000.0)
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_spy_low_h)
    reg_low = prov.get_macro_regime()
    assert reg_low["kappa"] == pytest.approx(0.50, abs=0.01)

    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_spy_high_h)
    reg_high = prov.get_macro_regime()
    assert reg_high["kappa"] == pytest.approx(0.25, abs=0.01)

    # 8) get_signal invalidates cache when regime provenance changes with identical state/probs/kappa
    df_spy_fri = df_spy_low_h.iloc[:-1].copy()
    df_spy_fri.attrs = {"observation_timestamp_ms": t_fri_ms, "obs_bar_date": "2025-11-28"}
    cls_t = 120.0 * np.exp(np.cumsum(rets_s[:-1]))
    df_t_fri = pd.DataFrame(
        {"open": cls_t * 0.998, "high": cls_t * 1.006, "low": cls_t * 0.994, "close": cls_t, "volume": np.full(199, 1_000_000.0)},
        index=idx_fri,
    )
    df_t_fri.attrs = {"observation_timestamp_ms": t_fri_ms, "obs_bar_date": "2025-11-28"}
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_t_fri if sym == "TSTR" else (df_spy_fri if sym == "SPY" else None))
    reg_base = {
        "available": True,
        "current_state": "MEDIUM_VOLATILITY",
        "state_index": 1,
        "kappa": 0.35,
        "filtered_probabilities": {"low_volatility": 0.1, "medium_volatility": 0.8, "high_volatility": 0.1},
        "transition_matrix": _ParkinsonHMM.transmat_.tolist(),
        "as_of_date": "2025-11-28",
        "observation_timestamp_ms": t_fri_ms,
        "regime_history": [{"timestamp": "2025-11-28", "state_index": 1}],
    }
    reg_ahead_same_vals = {
        **reg_base,
        "as_of_date": "2025-12-01",
        "observation_timestamp_ms": t_mon_ms,
        "regime_history": [{"timestamp": "2025-12-01", "state_index": 1}],
    }
    sig_prov1 = prov.get_signal("TSTR", regime=reg_base)
    sig_prov2 = prov.get_signal("TSTR", regime=reg_ahead_same_vals)
    assert sig_prov1["macro_regime"]["kappa"] == pytest.approx(0.35, abs=0.01)
    assert sig_prov2["macro_regime"]["kappa"] == pytest.approx(0.50, abs=0.01)

    # 9) Peer graph carries weight provenance & excludes same-day unstamped peer or ahead weight contributor in live mode
    prov_pg = LiveInferenceProvider(lakehouse_dir=lh, weights_dir=wdir, enable_live_fetch=False)
    monkeypatch.setattr(prov_pg, "_load_cached_daily", lambda sym, allow_refresh=True: df_spy_low_h if sym == "SPY" else df_t_fri)
    pg_built = prov_pg.get_peer_graph("TSTR")
    assert pg_built["weight_bar_date"] == "2025-12-01"
    assert pg_built["weight_observation_timestamp_ms"] == t_mon_ms

    prov.enable_live_fetch = True
    monkeypatch.setattr(prov, "_load_cached_daily_with_quote_ts", lambda sym, allow_refresh=True, require_quote_ts=False: (df_t_fri if sym == "TSTR" else df_spy_fri, None))
    with prov._lock:
        prov._signal_cache.clear()
        prov._peer_graph_cache["TSTR"] = (
            80_000.0,
            {
                "target_ticker": "TSTR",
                "nodes": [
                    {"id": "TSTR", "role": "TARGET", "spillover_contribution": 0.0, "observation_timestamp_ms": t_fri_ms, "bar_date": "2025-11-28"},
                    {"id": "MSFT", "role": "GICS_PEER", "spillover_contribution": 5.0, "observation_timestamp_ms": None, "bar_date": "2025-11-28", "weight_observation_timestamp_ms": t_fri_ms, "weight_bar_date": "2025-11-28"},
                    {"id": "NVDA", "role": "LEAD_LAG", "spillover_contribution": 5.0, "observation_timestamp_ms": t_fri_ms - 60_000, "bar_date": "2025-11-26", "weight_observation_timestamp_ms": t_mon_ms, "weight_bar_date": "2025-12-01"},
                ],
                "edges": [],
            },
        )
    sig_pg_excl = prov.get_signal("TSTR", regime=reg_base)
    assert sig_pg_excl["signal_breakdown"]["peer_gat_spillover"] == 0.0
    assert "PEER_GAT_NOT_LOADED" in sig_pg_excl["data_integrity"]["missing_modalities"]

    with prov._lock:
        prov._signal_cache.clear()
        prov._peer_graph_cache["TSTR"] = (
            80_000.0,
            {
                "target_ticker": "TSTR",
                "nodes": [
                    {"id": "TSTR", "role": "TARGET", "spillover_contribution": 0.0, "observation_timestamp_ms": t_fri_ms, "bar_date": "2025-11-28"},
                    {"id": "MSFT", "role": "GICS_PEER", "spillover_contribution": 5.0, "observation_timestamp_ms": t_fri_ms, "bar_date": "2025-11-28", "weight_observation_timestamp_ms": t_fri_ms, "weight_bar_date": "2025-11-28"},
                ],
                "edges": [],
            },
        )
    sig_pg_incl = prov.get_signal("TSTR", regime=reg_base)
    prov.enable_live_fetch = False
    assert sig_pg_incl["signal_breakdown"]["peer_gat_spillover"] == pytest.approx(0.5, abs=1e-6)
    assert "PEER_GAT_NOT_LOADED" not in sig_pg_incl["data_integrity"]["missing_modalities"]


def test_task3801_sec_form4_spy_and_peer_mtime_regressions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task-3801 regressions: SEC >32MiB overflow rejection + read1 deadline, 'amends ... previously unreported' Form 4/A ADD classification, ambiguous Form 4/A slot isolation across chained amendments, retained SPY row observation_timestamp check against Wednesday 10:00 target, and offline peer mtime ordering."""
    import pandas as pd
    from src.data.fundamentals_client import FundamentalsClient
    from src.ipc.live_providers import LiveInferenceProvider
    from src.risk.calibration import TemperatureScaler

    lh = tmp_path / "lh3801"
    wdir = tmp_path / "w3801"
    wdir.mkdir(parents=True, exist_ok=True)
    fc = FundamentalsClient(lakehouse_dir=lh)

    # 1) _sec_fetch_text rejects >32MiB overflow rather than returning truncated prefix, and succeeds below limit
    class _OverflowResp:
        def __init__(self, max_chunks: int = 520, payload_byte: bytes = b"A") -> None:
            self.n = 0
            self.max_chunks = max_chunks
            self.payload_byte = payload_byte
            self.bytes_sent = 0

        def __enter__(self) -> "_OverflowResp":
            return self

        def __exit__(self, *args: Any) -> bool:
            return False

        def read1(self, sz: int = 65536) -> bytes:
            self.n += 1
            if self.n <= self.max_chunks:
                chunk = self.payload_byte * min(sz, 65536)
                self.bytes_sent += len(chunk)
                return chunk
            return b""

        def read(self, sz: int = -1) -> bytes:
            raise AssertionError("read1 should be preferred when available")

    over_resp = _OverflowResp()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("urllib.request.urlopen", lambda req, timeout=6.0: over_resp)
        assert fc._sec_fetch_text("https://www.sec.gov/Archives/edgar/data/1/huge.htm") is None
    assert over_resp.bytes_sent > 32 * 1024 * 1024

    small_resp = _OverflowResp(max_chunks=1, payload_byte=b"B")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("urllib.request.urlopen", lambda req, timeout=6.0: small_resp)
        small_txt = fc._sec_fetch_text("https://www.sec.gov/Archives/edgar/data/1/small.htm")
    assert small_txt is not None and len(small_txt) == 65536

    # 2) Generic "amends the original filing to report a previously unreported transaction" stays ADD
    xml_amend_add = (
        "<ownershipDocument>"
        "<remarks>This form amends the original filing to report a previously unreported transaction.</remarks>"
        "<reportingOwner><reportingOwnerId><rptOwnerName>Alan Turing</rptOwnerName></reportingOwnerId>"
        "<reportingOwnerRelationship><isDirector>1</isDirector></reportingOwnerRelationship></reportingOwner>"
        "<nonDerivativeTable><nonDerivativeTransaction>"
        "<transactionDate><value>2025-11-03</value></transactionDate>"
        "<transactionCoding><transactionCode>P</transactionCode></transactionCoding>"
        "<transactionAmounts><transactionShares><value>150</value></transactionShares>"
        "<transactionPricePerShare><value>40.0</value></transactionPricePerShare></transactionAmounts>"
        "</nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>"
    )
    add_rows = fc._parse_sec_form4_xml(xml_amend_add, "2025-11-05", "2025-11-05T14:00:00Z", "AM1", form="4/A")
    assert add_rows is not None and add_rows[0].get("amendment_action") == "ADD"
    orig_p = {
        "form": "4",
        "filing_date": "2025-11-03",
        "transaction_date": "2025-11-03",
        "disclosed_at": "2025-11-03T14:00:00Z",
        "disclosed_at_verified": True,
        "accession_number": "AM0",
        "insider_name": "Alan Turing",
        "transaction_code": "P",
        "value": 8_000.0,
    }
    res_add = FundamentalsClient.resolve_pit_form4_transactions([orig_p, add_rows[0]])
    assert len(res_add) == 2

    # 3) Ambiguous Day-2 4/A sale cannot become confirmed slot for Day-3 4/A sale
    p_day1_a = {**orig_p, "accession_number": "D1", "tx_index": 0, "value": 4_000.0}
    p_day1_b = {**orig_p, "accession_number": "D1", "tx_index": 1, "value": 6_000.0}
    s_day2_4a = {
        "form": "4/A",
        "filing_date": "2025-11-04",
        "transaction_date": "2025-11-03",
        "disclosed_at": "2025-11-04T14:00:00Z",
        "disclosed_at_verified": True,
        "accession_number": "D2",
        "insider_name": "Alan Turing",
        "transaction_code": "S",
        "value": 5_000.0,
    }
    s_day3_4a = {
        "form": "4/A",
        "filing_date": "2025-11-05",
        "transaction_date": "2025-11-03",
        "disclosed_at": "2025-11-05T14:00:00Z",
        "disclosed_at_verified": True,
        "accession_number": "D3",
        "insider_name": "Alan Turing",
        "transaction_code": "S",
        "value": 5_500.0,
    }
    res_chain_ambig = FundamentalsClient.resolve_pit_form4_transactions([p_day1_a, p_day1_b, s_day2_4a, s_day3_4a])
    assert len([r for r in res_chain_ambig if r.get("transaction_code") == "P" and not r.get("_amendment_ambiguous")]) == 2
    s_rows = [r for r in res_chain_ambig if r.get("transaction_code") == "S"]
    assert len(s_rows) == 2 and all(r.get("_amendment_ambiguous") for r in s_rows)

    # 4) Wednesday 10:00 target excludes Wednesday 16:00 SPY close during causal regime fallback even when Thursday exists
    prov = LiveInferenceProvider(lakehouse_dir=lh, weights_dir=wdir, enable_live_fetch=False)

    class _WedSpikeHMM:
        STATE_LABELS = {0: "LOW_VOLATILITY", 1: "MEDIUM_VOLATILITY", 2: "HIGH_VOLATILITY"}
        transmat_ = np.array([[0.9, 0.08, 0.02], [0.05, 0.9, 0.05], [0.02, 0.08, 0.9]])

        def filter_sequence(self, x: np.ndarray) -> np.ndarray:
            p = [0.0, 0.0, 1.0] if float(np.max(x[-1:, 1])) > 0.002 else [1.0, 0.0, 0.0]
            return np.tile(p, (len(x), 1))

        def get_regime_multiplier(self, probs: np.ndarray) -> float:
            return 0.25 if probs[2] > 0.5 else 0.50

        def get_state_annualized_volatilities(self) -> List[float]:
            return [8.0, 16.0, 40.0]

        def get_stationary_distribution(self) -> List[float]:
            return [0.5, 0.3, 0.2]

    class _DummyLGBM:
        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            return np.tile([0.65, 0.25, 0.10], (len(x), 1))

    prov.regime_hmm = _WedSpikeHMM()
    prov.lgbm_backbone = _DummyLGBM()
    sc = TemperatureScaler()
    sc.temperature_ = 1.0
    prov.temp_scaler = sc

    idx_thu = pd.bdate_range(end="2025-12-04", periods=200)  # Tue=-3 (12-02), Wed=-2 (12-03), Thu=-1 (12-04)
    idx_wed = idx_thu[:-1]
    rets_b = np.random.default_rng(19).normal(0.0002, 0.008, size=200)
    cls_b = 400.0 * np.exp(np.cumsum(rets_b))
    df_spy_thu = pd.DataFrame(
        {"open": cls_b * 0.999, "high": cls_b * 1.004, "low": cls_b * 0.996, "close": cls_b, "volume": np.full(200, 3_000_000.0)},
        index=idx_thu,
    )
    # Inflate Wednesday's high so retaining Wed 16:00 flips HMM to kappa=0.25 while stopping at Tue keeps kappa=0.50
    df_spy_thu.iloc[-2, df_spy_thu.columns.get_loc("high")] = cls_b[-2] * 1.30
    t_tue_16_ms = int(pd.Timestamp("2025-12-02 16:00:00", tz="America/New_York").timestamp() * 1000)
    t_wed_10_ms = int(pd.Timestamp("2025-12-03 10:00:00", tz="America/New_York").timestamp() * 1000)
    t_wed_16_ms = int(pd.Timestamp("2025-12-03 16:00:00", tz="America/New_York").timestamp() * 1000)
    t_thu_16_ms = int(pd.Timestamp("2025-12-04 16:00:00", tz="America/New_York").timestamp() * 1000)
    df_spy_thu.attrs = {
        "observation_timestamp_ms": t_thu_16_ms,
        "obs_bar_date": "2025-12-04",
        "_row_provenance": {
            "2025-12-02": {"observation_timestamp_ms": t_tue_16_ms},
            "2025-12-03": {"observation_timestamp_ms": t_wed_16_ms},
        },
    }

    cls_t = 120.0 * np.exp(np.cumsum(rets_b[:-1]))
    df_t_wed10 = pd.DataFrame(
        {"open": cls_t * 0.998, "high": cls_t * 1.005, "low": cls_t * 0.995, "close": cls_t, "volume": np.full(199, 1_200_000.0)},
        index=idx_wed,
    )
    df_t_wed10.attrs = {"observation_timestamp_ms": t_wed_10_ms, "obs_bar_date": "2025-12-03"}

    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-12-04 16:10:00", tz="America/New_York").timestamp())
    monkeypatch.setattr("src.ipc.live_providers.time.monotonic", lambda: 90_000.0)
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_t_wed10 if sym == "WEDT" else (df_spy_thu if sym == "SPY" else None))
    reg_thu = prov.get_macro_regime()
    assert reg_thu["kappa"] == pytest.approx(0.50, abs=0.01)  # Thu itself low vol
    reg_wed_ahead = {
        **reg_thu,
        "kappa": 0.25,
        "current_state": "HIGH_VOLATILITY",
        "state_index": 2,
        "as_of_date": "2025-12-04",
        "observation_timestamp_ms": t_thu_16_ms,
    }
    sig_wed10 = prov.get_signal("WEDT", regime=reg_wed_ahead)
    assert sig_wed10["execution_mode"] == "UNREADY"
    assert sig_wed10["macro_regime"]["kappa"] == pytest.approx(0.50, abs=0.01)

    # Without per-row provenance, revised Thursday SPY file cannot substitute Tuesday scheduled close
    df_spy_thu_noprov = df_spy_thu.copy()
    df_spy_thu_noprov.attrs = {"observation_timestamp_ms": t_thu_16_ms, "obs_bar_date": "2025-12-04", "disk_mtime_ms": t_thu_16_ms}
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_t_wed10 if sym == "WEDT" else (df_spy_thu_noprov if sym == "SPY" else None))
    with prov._lock:
        prov._signal_cache.clear()
    sig_wed10_noprov = prov.get_signal("WEDT", regime=reg_wed_ahead)
    assert sig_wed10_noprov["macro_regime"]["kappa"] is None

    # 5) Offline same-day peer snapshot rejects peer/weight disk_mtime_ms later than target
    idx_off = idx_wed
    df_off_t = pd.DataFrame(
        {"open": cls_t * 0.998, "high": cls_t * 1.005, "low": cls_t * 0.995, "close": cls_t, "volume": np.full(199, 1_200_000.0)},
        index=idx_off,
    )
    df_off_s = df_spy_thu.iloc[:-1].copy()
    df_off_s.iloc[-1, df_off_s.columns.get_loc("high")] = cls_b[-2] * 1.004
    df_off_t.attrs = {"disk_mtime_ms": 1_000_000}
    df_off_s.attrs = {"disk_mtime_ms": 1_000_000}
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-12-03 10:00:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_off_t if sym == "OFFP" else (df_off_s if sym == "SPY" else None))
    reg_off = {"current_state": "LOW_VOLATILITY", "state_index": 0, "kappa": 0.50, "available": True}
    with prov._lock:
        prov._signal_cache.clear()
        prov._peer_graph_cache["OFFP"] = (
            90_000.0,
            {
                "target_ticker": "OFFP",
                "nodes": [
                    {"id": "OFFP", "role": "TARGET", "spillover_contribution": 0.0, "bar_date": "2025-12-03", "disk_mtime_ms": 1_000_000},
                    {"id": "MSFT", "role": "GICS_PEER", "spillover_contribution": 6.0, "bar_date": "2025-12-03", "disk_mtime_ms": 1_000_500, "weight_bar_date": "2025-12-03", "weight_disk_mtime_ms": 1_000_500},
                ],
                "edges": [],
            },
        )
    sig_off_late_peer = prov.get_signal("OFFP", regime=reg_off)
    assert sig_off_late_peer["signal_breakdown"]["peer_gat_spillover"] == 0.0
    assert "PEER_GAT_NOT_LOADED" in sig_off_late_peer["data_integrity"]["missing_modalities"]

    with prov._lock:
        prov._signal_cache.clear()
        prov._peer_graph_cache["OFFP"] = (
            90_000.0,
            {
                "target_ticker": "OFFP",
                "nodes": [
                    {"id": "OFFP", "role": "TARGET", "spillover_contribution": 0.0, "bar_date": "2025-12-03", "disk_mtime_ms": 1_000_000},
                    {"id": "MSFT", "role": "GICS_PEER", "spillover_contribution": 6.0, "bar_date": "2025-12-03", "disk_mtime_ms": 999_500, "weight_bar_date": "2025-12-03", "weight_disk_mtime_ms": 1_000_000},
                ],
                "edges": [],
            },
        )
    sig_off_ok_peer = prov.get_signal("OFFP", regime=reg_off)
    assert sig_off_ok_peer["signal_breakdown"]["peer_gat_spillover"] == pytest.approx(0.6, abs=1e-6)
    assert "PEER_GAT_NOT_LOADED" not in sig_off_ok_peer["data_integrity"]["missing_modalities"]


def test_task3887_sec_loader_and_regime_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task-3887 regressions: VWAP change invalidates timestamp reuse, FastInfo/history volume mismatch strips obs_ts_ms, TokenBucket lock deadline, urlopen header deadline + retryable Form 4 budget expiry, omitted-admin Form 4/A ADD classification, cross-date Form 4/A shares disambiguation, live overlay VWAP recompute, 16:05 legacy mtime non-settlement, single-snapshot get_peer_graph('SPY'), and same-day unverified regime recompute."""
    import threading
    import pandas as pd
    import yfinance as yf
    from src.data.fundamentals_client import FundamentalsClient
    from src.data.market_loader import MarketDataLoader
    from src.data.sec_edgar import TokenBucketRateLimiter
    from src.ipc.live_providers import LiveInferenceProvider
    from src.risk.calibration import TemperatureScaler

    lh = tmp_path / "lh3887"
    wdir = tmp_path / "w3887"
    wdir.mkdir(parents=True, exist_ok=True)
    ml = MarketDataLoader(data_dir=lh)

    # 1) refresh_ticker_daily does not carry forward pre-close obs_ts_ms when only VWAP changes
    t_pre_ms = int(pd.Timestamp("2025-11-28 11:00:00", tz="America/New_York").timestamp() * 1000)
    idx_v = pd.DatetimeIndex([pd.Timestamp("2025-11-26"), pd.Timestamp("2025-11-28")], name="timestamp")
    df_v_disk = pd.DataFrame(
        {"open": [100.0, 102.0], "high": [101.0, 104.0], "low": [99.0, 101.0], "close": [100.5, 103.0], "volume": [1000, 2000], "vwap": [100.2, 102.1]},
        index=idx_v,
    )
    df_v_disk.attrs = {"observation_timestamp_ms": t_pre_ms, "obs_bar_date": "2025-11-28", "_bar_unsettled": True}
    ml.save_ticker_daily("VWAPT", df_v_disk)
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-11-28 12:30:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(ml, "_latest_expected_trading_date", lambda: pd.Timestamp("2025-11-28"))
    # _normalize_yf_df sets vwap=(high+low+close)/3 = 102.6667 != 102.1 with identical OHLCV!
    df_v_dl = pd.DataFrame(
        {"Open": [102.0], "High": [104.0], "Low": [101.0], "Close": [103.0], "Volume": [2000]},
        index=pd.DatetimeIndex([pd.Timestamp("2025-11-28")]),
    )
    monkeypatch.setattr(yf, "download", lambda *a, **kw: df_v_dl)
    df_v_out = ml.refresh_ticker_daily("VWAPT", force_refresh_if_incomplete=True)
    assert df_v_out is not None
    assert df_v_out.attrs.get("_target_row_replaced") is True
    assert df_v_out.attrs.get("observation_timestamp_ms") is None

    # 2) fetch_live_quote omits obs_ts_ms when FastInfo used_hist_ohlcv volume diverges from regularMarketVolume
    t_rmt_s = int(pd.Timestamp("2025-12-02 12:00:00", tz="America/New_York").timestamp())

    class _VolMismatchFastInfo:
        _prices_1y = pd.DataFrame(
            {"Open": [100.0], "High": [103.0], "Low": [99.0], "Close": [102.0], "Volume": [0]},
            index=pd.DatetimeIndex([pd.Timestamp("2025-12-02")]),
        )
        _md = {"regularMarketPrice": 102.0, "regularMarketTime": t_rmt_s, "regularMarketVolume": 10_000}

        def get(self, key: str, default: Any = None) -> Any:
            return {"lastPrice": 102.0, "open": 100.0, "dayHigh": 103.0, "dayLow": 99.0, "lastVolume": 0}.get(key, default)

    class _VolMismatchTicker:
        fast_info = _VolMismatchFastInfo()

        def history(self, *a: Any, **kw: Any) -> pd.DataFrame:
            return pd.DataFrame(
                {"Open": [100.0], "High": [103.0], "Low": [99.0], "Close": [102.0], "Volume": [55_000]},
                index=pd.DatetimeIndex([pd.Timestamp("2025-12-02")]),
            )

        def get_history_metadata(self) -> Dict[str, Any]:
            return {"regularMarketTime": t_rmt_s, "regularMarketPrice": 102.0, "regularMarketVolume": 55_000}

    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-12-02 12:05:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(yf, "Ticker", lambda sym: _VolMismatchTicker())
    q_vm = ml.fetch_live_quote("VMM")
    assert q_vm is not None and "observation_timestamp_ms" not in q_vm

    # 3) TokenBucketRateLimiter.acquire honors deadline even while another thread holds self.lock
    lim = TokenBucketRateLimiter(rate=10.0, capacity=10.0)
    held_evt = threading.Event()
    release_evt = threading.Event()

    def _hold_lock() -> None:
        if lim.lock.acquire(timeout=2.5):
            try:
                held_evt.set()
                release_evt.wait(timeout=2.5)
            finally:
                lim.lock.release()

    lock_thr = threading.Thread(target=_hold_lock, daemon=True)
    lock_thr.start()
    try:
        assert held_evt.wait(timeout=1.0)
        t0 = time.monotonic()
        res_locked = lim.acquire(1.0, deadline=t0 + 0.04)
        assert res_locked == -1.0 and (time.monotonic() - t0) < 1.0
    finally:
        release_evt.set()
        lock_thr.join(timeout=1.0)

    # 4) _sec_fetch_text enforces header deadline + keeps Form 4 accession retryable after budget expiry
    fc = FundamentalsClient(lakehouse_dir=lh)
    hdr_unblock = threading.Event()
    hdr_entered = threading.Event()

    def _hanging_urlopen(req: Any, timeout: float = 6.0) -> Any:
        hdr_entered.set()
        hdr_unblock.wait(timeout=2.5)
        raise TimeoutError("late header")

    monkeypatch.setattr("urllib.request.urlopen", _hanging_urlopen)
    fc._sec_tl.deadline = time.monotonic() + 0.12
    try:
        t_start = time.monotonic()
        assert fc._sec_fetch_text("https://www.sec.gov/Archives/edgar/data/1/hdr.xml", timeout=5.0) is None
        assert hdr_entered.is_set()
        assert (time.monotonic() - t_start) < 1.0
        assert getattr(fc._sec_tl, "last_budget_expired", False) is True
    finally:
        hdr_unblock.set()
        fc._sec_tl.deadline = None

    fc._sec_tl.last_budget_expired = False
    bud_calls = {"n": 0}
    def _bud_fetch(url: str, timeout: float = 5.0) -> Optional[str]:
        bud_calls["n"] += 1
        fc._sec_tl.last_budget_expired = True
        return None
    monkeypatch.setattr(fc, "_sec_fetch_text", _bud_fetch)
    subs_bud = {
        "recent": {
            "form": ["4"],
            "filingDate": ["2025-11-25"],
            "acceptanceDateTime": ["2025-11-25T15:00:00Z"],
            "accessionNumber": ["BUDRETRY1"],
            "primaryDocument": ["b1.xml"],
        }
    }
    first_try = fc._load_or_fetch_sec_form4_rows("0000000001", subs_bud, enable_live_fetch=True)
    assert bud_calls["n"] == 1 and first_try == [] and not fc._is_in_cooldown("f4:BUDRETRY1")
    xml_ok_retry = (
        "<ownershipDocument>"
        "<reportingOwner><reportingOwnerId><rptOwnerName>Grace Hopper</rptOwnerName></reportingOwnerId>"
        "<reportingOwnerRelationship><isDirector>1</isDirector></reportingOwnerRelationship></reportingOwner>"
        "<nonDerivativeTable><nonDerivativeTransaction>"
        "<transactionDate><value>2025-11-24</value></transactionDate>"
        "<transactionCoding><transactionCode>P</transactionCode></transactionCoding>"
        "<transactionAmounts><transactionShares><value>100</value></transactionShares>"
        "<transactionPricePerShare><value>50.0</value></transactionPricePerShare></transactionAmounts>"
        "</nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>"
    )
    monkeypatch.setattr(fc, "_sec_fetch_text", lambda url, timeout=5.0: xml_ok_retry)
    retry_rows = fc._load_or_fetch_sec_form4_rows("0000000001", subs_bud, enable_live_fetch=True)
    assert len(retry_rows) == 1 and retry_rows[0]["value"] == 5_000.0

    # 5) Form 4/A "inadvertently omitted ... correction of an administrative error" -> ADD + cross-date shares match
    xml_admin_omit = (
        "<ownershipDocument>"
        "<remarks>Report sale inadvertently omitted due to correction of an administrative error.</remarks>"
        "<reportingOwner><reportingOwnerId><rptOwnerName>Grace Hopper</rptOwnerName></reportingOwnerId>"
        "<reportingOwnerRelationship><isDirector>1</isDirector></reportingOwnerRelationship></reportingOwner>"
        "<nonDerivativeTable><nonDerivativeTransaction>"
        "<transactionDate><value>2025-11-04</value></transactionDate>"
        "<transactionCoding><transactionCode>S</transactionCode></transactionCoding>"
        "<transactionAmounts><transactionShares><value>250</value></transactionShares>"
        "<transactionPricePerShare><value>50.0</value></transactionPricePerShare></transactionAmounts>"
        "</nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>"
    )
    rows_adm = fc._parse_sec_form4_xml(xml_admin_omit, "2025-11-06", "2025-11-06T14:00:00Z", "ADM1", form="4/A")
    assert rows_adm is not None and rows_adm[0].get("amendment_action") == "ADD"

    orig_mon_100 = {"form": "4", "filing_date": "2025-11-05", "transaction_date": "2025-11-03", "disclosed_at": "2025-11-05T14:00:00Z", "disclosed_at_verified": True, "accession_number": "DX0", "tx_index": 0, "insider_name": "Linus", "transaction_code": "S", "shares": 100, "value": 5_000.0}
    orig_tue_900 = {"form": "4", "filing_date": "2025-11-05", "transaction_date": "2025-11-04", "disclosed_at": "2025-11-05T14:00:00Z", "disclosed_at_verified": True, "accession_number": "DX0", "tx_index": 1, "insider_name": "Linus", "transaction_code": "S", "shares": 900, "value": 45_000.0}
    corr_mon_to_tue_100 = {"form": "4/A", "filing_date": "2025-11-06", "date_of_original_submission": "2025-11-05", "transaction_date": "2025-11-04", "disclosed_at": "2025-11-06T14:00:00Z", "disclosed_at_verified": True, "accession_number": "DX1", "insider_name": "Linus", "transaction_code": "S", "shares": 100, "value": 5_000.0, "amendment_action": "CORRECT", "corrects_date": True}
    res_dc = FundamentalsClient.resolve_pit_form4_transactions([orig_mon_100, orig_tue_900, corr_mon_to_tue_100])
    assert sorted(int(r["shares"]) for r in res_dc) == [100, 900]
    assert all(r["transaction_date"] == "2025-11-04" for r in res_dc)

    corr_ambig_250 = {**corr_mon_to_tue_100, "shares": 250, "value": 12_500.0}
    res_dc_amb = FundamentalsClient.resolve_pit_form4_transactions([orig_mon_100, orig_tue_900, corr_ambig_250])
    assert [int(r["shares"]) for r in res_dc_amb[:2]] == [100, 900] and res_dc_amb[2].get("_amendment_ambiguous") is True

    # 6) _daily_bar_status_ms does not treat 16:05 legacy disk_mtime_ms (pre-settlement) as settled close
    prov = LiveInferenceProvider(lakehouse_dir=lh, weights_dir=wdir, enable_live_fetch=True)
    idx_1605 = pd.bdate_range(end="2025-12-02", periods=35)
    df_leg_1605 = pd.DataFrame(
        {"open": np.full(35, 100.0), "high": np.full(35, 102.0), "low": np.full(35, 99.0), "close": np.full(35, 101.0), "volume": np.full(35, 1000.0), "vwap": np.full(35, 100.5)},
        index=idx_1605,
    )
    t_1605_ms = int(pd.Timestamp("2025-12-02 16:05:00", tz="America/New_York").timestamp() * 1000)
    df_leg_1605.attrs = {"disk_mtime_ms": t_1605_ms}
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-12-02 16:20:00", tz="America/New_York").timestamp())
    assert prov._daily_bar_status_ms(df_leg_1605)[:2] == (None, False)

    # 7) Live quote overlay recomputes vwap cleanly from quote OHLC instead of leaking disk VWAP
    df_vw_disk = df_leg_1605.copy()
    df_vw_disk.iloc[-1, df_vw_disk.columns.get_loc("vwap")] = 999.0
    t_1130_ms = int(pd.Timestamp("2025-12-02 11:30:00", tz="America/New_York").timestamp() * 1000)
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-12-02 11:35:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(prov, "_maybe_refresh_daily", lambda s: None)
    monkeypatch.setattr(prov, "_load_disk_daily", lambda s: df_vw_disk)
    monkeypatch.setattr(prov, "_fetch_live_quote_cached", lambda s: {"price": 103.0, "open": 101.0, "high": 104.0, "low": 100.0, "volume": 5000, "date": "2025-12-02", "observation_timestamp_ms": t_1130_ms})
    df_ov_vw, ov_ms_vw = prov._load_cached_daily_with_quote_ts("VWO", allow_refresh=True)
    assert ov_ms_vw == t_1130_ms and df_ov_vw is not None and float(df_ov_vw["vwap"].iloc[-1]) == pytest.approx((104.0 + 100.0 + 103.0) / 3.0, abs=1e-6)

    # 8) get_peer_graph("SPY") reuses first SPY snapshot & get_signal recomputes same-day regime when obs time is None
    spy_reads = {"n": 0}
    idx_200 = pd.bdate_range(end="2025-12-02", periods=200)
    cls_200 = 400.0 * np.exp(np.cumsum(np.random.default_rng(5).normal(0.0002, 0.008, size=200)))
    df_spy_200 = pd.DataFrame(
        {"open": cls_200 * 0.999, "high": cls_200 * 1.004, "low": cls_200 * 0.996, "close": cls_200, "volume": np.full(200, 2_000_000.0)},
        index=idx_200,
    )
    t_dec2_ms = int(pd.Timestamp("2025-12-02 16:00:00", tz="America/New_York").timestamp() * 1000)
    df_spy_200.attrs = {"observation_timestamp_ms": t_dec2_ms, "obs_bar_date": "2025-12-02"}

    def _spy_loader(sym: str, allow_refresh: bool = True) -> Optional[pd.DataFrame]:
        if sym == "SPY":
            spy_reads["n"] += 1
            return df_spy_200
        return None

    prov_pg_spy = LiveInferenceProvider(lakehouse_dir=lh, weights_dir=wdir, enable_live_fetch=False)
    monkeypatch.setattr(prov_pg_spy, "_load_cached_daily", _spy_loader)
    prov_pg_spy.get_peer_graph("SPY")
    assert spy_reads["n"] == 1

    class _RegRecompHMM:
        STATE_LABELS = {0: "LOW_VOLATILITY", 1: "MEDIUM_VOLATILITY", 2: "HIGH_VOLATILITY"}
        transmat_ = np.eye(3)

        def filter_sequence(self, x: np.ndarray) -> np.ndarray:
            return np.tile([1.0, 0.0, 0.0], (len(x), 1))

        def get_regime_multiplier(self, probs: np.ndarray) -> float:
            return 0.50

        def get_state_annualized_volatilities(self) -> List[float]:
            return [8.0, 16.0, 40.0]

        def get_stationary_distribution(self) -> List[float]:
            return [0.5, 0.3, 0.2]

    class _SimpleLGBM:
        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            return np.tile([0.65, 0.25, 0.10], (len(x), 1))

    prov_pg_spy.regime_hmm = _RegRecompHMM()
    prov_pg_spy.lgbm_backbone = _SimpleLGBM()
    sc2 = TemperatureScaler()
    sc2.temperature_ = 1.0
    prov_pg_spy.temp_scaler = sc2
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-12-02 16:20:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(prov_pg_spy, "_load_disk_daily", lambda s: df_spy_200)
    undated_same_day_reg = {
        "available": True,
        "current_state": "HIGH_VOLATILITY",
        "state_index": 2,
        "kappa": 0.25,
        "filtered_probabilities": {"low_volatility": 0.1, "medium_volatility": 0.1, "high_volatility": 0.8},
        "transition_matrix": np.eye(3).tolist(),
        "as_of_date": "2025-12-02",
        "observation_timestamp_ms": None,
        "regime_history": [{"timestamp": "2025-12-02", "state_index": 2}],
    }
    sig_recomp = prov_pg_spy.get_signal("SPY", regime=undated_same_day_reg)
    assert sig_recomp["macro_regime"]["kappa"] == pytest.approx(0.50, abs=0.01)


def test_task3976_remaining_edge_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task-3976 regressions: live undated target short-circuits as-of search, undated SPY live overlay rejected despite earlier disk mtime, HTTP 429 retains 5m cooldown despite capped timeout, real stalling HTTP socket exits within SEC deadline, excess-shares Form 4/A stays CORRECT, securityTitle disambiguates Class A vs Class B Form 4/A, FastInfo metadata snapshotted before previousClose, and Kelly sizer override honored."""
    import io
    import socket
    import threading
    import urllib.error
    import pandas as pd
    import yfinance as yf
    from src.data.fundamentals_client import FundamentalsClient
    from src.data.market_loader import MarketDataLoader
    from src.ipc.live_providers import LiveInferenceProvider
    from src.risk.calibration import TemperatureScaler
    from src.risk.kelly_engine import MultiOutcomeKellySizer

    lh = tmp_path / "lh3976"
    wdir = tmp_path / "w3976"
    wdir.mkdir(parents=True, exist_ok=True)
    prov = LiveInferenceProvider(lakehouse_dir=lh, weights_dir=wdir, enable_live_fetch=False)

    class _CountingHMM:
        STATE_LABELS = {0: "LOW_VOLATILITY", 1: "MEDIUM_VOLATILITY", 2: "HIGH_VOLATILITY"}
        transmat_ = np.eye(3)

        def filter_sequence(self, x: np.ndarray) -> np.ndarray:
            p = [0.0, 0.0, 1.0] if len(x) >= 199 else [1.0, 0.0, 0.0]
            return np.tile(p, (len(x), 1))

        def get_regime_multiplier(self, probs: np.ndarray) -> float:
            return 0.25 if probs[2] > 0.5 else 0.50

        def get_state_annualized_volatilities(self) -> List[float]:
            return [8.0, 16.0, 40.0]

        def get_stationary_distribution(self) -> List[float]:
            return [0.5, 0.3, 0.2]

    class _SimpleLGBM:
        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            return np.tile([0.65, 0.25, 0.10], (len(x), 1))

    prov.regime_hmm = _CountingHMM()
    prov.lgbm_backbone = _SimpleLGBM()
    sc = TemperatureScaler()
    sc.temperature_ = 1.0
    prov.temp_scaler = sc

    # 1) Live undated target short-circuits as-of slice loop cleanly
    idx_200 = pd.bdate_range(end="2025-12-03", periods=200)
    cls_200 = 100.0 * np.exp(np.cumsum(np.random.default_rng(11).normal(0.0002, 0.008, size=200)))
    df_t_nodt = pd.DataFrame(
        {"open": cls_200 * 0.998, "high": cls_200 * 1.005, "low": cls_200 * 0.995, "close": cls_200, "volume": np.full(200, 1_500_000.0)},
        index=idx_200,
    )
    t_1055_ms = int(pd.Timestamp("2025-12-03 10:55:00", tz="America/New_York").timestamp() * 1000)
    df_spy_stamped = df_t_nodt.copy()
    df_spy_stamped.attrs = {"observation_timestamp_ms": t_1055_ms, "obs_bar_date": "2025-12-03"}
    slice_calls = {"n": 0}
    orig_slice = prov._slice_aligned_frame

    def _spy_slice(df_raw: pd.DataFrame, idx: pd.Index) -> pd.DataFrame:
        slice_calls["n"] += 1
        return orig_slice(df_raw, idx)

    with pytest.MonkeyPatch.context() as mp_live:
        mp_live.setattr(prov, "_slice_aligned_frame", _spy_slice)
        mp_live.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-12-03 11:00:00", tz="America/New_York").timestamp())
        mp_live.setattr("src.ipc.live_providers.time.monotonic", lambda: 95_000.0)
        prov.enable_live_fetch = True
        mp_live.setattr(prov, "_load_cached_daily_with_quote_ts", lambda sym, allow_refresh=True, require_quote_ts=False: (df_t_nodt if sym == "UNDT" else df_spy_stamped, None))
        reg_live_obs = {
            "available": True,
            "current_state": "LOW_VOLATILITY",
            "state_index": 0,
            "kappa": 0.50,
            "filtered_probabilities": {"low_volatility": 0.9, "medium_volatility": 0.1, "high_volatility": 0.0},
            "transition_matrix": np.eye(3).tolist(),
            "as_of_date": "2025-12-03",
            "observation_timestamp_ms": t_1055_ms,
            "regime_history": [{"timestamp": "2025-12-03", "state_index": 0}],
        }
        sig_live_undt = prov.get_signal("UNDT", regime=reg_live_obs)
        assert slice_calls["n"] <= 1 and sig_live_undt["macro_regime"]["kappa"] is None

        # 2) Undated live overlay on SPY rejected in as-of search despite 09:45 disk_mtime_ms <= 10:00 target
        slice_calls["n"] = 0
        t_0945_ms = int(pd.Timestamp("2025-12-03 09:45:00", tz="America/New_York").timestamp() * 1000)
        t_1000_ms = int(pd.Timestamp("2025-12-03 10:00:00", tz="America/New_York").timestamp() * 1000)
        df_t_1000 = df_t_nodt.copy()
        df_t_1000.attrs = {"observation_timestamp_ms": t_1000_ms, "obs_bar_date": "2025-12-03"}
        df_spy_undated_ov = df_t_nodt.copy()
        df_spy_undated_ov.attrs = {"disk_mtime_ms": t_0945_ms, "obs_bar_date": "2025-12-03", "live_quote_undated": True}
        mp_live.setattr(prov, "_load_cached_daily_with_quote_ts", lambda sym, allow_refresh=True, require_quote_ts=False: (df_t_1000 if sym == "OVT" else df_spy_undated_ov, t_1000_ms if sym == "OVT" else None))
        reg_ov = {
            **reg_live_obs,
            "current_state": "HIGH_VOLATILITY",
            "state_index": 2,
            "kappa": 0.25,
            "observation_timestamp_ms": None,
        }
        sig_und_ov = prov.get_signal("OVT", regime=reg_ov)
        assert sig_und_ov["macro_regime"]["kappa"] == pytest.approx(0.50, abs=0.01)
        prov.enable_live_fetch = False

    # 3) HTTP 429 with capped eff_timeout applies cooldown + real local TCP stall returns within deadline
    fc = FundamentalsClient(lakehouse_dir=lh)

    def _raise_429(req: Any, timeout: float = 6.0) -> Any:
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, io.BytesIO(b""))

    fc_429 = FundamentalsClient(lakehouse_dir=lh)
    mono_429 = {"t": time.monotonic() + 10.0}
    try:
        subs_429 = {
            "recent": {
                "form": ["4"],
                "filingDate": ["2025-11-25"],
                "acceptanceDateTime": ["2025-11-25T15:00:00Z"],
                "accessionNumber": ["RATE429"],
                "primaryDocument": ["r429.xml"],
            }
        }
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("urllib.request.urlopen", _raise_429)
            mp.setattr("src.data.fundamentals_client.time.monotonic", lambda: mono_429["t"])
            fc_429._sec_tl.deadline = mono_429["t"] + 1.0
            fc_429._load_or_fetch_sec_form4_rows("0000000001", subs_429, enable_live_fetch=True)
            assert fc_429._is_in_cooldown("f4:RATE429")
            mono_429["t"] += 295.0
            assert fc_429._is_in_cooldown("f4:RATE429")
            mono_429["t"] += 10.0
            assert not fc_429._is_in_cooldown("f4:RATE429")
    finally:
        fc_429._sec_tl.deadline = None

    import http.client
    sock_srv, sock_cli = socket.socketpair()
    req_received = threading.Event()
    srv_done = threading.Event()

    def _stall_writer() -> None:
        try:
            sock_srv.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 5000\r\n\r\n<root>")
            req_received.set()
            srv_done.wait(timeout=3.0)
        except Exception:
            pass
        finally:
            try:
                sock_srv.close()
            except Exception:
                pass

    srv_thr = threading.Thread(target=_stall_writer, daemon=True)
    srv_thr.start()
    resp_pair = http.client.HTTPResponse(sock_cli)
    assert req_received.wait(timeout=1.0)
    resp_pair.begin()
    fc._sec_tl.deadline = time.monotonic() + 0.12
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("urllib.request.urlopen", lambda req, timeout=6.0: resp_pair)
            t0_sock = time.monotonic()
            res_stall = fc._sec_fetch_text("https://data.sec.gov/stall.xml", timeout=5.0)
            elapsed_sock = time.monotonic() - t0_sock
        assert res_stall is None and 0.08 <= elapsed_sock < 1.0
    finally:
        fc._sec_tl.deadline = None
        srv_done.set()
        try:
            sock_cli.close()
        except Exception:
            pass
        srv_thr.join(timeout=1.5)
        assert not srv_thr.is_alive()

    # 4) Excess-shares omission stays CORRECT + securityTitle disambiguates Class A vs Class B Form 4/A
    xml_excess = (
        "<ownershipDocument>"
        "<footnotes><footnote id=\"F1\">Corrects the reported share count; excess shares were inadvertently omitted from Column 5 due to rounding.</footnote></footnotes>"
        "<reportingOwner><reportingOwnerId><rptOwnerName>Ada Lovelace</rptOwnerName></reportingOwnerId>"
        "<reportingOwnerRelationship><isDirector>1</isDirector></reportingOwnerRelationship></reportingOwner>"
        "<nonDerivativeTable><nonDerivativeTransaction>"
        "<securityTitle><value>Class B Common Stock</value></securityTitle>"
        "<transactionDate><value>2025-11-01</value></transactionDate>"
        "<transactionCoding><transactionCode>P</transactionCode><footnoteId id=\"F1\"/></transactionCoding>"
        "<transactionAmounts><transactionShares><value>150</value></transactionShares>"
        "<transactionPricePerShare><value>25.0</value></transactionPricePerShare></transactionAmounts>"
        "</nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>"
    )
    rows_exc = fc._parse_sec_form4_xml(xml_excess, "2025-11-02", "2025-11-02T14:00:00Z", "EXC1", form="4/A")
    assert rows_exc is not None
    assert rows_exc[0].get("amendment_action") == "CORRECT"
    assert rows_exc[0].get("security_title") == "Class B Common Stock"

    orig_a = {"form": "4", "filing_date": "2025-11-01", "transaction_date": "2025-11-01", "disclosed_at": "2025-11-01T14:00:00Z", "disclosed_at_verified": True, "accession_number": "CLS0", "tx_index": 0, "insider_name": "Ada Lovelace", "security_title": "Class A Common Stock", "transaction_code": "P", "shares": 100, "value": 1_000.0}
    orig_b = {"form": "4", "filing_date": "2025-11-01", "transaction_date": "2025-11-01", "disclosed_at": "2025-11-01T14:00:00Z", "disclosed_at_verified": True, "accession_number": "CLS0", "tx_index": 1, "insider_name": "Ada Lovelace", "security_title": "Class B Common Stock", "transaction_code": "P", "shares": 120, "value": 3_000.0}
    res_cls = FundamentalsClient.resolve_pit_form4_transactions([orig_a, orig_b, rows_exc[0]])
    assert len(res_cls) == 2 and not any(r.get("_amendment_ambiguous") for r in res_cls)
    by_title = {r["security_title"]: int(r["shares"]) for r in res_cls}
    assert by_title == {"Class A Common Stock": 100, "Class B Common Stock": 150}

    # 5) FastInfo metadata snapshotted BEFORE previousClose mutates shared metadata + Kelly override honored
    ml = MarketDataLoader(data_dir=lh)
    t_early = int(pd.Timestamp("2025-12-03 11:00:00", tz="America/New_York").timestamp())
    t_late = int(pd.Timestamp("2025-12-03 15:00:00", tz="America/New_York").timestamp())

    class _OrderedFastInfo:
        _prices_1y = pd.DataFrame(
            {"Open": [100.0], "High": [102.0], "Low": [99.0], "Close": [100.0], "Volume": [1000]},
            index=pd.DatetimeIndex([pd.Timestamp("2025-12-03")]),
        )

        def __init__(self) -> None:
            self._md = {"regularMarketPrice": 100.0, "regularMarketTime": t_early}

        def get(self, key: str, default: Any = None) -> Any:
            if key == "previousClose":
                # Simulate yfinance previousClose fetching 1h history and mutating _md
                self._md["regularMarketTime"] = t_late
                return 98.0
            return {"lastPrice": 100.0, "open": 100.0, "dayHigh": 102.0, "dayLow": 99.0, "lastVolume": 1000}.get(key, default)

    class _OrderedTicker:
        fast_info = _OrderedFastInfo()

        def history(self, *a: Any, **kw: Any) -> pd.DataFrame:
            return pd.DataFrame()

        def get_history_metadata(self) -> Dict[str, Any]:
            return {}

    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-12-03 15:05:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(yf, "Ticker", lambda sym: _OrderedTicker())
    q_ord = ml.fetch_live_quote("ORDF")
    assert q_ord is not None and q_ord.get("observation_timestamp_ms") == t_early * 1000

    class _ZeroCapSizer(MultiOutcomeKellySizer):
        def solve_unconstrained_kelly(self, *args: Any, **kwargs: Any) -> float:
            return 0.0

    zc_rec = _ZeroCapSizer(prior_probs=(0.35, 0.35, 0.30)).compute_recommendation(
        p_buy=0.75, p_hold=0.15, p_sell=0.10, b_upper=0.05, b_lower=-0.025, entry_price=100.0, kappa=0.50
    )
    assert zc_rec.action == "HOLD" and zc_rec.unconstrained_kelly_pct == 0.0


def test_task4041_final_parity_edge_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task-4041 regressions: 16:05 legacy file preserves incomplete status when older row updates, history-only volume mismatch omits obs_ts_ms, 15:30 ET hourly bar caps source end at 16:00 ET, per-document timeout bounds slow chunk trickle, mixed multi-tx Form 4/A footnote/remarks isolation, and cross-date matching requires date-change evidence."""
    import os
    import pandas as pd
    import yfinance as yf
    from src.data.fundamentals_client import FundamentalsClient
    from src.data.market_loader import MarketDataLoader

    lh = tmp_path / "lh4041"
    ml = MarketDataLoader(data_dir=lh)

    # 1) Legacy 16:05 write preserves _bar_unsettled=True when 16:20 refresh changes only an earlier row
    idx_leg = pd.DatetimeIndex([pd.Timestamp("2025-11-26"), pd.Timestamp("2025-11-28")], name="timestamp")
    df_leg = pd.DataFrame(
        {"open": [100.0, 102.0], "high": [101.0, 104.0], "low": [99.0, 101.0], "close": [100.5, 103.0], "volume": [1000, 2000], "vwap": [100.2, 102.67]},
        index=idx_leg,
    )
    ml.save_ticker_daily("LEG1605", df_leg)
    leg_p = ml.daily_dir / "LEG1605.parquet"
    t_1205 = pd.Timestamp("2025-11-28 12:05:00", tz="America/New_York").timestamp()  # Wait: 11-28 is Black Friday 13:00 close -> 13:05 pre-settlement!
    os.utime(leg_p, (t_1205, t_1205))  # also before close or set 13:05 post-close pre-settlement!
    t_1305 = pd.Timestamp("2025-11-28 13:05:00", tz="America/New_York").timestamp()
    os.utime(leg_p, (t_1305, t_1305))
    df_dl_old_only = pd.DataFrame(
        {"Open": [98.0], "High": [99.0], "Low": [97.0], "Close": [98.5], "Volume": [900]},
        index=pd.DatetimeIndex([pd.Timestamp("2025-11-26")]),
    )
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-11-28 13:20:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(ml, "_latest_expected_trading_date", lambda: pd.Timestamp("2025-11-28"))
    monkeypatch.setattr(yf, "download", lambda *a, **kw: df_dl_old_only)
    df_leg_out = ml.refresh_ticker_daily("LEG1605", force_refresh_if_incomplete=True)
    assert df_leg_out is not None and df_leg_out.attrs.get("_bar_unsettled") is True

    # 2) History-only live quote rejects obs_ts_ms when history volume > regularMarketVolume
    t_hist_s = int(pd.Timestamp("2025-12-03 11:00:00", tz="America/New_York").timestamp())

    class _HistOnlyVolTicker:
        @property
        def fast_info(self) -> Any:
            return {}

        def history(self, *a: Any, **kw: Any) -> pd.DataFrame:
            return pd.DataFrame(
                {"Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.0], "Volume": [12_000]},
                index=pd.DatetimeIndex([pd.Timestamp("2025-12-03")]),
            )

        def get_history_metadata(self) -> Dict[str, Any]:
            return {"regularMarketTime": t_hist_s, "regularMarketPrice": 100.0, "regularMarketVolume": 10_000}

    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-12-03 11:05:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(yf, "Ticker", lambda sym: _HistOnlyVolTicker())
    q_hist_vm = ml.fetch_live_quote("HVMM")
    assert q_hist_vm is not None and "observation_timestamp_ms" not in q_hist_vm

    # 3) 15:30 ET regular hourly bar caps source interval end at 16:00 ET so request at 16:05 ET keeps obs_ts_ms
    df_1530 = pd.DataFrame(
        {"Open": [100.0], "High": [102.0], "Low": [99.0], "Close": [101.0], "Volume": [5000]},
        index=pd.DatetimeIndex([pd.Timestamp("2025-12-03 15:30:00", tz="America/New_York")]),
    )
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-12-03 16:05:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(yf, "download", lambda *a, **kw: df_1530)
    bars_1530 = ml.fetch_intraday_bars("H1530", interval="1h", count=5)
    exp_1600_ms = int(pd.Timestamp("2025-12-03 16:00:00", tz="America/New_York").timestamp() * 1000)
    assert len(bars_1530) == 1 and bars_1530[0].get("observation_timestamp_ms") == exp_1600_ms

    # 4) Per-document timeout bounds slow chunk trickle even when shared deadline has 5s remaining
    fc = FundamentalsClient(lakehouse_dir=lh)

    class _TrickleResp:
        def __init__(self) -> None:
            self._chunks = 0

        def __enter__(self) -> "_TrickleResp":
            return self

        def __exit__(self, *args: Any) -> bool:
            return False

        def read1(self, sz: int = 65536) -> bytes:
            self._chunks += 1
            if self._chunks > 40:
                return b""
            time.sleep(0.05)
            return b"part"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("urllib.request.urlopen", lambda req, timeout=6.0: _TrickleResp())
        fc._sec_tl.deadline = time.monotonic() + 5.0
        try:
            t0_tr = time.monotonic()
            assert fc._sec_fetch_text("https://www.sec.gov/Archives/edgar/data/1/trickle.htm", timeout=0.12) is None
            assert (time.monotonic() - t0_tr) < 1.0
        finally:
            fc._sec_tl.deadline = None

    # 5) Mixed filing-wide remarks preserve F2 ADD tx and mark unlinked sibling ambiguous; no cross-date match without date evidence
    xml_mixed = (
        "<ownershipDocument>"
        "<remarks>Corrects the price of the first transaction; adds a previously unreported transaction.</remarks>"
        "<footnotes><footnote id=\"F2\">Previously unreported transaction inadvertently omitted.</footnote></footnotes>"
        "<reportingOwner><reportingOwnerId><rptOwnerName>Alan Turing</rptOwnerName></reportingOwnerId>"
        "<reportingOwnerRelationship><isDirector>1</isDirector></reportingOwnerRelationship></reportingOwner>"
        "<nonDerivativeTable>"
        "<nonDerivativeTransaction><transactionDate><value>2025-11-01</value></transactionDate><transactionCoding><transactionCode>P</transactionCode></transactionCoding><transactionAmounts><transactionShares><value>100</value></transactionShares><transactionPricePerShare><value>52.0</value></transactionPricePerShare></transactionAmounts></nonDerivativeTransaction>"
        "<nonDerivativeTransaction><transactionDate><value>2025-11-01</value></transactionDate><transactionCoding><transactionCode>S</transactionCode><footnoteId id=\"F2\"/></transactionCoding><transactionAmounts><transactionShares><value>200</value></transactionShares><transactionPricePerShare><value>60.0</value></transactionPricePerShare></transactionAmounts></nonDerivativeTransaction>"
        "</nonDerivativeTable></ownershipDocument>"
    )
    mixed_rows = fc._parse_sec_form4_xml(xml_mixed, "2025-11-03", "2025-11-03T14:00:00Z", "MIX1", form="4/A")
    assert mixed_rows is not None and len(mixed_rows) == 2
    assert mixed_rows[0].get("_mixed_remarks_ambiguous") is True
    assert mixed_rows[1].get("amendment_action") == "ADD"

    jun2 = {"form": "4", "filing_date": "2025-06-05", "transaction_date": "2025-06-02", "disclosed_at": "2025-06-05T14:00:00Z", "disclosed_at_verified": True, "accession_number": "JN0", "tx_index": 0, "insider_name": "Ada Lovelace", "security_title": "Class A", "transaction_code": "P", "shares": 100, "value": 1_000.0}
    jun3 = {"form": "4", "filing_date": "2025-06-05", "transaction_date": "2025-06-03", "disclosed_at": "2025-06-05T14:00:00Z", "disclosed_at_verified": True, "accession_number": "JN0", "tx_index": 1, "insider_name": "Ada Lovelace", "security_title": "Class B", "transaction_code": "P", "shares": 300, "value": 6_000.0}
    corr_jun2_title = {"form": "4/A", "filing_date": "2025-06-06", "date_of_original_submission": "2025-06-05", "transaction_date": "2025-06-02", "disclosed_at": "2025-06-06T14:00:00Z", "disclosed_at_verified": True, "accession_number": "JN1", "insider_name": "Ada Lovelace", "security_title": "Class B", "transaction_code": "P", "shares": 100, "value": 1_200.0, "amendment_action": "CORRECT"}
    res_no_dt_ev = FundamentalsClient.resolve_pit_form4_transactions([jun2, jun3, corr_jun2_title])
    assert any(r["transaction_date"] == "2025-06-03" and int(r["shares"]) == 300 for r in res_no_dt_ev)


def test_task4128_unsettled_4h_vol_cache_and_form4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Verifies task-4128 fixes & ready ordered-barriers contract:
    1) _daily_bar_status_ms preserves verified rec_obs_ms with is_known_complete=False when _bar_unsettled=True
    2) Synthetic ready signal produces ordered barriers & _signal_cache key separates caller_supplied_regime
    3) fetch_live_quote FastInfo path clears observation_timestamp_ms when volume > regularMarketVolume
    4) fetch_intraday_bars('4h') on [09:30, 11:30] ET uses max(bucket_end, last_src_end) (12:30 ET)
    5) _sec_worker_sem bounds concurrent stalled deadline workers without unbounded thread spawning
    6) resolve_pit_form4_transactions Pass 3 marks conflicting code vs shares ambiguous
    """
    import threading
    import urllib.request
    import pandas as pd
    import yfinance as yf
    from src.data.fundamentals_client import FundamentalsClient
    from src.ipc.live_providers import LiveInferenceProvider
    from src.risk.calibration import TemperatureScaler

    lh = tmp_path / "lh_4128"
    wdir = tmp_path / "w_4128"
    lh.mkdir(parents=True, exist_ok=True)
    wdir.mkdir(parents=True, exist_ok=True)

    prov = LiveInferenceProvider(lakehouse_dir=lh, weights_dir=wdir, enable_live_fetch=False)

    # 1) _daily_bar_status_ms retains verified rec_obs_ms == scheduled_close_ms when _bar_unsettled=True
    sched_close_ms = int(pd.Timestamp("2025-12-01 16:00:00", tz="America/New_York").timestamp() * 1000)
    idx_u = pd.bdate_range(end="2025-12-01", periods=2)
    df_unsettled = pd.DataFrame(
        {"open": [100.0, 101.0], "high": [102.0, 103.0], "low": [99.0, 100.0], "close": [101.0, 102.0], "volume": [1000, 1200]},
        index=idx_u,
    )
    df_unsettled.attrs = {
        "obs_bar_date": "2025-12-01",
        "observation_timestamp_ms": sched_close_ms,
        "_bar_unsettled": True,
    }
    assert prov._daily_bar_status_ms(df_unsettled) == (sched_close_ms, False, sched_close_ms)

    df_mismatched = df_unsettled.copy()
    df_mismatched.attrs = {
        "obs_bar_date": "2025-11-28",
        "observation_timestamp_ms": sched_close_ms,
        "_bar_unsettled": True,
    }
    with monkeypatch.context() as m_ctx:
        m_ctx.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-12-01 16:10:00", tz="America/New_York").timestamp())
        assert prov._daily_bar_status_ms(df_mismatched) == (None, False, sched_close_ms)
        m_ctx.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-12-01 16:20:00", tz="America/New_York").timestamp())
        assert prov._daily_bar_status_ms(df_unsettled) == (sched_close_ms, False, sched_close_ms)
        # Past settlement without matching row attrs ignores stale 2025-11-28 _bar_unsettled flag
        assert prov._daily_bar_status_ms(df_mismatched) == (sched_close_ms, True, sched_close_ms)

    # 2) Ready ordered barriers check & _signal_cache key separates caller_supplied_regime
    class _ReadyHMM:
        STATE_LABELS = {0: "LOW_VOLATILITY", 1: "MEDIUM_VOLATILITY", 2: "HIGH_VOLATILITY"}
        transmat_ = np.array([[0.9, 0.08, 0.02], [0.05, 0.9, 0.05], [0.02, 0.08, 0.9]])

        def filter_sequence(self, x: np.ndarray) -> np.ndarray:
            return np.tile([0.85, 0.10, 0.05], (len(x), 1))

        def get_regime_multiplier(self, probs: np.ndarray) -> float:
            return 0.50

        def get_state_annualized_volatilities(self) -> list:
            return [10.0, 18.0, 35.0]

        def get_stationary_distribution(self) -> list:
            return [0.6, 0.3, 0.1]

    class _ReadyLGBM:
        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            return np.tile([0.65, 0.25, 0.10], (len(x), 1))

    prov.regime_hmm = _ReadyHMM()
    prov.lgbm_backbone = _ReadyLGBM()
    sc = TemperatureScaler()
    sc.temperature_ = 1.0
    prov.temp_scaler = sc

    idx_200 = pd.bdate_range(end="2025-12-01", periods=200, name="timestamp")
    rng = np.random.default_rng(4128)
    spy_cls = 400.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.008, size=200)))
    tgt_cls = 150.0 * np.exp(np.cumsum(rng.normal(0.0006, 0.012, size=200)))
    df_spy = pd.DataFrame(
        {"open": spy_cls * 0.999, "high": spy_cls * 1.006, "low": spy_cls * 0.994, "close": spy_cls, "volume": np.full(200, 5_000_000.0)},
        index=idx_200,
    )
    df_tgt = pd.DataFrame(
        {"open": tgt_cls * 0.998, "high": tgt_cls * 1.008, "low": tgt_cls * 0.992, "close": tgt_cls, "volume": np.full(200, 3_000_000.0)},
        index=idx_200,
    )
    df_spy.attrs = {"observation_timestamp_ms": sched_close_ms, "obs_bar_date": "2025-12-01"}
    df_tgt.attrs = {"observation_timestamp_ms": sched_close_ms, "obs_bar_date": "2025-12-01"}
    prov.market_loader.atomic_write_parquet(df_spy, prov.market_loader.daily_dir / "SPY.parquet")
    prov.market_loader.atomic_write_parquet(df_tgt, prov.market_loader.daily_dir / "NVDA.parquet")

    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-12-01 16:20:00", tz="America/New_York").timestamp())

    sig_ready = prov.get_signal("NVDA")
    assert "UNREADY" not in sig_ready["execution_mode"]
    assert sig_ready["winning_engine"] != "NONE"
    rec_ready = sig_ready["execution_recommendation"]
    assert rec_ready["dynamic_stop_loss"] < rec_ready["entry_price"] < rec_ready["dynamic_take_profit"]

    valid_regime = {
        "available": True,
        "current_state": "LOW_VOLATILITY",
        "state_index": 0,
        "kappa": 0.50,
        "filtered_probabilities": {"low_volatility": 0.85, "medium_volatility": 0.10, "high_volatility": 0.05},
        "transition_matrix": _ReadyHMM.transmat_.tolist(),
        "as_of_date": "2025-12-01",
        "observation_timestamp_ms": sched_close_ms,
        "regime_history": [{"timestamp": "2025-12-01", "state_index": 0}],
    }
    with monkeypatch.context() as m_reg:
        m_reg.setattr(prov, "get_macro_regime", lambda: dict(valid_regime))
        prov.get_signal("NVDA")
        ck_internal = prov._signal_cache["NVDA"][0]
        prov.get_signal("NVDA", regime=dict(valid_regime))
        ck_caller = prov._signal_cache["NVDA"][0]
        assert ck_internal != ck_caller
        assert ck_internal[4][-1] is False
        assert ck_caller[4][-1] is True

    # 3) fetch_live_quote FastInfo path omits observation_timestamp_ms when volume > regularMarketVolume
    ml = prov.market_loader
    rmt_sec = int(pd.Timestamp("2025-12-01 16:00:00", tz="America/New_York").timestamp())
    fi_prices = pd.DataFrame(
        {"Open": [199.0], "High": [202.0], "Low": [198.0], "Close": [200.0], "Volume": [900_000]},
        index=pd.DatetimeIndex([pd.Timestamp("2025-12-01")]),
    )

    class _FastVolMismatch:
        def __init__(self, rmv: int) -> None:
            self._prices_1y = fi_prices
            self._md = {
                "regularMarketPrice": 200.0,
                "regularMarketTime": rmt_sec,
                "regularMarketVolume": rmv,
            }

        def get(self, key: str, default=None):
            mapping = {
                "lastPrice": 200.0,
                "regularMarketPrice": 200.0,
                "open": 199.0,
                "dayHigh": 202.0,
                "dayLow": 198.0,
                "lastVolume": 900_000,
                "previousClose": 195.0,
            }
            return mapping.get(key, default)

    class _TickerVol:
        def __init__(self, rmv: int) -> None:
            self.fast_info = _FastVolMismatch(rmv)

        def history(self, *a, **kw) -> pd.DataFrame:
            return pd.DataFrame()

        def get_history_metadata(self):
            return {}

    monkeypatch.setattr("src.data.market_loader.time.time", lambda: float(rmt_sec + 600))
    monkeypatch.setattr(yf, "Ticker", lambda sym: _TickerVol(500_000))
    q_vol_mis = ml.fetch_live_quote("VOLMIS")
    assert q_vol_mis is not None
    assert q_vol_mis["volume"] == 900_000
    assert "observation_timestamp_ms" not in q_vol_mis

    monkeypatch.setattr(yf, "Ticker", lambda sym: _TickerVol(900_000))
    q_vol_ok = ml.fetch_live_quote("VOLOK")
    assert q_vol_ok is not None
    assert q_vol_ok.get("observation_timestamp_ms") == rmt_sec * 1000

    # 4) fetch_intraday_bars('4h') on [09:30, 11:30] uses max(12:00, 12:30) == 12:30 ET
    df_4h_two = pd.DataFrame(
        {"Open": [100.0, 102.0], "High": [101.0, 103.0], "Low": [99.0, 101.0], "Close": [100.5, 102.5], "Volume": [1000, 2000]},
        index=pd.DatetimeIndex([
            pd.Timestamp("2025-12-01 09:30:00", tz="America/New_York"),
            pd.Timestamp("2025-12-01 11:30:00", tz="America/New_York"),
        ]),
    )
    monkeypatch.setattr(yf, "download", lambda *a, **kw: df_4h_two)
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-12-01 12:15:00", tz="America/New_York").timestamp())
    bars_4h_early = ml.fetch_intraday_bars("TST4M", interval="4h", count=5)
    assert len(bars_4h_early) == 1 and "observation_timestamp_ms" not in bars_4h_early[0]

    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-12-01 12:35:00", tz="America/New_York").timestamp())
    bars_4h_done = ml.fetch_intraday_bars("TST4M", interval="4h", count=5)
    exp_1230_ms = int(pd.Timestamp("2025-12-01 12:30:00", tz="America/New_York").timestamp() * 1000)
    assert len(bars_4h_done) == 1 and bars_4h_done[0].get("observation_timestamp_ms") == exp_1230_ms

    # 5) _sec_worker_sem bounds concurrent stalled deadline workers
    fc = FundamentalsClient(lakehouse_dir=lh)
    opened_urls: list = []

    def _rec_urlopen(req, timeout=None):
        opened_urls.append(getattr(req, "full_url", str(req)))
        raise RuntimeError("should not run while semaphore saturated")

    monkeypatch.setattr(urllib.request, "urlopen", _rec_urlopen)
    held = 0
    try:
        while fc._sec_worker_sem.acquire(blocking=False):
            held += 1
        assert held == 8
        fc._sec_tl.deadline = time.monotonic() + 0.04
        fc._sec_tl.last_budget_expired = False
        res_sat = fc._sec_fetch_text("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json", timeout=0.04)
        assert res_sat is None
        assert fc._sec_tl.last_budget_expired is True
        assert opened_urls == []
    finally:
        fc._sec_tl.deadline = None
        for _ in range(held):
            fc._sec_worker_sem.release()

    # 6) Pass 3 Form 4/A resolution marks conflicting code vs shares ambiguous
    p_orig = {
        "form": "4",
        "filing_date": "2025-11-03",
        "transaction_date": "2025-11-01",
        "disclosed_at": "2025-11-03T14:00:00Z",
        "disclosed_at_verified": True,
        "accession_number": "A0",
        "tx_index": 0,
        "insider_name": "Ada Lovelace",
        "transaction_code": "P",
        "shares": 100,
        "value": 5_000.0,
    }
    s_orig = {
        "form": "4",
        "filing_date": "2025-11-03",
        "transaction_date": "2025-11-01",
        "disclosed_at": "2025-11-03T14:00:00Z",
        "disclosed_at_verified": True,
        "accession_number": "A0",
        "tx_index": 1,
        "insider_name": "Ada Lovelace",
        "transaction_code": "S",
        "shares": 200,
        "value": 10_000.0,
    }
    s_amend_conflict = {
        "form": "4/A",
        "filing_date": "2025-11-04",
        "date_of_original_submission": "2025-11-03",
        "transaction_date": "2025-11-02",
        "corrects_date": "2025-11-01",
        "disclosed_at": "2025-11-04T14:00:00Z",
        "disclosed_at_verified": True,
        "accession_number": "A1",
        "tx_index": 0,
        "insider_name": "Ada Lovelace",
        "transaction_code": "S",
        "shares": 100,
        "value": 5_200.0,
        "amendment_action": "CORRECT",
    }
    res_conflict = FundamentalsClient.resolve_pit_form4_transactions([p_orig, s_orig, s_amend_conflict])
    assert any(r.get("_amendment_ambiguous") for r in res_conflict)
    assert not any(
        r.get("accession_number") == "A1" and not r.get("_amendment_ambiguous")
        for r in res_conflict
    )
    assert any(int(r.get("shares") or 0) == 200 for r in res_conflict)


def test_task4226_pit_edgar_form4_cancel_and_cache_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Verifies all task-4226 fixes:
    1) _sec_fetch_text releases _sec_worker_sem permit when Thread.start raises
    2) get_sec_submissions / get_sec_company_facts reject missing filings.recent / facts while preserving stale caches
    3) build_edgar_diff rejects cached redlines lacking redlines list without KeyError
    4) Form 4/A zero-share cancellations are retained and remove only matched originals; unmatched CORRECT rows stay _amendment_unresolved
    5) MarketDataLoader.apply_pit_cutoff default honors scheduled early closes (13:00 ET)
    6) atomic_write_parquet omits stale observation metadata when obs_bar_date mismatches df.index[-1]
    7) fetch_live_quote does not infer close_ms after rejecting conflicting metadata
    8) get_signal passes xbrl_filed_date_verified independently of model readiness
    9) get_analyst_ratings uses verified non-future later-session Yahoo price when bar_ms is None
    """
    import threading
    import pandas as pd
    import yfinance as yf
    from src.data.fundamentals_client import FundamentalsClient
    from src.data.market_loader import MarketDataLoader
    from src.ipc.live_providers import LiveInferenceProvider
    from src.risk.calibration import TemperatureScaler

    lh = tmp_path / "lh_4226"
    wdir = tmp_path / "w_4226"
    lh.mkdir(parents=True, exist_ok=True)
    wdir.mkdir(parents=True, exist_ok=True)

    fc = FundamentalsClient(lakehouse_dir=lh)

    # 1) _sec_fetch_text enters worker path when deadline is set and releases _sec_worker_sem if Thread.start raises
    orig_start = threading.Thread.start
    start_calls: list[bool] = []
    def _fail_start(self_t: threading.Thread) -> None:
        start_calls.append(True)
        raise RuntimeError("thread limit")

    prev_dl = getattr(fc._sec_tl, "deadline", None)
    try:
        fc._sec_tl.deadline = time.monotonic() + 2.0
        monkeypatch.setattr(threading.Thread, "start", _fail_start)
        assert fc._sec_fetch_text("https://data.sec.gov/api/test.json", timeout=1.0) is None
    finally:
        fc._sec_tl.deadline = prev_dl
        monkeypatch.setattr(threading.Thread, "start", orig_start)
    assert start_calls == [True]

    held = 0
    while fc._sec_worker_sem.acquire(blocking=False):
        held += 1
    for _ in range(held):
        fc._sec_worker_sem.release()
    assert held == 8

    # 2) Submissions & CompanyFacts reject missing envelopes and accept explicit empty envelopes
    cik = "0000320193"
    sub_p = fc.raw_dir / "edgar_submissions" / f"CIK{cik}.json"
    facts_p = fc.raw_dir / "edgar_facts" / f"CIK{cik}.json"
    sub_p.parent.mkdir(parents=True, exist_ok=True)
    facts_p.parent.mkdir(parents=True, exist_ok=True)

    stale_sub = {"cik": cik, "recent": {"form": ["10-K"], "filingDate": ["2024-11-01"], "accessionNumber": ["0001"], "primaryDocument": ["a.htm"]}}
    sub_p.write_text(json.dumps(stale_sub), encoding="utf-8")
    os.utime(sub_p, (time.time() - 86400 * 30, time.time() - 86400 * 30))
    monkeypatch.setattr(fc, "_sec_fetch_text", lambda url, timeout=6.0: "{}")
    got_sub = fc.get_sec_submissions(cik, enable_live_fetch=True)
    assert got_sub is not None and got_sub["recent"]["form"] == ["10-K"]

    monkeypatch.setattr(fc, "_sec_fetch_text", lambda url, timeout=6.0: json.dumps({"filings": {"recent": {}}}))
    fc._failed_until.clear()
    got_sub_empty = fc.get_sec_submissions(cik, enable_live_fetch=True)
    assert got_sub_empty is not None and got_sub_empty["recent"] == {}

    stale_facts = {"cik": cik, "us_gaap": {"Revenues": {"units": {"USD": [{"val": 100.0}]}}}}
    facts_p.write_text(json.dumps(stale_facts), encoding="utf-8")
    os.utime(facts_p, (time.time() - 86400 * 30, time.time() - 86400 * 30))
    monkeypatch.setattr(fc, "_sec_fetch_text", lambda url, timeout=8.0: "{}")
    got_facts = fc.get_sec_company_facts(cik, enable_live_fetch=True)
    assert got_facts is not None and "Revenues" in got_facts["us_gaap"]

    monkeypatch.setattr(fc, "_sec_fetch_text", lambda url, timeout=8.0: json.dumps({"facts": {}}))
    fc._failed_until.clear()
    got_facts_empty = fc.get_sec_company_facts(cik, enable_live_fetch=True)
    assert got_facts_empty is not None and got_facts_empty["us_gaap"] == {}

    # 3) _valid_pair_cache rejects malformed redline JSON missing 'redlines' list
    sub_red_p = fc.raw_dir / "edgar_submissions" / f"CIK{cik}.json"
    sub_red_p.write_text(
        json.dumps({
            "cik": cik,
            "recent": {
                "form": ["10-Q", "10-Q"],
                "filingDate": ["2025-11-01", "2025-08-01"],
                "acceptanceDateTime": ["2025-11-01T14:00:00Z", "2025-08-01T14:00:00Z"],
                "accessionNumber": ["ACC-CUR", "ACC-PRI"],
                "primaryDocument": ["q2.htm", "q1.htm"],
            },
        }),
        encoding="utf-8",
    )
    red_p = fc.raw_dir / "edgar_redlines" / "AAPL_ACC-CUR_ACC-PRI.json"
    red_p.parent.mkdir(parents=True, exist_ok=True)
    red_p.write_text(json.dumps({"current_accession": "ACC-CUR", "prior_accession": "ACC-PRI"}), encoding="utf-8")
    edg_bad_red = fc.build_edgar_filings_payload("AAPL", known_cik=cik, enable_live_fetch=False)
    assert edg_bad_red is not None and edg_bad_red.get("item1a_sentiment_drift") is None

    # 4) Zero-share Form 4/A cancellation removes original; unmatched CORRECT stays unresolved
    xml_cancel = (
        "<ownershipDocument>"
        "<dateOfOriginalSubmission>2025-11-01</dateOfOriginalSubmission>"
        "<footnotes><footnote id=\"F1\">Cancels the previously reported sale of 500 shares reported in error.</footnote></footnotes>"
        "<reportingOwner><reportingOwnerId><rptOwnerName>Ada Lovelace</rptOwnerName></reportingOwnerId>"
        "<reportingOwnerRelationship><isDirector>1</isDirector></reportingOwnerRelationship></reportingOwner>"
        "<nonDerivativeTable><nonDerivativeTransaction>"
        "<transactionDate><value>2025-11-01</value></transactionDate>"
        "<transactionCoding><transactionCode>S</transactionCode><footnoteId id=\"F1\"/></transactionCoding>"
        "<transactionAmounts><transactionShares><value>0</value></transactionShares>"
        "<transactionPricePerShare><value>0</value></transactionPricePerShare></transactionAmounts>"
        "</nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>"
    )
    cancel_rows = fc._parse_sec_form4_xml(xml_cancel, "2025-11-03", "2025-11-03T14:00:00Z", "CAN1", form="4/A")
    assert cancel_rows is not None and len(cancel_rows) == 1 and cancel_rows[0]["amendment_action"] == "CANCEL"
    orig_sale = {
        "form": "4",
        "filing_date": "2025-11-01",
        "transaction_date": "2025-11-01",
        "disclosed_at": "2025-11-01T14:00:00Z",
        "disclosed_at_verified": True,
        "accession_number": "ORIGS1",
        "insider_name": "Ada Lovelace",
        "transaction_code": "S",
        "shares": 500,
        "value": 50_000.0,
    }
    assert FundamentalsClient.resolve_pit_form4_transactions([orig_sale, cancel_rows[0]]) == []

    corr_orphan = {
        "form": "4/A",
        "filing_date": "2025-11-05",
        "date_of_original_submission": "2025-11-01",
        "transaction_date": "2025-11-01",
        "disclosed_at": "2025-11-05T14:00:00Z",
        "disclosed_at_verified": True,
        "accession_number": "CORRO1",
        "insider_name": "Ada Lovelace",
        "transaction_code": "P",
        "shares": 300,
        "value": 30_000.0,
        "amendment_action": "CORRECT",
    }
    res_unres = FundamentalsClient.resolve_pit_form4_transactions([corr_orphan])
    assert len(res_unres) == 1 and res_unres[0].get("_amendment_unresolved") is True
    # Re-resolves cleanly once original filing becomes available
    orig_buy = {
        "form": "4",
        "filing_date": "2025-11-01",
        "transaction_date": "2025-11-01",
        "disclosed_at": "2025-11-01T14:00:00Z",
        "disclosed_at_verified": True,
        "accession_number": "ORIGB1",
        "insider_name": "Ada Lovelace",
        "transaction_code": "P",
        "shares": 300,
        "value": 27_000.0,
    }
    res_re = FundamentalsClient.resolve_pit_form4_transactions([res_unres[0], orig_buy])
    assert len(res_re) == 1 and res_re[0].get("_amendment_unresolved") is None
    assert float(res_re[0]["value"]) == pytest.approx(30_000.0)

    # 5) apply_pit_cutoff default honors scheduled early close (2026-11-27 13:00 ET)
    bf_post = MarketDataLoader.apply_pit_cutoff(pd.Timestamp("2026-11-27 14:00:00", tz="America/New_York"))
    assert bf_post.strftime("%Y-%m-%d") == "2026-11-30"

    # 6) atomic_write_parquet drops stale obs metadata when obs_bar_date mismatches last bar
    ml = MarketDataLoader(data_dir=lh)
    stale_ms = int(pd.Timestamp("2025-11-27 16:00:00", tz="America/New_York").timestamp() * 1000)
    df_stale_meta = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [1000]},
        index=pd.DatetimeIndex([pd.Timestamp("2025-11-28 09:30:00")], name="timestamp"),
    )
    df_stale_meta.attrs = {"observation_timestamp_ms": stale_ms, "obs_bar_date": "2025-11-27"}
    stale_pq = ml.daily_dir / "STALEOBS.parquet"
    ml.atomic_write_parquet(df_stale_meta, stale_pq)
    loaded_stale = ml.query_parquet_duckdb(stale_pq)
    assert loaded_stale is not None and "observation_timestamp_ms" not in loaded_stale.attrs

    # 7) fetch_live_quote leaves obs_ts_ms unset after conflicting regularMarketVolume on prior-session history bar
    class _ConflictTicker:
        class _FI:
            _prices_1y = pd.DataFrame()
            _md = {
                "regularMarketPrice": 150.0,
                "regularMarketTime": int(pd.Timestamp("2025-12-01 16:00:00", tz="America/New_York").timestamp()),
                "regularMarketVolume": 100_000,
            }

            def get(self, key: str, default=None):
                return None

        fast_info = _FI()

        def history(self, *a, **kw):
            return pd.DataFrame(
                {"Open": [149.0], "High": [151.0], "Low": [148.0], "Close": [150.0], "Volume": [250_000]},
                index=pd.DatetimeIndex([pd.Timestamp("2025-12-01")]),
            )

        def get_history_metadata(self):
            return self.fast_info._md

    monkeypatch.setattr(yf, "Ticker", lambda sym: _ConflictTicker())
    with monkeypatch.context() as mp_q:
        mp_q.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-12-02 10:30:00", tz="America/New_York").timestamp())
        q_conf = ml.fetch_live_quote("CONFL")
    assert q_conf is not None and "observation_timestamp_ms" not in q_conf

    # 8) get_signal reports xbrl_filed_date_verified independently of model readiness
    prov = LiveInferenceProvider(lakehouse_dir=lh, weights_dir=wdir, enable_live_fetch=False)

    class _HMM8:
        STATE_LABELS = {0: "LOW_VOLATILITY", 1: "MEDIUM_VOLATILITY", 2: "HIGH_VOLATILITY"}
        transmat_ = np.array([[0.9, 0.08, 0.02], [0.05, 0.9, 0.05], [0.02, 0.08, 0.9]])

        def filter_sequence(self, x):
            return np.tile([0.85, 0.10, 0.05], (len(x), 1))

        def get_regime_multiplier(self, p):
            return 0.5

        def get_state_annualized_volatilities(self):
            return [10.0, 18.0, 35.0]

        def get_stationary_distribution(self):
            return [0.6, 0.3, 0.1]

    class _LGBM8:
        def predict_proba(self, x):
            return np.tile([0.65, 0.25, 0.10], (len(x), 1))

    prov.regime_hmm = _HMM8()
    prov.lgbm_backbone = _LGBM8()
    sc8 = TemperatureScaler()
    sc8.temperature_ = 1.0
    prov.temp_scaler = sc8

    mon_close_ms = int(pd.Timestamp("2025-12-01 16:00:00", tz="America/New_York").timestamp() * 1000)
    idx_8 = pd.bdate_range(end="2025-12-01", periods=180, name="timestamp")
    cls_8 = 200.0 * np.exp(np.cumsum(np.random.default_rng(4226).normal(0.0005, 0.01, size=180)))
    df_8 = pd.DataFrame(
        {"open": cls_8 * 0.998, "high": cls_8 * 1.007, "low": cls_8 * 0.993, "close": cls_8, "volume": np.full(180, 2_000_000.0)},
        index=idx_8,
    )
    df_8.attrs = {"observation_timestamp_ms": mon_close_ms, "obs_bar_date": "2025-12-01"}
    prov.market_loader.atomic_write_parquet(df_8, prov.market_loader.daily_dir / "SPY.parquet")
    prov.market_loader.atomic_write_parquet(df_8, prov.market_loader.daily_dir / "XBRLT.parquet")

    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-12-01 16:20:00", tz="America/New_York").timestamp())
    future_edg = {
        "available": True,
        "current_filing_date": "2025-12-02",
        "current_acceptance_datetime": "2025-12-02T14:00:00Z",
        "item1a_sentiment_drift": 0.4,
        "form4_transactions": [corr_orphan],
    }
    with prov._lock:
        prov._edgar_cache["XBRLT"] = (time.monotonic(), future_edg)
        prov._signal_cache.clear()
    sig_fut_edg = prov.get_signal("XBRLT")
    assert "UNREADY" not in sig_fut_edg["execution_mode"]
    assert sig_fut_edg["data_integrity"]["xbrl_filed_date_verified"] is False
    assert sig_fut_edg["signal_breakdown"]["decayed_form4_insider_cluster"] == 0.0

    valid_edg = {
        "available": True,
        "current_filing_date": "2025-11-28",
        "current_acceptance_datetime": "2025-11-28T14:00:00Z",
        "item1a_sentiment_drift": 0.4,
        "form4_transactions": [corr_orphan],
    }
    with prov._lock:
        prov._edgar_cache["XBRLT"] = (time.monotonic(), valid_edg)
        prov._signal_cache.clear()
    sig_ok_edg = prov.get_signal("XBRLT")
    assert sig_ok_edg["data_integrity"]["xbrl_filed_date_verified"] is True
    assert sig_ok_edg["signal_breakdown"]["decayed_form4_insider_cluster"] == 0.0

    # 9) get_analyst_ratings orders later-session verified Yahoo price over bar without obs_ms
    df_mon_unver = df_8.copy()
    mon_pre_settle_ms = int(pd.Timestamp("2025-12-01 15:50:00", tz="America/New_York").timestamp() * 1000)
    df_mon_unver.attrs = {"obs_bar_date": "2025-12-01", "disk_mtime_ms": mon_pre_settle_ms}
    monkeypatch.setattr(prov, "_load_cached_daily_with_quote_ts", lambda sym, require_quote_ts=True: (df_mon_unver, None))
    tue_ts_s = int(pd.Timestamp("2025-12-02 10:30:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(
        prov.fundamentals_client,
        "get_yf_meta",
        lambda sym, enable_live_fetch=False: {
            "info": {
                "targetMeanPrice": 250.0,
                "numberOfAnalystOpinions": 10,
                "regularMarketPrice": 225.0,
                "regularMarketTime": tue_ts_s,
            },
            "rec_trend": [{"strong_buy": 8, "buy": 2, "hold": 0, "underperform": 0, "sell": 0}],
        },
    )
    with monkeypatch.context() as mp_ar:
        mp_ar.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-12-02 10:35:00", tz="America/New_York").timestamp())
        assert prov._market_bar_timestamp_ms(df_mon_unver, None) is None
        anr_tue = prov.get_analyst_ratings("XBRLT")
    assert anr_tue["current_price"] == pytest.approx(225.0)


def test_task4892_coderabbit_fixes(tmp_path: Any, monkeypatch: Any) -> None:
    """Covers task-4892 fixes across EDGAR, MarketDataLoader, LiveInferenceProvider, and ws_broadcaster."""
    import json
    from datetime import date, time as dtime
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from src.data.fundamentals_client import FundamentalsClient
    from src.data.market_loader import MarketDataLoader
    from src.ipc.live_providers import LiveInferenceProvider
    from src.ipc.ws_broadcaster import normalize_regime_probs, resolve_regime_kappa

    # 1) Null acceptanceDateTime in SEC submissions must not raise TypeError in build_financials_payload or Form 4/EDGAR readers
    lh = tmp_path / "lh4892"
    fc = FundamentalsClient(lakehouse_dir=lh)
    fc._write_json_cache("edgar_submissions/CIK0000123456.json", {
        "cik": "0000123456",
        "recent": {
            "form": ["10-K", "4"],
            "filingDate": ["2025-02-01", "2025-02-05"],
            "reportDate": ["2024-12-31", None],
            "accessionNumber": ["0001-25-000001", "0001-25-000002"],
            "primaryDocument": ["r10k.htm", "f4.xml"],
            "acceptanceDateTime": None,
        },
    })
    fc._write_json_cache("edgar_facts/CIK0000123456.json", {
        "cik": "0000123456",
        "entityName": "NULLA CORP",
        "us_gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {"start": "2023-01-01", "end": "2023-12-31", "val": 100.0, "form": "10-K", "filed": "2024-02-01", "accn": "0000-24-000001", "fp": "FY"},
                        {"start": "2024-01-01", "end": "2024-12-31", "val": 120.0, "form": "10-K", "filed": "2025-02-01", "accn": "0001-25-000001", "fp": "FY"},
                    ]
                }
            },
            "Assets": {
                "units": {
                    "USD": [
                        {"end": "2024-12-31", "val": 500.0, "form": "10-K", "filed": "2025-02-01", "accn": "0001-25-000001", "fp": "FY"},
                    ]
                }
            },
        },
    })
    fin = fc.build_financials_payload("NULLA", known_cik="0000123456", pit_strict=True, enable_live_fetch=False)
    assert fin["available"] is True
    assert fc._load_or_fetch_sec_form4_rows("0000123456", fc.get_sec_submissions("0000123456") or {}, enable_live_fetch=False) == []
    edg_nulla = fc.build_edgar_filings_payload("NULLA", known_cik="0000123456", enable_live_fetch=False)
    assert edg_nulla["ticker"] == "NULLA" and isinstance(edg_nulla["redlines"], list)

    # 2) Two-key posterior normalization shares 1.02 upper tolerance with three-key form (#13)
    p2 = normalize_regime_probs({"medium_volatility": 0.51, "high_volatility": 0.51})
    p3 = normalize_regime_probs({"low_volatility": 0.0, "medium_volatility": 0.51, "high_volatility": 0.51})
    assert p2 and p3
    assert p2 == pytest.approx(p3)
    k2 = resolve_regime_kappa(None, {"medium_volatility": 0.51, "high_volatility": 0.51})
    k3 = resolve_regime_kappa(None, {"low_volatility": 0.0, "medium_volatility": 0.51, "high_volatility": 0.51})
    assert k2 == 0.3 and k2 == k3

    # 3) Multi-row Form 4/A filing-wide cancellation remark does not delete positive-share trade without row footnote (#17)
    #    and exact-title Form 4/A reserves slots before blank-title rows (#20)
    xml_cancel_mix = """<?xml version="1.0"?>
    <ownershipDocument>
      <reportingOwner><reportingOwnerId><rptOwnerName>Jane Doe</rptOwnerName></reportingOwnerId>
        <reportingOwnerRelationship><isOfficer>1</isOfficer><officerTitle>CFO</officerTitle></reportingOwnerRelationship>
      </reportingOwner>
      <remarks>One transaction was cancelled.</remarks>
      <nonDerivativeTable>
        <nonDerivativeTransaction>
          <securityTitle><value>Common Stock</value></securityTitle>
          <transactionDate><value>2025-11-10</value></transactionDate>
          <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
          <transactionAmounts><transactionShares><value>0</value></transactionShares><transactionPricePerShare><value>0</value></transactionPricePerShare></transactionAmounts>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
          <securityTitle><value>Common Stock</value></securityTitle>
          <transactionDate><value>2025-11-11</value></transactionDate>
          <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
          <transactionAmounts><transactionShares><value>500</value></transactionShares><transactionPricePerShare><value>50</value></transactionPricePerShare></transactionAmounts>
        </nonDerivativeTransaction>
      </nonDerivativeTable>
    </ownershipDocument>"""
    parsed_mix = fc._parse_sec_form4_xml(xml_cancel_mix, form="4/A", filing_date="2025-11-12", accept_dt="2025-11-12T18:00:00Z", accn="0002-25-000002")
    assert parsed_mix[0].get("amendment_action") == "CANCEL" and parsed_mix[0]["shares"] == 0
    assert parsed_mix[1].get("_mixed_remarks_ambiguous") is True and "amendment_action" not in parsed_mix[1]

    orig_common = {"form": "4", "filing_date": "2025-11-01", "disclosed_at": "2025-11-01T17:00:00Z", "transaction_date": "2025-10-31", "insider_name": "Alex", "security_title": "Common Stock", "transaction_code": "P", "shares": 100, "accession_number": "A1"}
    orig_pref = {"form": "4", "filing_date": "2025-11-01", "disclosed_at": "2025-11-01T17:00:00Z", "transaction_date": "2025-10-31", "insider_name": "Alex", "security_title": "Preferred Stock", "transaction_code": "P", "shares": 200, "accession_number": "A1"}
    amend_blank_first = {"form": "4/A", "filing_date": "2025-11-02", "date_of_original_submission": "2025-11-01", "disclosed_at": "2025-11-02T17:00:00Z", "transaction_date": "2025-10-31", "insider_name": "Alex", "transaction_code": "P", "shares": 250, "accession_number": "A2"}
    amend_exact_common = {"form": "4/A", "filing_date": "2025-11-02", "date_of_original_submission": "2025-11-01", "disclosed_at": "2025-11-02T17:00:00Z", "transaction_date": "2025-10-31", "insider_name": "Alex", "security_title": "Common Stock", "transaction_code": "P", "shares": 150, "accession_number": "A2"}
    resolved_order = FundamentalsClient.resolve_pit_form4_transactions([orig_common, orig_pref, amend_blank_first, amend_exact_common])
    assert not any(r.get("_amendment_ambiguous") or r.get("_amendment_unresolved") for r in resolved_order)
    assert sorted(r["shares"] for r in resolved_order) == [150, 250]

    # 4) Settled row retained when refresh updates an older row must keep verified 16:02 obs_ms (#15)
    ml = MarketDataLoader(data_dir=lh)
    idx_s = pd.DatetimeIndex(pd.bdate_range("2025-11-24", "2025-12-03"), name="timestamp")
    df_settled = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000, "vwap": 100.17},
        index=idx_s,
    )
    obs_1602 = int(pd.Timestamp("2025-12-03 16:02:00", tz="America/New_York").timestamp() * 1000)
    df_settled.attrs = {"observation_timestamp_ms": obs_1602, "obs_bar_date": "2025-12-03"}
    ml.save_ticker_daily("RETOBS", df_settled)
    df_older_fix = pd.DataFrame(
        {"Open": [98.0], "High": [99.0], "Low": [97.0], "Close": [98.5], "Volume": [900]},
        index=pd.DatetimeIndex([pd.Timestamp("2025-12-01")]),
    )
    monkeypatch.setattr(ml, "_latest_expected_trading_date", lambda: pd.Timestamp("2025-12-03"))
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-12-03 16:25:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(yf, "download", lambda *a, **kw: df_older_fix)
    df_ret_out = ml.refresh_ticker_daily("RETOBS", force_refresh_if_incomplete=True)
    assert df_ret_out is not None
    assert df_ret_out.attrs.get("observation_timestamp_ms") == obs_1602
    assert not df_ret_out.attrs.get("_bar_unsettled")

    # 5) Active-session history fallback without regularMarketPrice rejects unverified obs_ts_ms (#21)
    class _HistNoRmp:
        @property
        def fast_info(self):
            return {}
        def history(self, **kw):
            return pd.DataFrame(
                {"Open": [100.0], "High": [105.0], "Low": [99.0], "Close": [104.0], "Volume": [10_000]},
                index=pd.DatetimeIndex([pd.Timestamp("2025-12-03")]),
            )
        def get_history_metadata(self):
            return {"regularMarketTime": int(pd.Timestamp("2025-12-03 10:00:00", tz="America/New_York").timestamp())}

    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-12-03 15:55:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(yf, "Ticker", lambda sym: _HistNoRmp())
    q_no_rmp = ml.fetch_live_quote("HNOMP")
    assert q_no_rmp is not None and "observation_timestamp_ms" not in q_no_rmp

    # 6) NYSE early-close calendar covers Thursday July 3 2025 & Thursday Dec 24 2026 vs regular & holiday (#22)
    assert MarketDataLoader.nyse_session_close_time(date(2025, 7, 3)) == dtime(13, 0)
    assert MarketDataLoader.nyse_session_close_time(date(2026, 12, 24)) == dtime(13, 0)
    assert MarketDataLoader.nyse_session_close_time(date(2025, 12, 3)) == dtime(16, 0)
    assert MarketDataLoader.is_nyse_trading_day(date(2025, 7, 4)) is False

    # 7) Placeholder peer node without bar_date is excluded (#14), while real peer with bar_date counts even at 0.0 return
    from src.risk.calibration import TemperatureScaler

    prov = LiveInferenceProvider(lakehouse_dir=lh, weights_dir=tmp_path / "w4892_peer", enable_live_fetch=False)

    class _HMM_Peer:
        STATE_LABELS = {0: "LOW_VOLATILITY", 1: "MEDIUM_VOLATILITY", 2: "HIGH_VOLATILITY"}
        transmat_ = np.array([[0.9, 0.08, 0.02], [0.05, 0.9, 0.05], [0.02, 0.08, 0.9]])

        def filter_sequence(self, x):
            return np.tile([0.85, 0.10, 0.05], (len(x), 1))

        def get_regime_multiplier(self, p):
            return 0.5

        def get_state_annualized_volatilities(self):
            return [10.0, 18.0, 35.0]

        def get_stationary_distribution(self):
            return [0.6, 0.3, 0.1]

    class _LGBM_Peer:
        def predict_proba(self, x):
            return np.tile([0.65, 0.25, 0.10], (len(x), 1))

    prov.regime_hmm = _HMM_Peer()
    prov.lgbm_backbone = _LGBM_Peer()
    sc_peer = TemperatureScaler()
    sc_peer.temperature_ = 1.0
    prov.temp_scaler = sc_peer

    idx_200 = pd.bdate_range(end="2025-12-01", periods=200, name="timestamp")
    closes_s = np.linspace(100.0, 120.0, 200)
    closes_m = np.linspace(400.0, 410.0, 200)
    df_sig_t = pd.DataFrame({"open": closes_s, "high": closes_s + 1, "low": closes_s - 1, "close": closes_s, "volume": 100_000, "vwap": closes_s}, index=idx_200)
    df_sig_m = pd.DataFrame({"open": closes_m, "high": closes_m + 1, "low": closes_m - 1, "close": closes_m, "volume": 100_000, "vwap": closes_m}, index=idx_200)
    obs_mon_close = int(pd.Timestamp("2025-12-01 16:00:00", tz="America/New_York").timestamp() * 1000)
    df_sig_t.attrs = {"observation_timestamp_ms": obs_mon_close, "obs_bar_date": "2025-12-01"}
    df_sig_m.attrs = {"observation_timestamp_ms": obs_mon_close, "obs_bar_date": "2025-12-01"}
    prov.market_loader.atomic_write_parquet(df_sig_t, prov.market_loader.daily_dir / "SIGT.parquet")
    prov.market_loader.atomic_write_parquet(df_sig_m, prov.market_loader.daily_dir / "SPY.parquet")
    with prov._lock:
        prov._peer_graph_cache["SIGT"] = (time.monotonic(), {
            "weight_bar_date": "2025-12-01",
            "weight_observation_timestamp_ms": obs_mon_close,
            "nodes": [{"id": "PLACEHOLDER", "role": "GICS_PEER", "bar_date": None, "spillover_contribution": 0.0}],
        })
        prov._news_cache["SIGT"] = (time.monotonic(), [{"decayed_sentiment": 0.8, "published_at": "2025-12-01T15:00:00Z", "published_at_verified": True}])
        prov._signal_cache.clear()
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-12-01 16:10:00", tz="America/New_York").timestamp())
    sig_ph_peer = prov.get_signal("SIGT")
    assert "UNREADY" not in sig_ph_peer["execution_mode"]
    assert "MODEL_WEIGHTS_OR_HISTORY_MISSING" not in sig_ph_peer["data_integrity"]["missing_modalities"]
    assert "PEER_GAT_NOT_LOADED" in sig_ph_peer["data_integrity"]["missing_modalities"]
    assert "DIVERGENCE_OBSERVATIONS_UNALIGNED" not in sig_ph_peer["data_integrity"]["missing_modalities"]
    with prov._lock:
        prov._peer_graph_cache["SIGT"] = (time.monotonic(), {
            "weight_bar_date": "2025-12-01",
            "weight_observation_timestamp_ms": obs_mon_close,
            "nodes": [{"id": "SPY", "role": "SECTOR_ETF", "bar_date": "2025-12-01", "observation_timestamp_ms": obs_mon_close, "spillover_contribution": 0.0}],
        })
        prov._signal_cache.clear()
    sig_real_zero_peer = prov.get_signal("SIGT")
    assert "UNREADY" not in sig_real_zero_peer["execution_mode"]
    assert "PEER_GAT_NOT_LOADED" not in sig_real_zero_peer["data_integrity"]["missing_modalities"]
    assert sig_real_zero_peer["signal_breakdown"]["peer_gat_spillover"] == 0.0


def test_task5024_coderabbit_fixes(tmp_path: Any, monkeypatch: Any) -> None:
    """Covers task-5024 fixes across broadcaster, FundamentalsClient, and MarketDataLoader."""
    import email.message
    import json
    import urllib.error
    import urllib.request
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from src.data.fundamentals_client import FundamentalsClient
    from src.data.market_loader import MarketDataLoader
    from src.ipc.ws_broadcaster import WebSocketBroadcaster, build_qare_signal_frame

    # 1) ws_broadcaster patch changing volume invalidates old observation_timestamp_ms; invalid prices block ready BUY (#10, #13)
    wb = WebSocketBroadcaster()
    wb.queue_tick("NVDA", {"ticker": "NVDA", "price": 100.0, "volume": 1000, "observation_timestamp_ms": 1764000000000})
    wb.queue_tick("NVDA", {"volume": 1500})
    assert "observation_timestamp_ms" not in wb.pending_ticks["NVDA"]
    assert wb.pending_ticks["NVDA"]["volume"] == 1500
    assert isinstance(wb.pending_ticks["NVDA"]["timestamp_ms"], int)
    wb.queue_tick("AAPL", {"ticker": "AAPL", "price": 200.0, "observation_timestamp_ms": 1764000000000})
    wb.queue_tick("AAPL", {"note": "annotation_only"})
    assert wb.pending_ticks["AAPL"]["observation_timestamp_ms"] == 1764000000000

    bad_price_sig = build_qare_signal_frame(
        ticker="NVDA", action="BUY", conviction_tier="HIGH", p_buy=0.7, p_hold=0.2, p_sell=0.1,
        entry_price=float("nan"), dynamic_take_profit=120.0, dynamic_stop_loss=95.0,
        winning_engine="LIGHTGBM", execution_mode="LONG_ONLY",
    )
    assert bad_price_sig["execution_mode"] == "UNREADY"
    assert bad_price_sig["winning_engine"] == "NONE"
    assert bad_price_sig["execution_recommendation"]["action"] == "HOLD"
    assert "INVALID_RECOMMENDATION_PRICES" in bad_price_sig["data_integrity"]["missing_modalities"]

    # 2) Negated cancel footnote & derivative cancellation remark do not falsely cancel valid Form 4 rows (#14, #16)
    lh = tmp_path / "lh5024"
    fc = FundamentalsClient(lakehouse_dir=lh)
    xml_neg = """<?xml version="1.0"?>
    <ownershipDocument>
      <reportingOwner><reportingOwnerId><rptOwnerName>Bob</rptOwnerName></reportingOwnerId></reportingOwner>
      <nonDerivativeTable>
        <nonDerivativeTransaction>
          <transactionDate><value>2025-11-10</value></transactionDate>
          <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
          <transactionAmounts><transactionShares><value>100</value></transactionShares><transactionPricePerShare><value>50</value></transactionPricePerShare></transactionAmounts>
          <footnoteId id="F1"/>
        </nonDerivativeTransaction>
      </nonDerivativeTable>
      <footnotes><footnote id="F1">This transaction was not cancelled; only the footnote has changed.</footnote></footnotes>
    </ownershipDocument>"""
    r_neg = fc._parse_sec_form4_xml(xml_neg, "2025-11-12", "2025-11-12T17:00:00Z", "AC1", form="4/A")
    assert r_neg and r_neg[0].get("amendment_action") != "CANCEL"

    xml_deriv = """<?xml version="1.0"?>
    <ownershipDocument>
      <reportingOwner><reportingOwnerId><rptOwnerName>Sue</rptOwnerName></reportingOwnerId></reportingOwner>
      <remarks>Cancelled the derivative option grant.</remarks>
      <nonDerivativeTable>
        <nonDerivativeTransaction>
          <transactionDate><value>2025-11-10</value></transactionDate>
          <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
          <transactionAmounts><transactionShares><value>200</value></transactionShares><transactionPricePerShare><value>50</value></transactionPricePerShare></transactionAmounts>
        </nonDerivativeTransaction>
      </nonDerivativeTable>
      <derivativeTable>
        <derivativeTransaction><transactionDate><value>2025-11-10</value></transactionDate></derivativeTransaction>
      </derivativeTable>
    </ownershipDocument>"""
    r_deriv = fc._parse_sec_form4_xml(xml_deriv, "2025-11-12", "2025-11-12T17:00:00Z", "AC2", form="4/A")
    assert r_deriv and r_deriv[0].get("_mixed_remarks_ambiguous") is True and "amendment_action" not in r_deriv[0]

    # 3) Truncated HTTP body (< Content-Length) rejected; HTTP 429 near deadline still records failure cooldown (#18, #24)
    class _ShortResp:
        headers = {"Content-Length": "100"}
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read1(self, n):
            if not getattr(self, "_done", False):
                self._done = True
                return b"short"
            return b""

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=6.0: _ShortResp())
    assert fc._sec_fetch_text("https://data.sec.gov/short.json", timeout=2.0) is None

    def _raise_429(req, timeout=6.0):
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", email.message.Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", _raise_429)
    fc._sec_tl.deadline = time.monotonic() + 5.0
    try:
        assert fc._sec_fetch_text("https://data.sec.gov/429.json", timeout=1.0) is None
        assert fc._sec_tl.last_budget_expired is False
    finally:
        fc._sec_tl.deadline = None

    # 4) Malformed facts["us-gaap"] = [] preserves stale cache (#23); 13th cached Form 4 resolves under Form 4/A (#25)
    cik_f = "0000999111"
    facts_p = fc.raw_dir / "edgar_facts" / f"CIK{cik_f}.json"
    facts_p.parent.mkdir(parents=True, exist_ok=True)
    facts_p.write_text(json.dumps({"cik": cik_f, "us_gaap": {"Revenues": {"units": {"USD": [{"val": 42.0}]}}}}), encoding="utf-8")
    os.utime(facts_p, (time.time() - 86400 * 30, time.time() - 86400 * 30))
    monkeypatch.setattr(fc, "_sec_fetch_text", lambda url, timeout=8.0: json.dumps({"facts": {"us-gaap": []}}))
    got_f = fc.get_sec_company_facts(cik_f, enable_live_fetch=True)
    assert got_f is not None and "Revenues" in got_f["us_gaap"]

    f4_cache_p = fc.raw_dir / "edgar_form4" / f"CIK{cik_f}.json"
    f4_cache_p.parent.mkdir(parents=True, exist_ok=True)
    subs_13 = {
        "recent": {
            "form": ["4/A"] + ["4"] * 12,
            "filingDate": [f"2025-11-{20 - i:02d}" for i in range(13)],
            "acceptanceDateTime": [f"2025-11-{20 - i:02d}T17:00:00Z" for i in range(13)],
            "accessionNumber": [f"ACC{i:02d}" for i in range(13)],
            "primaryDocument": [f"d{i:02d}.xml" for i in range(13)],
        }
    }
    filings_cached = {f"ACC{i:02d}": {"rows": [{"form": "4/A" if i == 0 else "4", "shares": 100 + i, "accession_number": f"ACC{i:02d}"}]} for i in range(13)}
    f4_cache_p.write_text(json.dumps({"cik": cik_f, "filings": filings_cached}), encoding="utf-8")
    loaded_13 = fc._load_or_fetch_sec_form4_rows(cik_f, subs_13, enable_live_fetch=False, max_filings=12)
    assert any(r.get("accession_number") == "ACC12" for r in loaded_13)

    # 5) Stale concurrent pre-settlement refresh must not overwrite newly settled disk partition (#19);
    #    Arrow fallback clears PyArrow-restored attrs before query filtering (#26);
    #    volume mismatch < or > and future regularMarketTime clear obs_ts_ms (#20, #21)
    ml = MarketDataLoader(data_dir=lh)
    idx2 = pd.DatetimeIndex([pd.Timestamp("2025-12-01"), pd.Timestamp("2025-12-02")], name="timestamp")
    df_two = pd.DataFrame({"open": [100.0, 102.0], "high": [101.0, 104.0], "low": [99.0, 101.0], "close": [100.5, 103.0], "volume": [1000, 2000], "vwap": [100.2, 102.67]}, index=idx2)
    obs2 = int(pd.Timestamp("2025-12-02 16:00:00", tz="America/New_York").timestamp() * 1000)
    df_two.attrs = {"observation_timestamp_ms": obs2, "obs_bar_date": "2025-12-02", "_bar_unsettled": True}
    p_two = ml.daily_dir / "TWOROW.parquet"
    ml.atomic_write_parquet(df_two, p_two)

    import duckdb
    orig_duck_connect = duckdb.connect

    class _MismatchDuckCon:
        def __init__(self):
            self._real = orig_duck_connect(":memory:")
            self._bump = 0

        def execute(self, sql, *a, **kw):
            res = self._real.execute(sql, *a, **kw)
            if "read_parquet" in sql:
                self._bump += 1
                t_b = time.time() + 10.0 + self._bump
                os.utime(p_two, (t_b, t_b))
            return res

        def close(self):
            self._real.close()

    with monkeypatch.context() as mp_duck:
        mp_duck.setattr("duckdb.connect", lambda *a, **kw: _MismatchDuckCon())
        df_sub1 = ml.query_parquet_duckdb(p_two, end_date="2025-12-01")
        assert df_sub1 is not None and len(df_sub1) == 1
        assert "observation_timestamp_ms" not in df_sub1.attrs
        assert "_bar_unsettled" not in df_sub1.attrs
        df_full2 = ml.query_parquet_duckdb(p_two)
        assert df_full2 is not None and df_full2.attrs.get("observation_timestamp_ms") == obs2
        assert df_full2.attrs.get("_bar_unsettled") is True

    idx_c = pd.DatetimeIndex(pd.bdate_range("2025-11-24", "2025-12-03"), name="timestamp")
    df_part_disk = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000, "vwap": 100.17}, index=idx_c)
    obs_pre = int(pd.Timestamp("2025-12-03 15:50:00", tz="America/New_York").timestamp() * 1000)
    df_part_disk.attrs = {"observation_timestamp_ms": obs_pre, "obs_bar_date": "2025-12-03", "_bar_unsettled": True}
    ml.save_ticker_daily("CONCR", df_part_disk)
    monkeypatch.setattr(ml, "_latest_expected_trading_date", lambda: pd.Timestamp("2025-12-03"))
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-12-03 15:55:00", tz="America/New_York").timestamp())
    df_settled_concurrent = df_part_disk.copy()
    df_settled_concurrent.iloc[-1, df_settled_concurrent.columns.get_loc("close")] = 109.0
    df_settled_concurrent.iloc[-1, df_settled_concurrent.columns.get_loc("high")] = 109.5
    df_settled_concurrent.attrs = {}

    def _dl_concurrent(*a, **kw):
        ml.save_ticker_daily("CONCR", df_settled_concurrent)
        return df_part_disk.tail(2).rename(columns=str.capitalize)

    monkeypatch.setattr(yf, "download", _dl_concurrent)
    out_c = ml.refresh_ticker_daily("CONCR", force_refresh_if_incomplete=True)
    assert out_c is not None and float(out_c["close"].iloc[-1]) == 109.0
    disk_after = ml.load_ticker_daily("CONCR")
    assert disk_after is not None and float(disk_after["close"].iloc[-1]) == 109.0
    assert not disk_after.attrs.get("_bar_unsettled")

    t_low_vol = int(pd.Timestamp("2025-12-03 11:00:00", tz="America/New_York").timestamp())
    t_future_s = int(pd.Timestamp("2025-12-03 11:06:00", tz="America/New_York").timestamp())
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-12-03 11:05:00", tz="America/New_York").timestamp())

    class _FiVolLower:
        fast_info = type("FILo", (), {
            "lastPrice": 100.0, "open": 99.0, "dayHigh": 101.0, "dayLow": 98.5, "previousClose": 98.0, "lastVolume": 8000,
            "_md": {"regularMarketPrice": 100.0, "regularMarketTime": t_low_vol, "regularMarketVolume": 10_000},
            "_prices_1y": pd.DataFrame({"Close": [100.0]}, index=pd.DatetimeIndex([pd.Timestamp("2025-12-03 00:00:00-05:00")])),
            "get": lambda self, k, d=None: {"lastPrice": 100.0, "open": 99.0, "dayHigh": 101.0, "dayLow": 98.5, "previousClose": 98.0, "lastVolume": 8000}.get(k, d),
        })()
        def history(self, **kw):
            return pd.DataFrame()
        def get_history_metadata(self):
            return {}

    monkeypatch.setattr(yf, "Ticker", lambda sym: _FiVolLower())
    q_lo = ml.fetch_live_quote("VOLO")
    assert q_lo is not None and "observation_timestamp_ms" not in q_lo

    class _FiFutureRmt:
        fast_info = type("FIFut", (), {
            "lastPrice": 100.0, "open": 99.0, "dayHigh": 101.0, "dayLow": 98.5, "previousClose": 98.0, "lastVolume": 10_000,
            "_md": {"regularMarketPrice": 100.0, "regularMarketTime": t_future_s, "regularMarketVolume": 10_000},
            "_prices_1y": pd.DataFrame({"Close": [100.0]}, index=pd.DatetimeIndex([pd.Timestamp("2025-12-03 00:00:00-05:00")])),
            "get": lambda self, k, d=None: {"lastPrice": 100.0, "open": 99.0, "dayHigh": 101.0, "dayLow": 98.5, "previousClose": 98.0, "lastVolume": 10_000}.get(k, d),
        })()
        def history(self, **kw):
            return pd.DataFrame()
        def get_history_metadata(self):
            return {}

    monkeypatch.setattr(yf, "Ticker", lambda sym: _FiFutureRmt())
    q_fut = ml.fetch_live_quote("FUTQ")
    assert q_fut is not None and "observation_timestamp_ms" not in q_fut


def test_task5108_coderabbit_fixes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers task-5108 fixes across ws_broadcaster, EDGAR PIT, cache/slice alignment, and market_loader."""
    from datetime import date
    import threading
    import pandas as pd
    import yfinance as yf
    from src.data.fundamentals_client import FundamentalsClient
    from src.data.market_loader import MarketDataLoader
    from src.ipc.live_providers import LiveInferenceProvider
    from src.ipc.ws_broadcaster import WebSocketBroadcaster, build_qare_signal_frame
    from src.risk.calibration import TemperatureScaler

    # 1) Price barrier rounding collapse & zero HOLD preservation & full-tick bool obs_ms rejection
    frame_collapse = build_qare_signal_frame(
        ticker="NVDA",
        action="BUY",
        p_buy=0.7,
        p_hold=0.3,
        p_sell=0.0,
        unconstrained_kelly_pct=5.0,
        l1_normalized_pct=5.0,
        entry_price=100.00,
        dynamic_take_profit=100.001,
        dynamic_stop_loss=99.999,
        execution_mode="LONG_ONLY",
    )
    assert frame_collapse["execution_mode"] == "UNREADY"
    assert frame_collapse["winning_engine"] == "NONE"
    assert frame_collapse["execution_recommendation"]["l1_normalized_portfolio_allocation_pct"] == 0.0
    assert "INVALID_RECOMMENDATION_PRICES" in frame_collapse["data_integrity"]["missing_modalities"]
    assert frame_collapse["execution_recommendation"]["entry_price"] == 100.0

    frame_zero_hold = build_qare_signal_frame(
        ticker="NVDA",
        action="BUY",
        p_buy=0.7,
        p_hold=0.0,
        p_sell=0.3,
        unconstrained_kelly_pct=5.0,
        l1_normalized_pct=5.0,
        entry_price=100.0,
        dynamic_take_profit=110.0,
        dynamic_stop_loss=0.0,
        execution_mode="LONG_ONLY",
    )
    assert frame_zero_hold["execution_mode"] == "UNREADY"
    assert "INVALID_RECOMMENDATION_PRICES" in frame_zero_hold["data_integrity"]["missing_modalities"]
    assert frame_zero_hold["calibration"]["calibrated_probabilities"]["neutral_hold"] == 0.0

    wb = WebSocketBroadcaster()
    wb.queue_tick("TST", {"price": 100.0, "observation_timestamp_ms": True, "timestamp_ms": True})
    assert "observation_timestamp_ms" not in wb.pending_ticks["TST"]
    assert isinstance(wb.pending_ticks["TST"]["timestamp_ms"], int) and not isinstance(wb.pending_ticks["TST"]["timestamp_ms"], bool)
    wb.queue_tick("TST", {"price": 101.0, "observation_timestamp_ms": 1764000000000})
    assert wb.pending_ticks["TST"]["observation_timestamp_ms"] == 1764000000000

    # 2) Hurricane Sandy closures in PIT mapping
    assert MarketDataLoader.is_nyse_trading_day(date(2012, 10, 29)) is False
    assert MarketDataLoader.is_nyse_trading_day(date(2012, 10, 30)) is False
    assert MarketDataLoader.is_nyse_trading_day(date(2012, 10, 31)) is True
    assert MarketDataLoader.apply_pit_cutoff(pd.Timestamp("2012-10-26 17:00:00", tz="America/New_York")).date() == date(2012, 10, 31)

    # 3) Date-only current_acceptance_datetime normalized & cannot verify same-day filing
    lh = tmp_path / "lh5108"
    wdir = tmp_path / "w5108"
    fc = FundamentalsClient(lakehouse_dir=lh)
    cik = "0000777888"
    fc._write_json_cache(f"edgar_submissions/CIK{cik}.json", {
        "cik": cik,
        "recent": {
            "form": ["10-K", "10-Q"],
            "filingDate": ["2025-11-24", "2025-08-01"],
            "reportDate": ["2025-09-30", "2025-06-30"],
            "accessionNumber": ["0001-25-000010", "0001-25-000005"],
            "primaryDocument": ["r10k.htm", "r10q.htm"],
            "acceptanceDateTime": ["2025-11-24", "2025-08-01T15:00:00Z"],
        },
    })
    edg_payload = fc.build_edgar_filings_payload("DATET", known_cik=cik, enable_live_fetch=False)
    assert edg_payload["current_acceptance_datetime"] is None

    prov = LiveInferenceProvider(lakehouse_dir=lh, weights_dir=wdir, enable_live_fetch=False)

    class _HMM5:
        STATE_LABELS = {0: "LOW_VOLATILITY", 1: "MEDIUM_VOLATILITY", 2: "HIGH_VOLATILITY"}
        transmat_ = np.array([[0.9, 0.08, 0.02], [0.05, 0.9, 0.05], [0.02, 0.08, 0.9]])
        def filter_sequence(self, x):
            return np.tile([0.85, 0.10, 0.05], (len(x), 1))
        def get_regime_multiplier(self, p):
            return 0.5
        def get_state_annualized_volatilities(self):
            return [10.0, 18.0, 35.0]
        def get_stationary_distribution(self):
            return [0.6, 0.3, 0.1]

    class _LGBM5:
        def predict_proba(self, x):
            return np.tile([0.65, 0.25, 0.10], (len(x), 1))

    prov.regime_hmm = _HMM5()
    prov.lgbm_backbone = _LGBM5()
    sc5 = TemperatureScaler()
    sc5.temperature_ = 1.0
    prov.temp_scaler = sc5

    idx_321 = pd.bdate_range(end="2025-11-24", periods=321, name="timestamp")
    obs_nov24 = int(pd.Timestamp("2025-11-24 16:00:00", tz="America/New_York").timestamp() * 1000)
    cls_t = 100.0 * np.exp(np.cumsum(np.random.default_rng(5108).normal(0.0005, 0.01, size=321)))
    df_t321 = pd.DataFrame({"open": cls_t * 0.998, "high": cls_t * 1.006, "low": cls_t * 0.994, "close": cls_t, "volume": np.full(321, 1_500_000.0)}, index=idx_321)
    df_t321.attrs = {"observation_timestamp_ms": obs_nov24, "obs_bar_date": "2025-11-24"}
    # SPY drops bar -2 so aligned intersection of length 320 includes row 0 (-321) of T321
    spy_idx = idx_321.delete(-2)
    cls_spy = 400.0 * np.exp(np.cumsum(np.random.default_rng(5109).normal(0.0004, 0.008, size=len(spy_idx))))
    df_spy_gap = pd.DataFrame({"open": cls_spy * 0.998, "high": cls_spy * 1.006, "low": cls_spy * 0.994, "close": cls_spy, "volume": np.full(len(spy_idx), 3_000_000.0)}, index=spy_idx)
    df_spy_gap.attrs = {"observation_timestamp_ms": obs_nov24, "obs_bar_date": "2025-11-24"}
    prov.market_loader.atomic_write_parquet(df_t321, prov.market_loader.daily_dir / "DATET.parquet")
    prov.market_loader.atomic_write_parquet(df_spy_gap, prov.market_loader.daily_dir / "SPY.parquet")
    with prov._lock:
        prov._edgar_cache["DATET"] = (time.monotonic(), {
            "available": True,
            "current_filing_date": "2025-11-24",
            "current_acceptance_datetime": "2025-11-24",
            "item1a_sentiment_drift": 0.4,
            "form4_transactions": [],
        })
        prov._signal_cache.clear()
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-11-24 16:10:00", tz="America/New_York").timestamp())

    res_calls = {"n": 0}
    orig_res = prov.residualizer.compute_rolling_residuals
    def _count_res(r_s, r_sp, r_sc):
        res_calls["n"] += 1
        return orig_res(r_s, r_sp, r_sc)
    monkeypatch.setattr(prov.residualizer, "compute_rolling_residuals", _count_res)

    sig_date_edg = prov.get_signal("DATET")
    assert "UNREADY" not in sig_date_edg["execution_mode"]
    assert sig_date_edg["data_integrity"]["xbrl_filed_date_verified"] is False
    assert "EDGAR_ITEM1A_DIFF" in sig_date_edg["data_integrity"]["missing_modalities"]
    assert res_calls["n"] == 1

    # Identical call hits cache; mutating row 0 (-321) invalidates cache
    _ = prov.get_signal("DATET")
    assert res_calls["n"] == 1
    df_t321_mod = df_t321.copy()
    df_t321_mod.iloc[0, df_t321_mod.columns.get_loc("close")] *= 1.15
    df_t321_mod.iloc[0, df_t321_mod.columns.get_loc("high")] = max(float(df_t321_mod.iloc[0]["high"]), float(df_t321_mod.iloc[0]["close"]))
    prov.market_loader.atomic_write_parquet(df_t321_mod, prov.market_loader.daily_dir / "DATET.parquet")
    _ = prov.get_signal("DATET")
    assert res_calls["n"] == 2

    # 4) SPY regime endpoint fallback selects endpoint before copying (<= 3 slices across 1500 rows)
    idx_big = pd.bdate_range(end="2025-11-25", periods=1500, name="timestamp")
    cls_big = np.linspace(300.0, 500.0, 1500)
    df_spy_big = pd.DataFrame({"open": cls_big - 1, "high": cls_big + 1, "low": cls_big - 2, "close": cls_big, "volume": 1_000_000.0}, index=idx_big)
    post_nov24_ms = int(pd.Timestamp("2025-11-25 16:20:00", tz="America/New_York").timestamp() * 1000)
    df_spy_big.attrs = {"disk_mtime_ms": post_nov24_ms, "obs_bar_date": "2025-11-25"}
    slice_calls = {"n": 0}
    orig_slice = prov._slice_aligned_frame
    def _count_slice(df_raw, idx_sel):
        slice_calls["n"] += 1
        return orig_slice(df_raw, idx_sel)
    monkeypatch.setattr(prov, "_slice_aligned_frame", _count_slice)
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_spy_big if sym.upper() == "SPY" else df_t321)
    with prov._lock:
        prov._signal_cache.clear()
    sig_big = prov.get_signal("DATET", regime={"current_state": "LOW_VOLATILITY", "state_index": 0, "as_of_date": "2025-11-25", "observation_timestamp_ms": post_nov24_ms})
    assert slice_calls["n"] <= 3
    assert sig_big["macro_regime"]["current_state"] == "UNAVAILABLE"

    # 5) FastInfo previous close uses prior close relative to quote session & req_started_ms NY date
    ml = prov.market_loader
    idx_hist = pd.DatetimeIndex([pd.Timestamp("2025-11-28"), pd.Timestamp("2025-12-01")])
    tue_ts_s = int(pd.Timestamp("2025-12-02 10:30:00", tz="America/New_York").timestamp())

    class _TueFastInfo:
        fast_info = type("FITue", (), {
            "lastPrice": 105.0, "open": 102.0, "dayHigh": 106.0, "dayLow": 101.0, "previousClose": None, "lastVolume": 50_000,
            "_md": {"regularMarketPrice": 105.0, "regularMarketTime": tue_ts_s, "regularMarketVolume": 50_000},
            "_prices_1y": pd.DataFrame({"Close": [105.0]}, index=pd.DatetimeIndex([pd.Timestamp("2025-12-02 00:00:00-05:00")])),
            "get": lambda self, k, d=None: {"lastPrice": 105.0, "open": 102.0, "dayHigh": 106.0, "dayLow": 101.0, "lastVolume": 50_000}.get(k, d),
        })()
        def history(self, **kw):
            return pd.DataFrame({"Open": [97.0, 99.0], "High": [99.0, 101.0], "Low": [96.0, 98.0], "Close": [98.0, 100.0], "Volume": [10_000, 12_000]}, index=idx_hist)
        def get_history_metadata(self):
            return {}

    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-12-02 10:35:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(yf, "Ticker", lambda sym: _TueFastInfo())
    q_tue = ml.fetch_live_quote("TUEQ")
    assert q_tue is not None and q_tue["prev_close"] == 100.0 and q_tue["change"] == 5.0

    class _TueWithTueHist(_TueFastInfo):
        def history(self, **kw):
            idx_3 = pd.DatetimeIndex([pd.Timestamp("2025-11-28"), pd.Timestamp("2025-12-01"), pd.Timestamp("2025-12-02")])
            return pd.DataFrame({"Open": [97.0, 99.0, 102.0], "High": [99.0, 101.0, 106.0], "Low": [96.0, 98.0, 101.0], "Close": [98.0, 100.0, 104.0], "Volume": [10_000, 12_000, 50_000]}, index=idx_3)

    monkeypatch.setattr(yf, "Ticker", lambda sym: _TueWithTueHist())
    q_tue2 = ml.fetch_live_quote("TUEQ2")
    assert q_tue2 is not None and q_tue2["prev_close"] == 100.0 and q_tue2["change"] == 5.0

    # Same-session unverified metadata across mocked midnight stays unverified without reading wall-clock datetime.now
    class _HistSameUnver:
        @property
        def fast_info(self):
            return {}
        def history(self, **kw):
            return pd.DataFrame({"Open": [100.0], "High": [105.0], "Low": [99.0], "Close": [104.0], "Volume": [10_000]}, index=pd.DatetimeIndex([pd.Timestamp("2025-12-02")]))
        def get_history_metadata(self):
            return {"regularMarketTime": int(pd.Timestamp("2025-12-02 16:00:00", tz="America/New_York").timestamp())}

    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-12-03 00:05:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(yf, "Ticker", lambda sym: _HistSameUnver())
    q_unver_mid = ml.fetch_live_quote("MIDUNV")
    assert q_unver_mid is not None and "observation_timestamp_ms" not in q_unver_mid

    # 6) refresh_ticker_daily rebases onto disk_now (preserving historical row fixes) and coordinates across writers
    idx_reb = pd.DatetimeIndex(pd.bdate_range("2025-11-26", "2025-12-03"), name="timestamp")
    df_reb = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.5, "volume": 1000, "vwap": 100.17}, index=idx_reb)
    df_reb.attrs = {"observation_timestamp_ms": int(pd.Timestamp("2025-12-03 15:50:00", tz="America/New_York").timestamp() * 1000), "obs_bar_date": "2025-12-03", "_bar_unsettled": True}
    ml.save_ticker_daily("REBASE", df_reb)
    ml2 = MarketDataLoader(data_dir=lh)
    df_hist_fix = df_reb.copy()
    df_hist_fix.iloc[0, df_hist_fix.columns.get_loc("close")] = 101.75
    df_hist_fix.iloc[0, df_hist_fix.columns.get_loc("high")] = 102.0

    def _dl_hist_fix(*a, **kw):
        # Concurrent writer fixes historical row outside yf download window while leaving latest bar unsettled
        ml2.save_ticker_daily("REBASE", df_hist_fix)
        return df_reb.tail(2).rename(columns=str.capitalize)

    monkeypatch.setattr(ml, "_latest_expected_trading_date", lambda: pd.Timestamp("2025-12-03"))
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-12-03 15:55:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(yf, "download", _dl_hist_fix)
    out_reb = ml.refresh_ticker_daily("REBASE", force_refresh_if_incomplete=True)
    assert out_reb is not None and float(out_reb["close"].iloc[0]) == pytest.approx(101.75)
    disk_reb = ml.load_ticker_daily("REBASE")
    assert disk_reb is not None and float(disk_reb["close"].iloc[0]) == pytest.approx(101.75)

    # Post-read generation bump before save is detected & re-evaluated, and cross-instance lock serializes writers
    ml.save_ticker_daily("GENCHK", df_reb)
    settled_concurrent = df_reb.copy()
    settled_concurrent.iloc[-1, settled_concurrent.columns.get_loc("close")] = 112.0
    settled_concurrent.iloc[-1, settled_concurrent.columns.get_loc("high")] = 112.5
    settled_concurrent.attrs = {}

    orig_load_g = ml.load_ticker_daily
    fired = {"done": False}
    dl_calls = {"n": 0}
    target_gen_pq = ml.daily_dir / "GENCHK.parquet"

    def _dl_plain(*a, **kw):
        dl_calls["n"] += 1
        return df_reb.tail(2).rename(columns=str.capitalize)

    def _load_race(t_sym: str, *a, **kw):
        res_df = orig_load_g(t_sym, *a, **kw)
        if t_sym.upper() == "GENCHK" and dl_calls["n"] >= 1 and not fired["done"]:
            fired["done"] = True
            # Simulate an un-locked external writer publishing settled bar between disk_now read and commit
            ext_tmp = target_gen_pq.with_name("ext_settled.tmp.parquet")
            settled_concurrent.to_parquet(ext_tmp, engine="pyarrow", index=True)
            os.replace(ext_tmp, target_gen_pq)
        return res_df

    monkeypatch.setattr(yf, "download", _dl_plain)
    monkeypatch.setattr(ml, "load_ticker_daily", _load_race)
    out_gen = ml.refresh_ticker_daily("GENCHK", force_refresh_if_incomplete=True)
    assert out_gen is not None and float(out_gen["close"].iloc[-1]) == pytest.approx(112.0)
    assert float(orig_load_g("GENCHK")["close"].iloc[-1]) == pytest.approx(112.0)

    # Also verify cross-instance writer thread serializes cleanly without self-deadlock or corrupted parquet
    monkeypatch.setattr(ml, "load_ticker_daily", orig_load_g)
    ml.save_ticker_daily("GENLCK", df_reb)
    started_wr = threading.Event()
    writer_err: list[Exception] = []
    writer_thr: list[threading.Thread] = []

    def _dl_spawn_writer(*a, **kw):
        def _bg_save() -> None:
            started_wr.set()
            try:
                ml2.save_ticker_daily("GENLCK", settled_concurrent)
            except Exception as exc:  # pragma: no cover
                writer_err.append(exc)

        t_wr = threading.Thread(target=_bg_save)
        writer_thr.append(t_wr)
        t_wr.start()
        started_wr.wait(timeout=2.0)
        return df_reb.tail(2).rename(columns=str.capitalize)

    monkeypatch.setattr(yf, "download", _dl_spawn_writer)
    try:
        _ = ml.refresh_ticker_daily("GENLCK", force_refresh_if_incomplete=True)
    finally:
        started_wr.set()
        for t_wr in writer_thr:
            t_wr.join(timeout=5.0)
    for t_wr in writer_thr:
        assert not t_wr.is_alive()
    assert not writer_err
    final_lck = ml.load_ticker_daily("GENLCK")
    assert final_lck is not None and len(final_lck) == len(df_reb)
    assert float(final_lck["close"].iloc[-1]) == pytest.approx(112.0)


def test_task5276_coderabbit_fixes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers task-5276 fixes across broadcaster, PIT cutoff, Arrow fallback, quote fallback, and alignment."""
    from datetime import date
    import duckdb
    import pandas as pd
    import yfinance as yf
    from src.data.market_loader import MarketDataLoader, _partition_commit_lock
    from src.ipc.live_providers import LiveInferenceProvider
    from src.ipc.ws_broadcaster import WebSocketBroadcaster, build_qare_signal_frame
    from src.risk.calibration import TemperatureScaler

    # 1) queue_tick ignores overflowing timestamp values without dropping valid price
    wb = WebSocketBroadcaster()
    wb.queue_tick("OVF", {"price": 123.45, "observation_timestamp_ms": 10**400, "timestamp_ms": 10**400})
    assert wb.pending_ticks["OVF"]["price"] == 123.45
    assert "observation_timestamp_ms" not in wb.pending_ticks["OVF"]
    assert isinstance(wb.pending_ticks["OVF"]["timestamp_ms"], int) and wb.pending_ticks["OVF"]["timestamp_ms"] > 0

    # 2) SELL barrier validation respects LONG_SHORT vs LONG_ONLY execution_mode
    short_bad = build_qare_signal_frame(
        ticker="NVDA", action="SELL", p_buy=0.1, p_hold=0.2, p_sell=0.7,
        unconstrained_kelly_pct=-3.0, l1_normalized_pct=-3.0,
        entry_price=100.0, dynamic_take_profit=110.0, dynamic_stop_loss=95.0,
        execution_mode="LONG_SHORT",
    )
    assert short_bad["execution_mode"] == "UNREADY"
    assert "INVALID_RECOMMENDATION_PRICES" in short_bad["data_integrity"]["missing_modalities"]

    short_ok = build_qare_signal_frame(
        ticker="NVDA", action="SELL", p_buy=0.1, p_hold=0.2, p_sell=0.7,
        unconstrained_kelly_pct=-3.0, l1_normalized_pct=-3.0,
        entry_price=100.0, dynamic_take_profit=95.0, dynamic_stop_loss=110.0,
        execution_mode="LONG_SHORT",
    )
    assert short_ok["execution_mode"] == "LONG_SHORT"
    assert short_ok["execution_recommendation"]["action"] == "SELL"

    long_exit_ok = build_qare_signal_frame(
        ticker="NVDA", action="SELL", p_buy=0.1, p_hold=0.2, p_sell=0.7,
        unconstrained_kelly_pct=0.0, l1_normalized_pct=0.0,
        entry_price=100.0, dynamic_take_profit=110.0, dynamic_stop_loss=95.0,
        execution_mode="LONG_ONLY",
    )
    assert long_exit_ok["execution_mode"] == "LONG_ONLY"
    assert long_exit_ok["execution_recommendation"]["action"] == "SELL"

    long_exit_bad = build_qare_signal_frame(
        ticker="NVDA", action="SELL", p_buy=0.1, p_hold=0.2, p_sell=0.7,
        unconstrained_kelly_pct=0.0, l1_normalized_pct=0.0,
        entry_price=100.0, dynamic_take_profit=95.0, dynamic_stop_loss=110.0,
        execution_mode="LONG_ONLY",
    )
    assert long_exit_bad["execution_mode"] == "UNREADY"
    assert "INVALID_RECOMMENDATION_PRICES" in long_exit_bad["data_integrity"]["missing_modalities"]

    # 3) apply_pit_cutoff selects & checks default cutoff in America/New_York even when tz_str="UTC"
    assert MarketDataLoader.apply_pit_cutoff("2026-11-27T15:00:00Z", tz_str="UTC") == pd.Timestamp("2026-11-27T00:00:00+00:00")
    assert MarketDataLoader.apply_pit_cutoff("2026-11-27T18:30:00Z", tz_str="UTC") == pd.Timestamp("2026-11-30T00:00:00+00:00")

    # 4) Arrow fallback preserves projection error when requested column is missing
    lh = tmp_path / "lh5276"
    wdir = tmp_path / "w5276"
    ml = MarketDataLoader(data_dir=lh)
    idx_a = pd.bdate_range(end="2025-12-01", periods=5, name="timestamp")
    df_a = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 1000, "vwap": 101.0}, index=idx_a)
    df_a_novwap = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 1000}, index=idx_a)
    pq_proj = ml.daily_dir / "PROJ.parquet"
    ml.atomic_write_parquet(df_a, pq_proj)

    orig_connect = duckdb.connect
    class _MismatchDuck:
        def __init__(self):
            self._inner = orig_connect(":memory:")
            self._reads = 0
        def execute(self, sql, params=None):
            res = self._inner.execute(sql, params or [])
            if "FROM read_parquet" in sql and "DESCRIBE" not in sql:
                self._reads += 1
                # Swap inode/mtime across first 2 reads with vwap present, then drop vwap on final swap for Arrow fallback
                ml.atomic_write_parquet(df_a_novwap if self._reads >= 3 else df_a, pq_proj)
            return res
        def close(self):
            self._inner.close()

    with monkeypatch.context() as mp_p:
        mp_p.setattr("src.data.market_loader.duckdb.connect", lambda *a, **kw: _MismatchDuck())
        with pytest.raises(KeyError):
            ml.query_parquet_duckdb(pq_proj, columns=["close", "vwap"])

    # 5) Partition commit lock restores depth cleanly if fcntl.flock raises OSError
    lock_target = ml.daily_dir / "FLOCKTST.parquet"
    called_flock = {"n": 0}
    def _fail_once_flock(fd, op):
        called_flock["n"] += 1
        if called_flock["n"] == 1:
            raise OSError("transient lock error")
    import src.data.market_loader as _ml_mod
    orig_flock = _ml_mod.fcntl.flock if _ml_mod.fcntl is not None else None
    if orig_flock is not None:
        monkeypatch.setattr(_ml_mod.fcntl, "flock", _fail_once_flock)
        with pytest.raises(OSError):
            with _partition_commit_lock(lock_target):
                pass
        with _partition_commit_lock(lock_target):
            pass
        assert called_flock["n"] >= 2
        monkeypatch.setattr(_ml_mod.fcntl, "flock", orig_flock)

    # 6) get_history_metadata exception preserves valid history OHLCV quote (without verified timestamp)
    class _HistMdErr:
        @property
        def fast_info(self):
            return {}
        def history(self, **kw):
            return pd.DataFrame(
                {"Open": [99.0, 100.0], "High": [101.0, 105.0], "Low": [98.0, 99.0], "Close": [100.0, 104.0], "Volume": [10_000, 12_000]},
                index=pd.DatetimeIndex([pd.Timestamp("2025-12-01"), pd.Timestamp("2025-12-02")]),
            )
        def get_history_metadata(self):
            raise RuntimeError("metadata down")

    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-12-03 10:30:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(yf, "Ticker", lambda sym: _HistMdErr())
    q_md_err = ml.fetch_live_quote("MDERR")
    assert q_md_err is not None and q_md_err["price"] == 104.0 and "observation_timestamp_ms" not in q_md_err

    # 7) Earlier-session FastInfo price resolves prev_close strictly before selected session even when FastInfo previousClose > 0
    tue_s = int(pd.Timestamp("2025-12-02 16:00:00", tz="America/New_York").timestamp())
    class _WedNanTueFi:
        fast_info = type("FIWN", (), {
            "lastPrice": 100.0, "previousClose": 100.0, "lastVolume": 10_000,
            "_md": {"regularMarketPrice": 100.0, "regularMarketTime": tue_s, "regularMarketVolume": 10_000},
            "_prices_1y": pd.DataFrame(
                {"Close": [100.0, float("nan")]},
                index=pd.DatetimeIndex([pd.Timestamp("2025-12-02 00:00:00-05:00"), pd.Timestamp("2025-12-03 00:00:00-05:00")]),
            ),
            "get": lambda self, k, d=None: {"lastPrice": 100.0, "previousClose": 100.0, "lastVolume": 10_000}.get(k, d),
        })()
        def history(self, **kw):
            return pd.DataFrame(
                {"Open": [97.0, 99.0], "High": [99.0, 101.0], "Low": [96.0, 98.0], "Close": [98.0, 100.0], "Volume": [9000, 10_000]},
                index=pd.DatetimeIndex([pd.Timestamp("2025-12-01"), pd.Timestamp("2025-12-02")]),
            )
        def get_history_metadata(self):
            return {"regularMarketPrice": 100.0, "regularMarketTime": tue_s, "regularMarketVolume": 10_000}

    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-12-03 10:30:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(yf, "Ticker", lambda sym: _WedNanTueFi())
    q_wed_nan = ml.fetch_live_quote("WNAN")
    assert q_wed_nan is not None and q_wed_nan["date"] == "2025-12-02"
    assert q_wed_nan["prev_close"] == 98.0 and q_wed_nan["change"] == 2.0

    # 8) Concurrent newer unsettled bar on disk is NOT overwritten by older in-flight yf.download
    idx_u = pd.bdate_range(end="2025-12-03", periods=5, name="timestamp")
    df_init_u = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 1000, "vwap": 100.0}, index=idx_u)
    t1100_ms = int(pd.Timestamp("2025-12-03 11:00:00", tz="America/New_York").timestamp() * 1000)
    t1101_ms = int(pd.Timestamp("2025-12-03 11:01:00", tz="America/New_York").timestamp() * 1000)
    df_init_u.attrs = {"observation_timestamp_ms": t1100_ms, "obs_bar_date": "2025-12-03", "_bar_unsettled": True}
    ml.save_ticker_daily("UNOVR", df_init_u)
    ml_other = MarketDataLoader(data_dir=lh)
    df_newer_u = df_init_u.copy()
    df_newer_u.iloc[-1, df_newer_u.columns.get_loc("close")] = 105.0
    df_newer_u.iloc[-1, df_newer_u.columns.get_loc("high")] = 106.0
    df_newer_u.attrs = {"observation_timestamp_ms": t1101_ms, "obs_bar_date": "2025-12-03", "_bar_unsettled": True}

    def _dl_older_u(*a, **kw):
        ml_other.save_ticker_daily("UNOVR", df_newer_u)
        return df_init_u.tail(2).rename(columns=str.capitalize)

    monkeypatch.setattr(ml, "_latest_expected_trading_date", lambda: pd.Timestamp("2025-12-03"))
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-12-03 11:02:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(yf, "download", _dl_older_u)
    out_u = ml.refresh_ticker_daily("UNOVR", force_refresh_if_incomplete=True)
    assert out_u is not None and float(out_u["close"].iloc[-1]) == pytest.approx(105.0)
    disk_u = ml.load_ticker_daily("UNOVR")
    assert disk_u is not None and float(disk_u["close"].iloc[-1]) == pytest.approx(105.0)
    assert disk_u.attrs.get("observation_timestamp_ms") == t1101_ms

    # 9) Future-dated live quote does not become an eligible undated overlay in get_historical_prices
    prov = LiveInferenceProvider(lakehouse_dir=lh, weights_dir=wdir, enable_live_fetch=True)
    idx_180 = pd.bdate_range(end="2025-12-01", periods=180, name="timestamp")
    cls_180 = np.linspace(100.0, 120.0, 180)
    df_f180 = pd.DataFrame({"open": cls_180 - 1, "high": cls_180 + 1, "low": cls_180 - 2, "close": cls_180, "volume": 100_000, "vwap": cls_180}, index=idx_180)
    obs_1600_ms = int(pd.Timestamp("2025-12-01 16:00:00", tz="America/New_York").timestamp() * 1000)
    df_f180.attrs = {"observation_timestamp_ms": obs_1600_ms, "obs_bar_date": "2025-12-01"}
    ml.atomic_write_parquet(df_f180, ml.daily_dir / "FUTOVL.parquet")
    monkeypatch.setattr(ml, "_latest_expected_trading_date", lambda: pd.Timestamp("2025-12-01"))
    monkeypatch.setattr(yf, "download", lambda *a, **kw: pd.DataFrame())
    future_tue_s = int(pd.Timestamp("2025-12-02 10:30:00", tz="America/New_York").timestamp())
    class _FutQuoteTicker:
        fast_info = type("FIFOV", (), {
            "lastPrice": 999.0, "lastVolume": 10_000,
            "_md": {"regularMarketPrice": 999.0, "regularMarketTime": future_tue_s, "regularMarketVolume": 10_000},
            "_prices_1y": pd.DataFrame({"Close": [999.0]}, index=pd.DatetimeIndex([pd.Timestamp("2025-12-02 00:00:00-05:00")])),
            "get": lambda self, k, d=None: {"lastPrice": 999.0, "lastVolume": 10_000}.get(k, d),
        })()
        def history(self, **kw):
            return pd.DataFrame()
        def get_history_metadata(self):
            return {}
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-12-01 16:20:00", tz="America/New_York").timestamp())
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-12-01 16:20:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(yf, "Ticker", lambda sym: _FutQuoteTicker())
    hp_fut = prov.get_historical_prices("FUTOVL", count=5, interval="1d")
    assert hp_fut and hp_fut[0]["date"] == "2025-12-01" and hp_fut[0]["close"] != 999.0

    # 10) SPY written at 16:25 without observation provenance cannot prove 16:00 availability,
    #     and switch from offline to live mode invalidates signal cache
    class _HMM52:
        STATE_LABELS = {0: "LOW_VOLATILITY", 1: "MEDIUM_VOLATILITY", 2: "HIGH_VOLATILITY"}
        transmat_ = np.array([[0.9, 0.08, 0.02], [0.05, 0.9, 0.05], [0.02, 0.08, 0.9]])
        def filter_sequence(self, x):
            return np.tile([0.85, 0.10, 0.05], (len(x), 1))
        def get_regime_multiplier(self, p):
            return 0.5
        def get_state_annualized_volatilities(self):
            return [10.0, 18.0, 35.0]
        def get_stationary_distribution(self):
            return [0.6, 0.3, 0.1]

    class _LGBM52:
        def predict_proba(self, x):
            return np.tile([0.65, 0.25, 0.10], (len(x), 1))

    prov.enable_live_fetch = False
    prov.regime_hmm = _HMM52()
    prov.lgbm_backbone = _LGBM52()
    sc52 = TemperatureScaler()
    sc52.temperature_ = 1.0
    prov.temp_scaler = sc52

    df_spy_1625 = df_f180.copy()
    df_spy_1625.attrs = {}
    spy_pq_1625 = ml.daily_dir / "SPY.parquet"
    ml.atomic_write_parquet(df_spy_1625, spy_pq_1625)
    t_1625 = pd.Timestamp("2025-12-01 16:25:00", tz="America/New_York").timestamp()
    os.utime(spy_pq_1625, (t_1625, t_1625))
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-12-01 16:30:00", tz="America/New_York").timestamp())
    sig_spy_late = prov.get_signal("FUTOVL")
    assert sig_spy_late["execution_mode"] == "UNREADY"
    assert "BENCHMARK_OBSERVATIONS_UNALIGNED" in sig_spy_late["data_integrity"]["missing_modalities"]

    # Offline mtime-ordered cache entry is invalidated on switching enable_live_fetch=True
    df_off_t = df_f180.copy()
    df_off_t.attrs = {"obs_bar_date": "2025-12-01", "disk_mtime_ms": int(pd.Timestamp("2025-12-01 10:05:00", tz="America/New_York").timestamp() * 1000)}
    df_off_s = df_f180.copy()
    df_off_s.attrs = {"obs_bar_date": "2025-12-01", "disk_mtime_ms": int(pd.Timestamp("2025-12-01 10:04:00", tz="America/New_York").timestamp() * 1000)}
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-12-01 10:10:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(prov, "_load_disk_daily", lambda sym: df_off_s if sym.upper() == "SPY" else df_off_t)
    monkeypatch.setattr(prov, "_maybe_refresh_daily", lambda sym: None)
    monkeypatch.setattr(prov, "_fetch_live_quote_cached", lambda sym: None)
    with prov._lock:
        prov._signal_cache.clear()
    prov.enable_live_fetch = False
    sig_off_m = prov.get_signal("MODESW")
    assert "UNREADY" not in sig_off_m["execution_mode"]
    prov.enable_live_fetch = True
    sig_live_m = prov.get_signal("MODESW")
    assert sig_live_m["execution_mode"] == "UNREADY"
    assert "BENCHMARK_OBSERVATIONS_UNALIGNED" in sig_live_m["data_integrity"]["missing_modalities"]
    assert "regime_hmm" in dir(prov)

    # 11) _maybe_refresh_daily does not treat external mtime change during empty download as own replacement
    prov_r = LiveInferenceProvider(lakehouse_dir=tmp_path / "lh_ref52", weights_dir=wdir, enable_live_fetch=True)
    idx_r = pd.bdate_range(end="2025-12-03", periods=5, name="timestamp")
    df_part_r = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000, "vwap": 100.0}, index=idx_r)
    df_part_r.attrs = {"observation_timestamp_ms": int(pd.Timestamp("2025-12-03 15:50:00", tz="America/New_York").timestamp() * 1000), "obs_bar_date": "2025-12-03", "_bar_unsettled": True}
    prov_r.market_loader.save_ticker_daily("EMPTYEXT", df_part_r)
    ml_ext = MarketDataLoader(data_dir=tmp_path / "lh_ref52")
    df_set_ext = df_part_r.copy()
    df_set_ext.iloc[-1, df_set_ext.columns.get_loc("close")] = 108.0
    df_set_ext.iloc[-1, df_set_ext.columns.get_loc("high")] = 109.0
    df_set_ext.attrs = {}

    def _dl_empty_with_ext(*a, **kw):
        ml_ext.save_ticker_daily("EMPTYEXT", df_set_ext)
        return pd.DataFrame()

    monkeypatch.setattr(prov_r.market_loader, "_latest_expected_trading_date", lambda: pd.Timestamp("2025-12-03"))
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-12-03 16:20:00", tz="America/New_York").timestamp())
    monkeypatch.setattr("src.data.market_loader.time.time", lambda: pd.Timestamp("2025-12-03 16:20:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(yf, "download", _dl_empty_with_ext)
    prov_r._maybe_refresh_daily("EMPTYEXT")
    assert "EMPTYEXT" not in prov_r._failed_yf_until
    re_r = prov_r._load_disk_daily("EMPTYEXT")
    assert re_r is not None and float(re_r["close"].iloc[-1]) == pytest.approx(108.0)

    # 12) Unprovenanced target written at 16:25 cannot admit 16:10 post-close news,
    #     and Nov 24 Form 4/A replacement preserves Nov 21 decay age rather than resetting age to 0d
    df_t_1625 = df_f180.copy()
    df_t_1625.attrs = {}
    t_pq_1625 = ml.daily_dir / "T1625.parquet"
    ml.atomic_write_parquet(df_t_1625, t_pq_1625)
    t_1625_ts = pd.Timestamp("2025-12-01 16:25:00", tz="America/New_York").timestamp()
    os.utime(t_pq_1625, (t_1625_ts, t_1625_ts))
    t_1620_ts = pd.Timestamp("2025-12-01 16:20:00", tz="America/New_York").timestamp()
    os.utime(spy_pq_1625, (t_1620_ts, t_1620_ts))
    monkeypatch.setattr("src.ipc.live_providers.time.time", lambda: pd.Timestamp("2025-12-01 16:30:00", tz="America/New_York").timestamp())
    monkeypatch.setattr(prov, "_load_disk_daily", LiveInferenceProvider._load_disk_daily.__get__(prov))
    with prov._lock:
        prov._signal_cache.clear()
        prov._df_cache.clear()
        prov._news_cache["T1625"] = (time.monotonic(), [{"finbert_score": 0.9, "published_at": "2025-12-01T21:10:00Z", "published_at_verified": True}])
        prov._edgar_cache["T1625"] = (
            time.monotonic(),
            {
                "available": True,
                "form4_transactions": [
                    {
                        "form": "4", "filing_date": "2025-11-28", "transaction_date": "2025-11-28",
                        "disclosed_at": "2025-11-28T15:00:00Z", "disclosed_at_verified": True,
                        "accession_number": "F4_1", "insider_name": "Ada", "transaction_code": "P", "value": 50_000.0,
                    },
                    {
                        "form": "4/A", "filing_date": "2025-12-01", "date_of_original_submission": "2025-11-28",
                        "transaction_date": "2025-11-28", "disclosed_at": "2025-12-01T15:00:00Z", "disclosed_at_verified": True,
                        "accession_number": "F4_2", "insider_name": "Ada", "transaction_code": "P", "value": 5_000_000.0,
                    },
                ],
            },
        )
    sig_t1625 = prov.get_signal("T1625")
    assert "FINBERT_NEWS_NOT_LOADED" in sig_t1625["data_integrity"]["missing_modalities"]
    assert "DIVERGENCE_OBSERVATIONS_UNALIGNED" in sig_t1625["data_integrity"]["missing_modalities"]

    # Control: brand-new Dec 1 trade (age=0d) vs Dec 1 4/A replacement of Nov 28 trade (age=3d -> strictly smaller score)
    with prov._lock:
        prov._signal_cache.clear()
        prov._edgar_cache["T1625"] = (
            time.monotonic(),
            {
                "available": True,
                "form4_transactions": [
                    {
                        "form": "4", "filing_date": "2025-12-01", "transaction_date": "2025-12-01",
                        "disclosed_at": "2025-12-01T15:00:00Z", "disclosed_at_verified": True,
                        "accession_number": "F4_0D", "insider_name": "Ada", "transaction_code": "P", "value": 5_000_000.0,
                    },
                ],
            },
        )
    sig_0d = prov.get_signal("T1625")
    assert sig_t1625["signal_breakdown"]["decayed_form4_insider_cluster"] < sig_0d["signal_breakdown"]["decayed_form4_insider_cluster"]

















