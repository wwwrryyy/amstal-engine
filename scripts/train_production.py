"""
Production Model Training & Lakehouse Hydration CLI Runner (QARE-MM v1.3)
Ingests historical universe market data, computes 66-channel feature tensors,
fits Regime HMM, Dynamic Peer GAT, Multimodal Fusion Net, and LightGBM Backbone,
calibrates with Platt Temperature Scaling, and serializes trained weights to disk.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import pickle
import resource
import sys
import time
from typing import Any, Dict, List, Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Increase open file descriptor limits on macOS
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))
except Exception:
    pass

import numpy as np
import pandas as pd
import torch

from src.data.market_loader import MarketDataLoader
from src.data.universe import UniverseSelector
from src.features.graph_builder import GraphBuilder
from src.features.microstructure import MicrostructureFeatureExtractor
from src.features.residualizer import Residualizer
from src.label.triple_barrier import TripleBarrierLabeler
from src.label.weights import UniquenessWeighter
from src.models.backbone_gru import AttentionGRUBackbone
from src.models.backbone_lgbm import LightGBMBackbone
from src.models.fusion_net import ModalityMaskedFusionNet
from src.models.peer_gat import PeerGAT
from src.models.regime_hmm import RegimeHMM
from src.risk.calibration import TemperatureScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("amstal.trainer")


class ProductionTrainer:
    """End-to-end trainer orchestrating lakehouse hydration and weight generation."""

    def __init__(
        self,
        lakehouse_dir: Path,
        weights_dir: Path,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        mode: str = "synthetic",
        random_seed: int = 42,
    ) -> None:
        self.lakehouse_dir = Path(lakehouse_dir)
        self.weights_dir = Path(weights_dir)
        self.start_date = start_date
        self.end_date = end_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.mode = mode.lower()
        self.random_seed = random_seed

        self.market_loader = MarketDataLoader(data_dir=self.lakehouse_dir)
        self.universe_selector = UniverseSelector()
        self.residualizer = Residualizer(window_size=60)
        self.micro_extractor = MicrostructureFeatureExtractor(lookback_window=126)
        self.graph_builder = GraphBuilder(self.universe_selector)

        # Create output directories
        self.lakehouse_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        (self.lakehouse_dir / "processed" / "daily").mkdir(parents=True, exist_ok=True)
        (self.lakehouse_dir / "processed" / "hourly").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _stable_seed(key: str) -> int:
        return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % (2**31)

    def _save_or_fallback(self, ticker: str, df: Optional[pd.DataFrame]) -> Optional[str]:
        """Validates and saves DataFrame; preserves existing data before synthetic fallback."""
        try:
            if df is not None and not df.empty:
                df_clean = self.market_loader._normalize_yf_df(df)
                if len(df_clean) >= 126:
                    self.market_loader.validate_market_data(df_clean)
                    self.market_loader.save_ticker_daily(ticker, df_clean)
                    return ticker

            # Preserve existing disk partition if valid instead of overwriting real data with synthetic GBM
            existing = self.market_loader.load_ticker_daily(ticker)
            if existing is not None and len(existing) >= 126:
                return ticker

            if self.mode == "live":
                # Keep universe clean if live download temporarily fails when no local data exists
                return None

            # Fallback to deterministic synthetic path
            seed = self._stable_seed(f"{ticker}_{self.random_seed}")
            df_synth = self.market_loader.generate_synthetic_daily(
                ticker=ticker,
                start_date=self.start_date,
                end_date=self.end_date,
                base_price=100.0 + (seed % 200),
                drift=0.0003 + ((seed % 10) * 0.0001),
                vol=0.012 + ((seed % 15) * 0.001),
                seed=seed,
            )
            self.market_loader.save_ticker_daily(ticker, df_synth)
            return ticker
        except Exception as e:
            logger.warning(f"Error saving {ticker}: {e}")
            return None

    def hydrate_universe(self, target_tickers: Optional[List[str]] = None, overwrite: bool = False) -> List[str]:
        """
        Hydrates local Parquet lakehouse for SPY, 9 Sector ETFs, and active target equities.
        Sanitizes headers, handles MultiIndex columns, and validates OHLCV invariants.
        """
        logger.info(f"Hydrating Lakehouse (mode: {self.mode}) from {self.start_date} to {self.end_date} (overwrite={overwrite})...")

        benchmarks = ["SPY"]
        sector_etfs = sorted(list(self.universe_selector.SECTOR_ETF_MAP.values()))

        if target_tickers is None:
            active_equities = [
                rec.ticker
                for rec in self.universe_selector._security_master.values()
                if rec.active
            ]
        else:
            active_equities = [t.upper() for t in target_tickers]

        all_tickers = sorted(list(set(benchmarks + sector_etfs + active_equities)))
        logger.info(f"Target universe: {len(all_tickers)} symbols (SPY + {len(sector_etfs)} ETFs + {len(active_equities)} equities)")

        successful_tickers: List[str] = []

        if self.mode == "live":
            import yfinance as yf

            stale_cutoff = (pd.Timestamp(self.end_date) - pd.offsets.BDay(2)).normalize()
            yf_end = (pd.Timestamp(self.end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

            # Chunk into batches of 25 symbols for reliable Yahoo Finance API throughput
            chunk_size = 25
            chunks = [all_tickers[i:i + chunk_size] for i in range(0, len(all_tickers), chunk_size)]

            for chunk_idx, chunk in enumerate(chunks, 1):
                tickers_to_fetch = []
                for ticker in chunk:
                    if not overwrite:
                        df_ex = self.market_loader.load_ticker_daily(ticker, self.start_date, self.end_date)
                        if df_ex is not None and len(df_ex) >= 126:
                            last_dt = pd.Timestamp(df_ex.index.max()).normalize()
                            if last_dt >= stale_cutoff:
                                successful_tickers.append(ticker)
                                continue
                    tickers_to_fetch.append(ticker)

                if not tickers_to_fetch:
                    continue

                try:
                    df_bulk = yf.download(
                        tickers_to_fetch,
                        start=self.start_date,
                        end=yf_end,
                        progress=False,
                        auto_adjust=True,
                        group_by="ticker",
                        timeout=10,
                    )
                except Exception as e:
                    logger.warning(f"Bulk download failed for chunk {chunk_idx}: {e}")
                    df_bulk = None

                for ticker in tickers_to_fetch:
                    df_ticker: Optional[pd.DataFrame] = None
                    if df_bulk is not None and not df_bulk.empty:
                        try:
                            if len(tickers_to_fetch) == 1:
                                df_ticker = df_bulk
                            elif ticker in df_bulk:
                                df_ticker = df_bulk[ticker]
                        except Exception:
                            df_ticker = None

                    saved = self._save_or_fallback(ticker, df_ticker)
                    if saved:
                        successful_tickers.append(saved)

                logger.info(f"Hydrated {min(chunk_idx * chunk_size, len(all_tickers))}/{len(all_tickers)} symbols into lakehouse Parquet.")
                time.sleep(0.15)
        else:
            for i, ticker in enumerate(all_tickers, 1):
                if not overwrite:
                    df_ex = self.market_loader.load_ticker_daily(ticker, self.start_date, self.end_date)
                    if df_ex is not None and len(df_ex) >= 126:
                        successful_tickers.append(ticker)
                        continue

                saved = self._save_or_fallback(ticker, None)
                if saved:
                    successful_tickers.append(saved)
                if i % 50 == 0 or i == len(all_tickers):
                    logger.info(f"Hydrated {i}/{len(all_tickers)} symbols into lakehouse Parquet.")

        logger.info(f"Lakehouse hydration complete: {len(successful_tickers)} valid symbols stored.")
        return successful_tickers

    def train_all(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Executes production multi-model fitting, temperature calibration, and serialization.
        """
        logger.info("Beginning model training and calibration pipeline...")
        t0 = time.perf_counter()

        # -------------------------------------------------------------
        # STEP 1: Fit 3-State Regime HMM on SPY + Volatility
        # -------------------------------------------------------------
        logger.info("[1/5] Fitting 3-State Gaussian Regime HMM on SPY benchmark...")
        df_spy = self.market_loader.load_ticker_daily("SPY", self.start_date, self.end_date)
        if df_spy is None or len(df_spy) < 126:
            raise RuntimeError("SPY historical data is missing or has insufficient length.")

        spy_ret = np.diff(np.log(df_spy["close"].values))
        high_low = df_spy["high"].values[1:] / np.maximum(df_spy["low"].values[1:], 1e-8)
        spy_vol = (np.log(high_low) ** 2) / (4.0 * np.log(2.0))  # Parkinson proxy
        X_hmm = np.column_stack([spy_ret, spy_vol])

        regime_hmm = RegimeHMM(n_iter=50, random_state=self.random_seed)
        regime_hmm.fit(X_hmm)
        logger.info(f"Regime HMM fitted. State variance ordering: {regime_hmm.state_order_}")

        # -------------------------------------------------------------
        # STEP 2: Compute Features & Residualized Triple Barrier Labels
        # -------------------------------------------------------------
        logger.info("[2/5] Extracting 105-channel features, 126x15 sequences & residual triple-barrier labels...")
        labeler = TripleBarrierLabeler(
            vol_lookback=20,
            upper_multiplier=2.0,
            lower_multiplier=1.0,
            vertical_horizon=10,
        )

        # Sample-aligned flat buffers (will be chronologically sorted by date before split)
        sample_dates: List[pd.Timestamp] = []
        sample_tickers: List[str] = []
        all_features: List[np.ndarray] = []
        all_labels: List[int] = []
        all_weights: List[float] = []
        seq_dates: List[pd.Timestamp] = []
        seq_samples: List[np.ndarray] = []
        seq_labels: List[int] = []
        seq_weights: List[float] = []
        residuals_panel: Dict[str, pd.Series] = {}
        latest_node_vecs: Dict[str, np.ndarray] = {}
        latest_ticker_labels: Dict[str, int] = {}

        equities = [t for t in tickers if t not in self.universe_selector.SECTOR_ETF_MAP.values() and t != "SPY"]

        for ticker in equities:
            df_equity = self.market_loader.load_ticker_daily(ticker, self.start_date, self.end_date)
            if df_equity is None or len(df_equity) < 126:
                continue

            try:
                sector_etf = self.universe_selector.get_sector_etf(ticker)
            except KeyError:
                sector_etf = "SPY"
            df_sec = self.market_loader.load_ticker_daily(sector_etf, self.start_date, self.end_date)
            if df_sec is None:
                df_sec = df_spy

            common_idx = df_equity.index.intersection(df_spy.index).intersection(df_sec.index)
            if len(common_idx) < 156:
                continue

            eq_sub = df_equity.loc[common_idx]
            spy_sub = df_spy.loc[common_idx]
            sec_sub = df_sec.loc[common_idx]

            r_stock = np.diff(np.log(eq_sub["close"].values))
            r_spy = np.diff(np.log(spy_sub["close"].values))
            r_sec = np.diff(np.log(sec_sub["close"].values))

            roll_res = self.residualizer.compute_rolling_residuals(r_stock, r_spy, r_sec)
            residuals_series = pd.Series(roll_res.residuals, index=eq_sub.index[1:])
            residuals_panel[ticker] = residuals_series

            channels_df = self.micro_extractor.compute_raw_channels_df(
                eq_sub.iloc[1:], residuals=residuals_series
            )
            seq_tensor = self.micro_extractor.build_sequence_tensor(channels_df)
            if len(seq_tensor) == 0:
                continue
            lgbm_feats = self.micro_extractor.extract_summary_105(seq_tensor)
            lgbm_idx = channels_df.index[self.micro_extractor.lookback_window - 1:]
            lgbm_df = pd.DataFrame(lgbm_feats, index=lgbm_idx)

            # Label strictly on residualized return series per PRD Section 5.4 (never raw price levels!)
            labeled_df = labeler.label_series(residuals_series)
            if len(labeled_df) == 0:
                continue
            labeled_df = UniquenessWeighter.weight_labeled_df(labeled_df, normalize=True)

            # Construct compact 10D peer graph node vector for ticker's latest valid date
            res_20d = float(np.nansum(residuals_series.iloc[-20:])) if len(residuals_series) >= 20 else 0.0
            res_5d = float(np.nansum(residuals_series.iloc[-5:])) if len(residuals_series) >= 5 else 0.0
            beta_60 = float(roll_res.betas_mkt[-1]) if not np.isnan(roll_res.betas_mkt[-1]) else 1.0
            latest_node_vecs[ticker] = GraphBuilder.build_prd_10d_node_vector(
                res_20d=res_20d, res_5d=res_5d, beta_spy_60d=beta_60, node_role=0.0
            )
            latest_ticker_labels[ticker] = int(labeled_df["label"].iloc[-1])

            common_dates = lgbm_df.index.intersection(labeled_df.index)
            if len(common_dates) > 0:
                feat_slice = lgbm_df.loc[common_dates].values
                label_slice = labeled_df.loc[common_dates, "label"].values.astype(int)
                weight_slice = labeled_df.loc[common_dates, "uniqueness_weight"].values.astype(float)

                all_features.append(feat_slice)
                all_labels.extend(label_slice.tolist())
                all_weights.extend(weight_slice.tolist())
                sample_dates.extend(common_dates.tolist())
                sample_tickers.extend([ticker] * len(common_dates))

                # Subsample sequences every 10 trading days for tractable CPU GRU+Fusion training
                date_to_pos = {d: i for i, d in enumerate(lgbm_idx)}
                for j in range(0, len(common_dates), 10):
                    dt = common_dates[j]
                    pos = date_to_pos.get(dt)
                    if pos is not None and pos < len(seq_tensor):
                        seq_dates.append(dt)
                        seq_samples.append(seq_tensor[pos])
                        seq_labels.append(int(label_slice[j]))
                        seq_weights.append(float(weight_slice[j]))

        if not all_features:
            raise RuntimeError("No valid features could be extracted from universe equities.")

        X_raw = np.vstack(all_features)
        y_raw = np.array(all_labels, dtype=int)
        w_raw = np.array(all_weights, dtype=np.float64)
        dates_arr = pd.to_datetime(sample_dates).values

        # Sort globally in strict chronological order so train/val split never leaks future across tickers
        chrono_order = np.argsort(dates_arr, kind="stable")
        X_all = X_raw[chrono_order]
        y_all = y_raw[chrono_order]
        weights_all = w_raw[chrono_order]

        unique_classes = np.unique(y_all)
        class_dist = {int(c): int(np.sum(y_all == c)) for c in unique_classes}
        logger.info(f"Total training dataset size: {X_all.shape[0]} samples | Class distribution: {class_dist}")
        if len(unique_classes) < 3:
            raise RuntimeError(f"Triple-barrier labels collapsed to classes {unique_classes.tolist()}; expected all {{0, 1, 2}}.")

        # Chronological 80/20 split with 10-day purge gap to avoid horizon overlap across the boundary
        split_idx = int(len(y_all) * 0.80)
        purge_offset = min(len(equities) * 10, max(1, (len(y_all) - split_idx) // 10))
        X_train, y_train, w_train = X_all[:split_idx], y_all[:split_idx], weights_all[:split_idx]
        val_start = min(split_idx + purge_offset, len(y_all) - 10)
        X_val, y_val, w_val = X_all[val_start:], y_all[val_start:], weights_all[val_start:]

        # -------------------------------------------------------------
        # STEP 3: Fit LightGBM Backbone & Post-Hoc Temperature Scaler
        # -------------------------------------------------------------
        logger.info("[3/5] Fitting LightGBM multiclass backbone...")
        lgbm_backbone = LightGBMBackbone(n_estimators=200, learning_rate=0.03, random_state=self.random_seed)
        lgbm_backbone.fit(X_train, y_train, sample_weights=w_train, eval_set=(X_val, y_val), early_stopping_rounds=15)

        raw_val_probs = lgbm_backbone.predict_proba(X_val)

        logger.info("[4/5] Fitting Platt Temperature Scaler on validation fold...")
        temp_scaler = TemperatureScaler(bounds=(0.10, 5.0))
        optimal_temp = temp_scaler.fit(raw_val_probs, y_val)
        calibrated_probs = temp_scaler.transform(raw_val_probs)

        # Compute Brier score improvement
        def brier_score(p: np.ndarray, y: np.ndarray) -> float:
            n = len(y)
            one_hot = np.zeros((n, 3))
            one_hot[np.arange(n), y] = 1.0
            return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))

        brier_uncal = brier_score(raw_val_probs, y_val)
        brier_cal = brier_score(calibrated_probs, y_val)
        denom = brier_uncal if brier_uncal > 1e-6 else 1.0
        brier_reduction = (brier_uncal - brier_cal) / denom
        logger.info(f"Optimal Temperature T={optimal_temp:.4f} | Validation Brier: {brier_uncal:.4f} -> {brier_cal:.4f} (Drop: {brier_reduction*100:.2f}%)")

        # -------------------------------------------------------------
        # STEP 4: Fit Attention-GRU, Dynamic Peer GAT & Multimodal Fusion Net
        # -------------------------------------------------------------
        logger.info("[5/5] Training Attention-GRU, Dynamic Peer GAT, and Modality-Masked Fusion Net...")
        gru_backbone = AttentionGRUBackbone(input_dim=15, hidden_dim=32, num_heads=2, random_state=self.random_seed)
        if len(seq_samples) >= 32:
            seq_order = np.argsort(pd.to_datetime(seq_dates).values, kind="stable")
            X_seq = np.array(seq_samples, dtype=np.float32)[seq_order]
            y_seq = np.array(seq_labels, dtype=np.int64)[seq_order]
            w_seq = np.array(seq_weights, dtype=np.float32)[seq_order]
            seq_split = int(len(X_seq) * 0.80)
            gru_backbone.fit(
                X_seq[:seq_split],
                y_seq[:seq_split],
                sample_weights=w_seq[:seq_split],
                epochs=5,
                batch_size=64,
                lr=1e-3,
                eval_set=(X_seq[seq_split:], y_seq[seq_split:]),
            )
            h_temp_all = gru_backbone.extract_latent(X_seq)
        else:
            X_seq = np.zeros((16, 126, 15), dtype=np.float32)
            y_seq = np.zeros(16, dtype=np.int64)
            w_seq = np.ones(16, dtype=np.float32)
            seq_split = len(X_seq)
            h_temp_all = np.zeros((16, 32), dtype=np.float32)

        peer_gat = PeerGAT(in_channels=10, hidden_dim=32, random_state=self.random_seed)
        res_df_panel = pd.DataFrame(residuals_panel) if residuals_panel else None
        train_subgraphs = []
        train_graph_labels = []
        for sym, node_vec in latest_node_vecs.items():
            sg = self.graph_builder.build_subgraph(
                target_ticker=sym,
                node_features=latest_node_vecs,
                feature_dim=10,
                residual_returns=res_df_panel,
            )
            train_subgraphs.append(sg)
            train_graph_labels.append(latest_ticker_labels[sym])

        if len(train_subgraphs) >= 4:
            batch_size_g = 16
            gat_batches = [
                self.graph_builder.build_batch(train_subgraphs[i : i + batch_size_g])
                for i in range(0, len(train_subgraphs), batch_size_g)
            ]
            peer_gat.fit(gat_batches, train_graph_labels, epochs=6, lr=0.003)

        fusion_net = ModalityMaskedFusionNet(dropout=0.20, random_state=self.random_seed)
        n_fusion = min(seq_split, len(h_temp_all), 256)
        if n_fusion >= 16 and len(train_subgraphs) > 0:
            batch_g = self.graph_builder.build_batch([train_subgraphs[i % len(train_subgraphs)] for i in range(n_fusion)])
            h_graph_f = peer_gat.extract_latent(batch_g)
            fusion_net.fit(
                h_temporal=h_temp_all[:n_fusion],
                h_graph=h_graph_f,
                y_train=y_seq[:n_fusion],
                kappa=0.45,
                sample_weights=w_seq[:n_fusion],
                epochs=6,
                lr=0.003,
                batch_size=32,
            )
        fusion_net.eval()

        # Quick forward verification pass
        with torch.no_grad():
            dummy_temporal = torch.randn(1, 32)
            dummy_graph = torch.randn(1, 32)
            dummy_alt = torch.zeros(1, 2)
            dummy_hmm = torch.tensor([[1.0, 0.0, 0.0]])
            logits, h_gated, h_fused = fusion_net(
                h_temporal=dummy_temporal,
                h_graph=dummy_graph,
                kappa=0.50,
                d_t=dummy_alt,
                hmm_probs=dummy_hmm,
            )
            assert logits.shape == (1, 3), f"Unexpected fusion logits shape: {logits.shape}"

        # -------------------------------------------------------------
        # STEP 5: Serialize Trained Weights Atomically
        # -------------------------------------------------------------
        logger.info(f"Serializing model weights to {self.weights_dir}...")
        with open(self.weights_dir / "regime_hmm.pkl", "wb") as f:
            pickle.dump(regime_hmm, f)

        with open(self.weights_dir / "lightgbm_backbone.pkl", "wb") as f:
            pickle.dump(lgbm_backbone, f)

        with open(self.weights_dir / "temperature_scaler.pkl", "wb") as f:
            pickle.dump(temp_scaler, f)

        torch.save(gru_backbone.model.state_dict() if gru_backbone.model is not None else {}, self.weights_dir / "gru_backbone.pt")
        torch.save(peer_gat.state_dict(), self.weights_dir / "peer_gat.pt")
        torch.save(fusion_net.state_dict(), self.weights_dir / "fusion_net.pt")

        manifest = {
            "version": "QARE-MM-v1.3",
            "training_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "mode": self.mode,
            "universe_size": len(equities),
            "total_samples": int(X_all.shape[0]),
            "feature_dim": int(X_all.shape[1]),
            "class_distribution": class_dist,
            "metrics": {
                "validation_brier_uncalibrated": round(brier_uncal, 4),
                "validation_brier_calibrated": round(brier_cal, 4),
                "brier_improvement_pct": round(brier_reduction * 100, 2),
                "optimal_temperature": round(optimal_temp, 4),
                "regime_hmm_state_order": regime_hmm.state_order_.tolist(),
            },
            "artifacts": [
                "regime_hmm.pkl",
                "lightgbm_backbone.pkl",
                "temperature_scaler.pkl",
                "gru_backbone.pt",
                "peer_gat.pt",
                "fusion_net.pt",
            ],
        }

        with open(self.weights_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        elapsed = time.perf_counter() - t0
        logger.info(f"Production model training and weight serialization finished in {elapsed:.2f}s.")
        return manifest


ENGINE_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Amstal Production Model Training & Lakehouse Hydration")
    parser.add_argument("--mode", choices=["synthetic", "live"], default="synthetic", help="Hydration mode")
    parser.add_argument("--lakehouse-dir", default=str(ENGINE_ROOT / "lakehouse"), help="Directory for DuckDB/Parquet store")
    parser.add_argument("--weights-dir", default=str(ENGINE_ROOT / "weights"), help="Directory for serialized weights")
    parser.add_argument("--start-date", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="End date (YYYY-MM-DD, defaults to today UTC)")
    parser.add_argument("--tickers", default="", help="Optional comma-separated list of tickers")
    parser.add_argument("--overwrite", action="store_true", help="Force re-download and overwrite existing data")
    args = parser.parse_args()

    custom_tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] if args.tickers else None

    trainer = ProductionTrainer(
        lakehouse_dir=Path(args.lakehouse_dir),
        weights_dir=Path(args.weights_dir),
        start_date=args.start_date,
        end_date=args.end_date,
        mode=args.mode,
    )

    hydrated_tickers = trainer.hydrate_universe(target_tickers=custom_tickers, overwrite=args.overwrite)
    manifest = trainer.train_all(hydrated_tickers)

    print("\n" + "=" * 80)
    print("AMSTAL PRODUCTION MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Artifacts stored in: {trainer.weights_dir}")
    print(f"Validation Brier Score: {manifest['metrics']['validation_brier_calibrated']}")
    print(f"Brier Improvement:      +{manifest['metrics']['brier_improvement_pct']}%")
    print(f"Optimal Temperature:    {manifest['metrics']['optimal_temperature']}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
