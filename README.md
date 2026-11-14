# Amstal Engine (`amstal-engine`)

Quantitative alpha, risk, and live IPC inference engine (**QARE-MM v1.3**) for cross-sectional US equity swing analysis, built with Python, LightGBM, PyTorch, PyTorch Geometric, DuckDB/Parquet, and SEC EDGAR XBRL.

For full mathematical specifications and system design, see [`PRD.md`](./PRD.md) and [`architecture.md`](./architecture.md).

## Core Pipeline

- **Data & PIT Integrity**: Daily/hourly OHLCV (`yfinance` / Parquet lakehouse), SEC EDGAR XBRL & Form 4 (`edgartools`, `fundamentals_client.py`), and FinBERT news sentiment with post-16:00 ET point-in-time stamping.
- **Residualization & Features**: 2-stage sequential OLS orthogonalizing `SPY` and GICS Sector ETF betas, 15-channel 126-day microstructure sequences, 105-channel LightGBM tabular summaries, and 10-node dynamic peer/GICS graphs.
- **Models**: Sample-weighted `LightGBMBackbone` (primary live inference), `RegimeHMM` (forward-filtered 3-state Gaussian volatility HMM), plus PyTorch/PyG neural components (`AttentionGRUBackbone`, `PeerGAT`, `ModalityMaskedFusionNet`).
- **Risk & Execution**: Temperature-scaled probabilities, L1-normalized fractional Kelly sizing (`LONG_ONLY` default), and 15 bps round-trip friction gates.
- **Local IPC**: Unix Domain Socket JSON-RPC (`/tmp/amstal.sock`) and 100 ms conflated WebSocket stream (`ws://127.0.0.1:8194`).

## Directory Layout

```text
amstal-engine/
├── src/
│   ├── main.py              # Engine daemon entrypoint & market-hours stream loop
│   ├── data/                # market_loader, sec_edgar, fundamentals_client, universe
│   ├── features/            # residualizer, microstructure, sec_diff, news_sentiment, graph_builder
│   ├── models/              # backbone_lgbm, backbone_gru, peer_gat, regime_hmm, fusion_net
│   ├── label/               # triple_barrier, weights
│   ├── risk/                # calibration, kelly_engine
│   ├── ipc/                 # uds_server, ws_broadcaster, live_providers
│   └── backtest/            # walk_forward, metrics
├── scripts/                 # audit_specs.py, train_production.py, run_gate[1-4]_battery.py
└── tests/                   # Unit, IPC, and Gate 1-4 falsification suites
```

## Quick Start

```bash
git clone git@github.com:wwwrryyy/amstal-engine.git
cd amstal-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest tests/
python scripts/audit_specs.py
python src/main.py
```

Set `SEC_EDGAR_USER_AGENT` before querying SEC EDGAR endpoints.

## License

[MIT](./LICENSE). Quantitative research software provided as-is; not investment advice.
