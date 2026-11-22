"""
Market Data Loader with Atomic POSIX Writes and Stateless DuckDB Queries
Loads daily OHLCV and intraday hourly bars with stateless DuckDB query caching.
Enforces Point-in-Time (PIT) cutoff at 16:00 America/New_York to eliminate look-ahead bias.
"""

from collections.abc import Mapping
from contextlib import contextmanager
from datetime import date, datetime, time as dtime, timedelta, timezone
try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]
import hashlib
import logging
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger("amstal.market_loader")

_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCK_ENTRIES: Dict[str, List[Any]] = {}


def _get_path_lock_entry(real_key: str) -> List[Any]:
    with _PATH_LOCKS_GUARD:
        entry = _PATH_LOCK_ENTRIES.get(real_key)
        if entry is None:
            entry = [threading.RLock(), 0]
            _PATH_LOCK_ENTRIES[real_key] = entry
        return entry


@contextmanager
def _partition_commit_lock(target_path: Path):
    """Coordinates partition writers across threads, instances, and OS processes (re-entrant per thread)."""
    real_p = Path(os.path.realpath(target_path))
    entry = _get_path_lock_entry(str(real_p))
    rlock = entry[0]
    with rlock:
        depth = entry[1]
        flock_fh = None
        flock_acquired = False
        entry[1] = depth + 1
        try:
            if depth == 0 and fcntl is not None and real_p.parent.exists():
                lock_file = real_p.with_name(f".{real_p.name}.lock")
                flock_fh = open(lock_file, "a+")
                fcntl.flock(flock_fh.fileno(), fcntl.LOCK_EX)
                flock_acquired = True
            yield
        finally:
            entry[1] = depth
            if flock_fh is not None:
                try:
                    if flock_acquired:
                        fcntl.flock(flock_fh.fileno(), fcntl.LOCK_UN)
                finally:
                    flock_fh.close()


def _file_stat_gen(target_path: Path) -> Optional[Tuple[int, int, int, int]]:
    try:
        st = os.stat(target_path)
        return (st.st_dev, st.st_ino, st.st_mtime_ns, st.st_size)
    except FileNotFoundError:
        return None


class MarketDataLoader:
    """Manages raw/processed market data ingestion, atomic file replacement, and stateless DuckDB queries."""

    REQUIRED_DAILY_COLUMNS = ["open", "high", "low", "close", "volume"]
    _POST_CLOSE_SETTLEMENT_MS = 15 * 60 * 1000

    def __init__(self, data_dir: Union[str, Path]) -> None:
        self.data_dir = Path(data_dir)
        self.daily_dir = self.data_dir / "processed" / "daily"
        self.hourly_dir = self.data_dir / "processed" / "hourly"
        self.universe_dir = self.data_dir / "universe"

        # Ensure directory tree exists
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.hourly_dir.mkdir(parents=True, exist_ok=True)
        self.universe_dir.mkdir(parents=True, exist_ok=True)
        self._refresh_locks_guard = threading.Lock()
        self._ticker_refresh_locks: Dict[str, threading.RLock] = {}

    def _get_refresh_lock(self, ticker: str) -> threading.RLock:
        sym = ticker.upper()
        with self._refresh_locks_guard:
            lock = self._ticker_refresh_locks.get(sym)
            if lock is None:
                lock = threading.RLock()
                self._ticker_refresh_locks[sym] = lock
            return lock

    @staticmethod
    def is_nyse_trading_day(day_val: Any) -> bool:
        """Returns True iff day_val is a weekday and not an NYSE market holiday."""
        d = day_val.date() if isinstance(day_val, datetime) else day_val
        if not isinstance(d, date) or d.weekday() >= 5:
            return False
        y = d.year

        def _observed_fixed(month: int, day: int) -> date:
            dt = date(y, month, day)
            if dt.weekday() == 5:
                return dt - timedelta(days=1)
            if dt.weekday() == 6:
                return dt + timedelta(days=1)
            return dt

        def _nth_weekday(month: int, weekday: int, nth: int) -> date:
            first = date(y, month, 1)
            offset = (weekday - first.weekday()) % 7
            return first + timedelta(days=offset + 7 * (nth - 1))

        def _last_weekday(month: int, weekday: int) -> date:
            nxt_first = date(y + (1 if month == 12 else 0), (month % 12) + 1, 1)
            last = nxt_first - timedelta(days=1)
            offset = (last.weekday() - weekday) % 7
            return last - timedelta(days=offset)

        # Anonymous Gregorian Easter Computus -> Good Friday
        a = y % 19
        b, c = divmod(y, 100)
        d_c, e = divmod(b, 4)
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d_c - g + 15) % 30
        i_c, k = divmod(c, 4)
        l_e = (32 + 2 * e + 2 * i_c - h - k) % 7
        m_e = (a + 11 * h + 22 * l_e) // 451
        easter_month = (h + l_e - 7 * m_e + 114) // 31
        easter_day = ((h + l_e - 7 * m_e + 114) % 31) + 1
        good_friday = date(y, easter_month, easter_day) - timedelta(days=2)

        holidays = {
            good_friday,
            _nth_weekday(2, 0, 3),   # Presidents' Day
            _last_weekday(5, 0),     # Memorial Day
            _observed_fixed(7, 4),   # Independence Day
            _nth_weekday(9, 0, 1),   # Labor Day
            _nth_weekday(11, 3, 4),  # Thanksgiving
            _observed_fixed(12, 25), # Christmas
            date(2001, 9, 11),       # September 11 attacks
            date(2001, 9, 12),
            date(2001, 9, 13),
            date(2001, 9, 14),
            date(2004, 6, 11),       # National Day of Mourning (R. Reagan)
            date(2007, 1, 2),        # National Day of Mourning (G.R. Ford)
            date(2012, 10, 29),      # Hurricane Sandy closure
            date(2012, 10, 30),      # Hurricane Sandy closure
            date(2018, 12, 5),       # National Day of Mourning (G.H.W. Bush)
            date(2025, 1, 9),        # National Day of Mourning (J. Carter)
        }
        # New Year's Day on Saturday is NOT observed on prior Friday Dec 31 by NYSE; check Sat/Sun/Mon explicitly
        nyd = date(y, 1, 1)
        if nyd.weekday() == 6:
            holidays.add(date(y, 1, 2))
        elif nyd.weekday() < 5:
            holidays.add(nyd)
        if y >= 1998:
            holidays.add(_nth_weekday(1, 0, 3))  # MLK Jr. Day
        if y >= 2022:
            holidays.add(_observed_fixed(6, 19))  # Juneteenth
        return d not in holidays

    @staticmethod
    def nyse_session_close_time(session_date: Any) -> dtime:
        """Returns NYSE regular close time (13:00 ET on scheduled half-days, else 16:00 ET)."""
        d = session_date.date() if isinstance(session_date, datetime) else session_date
        if isinstance(d, date):
            is_black_friday = d.month == 11 and d.weekday() == 4 and 23 <= d.day <= 29
            is_xmas_eve = d.month == 12 and d.day == 24 and d.weekday() < 4
            is_july3_early = d.month == 7 and d.day == 3 and d.weekday() < 4
            if is_black_friday or is_xmas_eve or is_july3_early:
                return dtime(13, 0)
        return dtime(16, 0)

    @staticmethod
    def atomic_write_parquet(df: pd.DataFrame, target_path: Path) -> None:
        """
        Atomically writes Parquet to avoid partial reads or file locking.
        Persists row-scoped observation metadata atomically alongside Parquet schema metadata.
        """
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with _partition_commit_lock(target_path):
            tmp_name = f"{target_path.stem}.{os.getpid()}.{time.time_ns()}.tmp.parquet"
            tmp_path = target_path.with_name(tmp_name)

            attrs = getattr(df, "attrs", {}) or {}
            obs_ms = attrs.get("observation_timestamp_ms")
            obs_date = attrs.get("obs_bar_date")
            is_unsettled = bool(attrs.get("_bar_unsettled"))
            has_valid_obs = (
                isinstance(obs_ms, (int, float))
                and not isinstance(obs_ms, bool)
                and math.isfinite(float(obs_ms))
                and float(obs_ms) > 0.0
            )
            if (has_valid_obs or is_unsettled) and len(df) > 0:
                last_ts_p = pd.Timestamp(df.index[-1])
                last_dt = (
                    last_ts_p.strftime("%Y-%m-%d")
                    if hasattr(last_ts_p, "strftime")
                    else str(df.index[-1])[:10]
                )
                if obs_date is None or str(obs_date)[:10] == last_dt:
                    import pyarrow as pa
                    import pyarrow.parquet as pq

                    eff_obs_date = str(obs_date or last_dt)[:10]
                    table = pa.Table.from_pandas(df, preserve_index=True)
                    meta = dict(table.schema.metadata or {})
                    meta[b"amstal_obs_bar_date"] = eff_obs_date.encode("utf-8")
                    meta[b"amstal_obs_bar_iso"] = last_ts_p.isoformat().encode("utf-8")
                    if has_valid_obs:
                        meta[b"amstal_obs_ts_ms"] = str(int(obs_ms)).encode("utf-8")
                    if is_unsettled:
                        meta[b"amstal_bar_unsettled"] = b"1"
                    table = table.replace_schema_metadata(meta)
                    pq.write_table(table, tmp_path)
                else:
                    df.to_parquet(tmp_path, engine="pyarrow", index=True)
            else:
                df.to_parquet(tmp_path, engine="pyarrow", index=True)
            os.replace(tmp_path, target_path)

    @staticmethod
    def apply_pit_cutoff(
        dt_val: Union[str, pd.Timestamp, datetime],
        cutoff_time: Optional[str] = None,
        tz_str: str = "America/New_York",
        skip_weekends: bool = True,
    ) -> pd.Timestamp:
        """
        Strict Point-in-Time Cutoff Enforcement.
        Any data arriving after cutoff_time (defaulting to the NYSE session close for
        ts.date(), i.e., 16:00 America/New_York or 13:00 on early-close sessions) on day t
        is shifted to day t+1 (or next trading session if skip_weekends=True) to prevent look-ahead.
        """
        if isinstance(dt_val, str):
            ts = pd.to_datetime(dt_val)
        elif isinstance(dt_val, datetime) and not isinstance(dt_val, pd.Timestamp):
            ts = pd.Timestamp(dt_val)
        else:
            ts = dt_val

        # Ensure timezone-aware in tz_str
        if ts.tz is None:
            ts = ts.tz_localize("UTC").tz_convert(tz_str)
        else:
            ts = ts.tz_convert(tz_str)

        if cutoff_time is None:
            ts_ny = ts.tz_convert("America/New_York")
            cutoff = MarketDataLoader.nyse_session_close_time(ts_ny.date())
            ref_date = ts_ny.date()
            ref_time = ts_ny.time()
        else:
            cutoff_hour, cutoff_minute = map(int, cutoff_time.split(":"))
            cutoff = dtime(cutoff_hour, cutoff_minute, 0)
            ref_date = ts.date()
            ref_time = ts.time()

        # Check if strictly after cutoff time or arriving on a closed session
        if not MarketDataLoader.is_nyse_trading_day(ref_date) or ref_time > cutoff:
            # Shift forward to next calendar day (then skip non-trading sessions below)
            next_date = ref_date + timedelta(days=1)
        else:
            next_date = ref_date

        if skip_weekends:
            while not MarketDataLoader.is_nyse_trading_day(next_date):
                next_date += timedelta(days=1)

        return pd.Timestamp(next_date).tz_localize(tz_str)

    @staticmethod
    def validate_market_data(df: pd.DataFrame) -> None:
        """
        Validates OHLCV market data invariants:
        - Non-empty DataFrame
        - Monotonic DatetimeIndex without duplicates
        - Required columns: open, high, low, close, volume
        - Positive finite prices and non-negative finite volume
        - High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close
        """
        if df.empty:
            raise ValueError("Market data DataFrame cannot be empty.")

        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                temp_idx = pd.to_datetime(df["timestamp"])
                if not temp_idx.is_monotonic_increasing:
                    raise ValueError("Market data timestamps must be strictly monotonic increasing.")
                if temp_idx.has_duplicates:
                    raise ValueError("Market data timestamps contain duplicates.")
            else:
                raise ValueError("DataFrame index must be a DatetimeIndex or contain 'timestamp' column.")
        else:
            if not df.index.is_monotonic_increasing:
                raise ValueError("Market data timestamps must be strictly monotonic increasing.")
            if df.index.has_duplicates:
                raise ValueError("Market data timestamps contain duplicates.")

        lower_cols = [c.lower() for c in df.columns]
        for col in MarketDataLoader.REQUIRED_DAILY_COLUMNS:
            if col not in lower_cols:
                raise ValueError(f"Missing required market data column: '{col}'. Present: {df.columns.tolist()}")

        # Ensure columns are accessible in lowercase
        col_map = {c: c.lower() for c in df.columns}
        renamed = df.rename(columns=col_map)

        for price_col in ["open", "high", "low", "close"]:
            vals = renamed[price_col].values
            if np.any(np.isnan(vals)) or np.any(np.isinf(vals)):
                raise ValueError(f"Column '{price_col}' contains NaN or infinite values.")
            if np.any(vals <= 0):
                raise ValueError(f"Column '{price_col}' contains non-positive values.")

        vols = renamed["volume"].values
        if np.any(np.isnan(vols)) or np.any(np.isinf(vols)):
            raise ValueError("Column 'volume' contains NaN or infinite values.")
        if np.any(vols < 0):
            raise ValueError("Column 'volume' contains negative values.")

        # High / Low boundary invariants
        if np.any(renamed["high"] < renamed["low"]):
            raise ValueError("High price is less than Low price in market data.")
        if np.any(renamed["high"] < renamed["open"]) or np.any(renamed["high"] < renamed["close"]):
            raise ValueError("High price must be >= max(Open, Close).")
        if np.any(renamed["low"] > renamed["open"]) or np.any(renamed["low"] > renamed["close"]):
            raise ValueError("Low price must be <= min(Open, Close).")

    def query_parquet_duckdb(
        self,
        parquet_path: Union[str, Path],
        columns: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Executes a stateless analytical query in DuckDB (:memory:) directly on Parquet.
        Zero file lock contention and sub-15ms latency.
        Guarantees returned DataFrame is indexed by DatetimeIndex.
        """
        p_path = Path(parquet_path)
        if not p_path.exists():
            raise FileNotFoundError(f"Parquet file does not exist: {p_path}")

        con = duckdb.connect(":memory:")
        try:
            con.execute("SET TimeZone = 'UTC'")
            schema_cols = con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(p_path)]).fetchall()
            col_by_lower = {str(r[0]).lower(): str(r[0]) for r in schema_cols}
            ts_actual = col_by_lower.get("timestamp", "timestamp")

            # Guarantee timestamp is fetched and normalized so date filtering & index assignment work
            if columns:
                col_regex = re.compile(r"^[A-Za-z0-9_]+$")
                for c in columns:
                    if not col_regex.match(c):
                        raise ValueError(f"Invalid column identifier: {c!r}")
                query_cols: List[str] = []
                seen_lower: set[str] = set()
                for c in columns:
                    cl = c.lower()
                    actual = col_by_lower.get(cl, c)
                    if cl == "timestamp":
                        if "timestamp" not in seen_lower:
                            query_cols.append(f'"{actual}" AS "timestamp"')
                            seen_lower.add("timestamp")
                    elif cl not in seen_lower:
                        query_cols.append(f'"{actual}"')
                        seen_lower.add(cl)
                if "timestamp" not in seen_lower:
                    query_cols.append(f'"{ts_actual}" AS "timestamp"')
                col_sql = ", ".join(query_cols)
            else:
                col_sql = "*"

            where_clauses = []
            params = [str(p_path)]

            if start_date is not None:
                where_clauses.append(f'"{ts_actual}" >= ?')
                params.append(start_date)
            if end_date is not None:
                where_clauses.append(f'"{ts_actual}" <= ?')
                params.append(end_date)

            where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            sql = f'SELECT {col_sql} FROM read_parquet(?) {where_str} ORDER BY "{ts_actual}" ASC'

            import pyarrow.parquet as pq

            def _normalize_ts_col(frame: pd.DataFrame) -> pd.DataFrame:
                if "timestamp" in frame.columns:
                    return frame
                ts_matches = [col_n for col_n in frame.columns if str(col_n).lower() == "timestamp"]
                if ts_matches:
                    return frame.rename(columns={ts_matches[0]: "timestamp"})
                if str(frame.index.name or "").lower() == "timestamp" or isinstance(frame.index, pd.DatetimeIndex):
                    return frame.rename_axis("timestamp").reset_index()
                return frame

            df = pd.DataFrame()
            schema_meta: Dict[bytes, bytes] = {}
            matched_mtime_ns: Optional[int] = None
            matched_generation = False
            for _ in range(3):
                st_before = os.stat(p_path)
                cand_df = con.execute(sql, params).fetchdf()
                with open(p_path, "rb") as fh:
                    st_meta = os.fstat(fh.fileno())
                    cand_meta = pq.read_schema(fh).metadata or {}
                if (
                    st_before.st_dev == st_meta.st_dev
                    and st_before.st_ino == st_meta.st_ino
                    and st_before.st_mtime_ns == st_meta.st_mtime_ns
                    and st_before.st_size == st_meta.st_size
                ):
                    df = _normalize_ts_col(cand_df)
                    schema_meta = cand_meta
                    matched_mtime_ns = st_meta.st_mtime_ns
                    matched_generation = True
                    break

            if not matched_generation:
                with open(p_path, "rb") as fh:
                    st_meta = os.fstat(fh.fileno())
                    tbl = pq.read_table(fh)
                    schema_meta = tbl.schema.metadata or {}
                    df = tbl.to_pandas()
                    matched_mtime_ns = st_meta.st_mtime_ns
                df = _normalize_ts_col(df)
                if columns:
                    col_by_lower = {c.lower(): c for c in df.columns}
                    sel = []
                    for req in columns:
                        actual = col_by_lower.get(req.lower())
                        if actual is None:
                            raise KeyError(f"Requested column {req!r} not found in Parquet schema {list(df.columns)}")
                        if actual not in sel:
                            sel.append(actual)
                    ts_norm_col = col_by_lower.get("timestamp")
                    if ts_norm_col is not None and ts_norm_col not in sel:
                        sel.append(ts_norm_col)
                    df = df[sel]
                if "timestamp" in df.columns:
                    ts_col = pd.to_datetime(df["timestamp"])
                    ts_tz = ts_col.dt.tz

                    def _coerce_bound(raw_d: Any) -> pd.Timestamp:
                        b = pd.to_datetime(raw_d)
                        if ts_tz is not None:
                            return b.tz_localize("UTC").tz_convert(ts_tz) if b.tzinfo is None else b.tz_convert(ts_tz)
                        return b.tz_convert("UTC").tz_localize(None) if b.tzinfo is not None else b

                    mask = pd.Series(True, index=df.index)
                    if start_date is not None:
                        mask = mask & (ts_col >= _coerce_bound(start_date))
                    if end_date is not None:
                        mask = mask & (ts_col <= _coerce_bound(end_date))
                    df = df.loc[mask].sort_values("timestamp", ascending=True)

            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")
            df.attrs.pop("observation_timestamp_ms", None)
            df.attrs.pop("obs_bar_date", None)
            df.attrs.pop("_bar_unsettled", None)
            if matched_mtime_ns is not None:
                df.attrs["disk_mtime_ms"] = int(matched_mtime_ns // 1_000_000)
                df.attrs["_disk_mtime"] = matched_mtime_ns / 1e9
            try:
                raw_obs_ms = schema_meta.get(b"amstal_obs_ts_ms")
                raw_obs_dt = schema_meta.get(b"amstal_obs_bar_date")
                raw_obs_iso = schema_meta.get(b"amstal_obs_bar_iso")
                raw_unsettled = schema_meta.get(b"amstal_bar_unsettled")
                if raw_obs_dt and len(df) > 0 and (raw_obs_ms or raw_unsettled == b"1"):
                    last_ts_r = pd.Timestamp(df.index[-1])
                    last_dt = (
                        last_ts_r.strftime("%Y-%m-%d")
                        if hasattr(last_ts_r, "strftime")
                        else str(df.index[-1])[:10]
                    )
                    obs_dt_str = raw_obs_dt.decode("utf-8", errors="ignore")[:10]
                    is_intraday_row = bool(last_ts_r != last_ts_r.normalize())
                    obs_iso_str = raw_obs_iso.decode("utf-8", errors="ignore").strip() if raw_obs_iso else ""
                    if obs_iso_str:
                        obs_iso_ts = pd.Timestamp(obs_iso_str)
                        is_intraday_meta = bool(obs_iso_ts != obs_iso_ts.normalize())
                        if is_intraday_row or is_intraday_meta:
                            row_matches_meta = bool(last_ts_r == obs_iso_ts)
                        else:
                            row_matches_meta = bool(last_dt == obs_dt_str)
                    else:
                        row_matches_meta = (not is_intraday_row) and bool(last_dt == obs_dt_str)
                    if row_matches_meta:
                        df.attrs["obs_bar_date"] = obs_dt_str
                        if raw_obs_ms:
                            df.attrs["observation_timestamp_ms"] = int(raw_obs_ms.decode("utf-8"))
                        if raw_unsettled == b"1":
                            df.attrs["_bar_unsettled"] = True
            except Exception:
                pass
            return df
        finally:
            con.close()

    def load_ticker_daily(
        self, ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Loads daily OHLCV for ticker from local Parquet store via stateless DuckDB."""
        target = self.daily_dir / f"{ticker}.parquet"
        if not target.exists():
            return None
        return self.query_parquet_duckdb(target, start_date=start_date, end_date=end_date)

    def save_ticker_daily(self, ticker: str, df: pd.DataFrame) -> Path:
        """Validates and atomically writes daily OHLCV Parquet partition for ticker."""
        df_copy = df.copy()
        if not isinstance(df_copy.index, pd.DatetimeIndex) and "timestamp" in df_copy.columns:
            df_copy.index = pd.to_datetime(df_copy["timestamp"])
        if "timestamp" in df_copy.columns:
            df_copy = df_copy.drop(columns=["timestamp"])
        df_copy.index.name = "timestamp"
        self.validate_market_data(df_copy)
        target = self.daily_dir / f"{ticker}.parquet"
        self.atomic_write_parquet(df_copy, target)
        return target

    @staticmethod
    def _ticker_seed(ticker: str) -> int:
        """Process-independent deterministic 31-bit seed derived from SHA-256 of ticker."""
        digest = hashlib.sha256(ticker.upper().encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % (2**31)

    @staticmethod
    def _latest_expected_trading_date() -> pd.Timestamp:
        """Returns the most recent completed or active NYSE trading date in America/New_York."""
        now_et = pd.Timestamp.now(tz="America/New_York")
        cand_d = now_et.date()
        # Before US market open (09:30 ET) or on a closed market day, step back to prior session
        if not MarketDataLoader.is_nyse_trading_day(cand_d) or now_et.time() < dtime(9, 30):
            cand_d -= timedelta(days=1)
        while not MarketDataLoader.is_nyse_trading_day(cand_d):
            cand_d -= timedelta(days=1)
        return pd.Timestamp(cand_d)

    @staticmethod
    def _normalize_yf_df(df_yf: pd.DataFrame) -> pd.DataFrame:
        """Sanitizes yfinance OHLCV DataFrame to strict lowercase monotonic DatetimeIndex schema."""
        out = df_yf.copy()
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in out.columns]
        else:
            out.columns = [str(c).lower() for c in out.columns]

        idx = pd.to_datetime(out.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("America/New_York").tz_localize(None)
        out.index = idx.normalize() if len(out) > 0 and (idx == idx.normalize()).mean() > 0.5 else idx
        out.index.name = "timestamp"

        out = out.dropna(subset=["open", "high", "low", "close"])
        out = out[(out["open"] > 0) & (out["high"] > 0) & (out["low"] > 0) & (out["close"] > 0)]
        if "volume" not in out.columns:
            out["volume"] = 0
        out["volume"] = out["volume"].fillna(0).clip(lower=0)
        # Enforce strict OHLC envelope bounds against upstream adjusted-close rounding artifacts
        out["high"] = np.maximum(out["high"], np.maximum(out["open"], out["close"]))
        out["low"] = np.minimum(out["low"], np.minimum(out["open"], out["close"]))
        out["low"] = np.maximum(out["low"], 1e-6)
        if "vwap" not in out.columns:
            out["vwap"] = (out["high"] + out["low"] + out["close"]) / 3.0
        out = out[~out.index.duplicated(keep="last")].sort_index()
        return out

    def generate_synthetic_daily(
        self,
        ticker: str,
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
        base_price: float = 100.0,
        drift: float = 0.0004,
        vol: float = 0.015,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generates deterministic geometric Brownian motion daily OHLCV + VWAP data on US business days.
        Used for offline unit tests, Gate 1 falsification tests, and CI/CD without network access.
        """
        eff_end = end_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rng = np.random.default_rng(seed if seed is not None else self._ticker_seed(ticker))
        dates = pd.bdate_range(start=start_date, end=eff_end)
        n = len(dates)
        if n == 0:
            raise ValueError(f"No business days between {start_date} and {end_date}.")

        # Log returns
        daily_returns = rng.normal(loc=drift, scale=vol, size=n)
        price_path = base_price * np.exp(np.cumsum(daily_returns))

        # Generate realistic intraday high, low, open, close, volume, vwap
        intraday_noise = rng.exponential(scale=vol * 0.7, size=(n, 2))
        open_prices = np.roll(price_path, 1)
        open_prices[0] = base_price

        # Intraday drift between open and close
        close_prices = price_path
        high_prices = np.maximum(open_prices, close_prices) + intraday_noise[:, 0] * price_path
        low_prices = np.minimum(open_prices, close_prices) - intraday_noise[:, 1] * price_path
        low_prices = np.maximum(low_prices, 0.01)  # Bound above zero

        # Volume lognormally distributed
        volumes = rng.lognormal(mean=14.5, sigma=0.5, size=n).astype(int)

        # VWAP approximation between low and high
        typical_price = (high_prices + low_prices + close_prices) / 3.0
        vwap = typical_price + rng.normal(0, vol * 0.05, size=n)
        vwap = np.clip(vwap, low_prices, high_prices)

        df = pd.DataFrame(
            {
                "open": np.round(open_prices, 4),
                "high": np.round(high_prices, 4),
                "low": np.round(low_prices, 4),
                "close": np.round(close_prices, 4),
                "volume": volumes,
                "vwap": np.round(vwap, 4),
            },
            index=dates,
        )
        df.index.name = "timestamp"
        return df

    @staticmethod
    def _mark_download_settlement(
        df: pd.DataFrame,
        snap_before_ms: int,
        *,
        preserve_obs_ms: Optional[int] = None,
    ) -> None:
        """Tags unsettled downloaded bars separately from verified observation_timestamp_ms."""
        if df is None or df.empty:
            return
        last_ts = pd.Timestamp(df.index[-1])
        last_bar_d = last_ts.date()
        last_str = last_ts.strftime("%Y-%m-%d")
        close_t = MarketDataLoader.nyse_session_close_time(last_bar_d)
        close_ms = int(
            datetime.combine(last_bar_d, close_t, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1000
        )
        settle_ms = close_ms + MarketDataLoader._POST_CLOSE_SETTLEMENT_MS
        if snap_before_ms < settle_ms:
            if preserve_obs_ms is not None:
                df.attrs["observation_timestamp_ms"] = int(preserve_obs_ms)
            else:
                df.attrs.pop("observation_timestamp_ms", None)
            df.attrs["_bar_unsettled"] = True
            df.attrs["obs_bar_date"] = last_str
        else:
            df.attrs.pop("observation_timestamp_ms", None)
            df.attrs.pop("_bar_unsettled", None)
            df.attrs.pop("obs_bar_date", None)

    def fetch_or_load_daily(
        self,
        ticker: str,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        fallback_synthetic: bool = False,
        provider: Optional[str] = None,
        refresh_if_stale: bool = False,
    ) -> pd.DataFrame:
        """
        Retrieves daily data from local Parquet cache if available (optionally catching up stale dates).
        If missing, attempts fetch via provider (e.g. yfinance); falls back to synthetic data if requested.
        """
        eff_end = end_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        loaded = self.load_ticker_daily(ticker, start_date=start_date, end_date=eff_end)
        if loaded is not None and not loaded.empty:
            if refresh_if_stale and provider != "synthetic":
                refreshed = self.refresh_ticker_daily(ticker, target_end_date=eff_end)
                if refreshed is not None and not refreshed.empty:
                    return refreshed.loc[start_date:eff_end]
            return loaded

        fetched = None
        if provider == "synthetic":
            fetched = self.generate_synthetic_daily(ticker, start_date=start_date, end_date=eff_end)
        else:
            try:
                import yfinance as yf

                snap_before_ms = int(time.time() * 1000)
                yf_end = (pd.Timestamp(eff_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                df_yf = yf.download(
                    ticker, start=start_date, end=yf_end, progress=False, auto_adjust=True, timeout=5
                )
                if df_yf is not None and not df_yf.empty:
                    df_yf = self._normalize_yf_df(df_yf)
                    if not df_yf.empty:
                        self.validate_market_data(df_yf)
                        self._mark_download_settlement(df_yf, snap_before_ms)
                        fetched = df_yf
            except Exception:
                fetched = None

            if fetched is None and fallback_synthetic:
                fetched = self.generate_synthetic_daily(ticker, start_date=start_date, end_date=eff_end)
            elif fetched is None:
                raise RuntimeError(f"Unable to retrieve market data for '{ticker}' and fallback_synthetic is False.")

        self.save_ticker_daily(ticker, fetched)
        return fetched

    def refresh_ticker_daily(
        self,
        ticker: str,
        target_end_date: Optional[str] = None,
        timeout: float = 5.0,
        force_refresh_if_incomplete: bool = False,
    ) -> Optional[pd.DataFrame]:
        """Incrementally catches up a stale or incomplete Parquet daily partition via yfinance and atomically saves."""
        sym = ticker.upper()
        with self._get_refresh_lock(sym):
            return self._refresh_ticker_daily_locked(
                sym,
                target_end_date=target_end_date,
                timeout=timeout,
                force_refresh_if_incomplete=force_refresh_if_incomplete,
            )

    def _refresh_ticker_daily_locked(
        self,
        sym: str,
        target_end_date: Optional[str] = None,
        timeout: float = 5.0,
        force_refresh_if_incomplete: bool = False,
    ) -> Optional[pd.DataFrame]:
        target_path = self.daily_dir / f"{sym}.parquet"

        def _with_refresh_meta(
            frame: Optional[pd.DataFrame],
            *,
            wrote: bool,
            replaced: bool,
            gen: Optional[Tuple[int, int, int, int]],
        ) -> Optional[pd.DataFrame]:
            if frame is None:
                return None
            out = frame.copy()
            out.attrs["_refresh_wrote"] = bool(wrote)
            out.attrs["_target_row_replaced"] = bool(replaced)
            out.attrs["_refresh_gen"] = gen
            return out

        loaded_stat_gen = _file_stat_gen(target_path)
        loaded = self.load_ticker_daily(sym)
        expected_bday = self._latest_expected_trading_date()
        eff_end = pd.Timestamp(target_end_date) if target_end_date else expected_bday

        loaded_mtime_ms: Optional[int] = None
        prev_last_str: Optional[str] = None
        if loaded is not None and not loaded.empty:
            raw_loaded_mtime = (getattr(loaded, "attrs", {}) or {}).get("disk_mtime_ms")
            if isinstance(raw_loaded_mtime, (int, float)) and not isinstance(raw_loaded_mtime, bool):
                loaded_mtime_ms = int(raw_loaded_mtime)
            last_dt = pd.Timestamp(loaded.index.max()).normalize()
            prev_last_str = last_dt.strftime("%Y-%m-%d")
            if last_dt >= min(expected_bday, eff_end) and not force_refresh_if_incomplete:
                return _with_refresh_meta(loaded, wrote=False, replaced=False, gen=loaded_stat_gen)
            fetch_start = (last_dt - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        else:
            fetch_start = "2020-01-01"

        try:
            import yfinance as yf

            snap_before_ms = int(time.time() * 1000)
            yf_end = (eff_end + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
            df_new = yf.download(
                sym, start=fetch_start, end=yf_end, progress=False, auto_adjust=True, timeout=timeout
            )
            if df_new is None or df_new.empty:
                return _with_refresh_meta(loaded, wrote=False, replaced=False, gen=loaded_stat_gen)
            df_new = self._normalize_yf_df(df_new)
            if df_new.empty:
                return _with_refresh_meta(loaded, wrote=False, replaced=False, gen=loaded_stat_gen)

            dl_last_str = pd.Timestamp(df_new.index.max()).strftime("%Y-%m-%d")
            df_new_norm = df_new.copy()
            df_new_norm.index = pd.DatetimeIndex(df_new_norm.index).normalize()
            new_dates = {pd.Timestamp(ts).strftime("%Y-%m-%d") for ts in df_new_norm.index}
            loaded_norm_pre: Optional[pd.DataFrame] = None
            if loaded is not None and not loaded.empty:
                loaded_norm_pre = loaded.copy()
                loaded_norm_pre.index = pd.DatetimeIndex(loaded_norm_pre.index).normalize()

            with _partition_commit_lock(target_path):
                for _attempt in range(4):
                    gen_before = _file_stat_gen(target_path)
                    disk_now = self.load_ticker_daily(sym)
                    gen_after_read = _file_stat_gen(target_path)
                    if gen_after_read != gen_before:
                        continue

                    conflict_dates: set[str] = set()
                    if disk_now is not None and not disk_now.empty:
                        d_attrs = getattr(disk_now, "attrs", {}) or {}
                        d_mtime = d_attrs.get("disk_mtime_ms")
                        d_obs = d_attrs.get("observation_timestamp_ms")
                        d_uns = bool(d_attrs.get("_bar_unsettled"))
                        d_last_str = pd.Timestamp(disk_now.index.max()).strftime("%Y-%m-%d")
                        p_attrs = (getattr(loaded, "attrs", {}) or {}) if loaded is not None else {}
                        disk_changed = bool(
                            loaded is None
                            or loaded.empty
                            or d_mtime != loaded_mtime_ms
                            or d_last_str != prev_last_str
                            or d_obs != p_attrs.get("observation_timestamp_ms")
                            or d_uns != bool(p_attrs.get("_bar_unsettled"))
                        )
                        if disk_changed and (
                            d_last_str > dl_last_str or (d_last_str == dl_last_str and not d_uns)
                        ):
                            return _with_refresh_meta(
                                disk_now, wrote=False, replaced=False, gen=gen_after_read
                            )

                        base_df: Optional[pd.DataFrame] = disk_now
                        base_mtime_ms: Optional[int] = (
                            int(d_mtime)
                            if isinstance(d_mtime, (int, float)) and not isinstance(d_mtime, bool)
                            else None
                        )
                        if disk_changed:
                            disk_norm_cmp = disk_now.copy()
                            disk_norm_cmp.index = pd.DatetimeIndex(disk_norm_cmp.index).normalize()
                            if d_last_str in new_dates and (
                                loaded_norm_pre is None
                                or d_last_str != prev_last_str
                                or d_obs != p_attrs.get("observation_timestamp_ms")
                                or d_uns != bool(p_attrs.get("_bar_unsettled"))
                            ):
                                conflict_dates.add(d_last_str)
                            for idx_ts in disk_norm_cmp.index:
                                ds = pd.Timestamp(idx_ts).strftime("%Y-%m-%d")
                                if ds not in new_dates:
                                    continue
                                if loaded_norm_pre is None or idx_ts not in loaded_norm_pre.index:
                                    conflict_dates.add(ds)
                                    continue
                                r_l = loaded_norm_pre.loc[idx_ts]
                                r_d = disk_norm_cmp.loc[idx_ts]
                                if not all(
                                    math.isclose(float(r_l[c]), float(r_d[c]), rel_tol=1e-9, abs_tol=1e-9)
                                    for c in ("open", "high", "low", "close", "volume", "vwap")
                                    if c in loaded_norm_pre.columns and c in disk_norm_cmp.columns
                                ):
                                    conflict_dates.add(ds)
                    else:
                        base_df = None
                        base_mtime_ms = None

                    if conflict_dates:
                        keep_mask = [
                            pd.Timestamp(ts).strftime("%Y-%m-%d") not in conflict_dates
                            for ts in df_new_norm.index
                        ]
                        df_new_merge = df_new_norm.iloc[keep_mask]
                    else:
                        df_new_merge = df_new_norm

                    if df_new_merge.empty:
                        return _with_refresh_meta(
                            base_df if base_df is not None else loaded,
                            wrote=False,
                            replaced=False,
                            gen=gen_after_read,
                        )

                    merge_dates = {pd.Timestamp(ts).strftime("%Y-%m-%d") for ts in df_new_merge.index}
                    base_last_str: Optional[str] = None
                    if base_df is not None and not base_df.empty:
                        base_last_str = pd.Timestamp(base_df.index.max()).strftime("%Y-%m-%d")
                        base_norm = base_df.copy()
                        base_norm.index = pd.DatetimeIndex(base_norm.index).normalize()
                        keep_cols = [
                            c
                            for c in ["open", "high", "low", "close", "volume", "vwap"]
                            if c in base_norm.columns and c in df_new_merge.columns
                        ]
                        combined = pd.concat([base_norm[keep_cols], df_new_merge[keep_cols]])
                        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                        combined_last_str = pd.Timestamp(combined.index.max()).strftime("%Y-%m-%d")
                        target_replaced = combined_last_str in merge_dates
                        if not target_replaced and combined_last_str == base_last_str and base_norm.equals(combined):
                            return _with_refresh_meta(
                                base_df, wrote=False, replaced=False, gen=gen_after_read
                            )
                    else:
                        combined = df_new_merge.copy()
                        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                        combined_last_str = pd.Timestamp(combined.index.max()).strftime("%Y-%m-%d")
                        target_replaced = combined_last_str in merge_dates

                    self.validate_market_data(combined)
                    last_bar_d = pd.Timestamp(combined.index[-1]).date()
                    close_t = self.nyse_session_close_time(last_bar_d)
                    close_ms = int(
                        datetime.combine(last_bar_d, close_t, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1000
                    )
                    settle_ms = close_ms + self._POST_CLOSE_SETTLEMENT_MS

                    prev_attrs = (getattr(base_df, "attrs", {}) or {}) if base_df is not None else {}
                    prev_obs = prev_attrs.get("observation_timestamp_ms")
                    prev_unsettled = bool(prev_attrs.get("_bar_unsettled"))
                    cand_prev_ms = (
                        prev_obs
                        if isinstance(prev_obs, (int, float)) and not isinstance(prev_obs, bool)
                        else base_mtime_ms
                    )
                    prev_threshold_ms = (
                        close_ms
                        if isinstance(prev_obs, (int, float)) and not isinstance(prev_obs, bool)
                        else settle_ms
                    )
                    prev_is_partial = bool(
                        base_df is not None
                        and not base_df.empty
                        and (
                            prev_unsettled
                            or (
                                isinstance(cand_prev_ms, (int, float))
                                and not isinstance(cand_prev_ms, bool)
                                and int(cand_prev_ms) < prev_threshold_ms
                            )
                        )
                    )
                    row_unchanged = False
                    if (
                        base_df is not None
                        and not base_df.empty
                        and target_replaced
                        and base_last_str == combined_last_str
                    ):
                        new_rows = df_new_merge[df_new_merge.index == pd.Timestamp(combined_last_str)]
                        if not new_rows.empty:
                            r_old = base_df.iloc[-1]
                            r_new = new_rows.iloc[-1]
                            row_unchanged = all(
                                math.isclose(float(r_old[c]), float(r_new[c]), rel_tol=1e-9, abs_tol=1e-9)
                                for c in ("open", "high", "low", "close", "volume", "vwap")
                                if c in base_df.columns and c in df_new_merge.columns
                            )

                    if target_replaced and snap_before_ms < settle_ms:
                        # Never fabricate a pre-close or request-start observation timestamp for yf.download bars;
                        # preserve verified observation time only when an unchanged partial row stays identical.
                        preserved_obs = (
                            int(prev_obs)
                            if isinstance(prev_obs, (int, float)) and not isinstance(prev_obs, bool)
                            else None
                        )
                        if prev_is_partial and row_unchanged:
                            target_replaced = False
                            if base_df is not None and base_df.equals(combined):
                                out_unchanged = base_df.copy()
                                self._mark_download_settlement(
                                    out_unchanged, snap_before_ms, preserve_obs_ms=preserved_obs
                                )
                                out_unchanged.attrs["_refresh_wrote"] = False
                                out_unchanged.attrs["_target_row_replaced"] = False
                                out_unchanged.attrs["_refresh_gen"] = gen_after_read
                                return out_unchanged
                            self._mark_download_settlement(combined, snap_before_ms, preserve_obs_ms=preserved_obs)
                        else:
                            self._mark_download_settlement(combined, snap_before_ms)
                    elif target_replaced:
                        self._mark_download_settlement(combined, snap_before_ms)
                    elif base_df is not None and not base_df.empty and combined_last_str == base_last_str:
                        if isinstance(prev_obs, (int, float)) and not isinstance(prev_obs, bool):
                            combined.attrs["observation_timestamp_ms"] = int(prev_obs)
                            combined.attrs["obs_bar_date"] = combined_last_str
                        else:
                            combined.attrs.pop("observation_timestamp_ms", None)
                        if prev_is_partial:
                            combined.attrs["_bar_unsettled"] = True
                            combined.attrs["obs_bar_date"] = combined_last_str
                        else:
                            combined.attrs.pop("_bar_unsettled", None)

                    if _file_stat_gen(target_path) != gen_after_read:
                        continue
                    self.save_ticker_daily(sym, combined)
                    final_gen = _file_stat_gen(target_path)
                    return _with_refresh_meta(
                        combined, wrote=True, replaced=bool(target_replaced), gen=final_gen
                    )
                cur_disk = self.load_ticker_daily(sym)
                return _with_refresh_meta(
                    cur_disk, wrote=False, replaced=False, gen=_file_stat_gen(target_path)
                )
        except Exception:
            return _with_refresh_meta(loaded, wrote=False, replaced=False, gen=loaded_stat_gen)

    def fetch_live_quote(self, ticker: str, timeout: float = 4.0) -> Optional[Dict[str, Any]]:
        """Fetches real-time/latest trading-session quote snapshot via yfinance."""
        sym = ticker.upper()
        try:
            import yfinance as yf

            tk = yf.Ticker(sym)
            req_started_ms = int(time.time() * 1000)
            price = None
            open_p = None
            high_p = None
            low_p = None
            prev_c = None
            volume = 0

            fi_md: Optional[Dict[str, Any]] = None
            fi_price_date: Optional[str] = None
            fi_ohlcv_date: Optional[str] = None
            fi_last_close_nan = False
            fi_prices_empty = False
            used_fi_ohlcv = False
            try:
                fi = tk.fast_info
                price = fi.get("lastPrice") or fi.get("regularMarketPrice")
                open_p = fi.get("open") or fi.get("regularMarketOpen")
                high_p = fi.get("dayHigh") or fi.get("regularMarketDayHigh")
                low_p = fi.get("dayLow") or fi.get("regularMarketDayLow")
                vol_val = fi.get("lastVolume") or fi.get("regularMarketVolume")
                if vol_val:
                    volume = int(vol_val)
                for fv in (open_p, high_p, low_p):
                    if fv is not None and not isinstance(fv, bool):
                        try:
                            if math.isfinite(float(fv)) and float(fv) > 0.0:
                                used_fi_ohlcv = True
                                break
                        except (TypeError, ValueError):
                            pass
                if volume > 0:
                    used_fi_ohlcv = True
                raw_fi_md = getattr(fi, "_md", None)
                if isinstance(raw_fi_md, Mapping):
                    # Snapshot before previousClose / tk.history(period="5d") mutates yfinance's shared _history_metadata
                    fi_md = {
                        k: raw_fi_md.get(k)
                        for k in ("regularMarketTime", "regularMarketPrice", "regularMarketVolume")
                        if k in raw_fi_md
                    }
                raw_fi_prices = getattr(fi, "_prices_1y", None)
                prev_c = fi.get("previousClose") or fi.get("regularMarketPreviousClose")
                if raw_fi_prices is not None and not getattr(raw_fi_prices, "empty", True):
                    fi_last = pd.Timestamp(raw_fi_prices.index[-1])
                    if fi_last.tzinfo is not None:
                        fi_last = fi_last.tz_convert("America/New_York")
                    fi_ohlcv_date = fi_last.strftime("%Y-%m-%d")
                    last_close_raw = raw_fi_prices["Close"].iloc[-1] if "Close" in raw_fi_prices.columns else None
                    last_close_valid = False
                    if last_close_raw is not None:
                        try:
                            lc_f = float(last_close_raw)
                            last_close_valid = math.isfinite(lc_f) and lc_f > 0.0
                        except (TypeError, ValueError):
                            last_close_valid = False
                    if last_close_valid:
                        fi_price_date = fi_ohlcv_date
                    else:
                        # FastInfo.last_price falls back to md["regularMarketPrice"] when latest row Close is NaN
                        fi_last_close_nan = True
                elif raw_fi_prices is not None:
                    fi_prices_empty = True
            except Exception as e:
                logger.debug(f"Yahoo fast_info fallback triggered for {sym}: {e}")

            def _extract_rmt(md_obj: Optional[Any]) -> Tuple[Optional[int], Optional[str]]:
                try:
                    raw_rmt = md_obj.get("regularMarketTime") if isinstance(md_obj, Mapping) else None
                    cand_ms: Optional[int] = None
                    if raw_rmt is not None:
                        if hasattr(raw_rmt, "timestamp"):
                            cand_ms = int(raw_rmt.timestamp() * 1000)
                        elif isinstance(raw_rmt, (int, float)) and not isinstance(raw_rmt, bool):
                            f_rmt = float(raw_rmt)
                            if math.isfinite(f_rmt) and f_rmt > 0.0:
                                cand_ms = int(f_rmt if f_rmt >= 1e12 else f_rmt * 1000.0)
                        elif isinstance(raw_rmt, str) and raw_rmt.strip():
                            ts_rmt = pd.Timestamp(raw_rmt.strip())
                            if ts_rmt.tzinfo is None:
                                ts_rmt = ts_rmt.tz_localize("UTC")
                            cand_ms = int(ts_rmt.timestamp() * 1000)
                    if cand_ms is not None:
                        ny_d = (
                            datetime.fromtimestamp(cand_ms / 1000.0, tz=timezone.utc)
                            .astimezone(ZoneInfo("America/New_York"))
                            .strftime("%Y-%m-%d")
                        )
                        return cand_ms, ny_d
                except (OverflowError, OSError, ValueError, Exception):
                    pass
                return None, None

            def _md_matches_price(md_obj: Optional[Any], val: float, require_present: bool) -> bool:
                raw_rmp = md_obj.get("regularMarketPrice") if isinstance(md_obj, Mapping) else None
                if raw_rmp is None or isinstance(raw_rmp, bool):
                    return not require_present
                try:
                    rmp_f = float(raw_rmp)
                except (TypeError, ValueError):
                    return False
                if not math.isfinite(rmp_f) or rmp_f <= 0.0:
                    return not require_present
                return math.isclose(float(val), rmp_f, rel_tol=1e-6, abs_tol=1e-4)

            def _md_volume(md_o: Optional[Any]) -> Optional[int]:
                rv = md_o.get("regularMarketVolume") if isinstance(md_o, Mapping) else None
                if rv is None or isinstance(rv, bool):
                    return None
                try:
                    fv = float(rv)
                    return int(round(fv)) if math.isfinite(fv) and fv >= 0.0 else None
                except (TypeError, ValueError):
                    return None

            used_fast_info_price = bool(
                price is not None and math.isfinite(float(price)) and float(price) > 0
            )
            fi_resolved_price_date = fi_price_date
            if fi_resolved_price_date is None and used_fast_info_price and fi_md is not None:
                _, fi_rmt_date_pre = _extract_rmt(fi_md)
                if fi_rmt_date_pre is not None and (
                    fi_last_close_nan
                    or fi_prices_empty
                    or _md_matches_price(fi_md, float(price), require_present=True)
                ):
                    fi_resolved_price_date = fi_rmt_date_pre
            hist = tk.history(period="5d", interval="1d", auto_adjust=True, timeout=timeout)
            bar_date = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
            hist_norm = None
            hist_bar_date: Optional[str] = None
            used_hist_ohlcv = False

            def _hist_prev_close(session_d: Optional[str]) -> Optional[float]:
                if hist_norm is None or hist_norm.empty:
                    return None
                if session_d is not None:
                    prior_rows = hist_norm[hist_norm.index.strftime("%Y-%m-%d") < session_d]
                    return float(prior_rows["close"].iloc[-1]) if not prior_rows.empty else None
                return float(hist_norm["close"].iloc[-2]) if len(hist_norm) >= 2 else None

            if hist is not None and not hist.empty:
                hist_norm = self._normalize_yf_df(hist)
                if not hist_norm.empty:
                    last_row = hist_norm.iloc[-1]
                    hist_bar_date = pd.Timestamp(hist_norm.index[-1]).strftime("%Y-%m-%d")
                    bar_date = hist_bar_date
                    if not price or not np.isfinite(float(price)):
                        price = float(last_row["close"])
                        open_p = float(last_row["open"])
                        high_p = float(last_row["high"])
                        low_p = float(last_row["low"])
                        volume = int(last_row["volume"])
                        cand_pc = _hist_prev_close(hist_bar_date)
                        prev_c = cand_pc if cand_pc is not None else open_p
                        used_fast_info_price = False
                        used_hist_ohlcv = True
                    fi_ohlcv_target_date = fi_resolved_price_date
                    if used_fast_info_price and (fi_ohlcv_target_date is None or hist_bar_date == fi_ohlcv_target_date):
                        if not open_p or not np.isfinite(float(open_p)):
                            open_p = float(last_row["open"])
                            used_hist_ohlcv = True
                        if not high_p or not np.isfinite(float(high_p)):
                            high_p = float(last_row["high"])
                            used_hist_ohlcv = True
                        if not low_p or not np.isfinite(float(low_p)):
                            low_p = float(last_row["low"])
                            used_hist_ohlcv = True
                        if volume <= 0:
                            volume = int(last_row["volume"])
                            used_hist_ohlcv = True
                    earlier_session_fi = bool(
                        used_fast_info_price
                        and fi_resolved_price_date is not None
                        and fi_ohlcv_date is not None
                        and fi_resolved_price_date < fi_ohlcv_date
                    )
                    if earlier_session_fi or not prev_c or not np.isfinite(float(prev_c)):
                        prev_sess_d = fi_resolved_price_date if used_fast_info_price else hist_bar_date
                        cand_pc = _hist_prev_close(prev_sess_d)
                        if cand_pc is not None:
                            prev_c = cand_pc

            if price is None or not np.isfinite(float(price)) or float(price) <= 0:
                return None

            md_lookup_failed = False

            def _safe_history_metadata() -> Optional[Any]:
                nonlocal md_lookup_failed
                if not hasattr(tk, "get_history_metadata"):
                    return None
                try:
                    return tk.get_history_metadata()
                except Exception as md_err:
                    md_lookup_failed = True
                    logger.debug(f"Yahoo history metadata lookup failed for {sym}: {md_err}")
                    return None

            md_used = fi_md if used_fast_info_price else _safe_history_metadata()
            obs_ts_ms, rmt_ny_date = _extract_rmt(md_used)

            if used_fast_info_price and fi_last_close_nan and rmt_ny_date is None:
                if hist_norm is not None and not hist_norm.empty:
                    last_row = hist_norm.iloc[-1]
                    price = float(last_row["close"])
                    open_p = float(last_row["open"])
                    high_p = float(last_row["high"])
                    low_p = float(last_row["low"])
                    volume = int(last_row["volume"])
                    bar_date = pd.Timestamp(hist_norm.index[-1]).strftime("%Y-%m-%d")
                    cand_pc = _hist_prev_close(bar_date)
                    prev_c = cand_pc if cand_pc is not None else open_p
                    used_fast_info_price = False
                    md_used = _safe_history_metadata()
                    obs_ts_ms, rmt_ny_date = _extract_rmt(md_used)
                else:
                    return None

            if used_fast_info_price:
                # yfinance 1.8.0 FastInfo.last_price uses _prices_1y["Close"] (daily bar close) unless empty/NaN,
                # whereas md["regularMarketTime"] timestamps md["regularMarketPrice"] — retain obs_ts_ms only when paired.
                price_dates_rmt = bool(
                    fi_last_close_nan
                    or fi_prices_empty
                    or _md_matches_price(md_used, float(price), require_present=True)
                )
                if fi_price_date is not None:
                    bar_date = fi_price_date
                    if (rmt_ny_date is not None and rmt_ny_date != fi_price_date) or not price_dates_rmt:
                        obs_ts_ms = None
                else:
                    if not price_dates_rmt:
                        obs_ts_ms = None
                    elif rmt_ny_date is not None:
                        bar_date = rmt_ny_date
                fi_rmv_chk = _md_volume(md_used)
                if fi_rmv_chk is not None and volume != fi_rmv_chk:
                    obs_ts_ms = None
                if used_fi_ohlcv and fi_ohlcv_date is not None and bar_date != fi_ohlcv_date:
                    # FastInfo OHLCV accessors read _prices_1y[-1] (e.g., today with Close=NaN) while fallback price came from an earlier session
                    obs_ts_ms = None
                    used_fi_ohlcv = False
                    if hist_norm is not None and not hist_norm.empty and hist_bar_date == bar_date:
                        last_row = hist_norm.iloc[-1]
                        open_p = float(last_row["open"])
                        high_p = float(last_row["high"])
                        low_p = float(last_row["low"])
                        volume = int(last_row["volume"])
                        cand_pc = _hist_prev_close(bar_date)
                        prev_c = cand_pc if cand_pc is not None else open_p
                        used_hist_ohlcv = True
                    else:
                        open_p = None
                        high_p = None
                        low_p = None
                        prev_c = None
                        volume = 0
                if obs_ts_ms is not None and used_hist_ohlcv:
                    # Do not apply an older FastInfo regularMarketTime to OHLCV filled from a separate history request
                    if hist_norm is None or hist_norm.empty or hist_bar_date != bar_date:
                        obs_ts_ms = None
                    else:
                        hist_md = _safe_history_metadata()
                        hist_obs_ms, hist_rmt_d = _extract_rmt(hist_md)
                        fi_rmv = _md_volume(fi_md)
                        hist_rmv = _md_volume(hist_md)
                        vol_mismatch = bool(
                            (fi_rmv is not None and hist_rmv is not None and fi_rmv != hist_rmv)
                            or (fi_rmv is not None and volume != fi_rmv)
                            or (hist_rmv is not None and volume != hist_rmv)
                        )
                        if (
                            hist_obs_ms != obs_ts_ms
                            or hist_rmt_d != bar_date
                            or vol_mismatch
                            or not math.isclose(float(price), float(hist_norm.iloc[-1]["close"]), rel_tol=1e-6, abs_tol=1e-4)
                        ):
                            obs_ts_ms = None
            else:
                # Daily bar close fallback: distinguish missing or later-session metadata (which still permits
                # prior-session scheduled-close inference) from conflicting or unverified same-session metadata.
                hist_rmv_fb = _md_volume(md_used)
                same_or_missing_session = rmt_ny_date is None or rmt_ny_date == bar_date
                same_session_unverified = bool(
                    rmt_ny_date == bar_date
                    and not _md_matches_price(md_used, float(price), require_present=True)
                )
                metadata_conflict = bool(
                    same_or_missing_session
                    and (
                        not _md_matches_price(md_used, float(price), require_present=False)
                        or (hist_rmv_fb is not None and volume != hist_rmv_fb)
                    )
                )
                if metadata_conflict or rmt_ny_date != bar_date or same_session_unverified:
                    obs_ts_ms = None
                if (
                    obs_ts_ms is None
                    and not metadata_conflict
                    and not same_session_unverified
                    and not md_lookup_failed
                    and hist_norm is not None
                    and not hist_norm.empty
                ):
                    try:
                        bar_d = pd.Timestamp(bar_date).date()
                        req_ny_d = (
                            datetime.fromtimestamp(req_started_ms / 1000.0, tz=timezone.utc)
                            .astimezone(ZoneInfo("America/New_York"))
                            .date()
                        )
                        close_dt = datetime.combine(
                            bar_d,
                            self.nyse_session_close_time(bar_d),
                            tzinfo=ZoneInfo("America/New_York"),
                        )
                        close_ms = int(close_dt.timestamp() * 1000)
                        if bar_d < req_ny_d and req_started_ms >= close_ms:
                            obs_ts_ms = close_ms
                    except Exception:
                        obs_ts_ms = None

            p_f = float(price)
            o_f = float(open_p) if open_p and float(open_p) > 0 else p_f
            h_f = max(float(high_p) if high_p and float(high_p) > 0 else p_f, o_f, p_f)
            l_f = min(float(low_p) if low_p and float(low_p) > 0 else p_f, o_f, p_f)
            prev_f = float(prev_c) if prev_c and float(prev_c) > 0 else o_f
            chg = round(p_f - prev_f, 2)
            chg_pct = round(((p_f - prev_f) / prev_f) * 100.0, 2) if prev_f > 0 else 0.0

            now_ms = int(time.time() * 1000)
            future_obs_rejected = False
            if obs_ts_ms is not None and obs_ts_ms > now_ms:
                obs_ts_ms = None
                future_obs_rejected = True

            quote_out: Dict[str, Any] = {
                "ticker": sym,
                "price": round(p_f, 2),
                "open": round(o_f, 2),
                "high": round(h_f, 2),
                "low": round(l_f, 2),
                "prev_close": round(prev_f, 2),
                "change": chg,
                "change_pct": chg_pct,
                "volume": int(volume),
                "date": bar_date,
                "timestamp_ms": now_ms,
            }
            if future_obs_rejected:
                quote_out["_future_observation_rejected"] = True
            if obs_ts_ms is not None:
                quote_out["observation_timestamp_ms"] = obs_ts_ms
            return quote_out
        except Exception:
            return None

    def fetch_intraday_bars(
        self, ticker: str, interval: str = "1h", count: int = 90, timeout: float = 5.0
    ) -> List[Dict[str, Any]]:
        """Fetches recent 1H or 4H intraday bars via yfinance for ChartPanel timeframe views."""
        sym = ticker.upper()
        try:
            import yfinance as yf

            req_started_ms = int(time.time() * 1000)
            df = yf.download(
                sym, period="60d", interval="60m", progress=False, auto_adjust=True, timeout=timeout
            )
            if df is None or df.empty:
                return []

            out = df.copy()
            if isinstance(out.columns, pd.MultiIndex):
                out.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in out.columns]
            else:
                out.columns = [str(c).lower() for c in out.columns]
            out = out.dropna(subset=["open", "high", "low", "close"])
            out["volume"] = out["volume"].fillna(0).clip(lower=0)
            has_tz_idx = isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None
            if has_tz_idx:
                ny_tz = ZoneInfo("America/New_York")
                src_ends: List[int] = []
                for t in out.index:
                    ts_u = pd.Timestamp(t).tz_convert("UTC")
                    raw_end = int(ts_u.timestamp() * 1000) + 3_600_000
                    ts_ny = ts_u.tz_convert(ny_tz)
                    close_t = self.nyse_session_close_time(ts_ny.date())
                    if dtime(9, 30) <= ts_ny.time() <= close_t:
                        sess_close_ms = int(
                            datetime.combine(ts_ny.date(), close_t, tzinfo=ny_tz).timestamp() * 1000
                        )
                        raw_end = min(raw_end, sess_close_ms)
                    src_ends.append(raw_end)
                out["_src_end_ms"] = src_ends

            if interval.lower() == "4h":
                agg_spec: Dict[str, str] = {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
                if has_tz_idx:
                    agg_spec["_src_end_ms"] = "max"
                out = out.resample("4h").agg(agg_spec).dropna(subset=["open", "high", "low", "close"])

            out = out.tail(count + 1)
            records: List[Dict[str, Any]] = []
            closes = out["close"].values
            opens = out["open"].values
            highs = out["high"].values
            lows = out["low"].values
            vols = out["volume"].values
            dates = out.index
            start_idx = 1 if len(out) > count else 0

            for i in range(start_idx, len(out)):
                prev_c = closes[i - 1] if i > 0 else opens[i]
                tot_ret = float(((closes[i] - prev_c) / prev_c) * 100.0) if prev_c > 0 else 0.0
                on_ret = float(((opens[i] - prev_c) / prev_c) * 100.0) if prev_c > 0 else 0.0
                intra_ret = float(((closes[i] - opens[i]) / opens[i]) * 100.0) if opens[i] > 0 else 0.0
                ts = pd.Timestamp(dates[i])
                if ts.tz is not None:
                    ts = ts.tz_convert("UTC")
                records.append({
                    "date": int(ts.timestamp()),
                    "open": round(float(opens[i]), 2),
                    "high": round(float(highs[i]), 2),
                    "low": round(float(lows[i]), 2),
                    "close": round(float(closes[i]), 2),
                    "vwap": round(float((highs[i] + lows[i] + closes[i]) / 3.0), 2),
                    "volume": int(vols[i]),
                    "overnight_return": round(on_ret, 2),
                    "intraday_return": round(intra_ret, 2),
                    "total_return": round(tot_ret, 2),
                })

            if records and has_tz_idx and "_src_end_ms" in out.columns and len(out) > 0:
                last_src_end_ms = int(out["_src_end_ms"].iloc[-1])
                bucket_secs = 14400 if interval.lower() == "4h" else 3600
                latest_start_s = int(records[-1]["date"])
                latest_end_ms = (latest_start_s + bucket_secs) * 1000
                try:
                    bar_dt_ny = datetime.fromtimestamp(latest_start_s, tz=timezone.utc).astimezone(
                        ZoneInfo("America/New_York")
                    ).date()
                    session_close_ms = int(
                        datetime.combine(
                            bar_dt_ny,
                            self.nyse_session_close_time(bar_dt_ny),
                            tzinfo=ZoneInfo("America/New_York"),
                        ).timestamp()
                        * 1000
                    )
                    if latest_start_s * 1000 <= session_close_ms:
                        latest_end_ms = min(latest_end_ms, session_close_ms)
                except Exception:
                    pass
                # Only stamp a completed intraday bar observation time once both its session-capped bucket end
                # and last contributing source bar end precede request start; record the constituent end so a
                # bar spanning past the nominal bucket boundary (e.g. 11:30-12:30) is never backdated to 12:00,
                # nor is a partial bucket inflated past its last contributing bar.
                if latest_end_ms <= req_started_ms and last_src_end_ms <= req_started_ms:
                    records[-1]["observation_timestamp_ms"] = last_src_end_ms

            return records[::-1]
        except Exception:
            return []
