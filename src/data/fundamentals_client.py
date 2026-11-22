"""
SEC EDGAR & Yahoo Finance Fundamentals Client
Provides disk-cached, rate-limited access to official SEC EDGAR submissions,
US-GAAP XBRL CompanyFacts (Point-in-Time vs As-Restated), Yahoo Finance company metadata,
analyst consensus targets/recommendations, and insider Form 4 transactions.
Returns honest unavailable states rather than synthetic fallback numbers when offline.
"""

from collections.abc import Mapping
from datetime import datetime, timezone
import difflib
import json
import logging
import math
import os
from pathlib import Path
import re
import socket
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo

import numpy as np

from src.data.market_loader import MarketDataLoader
from src.data.sec_edgar import TokenBucketRateLimiter

logger = logging.getLogger("amstal.fundamentals")


class FundamentalsClient:
    """Retrieves and caches authentic SEC EDGAR & Yahoo Finance fundamentals."""

    DEFAULT_SEC_UA = "AmstalQuantTerminal/1.3 (research@amstal-quant.org)"
    _TICKER_MAP_TTL_SECS = 7 * 86400.0
    _SUBMISSIONS_TTL_SECS = 6 * 3600.0
    _FACTS_TTL_SECS = 24 * 3600.0
    _YF_META_TTL_SECS = 3600.0
    _REDLINE_TTL_SECS = 24 * 3600.0
    _FAILED_COOLDOWN_SECS = 180.0

    _REVENUE_CONCEPTS = (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    )
    _COST_OF_REV_CONCEPTS = (
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
        "CostOfGoodsSold",
        "CostOfServices",
    )
    _GROSS_PROFIT_CONCEPTS = ("GrossProfit",)
    _OPEX_CONCEPTS = ("OperatingExpenses",)
    _OP_INC_CONCEPTS = (
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    )
    _NET_INC_CONCEPTS = (
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    )
    _EPS_CONCEPTS = (
        "EarningsPerShareDiluted",
        "IncomeLossFromContinuingOperationsPerDilutedShare",
        "EarningsPerShareBasic",
    )
    _CASH_CONCEPTS = (
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
        "Cash",
    )
    _ASSETS_CONCEPTS = ("Assets",)
    _LT_DEBT_CONCEPTS = ("LongTermDebtNoncurrent", "LongTermDebt")
    _ST_DEBT_CONCEPTS = ("LongTermDebtCurrent", "ShortTermBorrowings", "DebtCurrent")
    _EQUITY_CONCEPTS = (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    )
    _OCF_CONCEPTS = (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    )
    _CAPEX_CONCEPTS = (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    )

    def __init__(self, lakehouse_dir: Path) -> None:
        self.raw_dir = Path(lakehouse_dir) / "raw"
        self.sec_ua = os.environ.get("SEC_EDGAR_USER_AGENT") or self.DEFAULT_SEC_UA
        self._sec_limiter = TokenBucketRateLimiter(rate=8.0, capacity=8.0)
        self._sec_tl = threading.local()
        self._sec_worker_sem = threading.BoundedSemaphore(8)
        self._lock = threading.RLock()
        self._mem_cache: Dict[str, Tuple[float, Any]] = {}
        self._mem_stat: Dict[str, Tuple[int, int, int, int]] = {}
        self._failed_write_snap: Dict[str, Tuple[float, Optional[Tuple[int, int, int, int]], Any]] = {}
        self._failed_until: Dict[str, float] = {}
        self._ticker_to_cik_map: Optional[Dict[str, str]] = None

    @staticmethod
    def _stat_signature(path: Path) -> Optional[Tuple[int, int, int, int]]:
        try:
            st = path.stat()
            return (st.st_dev, st.st_ino, st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    @staticmethod
    def _read_json_with_fd_stat(path: Path) -> Optional[Tuple[float, Tuple[int, int, int, int], Any]]:
        """Reads JSON and fstats the exact open file descriptor so data and signature share one generation."""
        try:
            with open(path, "r", encoding="utf-8") as fh:
                st = os.fstat(fh.fileno())
                raw_text = fh.read()
            sig = (st.st_dev, st.st_ino, st.st_mtime_ns, st.st_size)
            return st.st_mtime, sig, json.loads(raw_text)
        except Exception:
            return None

    def _is_in_cooldown(self, key: str) -> bool:
        return time.monotonic() < self._failed_until.get(key, 0.0)

    def _mark_failed(self, key: str, seconds: float = _FAILED_COOLDOWN_SECS) -> None:
        self._failed_until[key] = time.monotonic() + seconds

    def _read_json_cache(self, rel_path: str, max_age_secs: float) -> Optional[Any]:
        path = self.raw_dir / rel_path
        with self._lock:
            disk_sig = self._stat_signature(path)
            fw = self._failed_write_snap.get(rel_path)
            if fw is not None:
                fw_ts, fw_pre_sig, fw_data = fw
                if disk_sig == fw_pre_sig:
                    if (time.time() - fw_ts) < max_age_secs:
                        return fw_data
                    return None
                self._failed_write_snap.pop(rel_path, None)
            mem = self._mem_cache.get(rel_path)
            mem_sig = self._mem_stat.get(rel_path)
            if disk_sig is None:
                if mem is not None and mem_sig is None and (time.time() - mem[0]) < max_age_secs:
                    return mem[1]
                self._mem_cache.pop(rel_path, None)
                self._mem_stat.pop(rel_path, None)
                return None
            if mem is not None and mem_sig == disk_sig and (time.time() - mem[0]) < max_age_secs:
                return mem[1]
            loaded = self._read_json_with_fd_stat(path)
            if loaded is None:
                return None
            mtime, fd_sig, data = loaded
            if (time.time() - mtime) < max_age_secs:
                self._mem_cache[rel_path] = (mtime, data)
                self._mem_stat[rel_path] = fd_sig
                return data
        return None

    def _read_stale_json_cache_with_mtime(self, rel_path: str) -> Tuple[Optional[Any], Optional[float]]:
        """Atomically reads stale cached JSON alongside its captured mtime under the same lock."""
        path = self.raw_dir / rel_path
        with self._lock:
            disk_sig = self._stat_signature(path)
            fw = self._failed_write_snap.get(rel_path)
            if fw is not None:
                fw_ts, fw_pre_sig, fw_data = fw
                if disk_sig == fw_pre_sig:
                    return fw_data, fw_ts
                self._failed_write_snap.pop(rel_path, None)
            mem = self._mem_cache.get(rel_path)
            mem_sig = self._mem_stat.get(rel_path)
            if disk_sig is None:
                if mem is not None and mem_sig is None:
                    return mem[1], mem[0]
                self._mem_cache.pop(rel_path, None)
                self._mem_stat.pop(rel_path, None)
                return None, None
            if mem is not None and mem_sig == disk_sig:
                return mem[1], mem[0]
            loaded = self._read_json_with_fd_stat(path)
            if loaded is None:
                return None, None
            mtime, fd_sig, data = loaded
            self._mem_cache[rel_path] = (mtime, data)
            self._mem_stat[rel_path] = fd_sig
            return data, mtime

    def _read_stale_json_cache(self, rel_path: str) -> Optional[Any]:
        """Reads existing on-disk JSON regardless of TTL when network is unreachable."""
        data, _ = self._read_stale_json_cache_with_mtime(rel_path)
        return data

    def _read_dict_cache(self, rel_path: str, max_age_secs: Optional[float] = None) -> Optional[Dict[str, Any]]:
        data = (
            self._read_json_cache(rel_path, max_age_secs)
            if max_age_secs is not None
            else self._read_stale_json_cache(rel_path)
        )
        return data if isinstance(data, dict) else None

    def _read_stale_payload(self, rel_path: str) -> Optional[Dict[str, Any]]:
        """Reads stale cached payload dict with _cache_stale=True, _cache_hit=True and snapshot timestamp."""
        data, mtime = self._read_stale_json_cache_with_mtime(rel_path)
        if not isinstance(data, dict):
            return None
        out = {k: v for k, v in data.items() if k not in ("_cache_stale", "_cache_hit")}
        out["_cache_stale"] = True
        out["_cache_hit"] = True
        if not out.get("fetched_at") and mtime is not None:
            out["fetched_at"] = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return out

    @staticmethod
    def _sanitize_json_obj(obj: Any) -> Any:
        if isinstance(obj, bool) or obj is None or isinstance(obj, (str, int)):
            return obj
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        if isinstance(obj, dict):
            return {str(k): FundamentalsClient._sanitize_json_obj(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [FundamentalsClient._sanitize_json_obj(v) for v in obj]
        return obj

    def _write_json_cache(self, rel_path: str, payload: Any) -> None:
        path = self.raw_dir / rel_path
        clean_payload = (
            {k: v for k, v in payload.items() if k not in ("_cache_stale", "_cache_hit")}
            if isinstance(payload, dict)
            else payload
        )
        clean_payload = self._sanitize_json_obj(clean_payload)
        with self._lock:
            now = time.time()
            pre_sig = self._stat_signature(path)
            self._mem_cache[rel_path] = (now, clean_payload)
            self._mem_stat.pop(rel_path, None)
            tmp_name: Optional[str] = None
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=path.parent,
                    prefix=f".{path.stem}_",
                    suffix=".tmp",
                    delete=False,
                ) as tf:
                    tmp_name = tf.name
                    tf.write(json.dumps(clean_payload, allow_nan=False))
                    tf.flush()
                    st = os.fstat(tf.fileno())
                os.replace(tmp_name, path)
                tmp_name = None
                sig = (st.st_dev, st.st_ino, st.st_mtime_ns, st.st_size)
                self._failed_write_snap.pop(rel_path, None)
                self._mem_stat[rel_path] = sig
                self._mem_cache[rel_path] = (st.st_mtime, clean_payload)
            except Exception as e:
                self._failed_write_snap[rel_path] = (now, pre_sig, clean_payload)
                if tmp_name is not None:
                    try:
                        os.unlink(tmp_name)
                    except OSError:
                        pass
                logger.debug(f"Failed writing cache {path}: {e}")

    def _sec_fetch_text(self, url: str, timeout: float = 6.0) -> Optional[str]:
        self._sec_tl.last_budget_expired = False
        dl = getattr(getattr(self, "_sec_tl", None), "deadline", None)
        waited = self._sec_limiter.acquire(1.0, deadline=dl)
        if waited < 0.0:
            self._sec_tl.last_budget_expired = True
            return None
        eff_timeout = float(timeout)
        eff_dl: Optional[float] = None
        if dl is not None:
            now_m = time.monotonic()
            rem = dl - now_m
            if rem <= 0.1:
                self._sec_tl.last_budget_expired = True
                return None
            eff_timeout = min(eff_timeout, rem)
            eff_dl = min(dl, now_m + eff_timeout)
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.sec_ua,
                "Accept": "application/json, text/html, text/plain, */*",
            },
        )
        max_bytes = 32 * 1024 * 1024
        chunk_sz = 65536
        resp_holder: List[Any] = [None]
        timed_out_evt = threading.Event()
        budget_hit = [False]

        def _perform() -> Optional[str]:
            try:
                with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
                    resp_holder[0] = resp
                    expected_len: Optional[int] = None
                    try:
                        hdrs = getattr(resp, "headers", None)
                        raw_cl = hdrs.get("Content-Length") if hdrs is not None else None
                        raw_ce = hdrs.get("Content-Encoding") if hdrs is not None else None
                        if raw_cl is not None and not raw_ce:
                            cl_i = int(str(raw_cl).strip())
                            if cl_i >= 0:
                                expected_len = cl_i
                    except (TypeError, ValueError, Exception):
                        expected_len = None
                    read_fn = getattr(resp, "read1", None) or resp.read
                    chunks: List[bytes] = []
                    total = 0
                    while total <= max_bytes:
                        if timed_out_evt.is_set():
                            if dl is not None and time.monotonic() >= dl - 0.05:
                                budget_hit[0] = True
                            return None
                        if eff_dl is not None:
                            now_r = time.monotonic()
                            rem_read = eff_dl - now_r
                            if rem_read <= 0.02:
                                if dl is not None and (now_r >= dl - 0.05 or eff_dl >= dl - 1e-6):
                                    budget_hit[0] = True
                                return None
                            raw_sock = getattr(getattr(getattr(resp, "fp", None), "raw", None), "_sock", None)
                            if raw_sock is not None and hasattr(raw_sock, "settimeout"):
                                try:
                                    raw_sock.settimeout(min(eff_timeout, rem_read))
                                except Exception:
                                    pass
                        chunk = read_fn(min(chunk_sz, (max_bytes + 1) - total))
                        if not chunk:
                            break
                        if timed_out_evt.is_set() or (eff_dl is not None and time.monotonic() > eff_dl):
                            if dl is not None and (time.monotonic() >= dl - 0.05 or eff_dl >= dl - 1e-6):
                                budget_hit[0] = True
                            return None
                        chunks.append(chunk)
                        total += len(chunk)
                    if expected_len is not None and total < expected_len:
                        logger.warning(
                            f"SEC response truncated ({total} < {expected_len} bytes) for {url}"
                        )
                        return None
                    if total > max_bytes:
                        logger.warning(f"SEC response exceeded {max_bytes} bytes limit ({url}); rejecting truncated body")
                        return None
                    return b"".join(chunks).decode("utf-8", errors="ignore")
            except urllib.error.HTTPError as e:
                if timed_out_evt.is_set():
                    budget_hit[0] = True
                logger.debug(f"SEC HTTP fetch failed ({url}): {e}")
                return None
            except Exception as e:
                is_timeout_exc = (
                    isinstance(e, (TimeoutError, socket.timeout))
                    or (
                        isinstance(e, urllib.error.URLError)
                        and isinstance(getattr(e, "reason", None), (TimeoutError, socket.timeout))
                    )
                    or ("timed out" in str(e).lower())
                )
                if (timed_out_evt.is_set() and dl is not None and time.monotonic() >= dl - 0.05) or (
                    dl is not None
                    and (
                        time.monotonic() >= dl - 0.05
                        or (is_timeout_exc and eff_timeout < float(timeout))
                    )
                ):
                    budget_hit[0] = True
                logger.debug(f"SEC fetch failed ({url}): {e}")
                return None

        if eff_dl is None:
            res_text = _perform()
            if budget_hit[0]:
                self._sec_tl.last_budget_expired = True
            return res_text

        rem_sem = eff_dl - time.monotonic()
        if rem_sem <= 0.01 or not self._sec_worker_sem.acquire(timeout=rem_sem):
            if dl is not None:
                self._sec_tl.last_budget_expired = True
            return None

        out_box: List[Optional[str]] = [None]
        done_evt = threading.Event()

        def _worker() -> None:
            try:
                out_box[0] = _perform()
            finally:
                try:
                    self._sec_worker_sem.release()
                except ValueError:
                    pass
                done_evt.set()

        t = threading.Thread(target=_worker, daemon=True)
        try:
            t.start()
        except Exception:
            try:
                self._sec_worker_sem.release()
            except ValueError:
                pass
            return None
        rem_total = eff_dl - time.monotonic()
        if not done_evt.wait(timeout=max(0.01, rem_total)):
            timed_out_evt.set()
            if dl is not None and (time.monotonic() >= dl - 0.05 or eff_dl >= dl - 1e-6):
                self._sec_tl.last_budget_expired = True
            resp_open = resp_holder[0]
            if resp_open is not None:
                try:
                    raw_sock = getattr(getattr(getattr(resp_open, "fp", None), "raw", None), "_sock", None)
                    if raw_sock is not None:
                        if hasattr(raw_sock, "shutdown"):
                            try:
                                raw_sock.shutdown(socket.SHUT_RDWR)
                            except Exception:
                                pass
                        if hasattr(raw_sock, "close"):
                            raw_sock.close()
                except Exception:
                    pass
            return None
        if budget_hit[0]:
            self._sec_tl.last_budget_expired = True
        return out_box[0]

    def resolve_cik(
        self,
        ticker: str,
        known_cik: Optional[str] = None,
        enable_live_fetch: bool = False,
    ) -> Optional[str]:
        """Resolves zero-padded 10-digit SEC CIK via UniverseSelector or SEC ticker map."""
        sym = ticker.upper().strip()
        if known_cik and str(known_cik).strip().isdigit():
            return str(known_cik).strip().zfill(10)

        ticker_map = self._ticker_to_cik_map
        if ticker_map is None:
            cached = self._read_json_cache("sec_company_tickers.json", self._TICKER_MAP_TTL_SECS)
            if cached is None and enable_live_fetch and not self._is_in_cooldown("sec:tickers"):
                raw = self._sec_fetch_text("https://www.sec.gov/files/company_tickers.json", timeout=6.0)
                if raw:
                    try:
                        parsed = json.loads(raw)
                        mapping = {
                            str(v.get("ticker", "")).upper(): str(v.get("cik_str", "")).zfill(10)
                            for v in parsed.values()
                            if isinstance(v, dict) and v.get("ticker") and v.get("cik_str") is not None
                        }
                        self._write_json_cache("sec_company_tickers.json", mapping)
                        cached = mapping
                    except Exception:
                        self._mark_failed("sec:tickers")
                else:
                    self._mark_failed("sec:tickers")
            if isinstance(cached, dict):
                self._ticker_to_cik_map = cached
                ticker_map = cached
            else:
                stale_map = self._read_stale_json_cache("sec_company_tickers.json")
                if isinstance(stale_map, dict):
                    ticker_map = stale_map

        if ticker_map:
            return ticker_map.get(sym)
        return None

    @staticmethod
    def classify_insider_tx(raw_text: str) -> str:
        """Classifies Yahoo/SEC insider transaction text into standard Form 4 codes."""
        t = str(raw_text or "").lower()
        if re.search(
            r"\b(?:disposition|(?:sale|sell|sold|transfer(?:red)?|surrender(?:ed)?)(?:\s+of)?(?:\s+(?:\d[\d,]*\s+)?shares)?(?:\s+back)?)\s+to\s+(?:the\s+)?issuer\b",
            t,
        ) and not re.search(
            r"\b(?:not\s+(?:sold|(?:a|an)\s+(?:sale|disposition|transfer))|no\s+(?:[\w\-]+\s+){0,3}(?:shares\s+)?(?:were\s+|are\s+|was\s+)?sold|(?:were|was|are|is)\s+not\s+(?:[\w\-]+\s+){0,2}sold)\b",
            t,
        ):
            return "D"
        if re.search(r"\bgift(?:ed|s)?\b", t) and not re.search(
            r"\b(?:not\s+(?:a\s+|an\s+)?gift|non[\s\-]+gift|no\s+gift)\b",
            t,
        ):
            return "G"
        # Explicit market/sell-to-cover sales are Form 4 code "S" unless negated ("not a sale" / "no shares were sold" / "shares were not sold")
        if re.search(r"\b(sale|sell|sold)\b", t) and not re.search(
            r"\b(?:not\s+(?:sold|(?:a|an)\s+(?:open[\s\-]+market\s+)?(?:sale|sell))|no\s+(?:[\w\-]+\s+){0,3}(?:shares\s+)?(?:were\s+|are\s+|was\s+)?(?:sold|sale|sell)|(?:were|was|are|is)\s+not\s+(?:[\w\-]+\s+){0,2}(?:sold|sale|sell))\b",
            t,
        ):
            return "S"
        if re.search(
            r"\b(?:withholding|withheld|tax[\s\-]+(?:withhold(?:ing)?|withheld)|net[\s\-]+share\s+settlement|surrender(?:ed)?\s+(?:for|to\s+pay)\s+tax(?:es)?|payment\s+of\s+exercise\s+price\s+or\s+tax\s+liability\s+by\s+delivering|delivering\s+or\s+withholding)\b",
            t,
        ):
            return "F"
        if re.search(r"\b(?:exercised|conversion|converted|exercise(?!\s+price\b))\b", t):
            return "M"
        if re.search(r"\b(purchase|buy|bought)\b", t):
            return "P"
        if re.search(r"\b(award|grant|vest(?:ing|ed)?)\b", t):
            return "A"
        # "J" = SEC Form 4 Other acquisition/disposition (kept out of P/S scoring)
        return "J"

    def get_yf_meta(
        self,
        ticker: str,
        enable_live_fetch: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Fetches & disk-caches real Yahoo Finance info, analyst consensus, and insider rows."""
        sym = ticker.upper().strip()
        rel_path = f"yf_meta/{sym}.json"
        cached = self._read_json_cache(rel_path, self._YF_META_TTL_SECS)
        if isinstance(cached, dict):
            out = {k: v for k, v in cached.items() if k not in ("_cache_stale", "_cache_hit")}
            out["_cache_stale"] = False
            out["_cache_hit"] = True
            return out
        if not enable_live_fetch:
            return self._read_stale_payload(rel_path)

        fail_key = f"yf_meta:{sym}"
        if self._is_in_cooldown(fail_key):
            return self._read_stale_payload(rel_path)

        try:
            import yfinance as yf

            tk = yf.Ticker(sym)
            info = tk.info or {}
            if not isinstance(info, dict) or len(info) == 0:
                self._mark_failed(fail_key)
                return self._read_stale_payload(rel_path)

            # Clean JSON-serializable subset of tk.info
            safe_info: Dict[str, Any] = {}
            for k in (
                "longName", "shortName", "exchange", "fullExchangeName", "quoteType",
                "sector", "industry", "marketCap", "trailingPE", "forwardPE",
                "dividendYield", "trailingAnnualDividendYield",
                "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "longBusinessSummary",
                "city", "state", "country", "fullTimeEmployees", "companyOfficers",
                "currentPrice", "regularMarketPrice", "regularMarketTime", "targetMeanPrice",
                "targetMedianPrice", "targetHighPrice", "targetLowPrice",
                "recommendationMean", "recommendationKey", "numberOfAnalystOpinions",
            ):
                if k in info and info[k] is not None:
                    v = info[k]
                    if isinstance(v, float) and not math.isfinite(v):
                        continue
                    safe_info[k] = self._sanitize_json_obj(v)

            # Analyst recommendation distribution trend (0m, -1m, -2m, -3m)
            rec_trend: List[Dict[str, Any]] = []
            try:
                rec_df = tk.recommendations_summary
                if rec_df is not None and not rec_df.empty:
                    for _, row in rec_df.head(4).iterrows():
                        rec_trend.append({
                            "period": str(row.get("period", "")),
                            "strong_buy": int(row.get("strongBuy", 0) or 0),
                            "buy": int(row.get("buy", 0) or 0),
                            "hold": int(row.get("hold", 0) or 0),
                            "underperform": int(row.get("sell", 0) or 0),
                            "sell": int(row.get("strongSell", 0) or 0),
                        })
            except Exception as e:
                logger.debug(f"Yahoo recommendations_summary unavailable for {sym}: {e}")

            # Insider transactions
            insiders: List[Dict[str, Any]] = []
            try:
                ins_df = tk.insider_transactions
                if ins_df is not None and not ins_df.empty:
                    for _, row in ins_df.head(15).iterrows():
                        dt_val = row.get("Start Date")
                        dt_str = dt_val.strftime("%Y-%m-%d") if hasattr(dt_val, "strftime") else str(dt_val or "")[:10]
                        shares = float(row.get("Shares", 0) or 0)
                        val = float(row.get("Value", 0) or 0)
                        if not math.isfinite(val):
                            val = 0.0
                        if not math.isfinite(shares):
                            shares = 0.0
                        raw_txt = str(row.get("Text") or row.get("Transaction") or "").strip()
                        text_desc = raw_txt.lower()
                        tx_code = self.classify_insider_tx(raw_txt)
                        price_est = round(val / shares, 2) if shares > 0 and val > 0 else 0.0
                        insiders.append({
                            "filing_date": dt_str,
                            "disclosed_at": None,
                            "disclosed_at_verified": False,
                            "insider_name": str(row.get("Insider") or "Insider").strip(),
                            "relationship": str(row.get("Position") or "Officer/Director").strip(),
                            "transaction_code": tx_code,
                            "shares": int(round(abs(shares))),
                            "price": price_est,
                            "value": round(abs(val), 2),
                            "is_10b5_1": ("10b5-1" in text_desc or "automatic" in text_desc or "plan" in text_desc),
                            "raw_text": raw_txt,
                        })
            except Exception as e:
                logger.debug(f"Yahoo insider_transactions unavailable for {sym}: {e}")

            payload = {
                "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "info": safe_info,
                "rec_trend": rec_trend,
                "insider_transactions": insiders,
            }
            self._write_json_cache(rel_path, payload)
            return {**payload, "_cache_stale": False, "_cache_hit": False}
        except Exception as e:
            logger.debug(f"Yahoo meta fetch failed for {sym}: {e}")
            self._mark_failed(fail_key)
            return self._read_stale_payload(rel_path)

    @staticmethod
    def _recent_col(rec: Mapping[str, Any], key: str) -> List[Any]:
        val = rec.get(key)
        return val if isinstance(val, list) else []

    def get_sec_submissions(
        self,
        cik: str,
        enable_live_fetch: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Fetches & disk-caches SEC EDGAR submissions JSON for a 10-digit CIK."""
        cik10 = str(cik).strip().zfill(10)
        rel_path = f"edgar_submissions/CIK{cik10}.json"
        cached = self._read_dict_cache(rel_path, self._SUBMISSIONS_TTL_SECS)
        if cached is not None:
            return cached
        if not enable_live_fetch:
            return self._read_dict_cache(rel_path, None)

        fail_key = f"sec_sub:{cik10}"
        if self._is_in_cooldown(fail_key):
            return self._read_dict_cache(rel_path, None)

        raw = self._sec_fetch_text(f"https://data.sec.gov/submissions/CIK{cik10}.json", timeout=6.0)
        if not raw:
            self._mark_failed(fail_key)
            return self._read_dict_cache(rel_path, None)
        try:
            full = json.loads(raw)
            if not isinstance(full, dict):
                self._mark_failed(fail_key)
                return self._read_dict_cache(rel_path, None)
            filings_obj = full.get("filings")
            recent = filings_obj.get("recent") if isinstance(filings_obj, Mapping) else None
            if not isinstance(recent, Mapping):
                self._mark_failed(fail_key)
                return self._read_dict_cache(rel_path, None)
            # Store compact recent filings slice (~1000 filings) rather than multi-MB raw JSON
            compact = {
                "cik": cik10,
                "name": full.get("name", ""),
                "sicDescription": full.get("sicDescription", ""),
                "stateOfIncorporation": full.get("stateOfIncorporation", ""),
                "exchanges": full.get("exchanges", []),
                "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "recent": {
                    k: (v[:1000] if isinstance(v, list) else v)
                    for k, v in recent.items()
                },
            }
            self._write_json_cache(rel_path, compact)
            return compact
        except Exception as e:
            logger.debug(f"Failed parsing SEC submissions for {cik10}: {e}")
            self._mark_failed(fail_key)
            return self._read_dict_cache(rel_path, None)

    def get_sec_company_facts(
        self,
        cik: str,
        enable_live_fetch: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Fetches & disk-caches SEC EDGAR US-GAAP CompanyFacts JSON."""
        cik10 = str(cik).strip().zfill(10)
        rel_path = f"edgar_facts/CIK{cik10}.json"
        cached = self._read_dict_cache(rel_path, self._FACTS_TTL_SECS)
        if cached is not None:
            return cached
        if not enable_live_fetch:
            return self._read_dict_cache(rel_path, None)

        fail_key = f"sec_facts:{cik10}"
        if self._is_in_cooldown(fail_key):
            return self._read_dict_cache(rel_path, None)

        raw = self._sec_fetch_text(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json", timeout=8.0)
        if not raw:
            self._mark_failed(fail_key)
            return self._read_dict_cache(rel_path, None)
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                self._mark_failed(fail_key)
                return self._read_dict_cache(rel_path, None)
            facts_obj = parsed.get("facts")
            if not isinstance(facts_obj, Mapping):
                self._mark_failed(fail_key)
                return self._read_dict_cache(rel_path, None)
            raw_us_gaap = facts_obj.get("us-gaap")
            if "us-gaap" in facts_obj and raw_us_gaap is not None and not isinstance(raw_us_gaap, Mapping):
                self._mark_failed(fail_key)
                return self._read_dict_cache(rel_path, None)
            us_gaap = raw_us_gaap if isinstance(raw_us_gaap, Mapping) else {}
            needed = set(
                self._REVENUE_CONCEPTS
                + self._COST_OF_REV_CONCEPTS
                + self._GROSS_PROFIT_CONCEPTS
                + self._OPEX_CONCEPTS
                + self._OP_INC_CONCEPTS
                + self._NET_INC_CONCEPTS
                + self._EPS_CONCEPTS
                + self._CASH_CONCEPTS
                + self._ASSETS_CONCEPTS
                + self._LT_DEBT_CONCEPTS
                + self._ST_DEBT_CONCEPTS
                + self._EQUITY_CONCEPTS
                + self._OCF_CONCEPTS
                + self._CAPEX_CONCEPTS
            )
            trimmed_gaap = {k: v for k, v in us_gaap.items() if k in needed}
            compact = {
                "cik": cik10,
                "entityName": parsed.get("entityName", ""),
                "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "us_gaap": trimmed_gaap,
            }
            self._write_json_cache(rel_path, compact)
            return compact
        except Exception as e:
            logger.debug(f"Failed parsing SEC companyfacts for {cik10}: {e}")
            self._mark_failed(fail_key)
            return self._read_dict_cache(rel_path, None)

    @staticmethod
    def _parse_iso_utc_us(raw_ts: Any) -> Optional[int]:
        """Parses an ISO-8601 timestamp into integer UTC microseconds."""
        s = str(raw_ts or "").strip()
        if not s or ("T" not in s and ":" not in s):
            return None
        try:
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return int(dt.replace(microsecond=0).timestamp()) * 1_000_000 + dt.microsecond
        except ValueError:
            return None

    @staticmethod
    def _xbrl_filing_key(
        filed_dt: str,
        accn: str,
        form: str,
        acc_accept: Optional[Dict[str, str]] = None,
        use_accept: bool = False,
    ) -> Tuple[str, str, int, str]:
        f_dt = str(filed_dt or "")[:10]
        a_str = str(accn or "")
        accept = str((acc_accept or {}).get(a_str, "")) if use_accept else ""
        is_amend = 1 if str(form or "").endswith("/A") else 0
        return (f_dt, accept, is_amend, a_str)

    @staticmethod
    def _extract_annual_xbrl_series(
        us_gaap: Dict[str, Any],
        concepts: Tuple[str, ...],
        is_flow: bool,
        unit_pref: str = "USD",
        pit_strict: bool = True,
        valid_period_ends: Optional[Dict[str, str]] = None,
        acc_accept: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, Tuple[str, str, str, bool]]]:
        """
        Extracts annual values keyed by fiscal period end ('YYYY-MM-DD') -> val.
        When multiple concepts exist in hierarchy, selects the highest-priority concept
        available per fiscal period_end (enforcing PIT earliest-filed vs As-Restated latest-filed).
        Returns (period_end -> val, period_end -> (filed_date, accn, form, used_accept_order)).
        """
        out_vals: Dict[str, float] = {}
        out_meta: Dict[str, Tuple[str, str, str, bool]] = {}

        for concept in concepts:
            cdata = us_gaap.get(concept)
            if not cdata:
                continue
            units = cdata.get("units", {})
            unit_list = units.get(unit_pref)
            if unit_list is None and unit_pref == "USD":
                # fallback first currency unit if present
                for uk, uv in units.items():
                    if "USD" in uk:
                        unit_list = uv
                        break
            if not unit_list:
                continue

            by_end: Dict[str, List[Dict[str, Any]]] = {}
            for obs in unit_list:
                form = str(obs.get("form", ""))
                if form not in ("10-K", "10-K/A"):
                    continue
                end_dt = str(obs.get("end", ""))[:10]
                filed_dt = str(obs.get("filed", ""))[:10]
                if not end_dt or not filed_dt:
                    continue
                val = obs.get("val")
                if val is None or not isinstance(val, (int, float)) or not math.isfinite(val):
                    continue
                if is_flow:
                    start_dt = str(obs.get("start", ""))[:10]
                    if not start_dt:
                        continue
                    try:
                        days = (datetime.strptime(end_dt, "%Y-%m-%d") - datetime.strptime(start_dt, "%Y-%m-%d")).days
                    except ValueError:
                        continue
                    # Annual periods are ~330..385 days (filter out Q1/Q2/Q3/Q4 standalone fragments)
                    if days < 300 or days > 400:
                        continue
                else:
                    if obs.get("fp") not in ("FY", None, ""):
                        # Balance sheet FY rows sometimes omit fp or tag FY
                        if obs.get("fp") != "FY":
                            continue

                by_end.setdefault(end_dt, []).append(obs)

            for end_dt, obs_list in by_end.items():
                if valid_period_ends and end_dt not in valid_period_ends:
                    # Match close fiscal year-end dates within 7 days (52/53 week fiscal calendars like AAPL/MSFT)
                    matched_end = None
                    try:
                        t_end = datetime.strptime(end_dt, "%Y-%m-%d")
                        for target_end in valid_period_ends:
                            if abs((datetime.strptime(target_end, "%Y-%m-%d") - t_end).days) <= 7:
                                matched_end = target_end
                                break
                    except ValueError:
                        matched_end = None
                    if matched_end is None:
                        continue
                    canonical_end = matched_end
                else:
                    canonical_end = end_dt

                if canonical_end in out_vals:
                    continue

                def _accept_us(o: Dict[str, Any]) -> Optional[int]:
                    return FundamentalsClient._parse_iso_utc_us((acc_accept or {}).get(str(o.get("accn", "")), ""))

                def _is_amend(o: Dict[str, Any]) -> int:
                    return 1 if str(o.get("form", "") or "").endswith("/A") else 0

                target_filed = (
                    min(str(o.get("filed", ""))[:10] for o in obs_list)
                    if pit_strict
                    else max(str(o.get("filed", ""))[:10] for o in obs_list)
                )
                bucket = [o for o in obs_list if str(o.get("filed", ""))[:10] == target_filed]
                ts_known = [o for o in bucket if _accept_us(o) is not None]
                ts_missing = [o for o in bucket if _accept_us(o) is None]
                ts_known.sort(key=lambda o: (_accept_us(o) or 0, _is_amend(o), str(o.get("accn", ""))))
                missing_orig = sorted([o for o in ts_missing if _is_amend(o) == 0], key=lambda o: str(o.get("accn", "")))
                missing_amend = sorted([o for o in ts_missing if _is_amend(o) == 1], key=lambda o: str(o.get("accn", "")))
                if pit_strict:
                    if ts_known and (any(_is_amend(o) == 0 for o in ts_known) or not missing_orig):
                        chosen = ts_known[0]
                    else:
                        chosen = (missing_orig or missing_amend or ts_known)[0]
                else:
                    if missing_amend:
                        chosen = missing_amend[-1]
                    elif ts_known:
                        chosen = ts_known[-1]
                    else:
                        chosen = missing_orig[-1]

                chosen_f_d = str(chosen.get("filed", ""))[:10]
                ident_accs = {
                    str(o.get("accn") or "").strip()
                    for o in bucket
                    if str(o.get("accn") or "").strip()
                }
                missing_acc_count = sum(1 for o in bucket if not str(o.get("accn") or "").strip())
                acc_to_accept = {
                    str(o.get("accn") or "").strip(): _accept_us(o)
                    for o in bucket
                    if str(o.get("accn") or "").strip()
                }
                acc_to_amend = {
                    str(o.get("accn") or "").strip(): _is_amend(o)
                    for o in bucket
                    if str(o.get("accn") or "").strip()
                }
                has_missing_identity = bool(missing_acc_count > 0 and (len(ident_accs) + missing_acc_count) > 1)
                has_missing_or_tied_ts = len(acc_to_accept) > 1 and (
                    any(ts is None for ts in acc_to_accept.values())
                    or len(set(zip(acc_to_accept.values(), acc_to_amend.values()))) < len(acc_to_accept)
                )
                order_uncertain = bool(has_missing_identity or has_missing_or_tied_ts)
                out_vals[canonical_end] = float(chosen["val"])
                out_meta[canonical_end] = (
                    chosen_f_d,
                    str(chosen.get("accn", "")),
                    str(chosen.get("form", "")),
                    not order_uncertain,
                )

        return out_vals, out_meta

    def build_financials_payload(
        self,
        ticker: str,
        known_cik: Optional[str] = None,
        pit_strict: bool = True,
        enable_live_fetch: bool = False,
    ) -> Dict[str, Any]:
        """Builds 10-year US-GAAP financial statement payload from real SEC EDGAR XBRL CompanyFacts."""
        sym = ticker.upper().strip()
        cik = self.resolve_cik(sym, known_cik=known_cik, enable_live_fetch=enable_live_fetch)
        if not cik:
            return {
                "ticker": sym,
                "available": False,
                "source": "SEC_CIK_UNAVAILABLE",
                "pit_strict": pit_strict,
                "filings": [],
                "incomeStatement": [],
                "balanceSheet": [],
                "cashFlow": [],
                "ratios": [],
                "message": f"No SEC CIK mapping available for {sym} (ETF, ADR, or offline).",
            }

        subs = self.get_sec_submissions(cik, enable_live_fetch=enable_live_fetch)
        facts = self.get_sec_company_facts(cik, enable_live_fetch=enable_live_fetch)
        if not facts or not facts.get("us_gaap"):
            return {
                "ticker": sym,
                "available": False,
                "source": "EDGAR_FACTS_UNAVAILABLE",
                "pit_strict": pit_strict,
                "filings": [],
                "incomeStatement": [],
                "balanceSheet": [],
                "cashFlow": [],
                "ratios": [],
                "message": "SEC EDGAR US-GAAP CompanyFacts unreachable offline or non-reporting issuer.",
            }

        # Map 10-K filings from submissions
        end_to_10k: Dict[str, Dict[str, str]] = {}
        acc_accept: Dict[str, str] = {}
        if subs and isinstance(subs.get("recent"), Mapping):
            rec = subs["recent"]
            forms = self._recent_col(rec, "form")
            fdates = self._recent_col(rec, "filingDate")
            rdates = self._recent_col(rec, "reportDate")
            accns = self._recent_col(rec, "accessionNumber")
            docs = self._recent_col(rec, "primaryDocument")
            accepts = self._recent_col(rec, "acceptanceDateTime")
            for accn_raw, accept_raw in zip(accns, accepts):
                if accn_raw and accept_raw:
                    acc_accept[str(accn_raw)] = str(accept_raw)
            for form, fdt, rdt, accn, doc in zip(forms, fdates, rdates, accns, docs):
                if form in ("10-K", "10-K/A") and rdt:
                    rdt_clean = str(rdt)[:10]
                    accn_str = str(accn)
                    accn_nodash = accn_str.replace("-", "")
                    sec_url = (
                        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn_nodash}/{doc}"
                        if doc
                        else f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn_nodash}/"
                    )
                    prev = end_to_10k.get(rdt_clean)
                    if prev is None or form == "10-K":
                        end_to_10k[rdt_clean] = {
                            "filed_date": str(fdt)[:10],
                            "period_end_date": rdt_clean,
                            "accession_number": accn_str,
                            "sec_url": sec_url,
                            "form": form,
                        }

        us_gaap = facts["us_gaap"]
        # First pass extract revenue ends to identify canonical fiscal year-end dates
        rev_all_pit, _ = self._extract_annual_xbrl_series(
            us_gaap, self._REVENUE_CONCEPTS, is_flow=True, unit_pref="USD", pit_strict=True
        )
        canonical_ends = sorted(set(end_to_10k.keys()) | set(rev_all_pit.keys()))
        # Filter to annual closes >= 2016 and keep at most 10 most recent fiscal years
        canonical_ends = [d for d in canonical_ends if d[:4].isdigit() and int(d[:4]) >= 2016][-10:]
        if not canonical_ends:
            return {
                "ticker": sym,
                "available": False,
                "source": "EDGAR_NO_ANNUAL_PERIODS",
                "pit_strict": pit_strict,
                "filings": [],
                "incomeStatement": [],
                "balanceSheet": [],
                "cashFlow": [],
                "ratios": [],
                "message": f"No annual 10-K US-GAAP periods parsed for {sym}.",
            }

        valid_map = {d: d for d in canonical_ends}

        rev_pit, rev_meta_pit = self._extract_annual_xbrl_series(
            us_gaap, self._REVENUE_CONCEPTS, True, "USD", True, valid_map, acc_accept
        )
        rev_rest, rev_meta_rest = self._extract_annual_xbrl_series(
            us_gaap, self._REVENUE_CONCEPTS, True, "USD", False, valid_map, acc_accept
        )
        rev = rev_pit if pit_strict else rev_rest
        rev_meta = rev_meta_pit if pit_strict else rev_meta_rest

        cor, _ = self._extract_annual_xbrl_series(us_gaap, self._COST_OF_REV_CONCEPTS, True, "USD", pit_strict, valid_map, acc_accept)
        gp, _ = self._extract_annual_xbrl_series(us_gaap, self._GROSS_PROFIT_CONCEPTS, True, "USD", pit_strict, valid_map, acc_accept)
        opex, _ = self._extract_annual_xbrl_series(us_gaap, self._OPEX_CONCEPTS, True, "USD", pit_strict, valid_map, acc_accept)
        opinc, _ = self._extract_annual_xbrl_series(us_gaap, self._OP_INC_CONCEPTS, True, "USD", pit_strict, valid_map, acc_accept)
        ni_pit, ni_meta_pit = self._extract_annual_xbrl_series(us_gaap, self._NET_INC_CONCEPTS, True, "USD", True, valid_map, acc_accept)
        ni_rest, ni_meta_rest = self._extract_annual_xbrl_series(us_gaap, self._NET_INC_CONCEPTS, True, "USD", False, valid_map, acc_accept)
        ni = ni_pit if pit_strict else ni_rest
        eps, _ = self._extract_annual_xbrl_series(us_gaap, self._EPS_CONCEPTS, True, "USD/shares", pit_strict, valid_map, acc_accept)

        cash, _ = self._extract_annual_xbrl_series(us_gaap, self._CASH_CONCEPTS, False, "USD", pit_strict, valid_map, acc_accept)
        assets, _ = self._extract_annual_xbrl_series(us_gaap, self._ASSETS_CONCEPTS, False, "USD", pit_strict, valid_map, acc_accept)
        lt_debt, _ = self._extract_annual_xbrl_series(us_gaap, self._LT_DEBT_CONCEPTS, False, "USD", pit_strict, valid_map, acc_accept)
        st_debt, _ = self._extract_annual_xbrl_series(us_gaap, self._ST_DEBT_CONCEPTS, False, "USD", pit_strict, valid_map, acc_accept)
        equity, _ = self._extract_annual_xbrl_series(us_gaap, self._EQUITY_CONCEPTS, False, "USD", pit_strict, valid_map, acc_accept)

        ocf, _ = self._extract_annual_xbrl_series(us_gaap, self._OCF_CONCEPTS, True, "USD", pit_strict, valid_map, acc_accept)
        capex, _ = self._extract_annual_xbrl_series(us_gaap, self._CAPEX_CONCEPTS, True, "USD", pit_strict, valid_map, acc_accept)

        # Build sorted FY labels handling e.g. January fiscal year-end without collisions
        filings: List[Dict[str, Any]] = []
        end_to_fy: Dict[str, str] = {}
        used_labels: set = set()
        for end_dt in canonical_ends:
            if end_dt not in rev and end_dt not in assets:
                continue
            yr = int(end_dt[:4])
            label = f"FY{yr}"
            while label in used_labels:
                yr += 1
                label = f"FY{yr}"
            used_labels.add(label)
            end_to_fy[end_dt] = label

            k_info = end_to_10k.get(end_dt)
            filed_dt = (k_info["filed_date"] if k_info else rev_meta.get(end_dt, (end_dt, ""))[0]) or end_dt
            accn = (k_info["accession_number"] if k_info else rev_meta.get(end_dt, ("", ""))[1]) or "SEC-XBRL"
            sec_url = (
                k_info["sec_url"]
                if k_info
                else f"https://www.sec.gov/edgar/browse/?CIK={cik}"
            )
            delta_rev_raw = (rev_rest.get(end_dt, 0.0) or 0.0) - (rev_pit.get(end_dt, 0.0) or 0.0)
            delta_ni_raw = (ni_rest.get(end_dt, 0.0) or 0.0) - (ni_pit.get(end_dt, 0.0) or 0.0)

            def _is_subsequent_filing(
                pit_m: Optional[Tuple[str, str, str, bool]],
                rest_m: Optional[Tuple[str, str, str, bool]],
            ) -> bool:
                if not pit_m or not rest_m:
                    return False
                p_filed, p_acc, p_form, _ = pit_m
                r_filed, r_acc, r_form, _ = rest_m
                if r_filed and p_filed and r_filed != p_filed:
                    return r_filed > p_filed
                if p_acc and r_acc and acc_accept:
                    p_us = self._parse_iso_utc_us(acc_accept.get(p_acc, ""))
                    r_us = self._parse_iso_utc_us(acc_accept.get(r_acc, ""))
                    if p_us is not None and r_us is not None and r_us != p_us:
                        return r_us > p_us
                p_amend = 1 if str(p_form or "").endswith("/A") else 0
                r_amend = 1 if str(r_form or "").endswith("/A") else 0
                return r_amend > p_amend

            pit_rev_cert = (rev_meta_pit.get(end_dt) or ("", "", "", True))[3]
            rest_rev_cert = (rev_meta_rest.get(end_dt) or ("", "", "", True))[3]
            pit_ni_cert = (ni_meta_pit.get(end_dt) or ("", "", "", True))[3]
            rest_ni_cert = (ni_meta_rest.get(end_dt) or ("", "", "", True))[3]
            order_uncertain = not (pit_rev_cert and rest_rev_cert and pit_ni_cert and rest_ni_cert)

            rev_changed = (
                abs(delta_rev_raw) > 1e5
                and end_dt in rev_rest
                and _is_subsequent_filing(rev_meta_pit.get(end_dt), rev_meta_rest.get(end_dt))
            )
            ni_changed = (
                abs(delta_ni_raw) > 1e5
                and end_dt in ni_rest
                and _is_subsequent_filing(ni_meta_pit.get(end_dt), ni_meta_rest.get(end_dt))
            )
            is_restated = bool(rev_changed or ni_changed)
            note = None
            if is_restated:
                def _fmt_m(raw_delta: float) -> str:
                    m = raw_delta / 1e6
                    if abs(m - round(m)) < 0.05:
                        return f"{round(m):+d}M"
                    return f"{m:+.1f}M"

                grouped_changes: Dict[str, Tuple[str, List[str]]] = {}
                if rev_changed:
                    _, r_acc, r_form, _ = rev_meta_rest.get(end_dt) or ("", "", "", False)
                    g_key = r_acc or f"__rev_{r_form}"
                    if g_key not in grouped_changes:
                        grouped_changes[g_key] = (r_form, [])
                    grouped_changes[g_key][1].append(f"Revenue by {_fmt_m(delta_rev_raw)}")
                if ni_changed:
                    _, n_acc, n_form, _ = ni_meta_rest.get(end_dt) or ("", "", "", False)
                    g_key = n_acc or f"__ni_{n_form}"
                    if g_key not in grouped_changes:
                        grouped_changes[g_key] = (n_form, [])
                    grouped_changes[g_key][1].append(f"Net Income by {_fmt_m(delta_ni_raw)}")
                if grouped_changes:
                    notes_parts: List[str] = []
                    for src_form, metric_changes in grouped_changes.values():
                        if src_form == "10-K/A":
                            prefix = "Subsequent 10-K/A restatement changed"
                        elif src_form:
                            prefix = f"Subsequent {src_form} filing revised"
                        else:
                            prefix = "Subsequent filing revised"
                        notes_parts.append(f"{prefix} {', '.join(metric_changes)}")
                    note = "; ".join(notes_parts)
            filings.append({
                "period": label,
                "filed_date": filed_dt,
                "period_end_date": end_dt,
                "accession_number": accn,
                "sec_url": sec_url,
                "is_restated": is_restated,
                "order_uncertain": order_uncertain,
                "restatement_delta_notes": note,
            })

        def _millions(d: Dict[str, float], end_dt: str) -> Optional[float]:
            v = d.get(end_dt)
            return round(v / 1_000_000.0, 1) if v is not None else None

        rev_row: Dict[str, Optional[float]] = {}
        cor_row: Dict[str, Optional[float]] = {}
        gp_row: Dict[str, Optional[float]] = {}
        opex_row: Dict[str, Optional[float]] = {}
        opinc_row: Dict[str, Optional[float]] = {}
        ni_row: Dict[str, Optional[float]] = {}
        eps_row: Dict[str, Optional[float]] = {}

        cash_row: Dict[str, Optional[float]] = {}
        assets_row: Dict[str, Optional[float]] = {}
        debt_row: Dict[str, Optional[float]] = {}
        eq_row: Dict[str, Optional[float]] = {}

        ocf_row: Dict[str, Optional[float]] = {}
        capex_row: Dict[str, Optional[float]] = {}
        fcf_row: Dict[str, Optional[float]] = {}

        gm_row: Dict[str, Optional[float]] = {}
        om_row: Dict[str, Optional[float]] = {}
        nm_row: Dict[str, Optional[float]] = {}
        roe_row: Dict[str, Optional[float]] = {}

        for f in filings:
            end_dt = f["period_end_date"]
            p = f["period"]

            r_m = _millions(rev, end_dt)
            c_m = _millions(cor, end_dt)
            g_m = _millions(gp, end_dt)
            if g_m is None and r_m is not None and c_m is not None:
                g_m = round(r_m - c_m, 1)
            if c_m is None and r_m is not None and g_m is not None:
                c_m = round(r_m - g_m, 1)

            oi_m = _millions(opinc, end_dt)
            ox_m = _millions(opex, end_dt)
            if ox_m is None and g_m is not None and oi_m is not None:
                ox_m = round(g_m - oi_m, 1)

            n_m = _millions(ni, end_dt)
            e_v = eps.get(end_dt)
            eps_v = round(float(e_v), 2) if e_v is not None else None

            rev_row[p] = r_m
            cor_row[p] = c_m
            gp_row[p] = g_m
            opex_row[p] = ox_m
            opinc_row[p] = oi_m
            ni_row[p] = n_m
            eps_row[p] = eps_v

            ca_m = _millions(cash, end_dt)
            as_m = _millions(assets, end_dt)
            ltd = lt_debt.get(end_dt, 0.0) or 0.0
            std = st_debt.get(end_dt, 0.0) or 0.0
            d_m = round((ltd + std) / 1_000_000.0, 1) if (end_dt in lt_debt or end_dt in st_debt) else None
            eq_m = _millions(equity, end_dt)

            cash_row[p] = ca_m
            assets_row[p] = as_m
            debt_row[p] = d_m
            eq_row[p] = eq_m

            o_m = _millions(ocf, end_dt)
            cx_m = _millions(capex, end_dt)
            f_m = round(o_m - abs(cx_m), 1) if (o_m is not None and cx_m is not None) else None

            ocf_row[p] = o_m
            capex_row[p] = abs(cx_m) if cx_m is not None else None
            fcf_row[p] = f_m

            gm_row[p] = round((g_m / r_m) * 100.0, 1) if (r_m and g_m is not None and r_m != 0) else None
            om_row[p] = round((oi_m / r_m) * 100.0, 1) if (r_m and oi_m is not None and r_m != 0) else None
            nm_row[p] = round((n_m / r_m) * 100.0, 1) if (r_m and n_m is not None and r_m != 0) else None
            roe_row[p] = round((n_m / eq_m) * 100.0, 1) if (eq_m and n_m is not None and eq_m > 0) else None

        return {
            "ticker": sym,
            "available": len(filings) > 0,
            "source": "SEC_EDGAR_XBRL",
            "entity_name": facts.get("entityName") or sym,
            "cik": cik,
            "pit_strict": pit_strict,
            "filings": filings,
            "incomeStatement": [
                {"tag": "Revenues", "label": "Revenue ($M)", "values": rev_row, "format": "currency", "isHeader": True},
                {"tag": "CostOfRevenue", "label": "Cost of Revenue ($M)", "values": cor_row, "format": "currency", "indent": True},
                {"tag": "GrossProfit", "label": "Gross Profit ($M)", "values": gp_row, "format": "currency", "isHeader": True},
                {"tag": "OperatingExpenses", "label": "Operating Expenses ($M)", "values": opex_row, "format": "currency", "indent": True},
                {"tag": "OperatingIncome", "label": "Operating Income / EBIT ($M)", "values": opinc_row, "format": "currency", "isHeader": True},
                {"tag": "NetIncome", "label": "Net Income ($M)", "values": ni_row, "format": "currency", "isHeader": True},
                {"tag": "EPSDiluted", "label": "Diluted EPS ($)", "values": eps_row, "format": "number"},
            ],
            "balanceSheet": [
                {"tag": "CashAndEquivalents", "label": "Cash & Equivalents ($M)", "values": cash_row, "format": "currency"},
                {"tag": "TotalAssets", "label": "Total Assets ($M)", "values": assets_row, "format": "currency", "isHeader": True},
                {"tag": "TotalDebt", "label": "Total Debt ($M)", "values": debt_row, "format": "currency"},
                {"tag": "StockholdersEquity", "label": "Stockholders' Equity ($M)", "values": eq_row, "format": "currency", "isHeader": True},
            ],
            "cashFlow": [
                {"tag": "OperatingCashFlow", "label": "Cash from Operations ($M)", "values": ocf_row, "format": "currency", "isHeader": True},
                {"tag": "CapitalExpenditures", "label": "Capital Expenditures ($M)", "values": capex_row, "format": "currency", "indent": True},
                {"tag": "FreeCashFlow", "label": "Free Cash Flow ($M)", "values": fcf_row, "format": "currency", "isHeader": True},
            ],
            "ratios": [
                {"tag": "GrossMargin", "label": "Gross Margin (%)", "values": gm_row, "format": "percent"},
                {"tag": "OperatingMargin", "label": "Operating Margin (%)", "values": om_row, "format": "percent"},
                {"tag": "NetMargin", "label": "Net Profit Margin (%)", "values": nm_row, "format": "percent"},
                {"tag": "ROE", "label": "Return on Equity (%)", "values": roe_row, "format": "percent"},
            ],
        }

    @staticmethod
    def _extract_html_section_sentences(html_text: str, section: str) -> List[str]:
        """Extracts normalized prose sentences from Item 1A or Item 7 in SEC HTML/text."""
        clean = re.sub(r"<[^>]+>", " ", html_text)
        clean = re.sub(r"&nbsp;|&#160;|&amp;|&quot;", " ", clean)
        clean = re.sub(r"\s+", " ", clean)
        low = clean.lower()

        if section == "item1a":
            matches = [m.start() for m in re.finditer(r"item\s+1a[\.\:\s\-]+risk\s+factors", low)]
            # Skip table of contents match if a later body heading exists
            start = matches[-1] if len(matches) >= 2 else (matches[0] if matches else -1)
        else:
            matches = [m.start() for m in re.finditer(r"item\s+(?:7|2)[\.\:\s\-]+management", low)]
            start = matches[-1] if len(matches) >= 2 else (matches[0] if matches else -1)

        if start < 0:
            return []
        snippet = clean[start : start + 45000]
        raw_sents = re.split(r"(?<=[.!?])\s+", snippet)
        return [s.strip() for s in raw_sents if 50 <= len(s.strip()) <= 450][:140]

    @staticmethod
    def _bag_cosine_similarity(sents_a: List[str], sents_b: List[str]) -> float:
        if not sents_a or not sents_b:
            return 0.0
        toks_a: Dict[str, int] = {}
        toks_b: Dict[str, int] = {}
        for s in sents_a:
            for w in re.findall(r"[a-z]{4,}", s.lower()):
                toks_a[w] = toks_a.get(w, 0) + 1
        for s in sents_b:
            for w in re.findall(r"[a-z]{4,}", s.lower()):
                toks_b[w] = toks_b.get(w, 0) + 1
        if not toks_a or not toks_b:
            return 0.0
        dot = sum(cnt * toks_b.get(w, 0) for w, cnt in toks_a.items())
        na = math.sqrt(sum(c * c for c in toks_a.values()))
        nb = math.sqrt(sum(c * c for c in toks_b.values()))
        if na <= 0 or nb <= 0:
            return 0.0
        return round(float(dot / (na * nb)), 3)

    @staticmethod
    def _parse_sec_form4_xml(
        xml_text: str,
        filing_date: str,
        accept_dt: str,
        accn: str,
        form: str = "4",
    ) -> Optional[List[Dict[str, Any]]]:
        """Parses SEC EDGAR Form 4 XML non-derivative transactions; returns None on malformed XML."""
        if not xml_text:
            return None
        upper_head = xml_text[:1024].upper()
        if "<!ENTITY" in upper_head or "<!DOCTYPE" in upper_head:
            return None
        try:
            import xml.etree.ElementTree as ET

            root = ET.fromstring(xml_text)
        except Exception:
            return None

        def _lname(tag: str) -> str:
            return tag.split("}", 1)[-1] if "}" in tag else tag

        if _lname(str(root.tag)) != "ownershipDocument":
            return None

        def _child(elem: Any, name: str) -> Any:
            for ch in list(elem):
                if _lname(str(ch.tag)) == name:
                    return ch
            return None

        def _child_text(elem: Any, *path: str) -> str:
            cur = elem
            for step in path:
                if cur is None:
                    return ""
                cur = _child(cur, step)
            return str(cur.text or "").strip() if cur is not None else ""

        owners: List[str] = []
        rels: List[str] = []
        for ch in list(root):
            if _lname(str(ch.tag)) != "reportingOwner":
                continue
            nm = _child_text(ch, "reportingOwnerId", "rptOwnerName")
            if nm:
                owners.append(nm)
            rel_el = _child(ch, "reportingOwnerRelationship")
            if rel_el is not None:
                title = _child_text(rel_el, "officerTitle")
                if title:
                    rels.append(title)
                else:
                    roles: List[str] = []
                    if _child_text(rel_el, "isOfficer").lower() in ("1", "true"):
                        roles.append("Officer")
                    if _child_text(rel_el, "isDirector").lower() in ("1", "true"):
                        roles.append("Director")
                    if _child_text(rel_el, "isTenPercentOwner").lower() in ("1", "true"):
                        roles.append("10% Owner")
                    if roles:
                        rels.append("/".join(roles))

        insider_name = ", ".join(owners) if owners else "Insider"
        relationship = ", ".join(dict.fromkeys(rels)) if rels else "Officer/Director"
        root_10b5_1 = _child_text(root, "aff10b5One").lower() in ("1", "true")
        orig_sub_dt = _child_text(root, "dateOfOriginalSubmission")[:10]
        remarks_txt = _child_text(root, "remarks").lower()

        footnotes_by_id: Dict[str, str] = {}
        fn_table = _child(root, "footnotes")
        if fn_table is not None:
            for fn in list(fn_table):
                if _lname(str(fn.tag)) == "footnote":
                    fid = str(fn.attrib.get("id") or "").strip()
                    if fid:
                        footnotes_by_id[fid] = "".join(fn.itertext()).lower()

        deriv_table = _child(root, "derivativeTable")
        deriv_tx_count = (
            sum(1 for el in list(deriv_table) if _lname(str(el.tag)) == "derivativeTransaction")
            if deriv_table is not None
            else 0
        )

        nd_table = _child(root, "nonDerivativeTable")
        if nd_table is None:
            return []

        nd_tx_elems = [tx for tx in list(nd_table) if _lname(str(tx.tag)) == "nonDerivativeTransaction"]
        total_tx_count = len(nd_tx_elems) + deriv_tx_count

        rows: List[Dict[str, Any]] = []
        for tx_idx, tx in enumerate(nd_tx_elems):
            tx_code = _child_text(tx, "transactionCoding", "transactionCode").upper()
            if not tx_code:
                continue
            sec_title = _child_text(tx, "securityTitle", "value")
            raw_shares = _child_text(tx, "transactionAmounts", "transactionShares", "value")
            try:
                shares_f = float(raw_shares)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(shares_f) or shares_f < 0.0:
                continue

            tx_fids = [
                str(el.attrib.get("id") or "").strip()
                for el in tx.iter()
                if _lname(str(el.tag)) == "footnoteId"
            ]
            tx_note_txt = " ".join(footnotes_by_id.get(fid, "") for fid in tx_fids).strip()

            def _classify_ev(txt: str, is_zero_sh: bool = False) -> Tuple[bool, bool, bool, bool, bool]:
                if not txt or str(form or "").upper() != "4/A":
                    return False, False, False, False, False
                txt_aff_cancel = re.sub(
                    r"\b(?:was|were|is|are|has|have|had)?\s*(?:not|never|no|neither|nor|n't)\s+(?:been\s+)?(?:cancel(?:s|ed|led|ing|lation)?|rescind(?:s|ed|ing)?|void(?:s|ed)?|withdrawn)\b",
                    " ",
                    txt,
                    flags=re.IGNORECASE,
                )
                cancel_core = bool(
                    re.search(
                        r"\b(?:cancel(?:s|ed|led|ing|lation)?|rescind(?:s|ed|ing)?|void(?:s|ed)?|withdrawn|never\s+(?:occurred|took\s+place|executed)|did\s+not\s+occur)\b",
                        txt_aff_cancel,
                        flags=re.IGNORECASE,
                    )
                )
                zero_err = bool(
                    is_zero_sh
                    and re.search(
                        r"\b(?:reported\s+in\s+error|filed\s+in\s+error|erroneously\s+reported)\b",
                        txt,
                        flags=re.IGNORECASE,
                    )
                )
                row_corr = bool(
                    re.search(
                        r"\b(?:correct(?:ed|ing|s)?\s+(?:details|the\s+(?:previously\s+)?(?:reported\s+)?(?:transaction|date|price|shares|share\s+count|amount|code|number|ownership|column\s*\d+)|(?:previously\s+)?(?:reported\s+)?(?:transaction|price|shares|share\s+count|date|amount|code|column\s*\d+))|correction\s+(?:to|of)\s+(?:the\s+)?(?:previously\s+)?(?:reported\s+)?(?:transaction|price|shares|share\s+count|date|amount|code|number)|restate(?:s|d|ment)?|supersede(?:s|d)?|amend(?:ed|ing|s)?\s+(?:the\s+)?(?:previously\s+reported\s+)?(?:transaction(?:\s+(?:date|code|price|shares|amount))?|price|shares|share\s+count|date|table|entry|number)|(?:excess\s+shares\s+were\s+)?omitted\s+from\s+(?:the\s+)?(?:(?:date|price|share|shares|amount|code)\s+column|column\s*\d+))\b",
                        txt,
                        flags=re.IGNORECASE,
                    )
                )
                gen_corr = bool(
                    re.search(
                        r"\b(?:correct(?:ed|ing|s)?|correction|restate(?:s|d|ment)?|supersede(?:s|d)?)\b",
                        txt,
                        flags=re.IGNORECASE,
                    )
                )
                omit_ev = bool(
                    re.search(
                        r"\b(?:(?:inadvertently\s+)?(?:omitted|excluded)|previously\s+unreported|add\s+a\s+transaction)\b",
                        txt,
                        flags=re.IGNORECASE,
                    )
                )
                is_cancel = bool((cancel_core or zero_err) and (is_zero_sh or (not omit_ev and not gen_corr)))
                is_add = bool(omit_ev and not row_corr and not is_cancel)
                is_corr = bool((row_corr or gen_corr) and not is_add and not is_cancel)
                is_date = bool((is_corr or is_cancel) and re.search(r"\bdate\b", txt, flags=re.IGNORECASE))
                kinds = int(bool(omit_ev)) + int(bool((row_corr or gen_corr) and not is_cancel)) + int(bool(cancel_core or zero_err))
                is_mixed = bool(kinds > 1 and not tx_note_txt)
                return is_add, is_corr, is_cancel, is_date, is_mixed

            add_ev, corr_ev, cancel_ev, date_corr_ev, _ = _classify_ev(tx_note_txt, is_zero_sh=(shares_f == 0.0))
            mixed_filing_remarks = False
            if not (add_ev or corr_ev or cancel_ev) and remarks_txt:
                r_add, r_corr, r_cancel, r_date, r_mixed = _classify_ev(remarks_txt, is_zero_sh=(shares_f == 0.0))
                if shares_f == 0.0 and r_cancel:
                    cancel_ev, date_corr_ev = True, r_date
                elif total_tx_count > 1 and (r_mixed or r_cancel):
                    mixed_filing_remarks = True
                else:
                    add_ev, corr_ev, cancel_ev, date_corr_ev = r_add, r_corr, r_cancel, r_date

            if shares_f == 0.0 and not (str(form or "").upper() == "4/A" and cancel_ev):
                continue

            raw_price = _child_text(tx, "transactionAmounts", "transactionPricePerShare", "value")
            price_f = 0.0
            if raw_price:
                try:
                    cand_p = float(raw_price)
                    if math.isfinite(cand_p) and cand_p >= 0.0:
                        price_f = cand_p
                except (TypeError, ValueError):
                    price_f = 0.0

            tx_fn_10b5_1 = any(
                re.search(r"\b10b5[\-\s]*1\b", footnotes_by_id.get(fid, ""))
                and not re.search(
                    r"\bnot\s+(?:under|pursuant\s+to|subject\s+to|a)\s+(?:rule\s+)?10b5[\-\s]*1\b",
                    footnotes_by_id.get(fid, ""),
                )
                for fid in tx_fids
            )
            row_10b5_1 = _child_text(tx, "aff10b5One").lower() in ("1", "true")
            tx_10b5_1 = bool(tx_fn_10b5_1 or row_10b5_1 or (root_10b5_1 and total_tx_count == 1))
            tx_dt = _child_text(tx, "transactionDate", "value") or str(filing_date)[:10]
            raw_val = shares_f * price_f
            action_fields: Dict[str, str] = {}
            if cancel_ev:
                action_fields["amendment_action"] = "CANCEL"
            elif add_ev:
                action_fields["amendment_action"] = "ADD"
            elif corr_ev:
                action_fields["amendment_action"] = "CORRECT"
            rows.append({
                "form": str(form or "4").upper(),
                "filing_date": str(filing_date)[:10],
                "transaction_date": str(tx_dt)[:10],
                "date_of_original_submission": orig_sub_dt,
                "disclosed_at": str(accept_dt),
                "disclosed_at_verified": True,
                "accession_number": str(accn),
                "tx_index": tx_idx,
                "insider_name": insider_name,
                "relationship": relationship,
                **({"security_title": sec_title} if sec_title else {}),
                "transaction_code": tx_code,
                "shares": int(round(shares_f)),
                "price": round(price_f, 2),
                "value": round(raw_val, 2) if math.isfinite(raw_val) else None,
                "filing_has_10b5_1": bool(root_10b5_1),
                "is_10b5_1": bool(tx_10b5_1),
                **action_fields,
                **({"corrects_date": True} if date_corr_ev else {}),
                **({"_mixed_remarks_ambiguous": True} if mixed_filing_remarks else {}),
                "source": "SEC_FORM4_XML",
            })
        return rows

    @staticmethod
    def resolve_pit_form4_transactions(txs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolves Form 4/A amendments chronologically over already PIT-eligible rows so an
        amended filing replaces only the corrected transaction(s) against a snapshot of
        prior rows, preserves original filing identity across chained amendments, and
        flags ambiguous code-changing amendments without counting duplicate trades.
        """
        ordered: List[Tuple[int, str, int, Dict[str, Any]]] = []
        for idx, row in enumerate(txs):
            if not isinstance(row, dict):
                continue
            disc_raw = row.get("disclosed_at") or row.get("acceptance_datetime") or ""
            disc_us = FundamentalsClient._parse_iso_utc_us(disc_raw)
            sort_us = disc_us if disc_us is not None else -1
            accn = str(row.get("accession_number") or row.get("filing_date") or "")
            tx_i = int(row.get("tx_index") if isinstance(row.get("tx_index"), int) else idx)
            ordered.append((sort_us, accn, tx_i, row))
        ordered.sort(key=lambda item: (item[0], item[1], item[2]))

        resolved: List[Dict[str, Any]] = []
        # Group by (sort_us, accn) so all rows of a single Form 4/A match against the same prior snapshot
        i = 0
        while i < len(ordered):
            batch_accn = ordered[i][1]
            batch_us = ordered[i][0]
            j = i + 1
            while j < len(ordered) and ordered[j][0] == batch_us and ordered[j][1] == batch_accn:
                j += 1
            batch_rows = [dict(ordered[k][3]) for k in range(i, j)]
            for row_b in batch_rows:
                row_b.pop("_amendment_unresolved", None)
            i = j

            is_amend_batch = any(str(r.get("form") or "4").strip().upper() == "4/A" for r in batch_rows)
            if not is_amend_batch:
                for r in batch_rows:
                    r.setdefault("original_filing_date", str(r.get("filing_date") or "")[:10])
                    resolved.append(r)
                continue

            prior_snapshot = list(resolved)
            used_slots: set[int] = set()
            pending_replacements: Dict[int, Dict[str, Any]] = {}
            cancelled_slots: set[int] = set()
            unmatched_rows: List[Dict[str, Any]] = []

            def _norm_sec_title(val: Any) -> str:
                return " ".join(str(val or "").strip().lower().split())

            def _cand_base_matches(cand: Dict[str, Any], row: Dict[str, Any], accn: str) -> bool:
                if cand.get("_amendment_ambiguous") or cand.get("_amendment_unresolved"):
                    return False
                cand_accn = str(cand.get("accession_number") or "").strip()
                if accn and cand_accn == accn:
                    return False
                insider_ok = (
                    str(cand.get("insider_name") or "").strip().lower()
                    == str(row.get("insider_name") or "").strip().lower()
                )
                tx_dt_ok = (
                    str(cand.get("transaction_date") or cand.get("filing_date") or "")[:10]
                    == str(row.get("transaction_date") or row.get("filing_date") or "")[:10]
                )
                if not (insider_ok and tx_dt_ok):
                    return False
                r_st = _norm_sec_title(row.get("security_title"))
                c_st = _norm_sec_title(cand.get("security_title"))
                if r_st and c_st and r_st != c_st:
                    return False
                orig_sub = str(row.get("date_of_original_submission") or "")[:10]
                if orig_sub:
                    cand_orig_fd = str(cand.get("original_filing_date") or cand.get("filing_date") or "")[:10]
                    return cand_orig_fd == orig_sub
                return True

            # Pass 1: exact (insider_name, transaction_date, orig_sub, transaction_code, security_title) match;
            # replace only when unique or when batch row count matches candidate count identically.
            def _row_exact_key(row: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
                return (
                    str(row.get("insider_name") or "").strip().lower(),
                    str(row.get("transaction_date") or row.get("filing_date") or "")[:10],
                    str(row.get("date_of_original_submission") or "")[:10],
                    str(row.get("transaction_code") or "").strip().upper(),
                    _norm_sec_title(row.get("security_title")),
                )

            amend_rows = [
                r
                for r in batch_rows
                if str(r.get("form") or "4").strip().upper() == "4/A"
                and str(r.get("amendment_action") or "").strip().upper() != "ADD"
                and not r.get("_mixed_remarks_ambiguous")
            ]
            exact_key_totals: Dict[Tuple[str, str, str, str, str], int] = {}
            for r in amend_rows:
                ek = _row_exact_key(r)
                exact_key_totals[ek] = exact_key_totals.get(ek, 0) + 1
            ambiguous_exact_keys: set[Tuple[str, str, str, str, str]] = set()
            deferred_date_corr_keys: set[Tuple[str, str, str, str, str]] = set()
            key_slots_planned: Dict[Tuple[str, str, str, str, str], List[int]] = {}
            claimed_in_plan: set[int] = set()
            ordered_amend_rows = [x for x in amend_rows if _row_exact_key(x)[4]] + [
                x for x in amend_rows if not _row_exact_key(x)[4]
            ]
            for r in ordered_amend_rows:
                ek = _row_exact_key(r)
                if ek in key_slots_planned or ek in ambiguous_exact_keys or ek in deferred_date_corr_keys:
                    continue
                accn_r = str(r.get("accession_number") or "").strip()
                r_code = ek[3]
                orig_sub_r = ek[2]
                r_tx_dt = ek[1]
                r_st_ek = ek[4]
                cands = [
                    slot_idx
                    for slot_idx, cand in enumerate(prior_snapshot)
                    if slot_idx not in used_slots
                    and slot_idx not in claimed_in_plan
                    and str(cand.get("transaction_code") or "").strip().upper() == r_code
                    and _cand_base_matches(cand, r, accn_r)
                ]
                if r_st_ek:
                    exact_st_cands = [s for s in cands if _norm_sec_title(prior_snapshot[s].get("security_title")) == r_st_ek]
                    if exact_st_cands:
                        cands = exact_st_cands
                other_date_orig_cands = [
                    slot_idx
                    for slot_idx, cand in enumerate(prior_snapshot)
                    if slot_idx not in used_slots
                    and slot_idx not in claimed_in_plan
                    and not cand.get("_amendment_ambiguous")
                    and not cand.get("_amendment_unresolved")
                    and not (accn_r and str(cand.get("accession_number") or "").strip() == accn_r)
                    and str(cand.get("insider_name") or "").strip().lower() == ek[0]
                    and str(cand.get("transaction_code") or "").strip().upper() == r_code
                    and (not r_st_ek or not _norm_sec_title(cand.get("security_title")) or _norm_sec_title(cand.get("security_title")) == r_st_ek)
                    and str(cand.get("original_filing_date") or cand.get("filing_date") or "")[:10] == orig_sub_r
                    and str(cand.get("transaction_date") or cand.get("filing_date") or "")[:10] != r_tx_dt
                ] if orig_sub_r else []
                r_sh_raw = r.get("shares")
                r_sh_f = (
                    float(r_sh_raw)
                    if isinstance(r_sh_raw, (int, float)) and not isinstance(r_sh_raw, bool) and float(r_sh_raw) > 0.0
                    else None
                )
                same_date_shares_conflict = bool(
                    r_sh_f is not None
                    and cands
                    and all(
                        not isinstance(prior_snapshot[s].get("shares"), (int, float))
                        or not math.isclose(float(prior_snapshot[s]["shares"]), r_sh_f, rel_tol=1e-6, abs_tol=1e-6)
                        for s in cands
                    )
                    and any(
                        isinstance(prior_snapshot[s].get("shares"), (int, float))
                        and math.isclose(float(prior_snapshot[s]["shares"]), r_sh_f, rel_tol=1e-6, abs_tol=1e-6)
                        for s in other_date_orig_cands
                    )
                )
                if other_date_orig_cands and (r.get("corrects_date") or (str(r.get("amendment_action") or "").upper() in ("CORRECT", "CANCEL") and same_date_shares_conflict)):
                    deferred_date_corr_keys.add(ek)
                    continue
                k_total = exact_key_totals.get(ek, 0)
                if len(cands) > 1 and k_total != len(cands):
                    ambiguous_exact_keys.add(ek)
                elif len(cands) >= 1:
                    chosen_cands = cands[:k_total]
                    key_slots_planned[ek] = chosen_cands
                    claimed_in_plan.update(chosen_cands)

            def _replace_slot(s_idx: int, amend_row: Dict[str, Any]) -> None:
                orig_cand = prior_snapshot[s_idx]
                orig_fd = str(orig_cand.get("original_filing_date") or orig_cand.get("filing_date") or "")[:10]
                rep_row = {**amend_row, "original_filing_date": orig_fd}
                # Preserve original disclosure PIT effective date for decay aging when replacement amends an earlier filing
                if "_tx_eff_d" in orig_cand and orig_cand["_tx_eff_d"] is not None:
                    rep_row["_tx_eff_d"] = orig_cand["_tx_eff_d"]
                pending_replacements[s_idx] = rep_row

            ambig_out_rows: List[Dict[str, Any]] = []
            direct_add_rows: List[Dict[str, Any]] = []
            for r in batch_rows:
                form_u = str(r.get("form") or "4").strip().upper()
                if form_u != "4/A":
                    unmatched_rows.append(r)
                    continue
                if str(r.get("amendment_action") or "").strip().upper() == "ADD":
                    r.setdefault("original_filing_date", str(r.get("filing_date") or "")[:10])
                    direct_add_rows.append(r)
                    continue
                ek = _row_exact_key(r)
                if ek in ambiguous_exact_keys:
                    ambig_out_rows.append({**r, "_amendment_ambiguous": True})
                    continue
                if ek in deferred_date_corr_keys:
                    unmatched_rows.append({**r, "_deferred_date_corr": True})
                    continue
                if r.pop("_mixed_remarks_ambiguous", False):
                    ambig_out_rows.append({**r, "_amendment_ambiguous": True})
                    continue
                planned = key_slots_planned.get(ek)
                if planned:
                    matched_slot = planned.pop(0)
                    used_slots.add(matched_slot)
                    if str(r.get("amendment_action") or "").strip().upper() == "CANCEL":
                        cancelled_slots.add(matched_slot)
                    else:
                        _replace_slot(matched_slot, r)
                else:
                    unmatched_rows.append(r)

            # Pass 2: plan code-tolerant fallback across all remaining 4/A rows before claiming so order never matters
            appended_rows: List[Dict[str, Any]] = list(direct_add_rows) + list(ambig_out_rows)
            pass2_candidates: List[Tuple[Dict[str, Any], List[int]]] = []
            slot_claim_counts: Dict[int, int] = {}
            for r in unmatched_rows:
                form_u = str(r.get("form") or "4").strip().upper()
                accn = str(r.get("accession_number") or "").strip()
                if form_u != "4/A" or r.get("_deferred_date_corr") or r.get("corrects_date"):
                    if form_u != "4/A":
                        r.setdefault("original_filing_date", str(r.get("filing_date") or "")[:10])
                    pass2_candidates.append((r, []))
                    continue
                alt_slots = [
                    slot_idx
                    for slot_idx, cand in enumerate(prior_snapshot)
                    if slot_idx not in used_slots and _cand_base_matches(cand, r, accn)
                ]
                for s_idx in alt_slots:
                    slot_claim_counts[s_idx] = slot_claim_counts.get(s_idx, 0) + 1
                pass2_candidates.append((r, alt_slots))

            pass2_unmatched_4a: List[Dict[str, Any]] = []
            for r, alt_slots in pass2_candidates:
                form_u = str(r.get("form") or "4").strip().upper()
                if form_u != "4/A":
                    appended_rows.append(r)
                    continue
                if len(alt_slots) == 1 and slot_claim_counts.get(alt_slots[0], 0) == 1:
                    s_idx = alt_slots[0]
                    used_slots.add(s_idx)
                    if str(r.get("amendment_action") or "").strip().upper() == "CANCEL":
                        cancelled_slots.add(s_idx)
                    else:
                        _replace_slot(s_idx, r)
                elif alt_slots:
                    appended_rows.append({**r, "_amendment_ambiguous": True})
                else:
                    pass2_unmatched_4a.append(r)

            # Pass 3: match explicit date-corrected 4/A rows with original submission date across differing transaction_date
            pass3_candidates: List[Tuple[Dict[str, Any], List[int]]] = []
            pass3_claim_counts: Dict[int, int] = {}
            for r in pass2_unmatched_4a:
                orig_sub = str(r.get("date_of_original_submission") or "")[:10]
                has_date_corr_ev = bool(r.get("corrects_date") or r.get("_deferred_date_corr"))
                if not has_date_corr_ev or (str(r.get("amendment_action") or "").strip().upper() not in ("CORRECT", "CANCEL") and not r.get("_deferred_date_corr")) or not orig_sub:
                    r.pop("_deferred_date_corr", None)
                    r.setdefault("original_filing_date", str(r.get("filing_date") or "")[:10])
                    pass3_candidates.append((r, []))
                    continue
                r.pop("_deferred_date_corr", None)
                accn_r = str(r.get("accession_number") or "").strip()
                r_ins = str(r.get("insider_name") or "").strip().lower()
                r_code = str(r.get("transaction_code") or "").strip().upper()
                r_st_p3 = _norm_sec_title(r.get("security_title"))
                base_slots = [
                    slot_idx
                    for slot_idx, cand in enumerate(prior_snapshot)
                    if slot_idx not in used_slots
                    and not cand.get("_amendment_ambiguous")
                    and not cand.get("_amendment_unresolved")
                    and not (accn_r and str(cand.get("accession_number") or "").strip() == accn_r)
                    and str(cand.get("insider_name") or "").strip().lower() == r_ins
                    and (not r_st_p3 or not _norm_sec_title(cand.get("security_title")) or _norm_sec_title(cand.get("security_title")) == r_st_p3)
                    and str(cand.get("original_filing_date") or cand.get("filing_date") or "")[:10] == orig_sub
                ]
                code_slots = [
                    s_idx
                    for s_idx in base_slots
                    if r_code and str(prior_snapshot[s_idx].get("transaction_code") or "").strip().upper() == r_code
                ]
                st_matches_p3 = (
                    [s_idx for s_idx in base_slots if _norm_sec_title(prior_snapshot[s_idx].get("security_title")) == r_st_p3]
                    if r_st_p3
                    else []
                )
                r_sh_raw = r.get("shares")
                r_sh_f = (
                    float(r_sh_raw)
                    if isinstance(r_sh_raw, (int, float)) and not isinstance(r_sh_raw, bool) and float(r_sh_raw) > 0.0
                    else None
                )
                sh_matches_all = (
                    [
                        s_idx
                        for s_idx in base_slots
                        if isinstance(prior_snapshot[s_idx].get("shares"), (int, float))
                        and not isinstance(prior_snapshot[s_idx].get("shares"), bool)
                        and math.isclose(float(prior_snapshot[s_idx]["shares"]), r_sh_f, rel_tol=1e-6, abs_tol=1e-6)
                    ]
                    if r_sh_f is not None
                    else []
                )
                if len(base_slots) <= 1:
                    cands_3 = base_slots
                elif st_matches_p3 and code_slots and not any(s in code_slots for s in st_matches_p3):
                    cands_3 = list(dict.fromkeys(code_slots + st_matches_p3))
                elif code_slots and sh_matches_all:
                    joint = [s_idx for s_idx in code_slots if s_idx in sh_matches_all]
                    if st_matches_p3:
                        joint_st = [s_idx for s_idx in joint if s_idx in st_matches_p3]
                        if joint_st:
                            joint = joint_st
                    cands_3 = joint if joint else list(dict.fromkeys(code_slots + sh_matches_all))
                elif st_matches_p3 and sh_matches_all:
                    joint_ss = [s_idx for s_idx in st_matches_p3 if s_idx in sh_matches_all]
                    cands_3 = joint_ss if joint_ss else list(dict.fromkeys(st_matches_p3 + sh_matches_all))
                elif st_matches_p3:
                    cands_3 = st_matches_p3
                elif code_slots:
                    cands_3 = code_slots
                elif sh_matches_all:
                    cands_3 = sh_matches_all
                else:
                    cands_3 = base_slots
                for s_idx in cands_3:
                    pass3_claim_counts[s_idx] = pass3_claim_counts.get(s_idx, 0) + 1
                pass3_candidates.append((r, cands_3))

            for r, cands_3 in pass3_candidates:
                if len(cands_3) == 1 and pass3_claim_counts.get(cands_3[0], 0) == 1:
                    s_idx = cands_3[0]
                    used_slots.add(s_idx)
                    if str(r.get("amendment_action") or "").strip().upper() == "CANCEL":
                        cancelled_slots.add(s_idx)
                    else:
                        _replace_slot(s_idx, r)
                elif cands_3:
                    appended_rows.append({**r, "_amendment_ambiguous": True})
                elif str(r.get("amendment_action") or "").strip().upper() in ("CORRECT", "CANCEL"):
                    appended_rows.append({**r, "_amendment_unresolved": True})
                else:
                    r.setdefault("original_filing_date", str(r.get("filing_date") or "")[:10])
                    appended_rows.append(r)

            resolved = [
                pending_replacements.get(slot_idx, row)
                for slot_idx, row in enumerate(resolved)
                if slot_idx not in cancelled_slots
            ]
            resolved.extend(appended_rows)

        return resolved

    def _load_or_fetch_sec_form4_rows(
        self,
        cik: str,
        subs: Dict[str, Any],
        enable_live_fetch: bool = False,
        max_filings: int = 12,
        rpc_budget_secs: float = 3.5,
    ) -> List[Dict[str, Any]]:
        """Loads cached or bounded live SEC Form 4 XML transactions carrying authoritative SEC acceptanceDateTime."""
        cik10 = str(cik).strip().zfill(10)
        rec = subs.get("recent")
        if not isinstance(rec, Mapping):
            return []
        forms = self._recent_col(rec, "form")
        fdates = self._recent_col(rec, "filingDate")
        accs = self._recent_col(rec, "acceptanceDateTime")
        accns = self._recent_col(rec, "accessionNumber")
        docs = self._recent_col(rec, "primaryDocument")
        n_items = min(len(forms), len(fdates), len(accns), len(docs))

        rel_f4 = f"edgar_form4/CIK{cik10}.json"
        cache_obj = self._read_dict_cache(rel_f4, None) or {}
        raw_map = cache_obj.get("filings")
        filings_map: Dict[str, Any] = dict(raw_map) if isinstance(raw_map, dict) else {}

        all_valid: List[Tuple[str, str, str, str, str]] = []
        for i in range(n_items):
            form = str(forms[i] or "").strip().upper()
            if form not in ("4", "4/A"):
                continue
            acc_dt = str(accs[i] or "").strip() if i < len(accs) else ""
            accn = str(accns[i] or "").strip()
            doc = str(docs[i] or "").strip()
            fdt = str(fdates[i] or "")[:10]
            if not accn or not doc or self._parse_iso_utc_us(acc_dt) is None:
                continue
            raw_doc = re.sub(r"^xsl[^/]+/", "", doc.lstrip("/"))
            if not raw_doc.lower().endswith(".xml"):
                continue
            all_valid.append((form, fdt, acc_dt, accn, raw_doc))

        candidates = list(all_valid[:max_filings])
        if any(f == "4/A" for f, _, _, _, _ in candidates) and len(all_valid) > max_filings:
            for item in all_valid[max_filings:]:
                accn_old = item[3]
                ent_old = filings_map.get(accn_old)
                if isinstance(ent_old, dict) and isinstance(ent_old.get("rows"), list):
                    candidates.append(item)

        if not candidates:
            return []
        newly_fetched: Dict[str, Any] = {}
        out_rows: List[Dict[str, Any]] = []
        fetch_deadline = time.monotonic() + max(0.5, float(rpc_budget_secs))
        cold_fetches = 0
        max_cold_fetches = 4

        prev_dl = getattr(self._sec_tl, "deadline", None)
        self._sec_tl.deadline = fetch_deadline
        try:
            for form, fdt, acc_dt, accn, raw_doc in candidates:
                ent = filings_map.get(accn)
                if isinstance(ent, dict) and isinstance(ent.get("rows"), list):
                    for row in ent["rows"]:
                        if isinstance(row, dict):
                            out_rows.append({
                                **row,
                                "form": row.get("form") or form,
                                "disclosed_at": acc_dt,
                                "disclosed_at_verified": True,
                                "accession_number": accn,
                            })
                    continue
                remaining = fetch_deadline - time.monotonic()
                if (
                    not enable_live_fetch
                    or self._is_in_cooldown(f"f4:{accn}")
                    or cold_fetches >= max_cold_fetches
                    or remaining <= 0.4
                ):
                    continue
                cold_fetches += 1
                url = f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/{accn.replace('-', '')}/{raw_doc}"
                xml_txt = self._sec_fetch_text(url, timeout=min(2.5, remaining))
                if not xml_txt:
                    if not getattr(self._sec_tl, "last_budget_expired", False):
                        self._mark_failed(f"f4:{accn}", seconds=300.0)
                    continue
                parsed_rows = self._parse_sec_form4_xml(xml_txt, fdt, acc_dt, accn, form=form)
                if parsed_rows is None:
                    self._mark_failed(f"f4:{accn}", seconds=300.0)
                    continue
                newly_fetched[accn] = {
                    "form": form,
                    "filing_date": fdt,
                    "acceptance_datetime": acc_dt,
                    "rows": parsed_rows,
                }
                out_rows.extend(parsed_rows)
        finally:
            self._sec_tl.deadline = prev_dl

        if newly_fetched:
            with self._lock:
                latest_obj = self._read_dict_cache(rel_f4, None) or {}
                latest_raw = latest_obj.get("filings")
                merged_map: Dict[str, Any] = dict(latest_raw) if isinstance(latest_raw, dict) else {}
                merged_map.update(newly_fetched)
                if len(merged_map) > 64:
                    keep_keys = list(merged_map.keys())[-64:]
                    merged_map = {k: merged_map[k] for k in keep_keys}
                self._write_json_cache(rel_f4, {"cik": cik10, "filings": merged_map})

        return out_rows

    def build_edgar_filings_payload(
        self,
        ticker: str,
        known_cik: Optional[str] = None,
        enable_live_fetch: bool = False,
    ) -> Dict[str, Any]:
        """Builds authentic SEC EDGAR filing index, PIT cutoff, real section diffs (if fetched), and Form 4 rows."""
        sym = ticker.upper().strip()
        cik = self.resolve_cik(sym, known_cik=known_cik, enable_live_fetch=enable_live_fetch)
        subs = self.get_sec_submissions(cik, enable_live_fetch=enable_live_fetch) if cik else None
        yf_meta = self.get_yf_meta(sym, enable_live_fetch=enable_live_fetch)

        filings_10: List[Dict[str, str]] = []
        if subs and isinstance(subs.get("recent"), Mapping):
            rec = subs["recent"]
            forms = self._recent_col(rec, "form")
            fdates = self._recent_col(rec, "filingDate")
            accs = self._recent_col(rec, "acceptanceDateTime")
            accns = self._recent_col(rec, "accessionNumber")
            docs = self._recent_col(rec, "primaryDocument")
            n_items = min(len(forms), len(fdates), len(accns), len(docs))
            for i in range(n_items):
                form = forms[i]
                fdt = fdates[i]
                acc_dt = accs[i] if i < len(accs) else ""
                accn = accns[i]
                doc = docs[i]
                if form in ("10-K", "10-Q") and accn and doc:
                    filings_10.append({
                        "form": str(form),
                        "filing_date": str(fdt)[:10],
                        "acceptance_datetime": str(acc_dt or ""),
                        "accession_number": str(accn),
                        "primary_document": str(doc),
                    })
                if len(filings_10) >= 2:
                    break

        edgar_deadline = time.monotonic() + 6.5
        form4_rows: List[Dict[str, Any]] = (
            self._load_or_fetch_sec_form4_rows(
                cik,
                subs,
                enable_live_fetch=enable_live_fetch,
                rpc_budget_secs=min(3.0, max(0.5, edgar_deadline - time.monotonic())),
            )
            if (cik and subs and subs.get("recent"))
            else []
        )
        if not form4_rows and yf_meta and isinstance(yf_meta.get("insider_transactions"), list):
            for tx in yf_meta["insider_transactions"]:
                if tx.get("shares", 0) > 0:
                    row = {k: v for k, v in tx.items() if k != "raw_text"}
                    if tx.get("raw_text"):
                        row["transaction_code"] = self.classify_insider_tx(str(tx["raw_text"]))
                    form4_rows.append(row)

        if not filings_10:
            return {
                "ticker": sym,
                "available": bool(form4_rows),
                "filing_type": "UNAVAILABLE" if not form4_rows else "INSIDER_ONLY",
                "current_filing_date": None,
                "prior_filing_date": None,
                "acceptance_timestamp_et": "Unavailable offline",
                "item1a_similarity": None,
                "mda_similarity": None,
                "item1a_sentiment_drift": None,
                "mda_catalyst_score": None,
                "redlines": [
                    {
                        "section": "SEC EDGAR Filing Index",
                        "status": "UNAVAILABLE_OFFLINE" if not enable_live_fetch else "NO_10K_10Q_FILINGS",
                        "added_clauses": [
                            f"SEC EDGAR filing disclosures unavailable for {sym} (offline mode or non-reporting ETF)."
                        ],
                        "removed_clauses": [],
                    }
                ],
                "form4_transactions": form4_rows,
            }

        curr_f = filings_10[0]
        prior_f = filings_10[1] if len(filings_10) > 1 else None
        acc_raw = curr_f["acceptance_datetime"]
        acc_us = self._parse_iso_utc_us(acc_raw) if acc_raw else None
        if acc_us is not None:
            dt_utc = datetime.fromtimestamp(acc_us / 1_000_000.0, tz=timezone.utc)
            dt_ny = dt_utc.astimezone(ZoneInfo("America/New_York"))
            cutoff_str = MarketDataLoader.nyse_session_close_time(dt_ny.date()).strftime("%H:%M")
            pit_dt = MarketDataLoader.apply_pit_cutoff(dt_utc, cutoff_time=cutoff_str)
            shifted_note = f"{dt_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC -> PIT Effective {pit_dt.strftime('%Y-%m-%d')}"
        else:
            shifted_note = f"{curr_f['filing_date']} (SEC Filing Date)"

        # Cached section redline comparison keyed and validated by both filing accessions
        curr_acc = curr_f["accession_number"]
        prior_acc = prior_f["accession_number"] if prior_f else None
        rel_redline = f"edgar_redlines/{sym}_{curr_acc}_{prior_acc or 'NONE'}.json"

        def _valid_pair_cache(payload: Any) -> Optional[Dict[str, Any]]:
            if not isinstance(payload, dict):
                return None
            if payload.get("current_accession") != curr_acc or payload.get("prior_accession") != prior_acc:
                return None
            if not isinstance(payload.get("redlines"), list):
                return None
            return payload

        def _try_verified_legacy_redline(ttl_secs: Optional[float]) -> Optional[Dict[str, Any]]:
            if prior_f is None:
                return None
            legacy_rel = f"edgar_redlines/{sym}_{curr_acc}.json"
            raw_leg = (
                self._read_json_cache(legacy_rel, ttl_secs)
                if ttl_secs is not None
                else self._read_stale_json_cache(legacy_rel)
            )
            # Only honor single-accession legacy files that explicitly embed both pair accessions;
            # never infer prior_accession from form/date heuristics and never refresh expired TTLs.
            return _valid_pair_cache(raw_leg)

        cached_diff = _valid_pair_cache(self._read_json_cache(rel_redline, self._REDLINE_TTL_SECS))
        if cached_diff is None:
            cached_diff = _try_verified_legacy_redline(self._REDLINE_TTL_SECS)
        if (
            cached_diff is None
            and enable_live_fetch
            and cik
            and prior_f is not None
            and not self._is_in_cooldown(rel_redline)
            and (edgar_deadline - time.monotonic()) >= 0.8
        ):
            curr_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{curr_f['accession_number'].replace('-', '')}/{curr_f['primary_document']}"
            prior_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{prior_f['accession_number'].replace('-', '')}/{prior_f['primary_document']}"
            per_doc_t = min(3.0, max(0.5, (edgar_deadline - time.monotonic()) * 0.5))
            prev_dl_red = getattr(self._sec_tl, "deadline", None)
            self._sec_tl.deadline = edgar_deadline
            redline_budget_hit = False
            try:
                html_curr = self._sec_fetch_text(curr_url, timeout=per_doc_t)
                curr_budget_expired = bool(getattr(self._sec_tl, "last_budget_expired", False))
                rem_after_curr = edgar_deadline - time.monotonic()
                if html_curr and rem_after_curr >= 0.4:
                    html_prior = self._sec_fetch_text(prior_url, timeout=min(3.0, rem_after_curr))
                    prior_budget_expired = bool(getattr(self._sec_tl, "last_budget_expired", False))
                else:
                    html_prior = None
                    prior_budget_expired = bool(html_curr is not None and rem_after_curr < 0.4)
                redline_budget_hit = bool((html_curr is None and curr_budget_expired) or (html_prior is None and prior_budget_expired))
            finally:
                self._sec_tl.deadline = prev_dl_red
            if html_curr and html_prior:
                s1_curr = self._extract_html_section_sentences(html_curr, "item1a")
                s1_prior = self._extract_html_section_sentences(html_prior, "item1a")
                s7_curr = self._extract_html_section_sentences(html_curr, "item7")
                s7_prior = self._extract_html_section_sentences(html_prior, "item7")

                item1a_sim = self._bag_cosine_similarity(s1_curr, s1_prior) if (s1_curr and s1_prior) else None
                mda_sim = self._bag_cosine_similarity(s7_curr, s7_prior) if (s7_curr and s7_prior) else None

                # Find genuinely new / removed sentences
                set_prior_1a = set(s1_prior)
                set_curr_1a = set(s1_curr)
                added_1a = [s for s in s1_curr if s not in set_prior_1a and not any(difflib.SequenceMatcher(None, s, p).ratio() > 0.85 for p in s1_prior[:40])][:3]
                removed_1a = [s for s in s1_prior if s not in set_curr_1a and not any(difflib.SequenceMatcher(None, s, c).ratio() > 0.85 for c in s1_curr[:40])][:3]

                set_prior_7 = set(s7_prior)
                set_curr_7 = set(s7_curr)
                added_7 = [s for s in s7_curr if s not in set_prior_7][:3]
                removed_7 = [s for s in s7_prior if s not in set_curr_7][:3]

                # Lexical sentiment drift over Item 1A & Item 7 additions
                pos_w = {"growth", "expanded", "increase", "increased", "strong", "opportunity", "improved", "demand"}
                neg_w = {"adverse", "adversely", "risk", "uncertainty", "decline", "shortfall", "restriction", "tariffs", "litigation"}
                def _lex_score(clauses: List[str]) -> float:
                    if not clauses:
                        return 0.0
                    words = re.findall(r"[a-z]+", " ".join(clauses).lower())
                    if not words:
                        return 0.0
                    pos = sum(1 for w in words if w in pos_w)
                    neg = sum(1 for w in words if w in neg_w)
                    return round(float(np.clip((pos - neg) / max(len(words) * 0.08, 1.0), -1.0, 1.0)), 2)

                cached_diff = {
                    "current_accession": curr_acc,
                    "prior_accession": prior_acc,
                    "item1a_similarity": item1a_sim,
                    "mda_similarity": mda_sim,
                    "item1a_sentiment_drift": _lex_score(added_1a) if added_1a else None,
                    "mda_catalyst_score": _lex_score(added_7) if added_7 else None,
                    "redlines": [
                        {
                            "section": f"Item 1A Risk Factors ({curr_f['form']} {curr_f['filing_date']} vs {prior_f['form']} {prior_f['filing_date']})",
                            "status": "DIFF_VERIFIED" if (added_1a or removed_1a) else ("UNCHANGED" if item1a_sim else "SECTION_HEADER_NOT_FOUND"),
                            "added_clauses": added_1a or [f"Filing verified on SEC EDGAR: {curr_f['form']} Accn {curr_f['accession_number']} (filed {curr_f['filing_date']})."],
                            "removed_clauses": removed_1a,
                        },
                        {
                            "section": f"Item 7 MD&A ({curr_f['form']} {curr_f['filing_date']})",
                            "status": "DIFF_VERIFIED" if (added_7 or removed_7) else ("UNCHANGED" if mda_sim else "SECTION_HEADER_NOT_FOUND"),
                            "added_clauses": added_7,
                            "removed_clauses": removed_7,
                        },
                    ],
                }
                self._write_json_cache(rel_redline, cached_diff)
            elif not redline_budget_hit:
                self._mark_failed(rel_redline, seconds=300.0)
        if cached_diff is None:
            cached_diff = _valid_pair_cache(self._read_stale_json_cache(rel_redline))
        if cached_diff is None:
            cached_diff = _try_verified_legacy_redline(None)

        redlines = (
            cached_diff["redlines"]
            if cached_diff
            else [
                {
                    "section": f"SEC Filing {curr_f['form']} ({curr_f['filing_date']})",
                    "status": "INDEX_VERIFIED",
                    "added_clauses": [
                        f"Official SEC EDGAR Accession {curr_f['accession_number']} (filed {curr_f['filing_date']}). Prior filing: {prior_f['filing_date'] if prior_f else 'N/A'}."
                    ],
                    "removed_clauses": [],
                }
            ]
        )

        return {
            "ticker": sym,
            "available": True,
            "filing_type": curr_f["form"],
            "current_filing_date": curr_f["filing_date"],
            "current_acceptance_datetime": acc_raw if acc_us is not None else None,
            "prior_filing_date": prior_f["filing_date"] if prior_f else None,
            "acceptance_timestamp_et": shifted_note,
            "item1a_similarity": cached_diff.get("item1a_similarity") if cached_diff else None,
            "mda_similarity": cached_diff.get("mda_similarity") if cached_diff else None,
            "item1a_sentiment_drift": cached_diff.get("item1a_sentiment_drift") if cached_diff else None,
            "mda_catalyst_score": cached_diff.get("mda_catalyst_score") if cached_diff else None,
            "redlines": redlines,
            "form4_transactions": form4_rows,
        }
