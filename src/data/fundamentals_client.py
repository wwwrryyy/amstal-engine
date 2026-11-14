"""
SEC EDGAR & Yahoo Finance Fundamentals Client
Provides disk-cached, rate-limited access to official SEC EDGAR submissions,
US-GAAP XBRL CompanyFacts (Point-in-Time vs As-Restated), Yahoo Finance company metadata,
analyst consensus targets/recommendations, and insider Form 4 transactions.
Returns honest unavailable states rather than synthetic fallback numbers when offline.
"""

from datetime import datetime, timezone
import difflib
import json
import logging
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.request

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
        self._lock = threading.RLock()
        self._mem_cache: Dict[str, Tuple[float, Any]] = {}
        self._failed_until: Dict[str, float] = {}
        self._ticker_to_cik_map: Optional[Dict[str, str]] = None

    def _is_in_cooldown(self, key: str) -> bool:
        return time.monotonic() < self._failed_until.get(key, 0.0)

    def _mark_failed(self, key: str, seconds: float = _FAILED_COOLDOWN_SECS) -> None:
        self._failed_until[key] = time.monotonic() + seconds

    def _read_json_cache(self, rel_path: str, max_age_secs: float) -> Optional[Any]:
        path = self.raw_dir / rel_path
        with self._lock:
            mem = self._mem_cache.get(rel_path)
            if mem is not None and (time.time() - mem[0]) < max_age_secs:
                return mem[1]
            if path.exists():
                try:
                    mtime = path.stat().st_mtime
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if (time.time() - mtime) < max_age_secs:
                        self._mem_cache[rel_path] = (mtime, data)
                        return data
                except Exception:
                    return None
        return None

    def _read_stale_json_cache(self, rel_path: str) -> Optional[Any]:
        """Reads existing on-disk JSON regardless of TTL when network is unreachable."""
        path = self.raw_dir / rel_path
        with self._lock:
            if rel_path in self._mem_cache:
                return self._mem_cache[rel_path][1]
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    self._mem_cache[rel_path] = (path.stat().st_mtime, data)
                    return data
                except Exception:
                    return None
        return None

    def _write_json_cache(self, rel_path: str, payload: Any) -> None:
        path = self.raw_dir / rel_path
        with self._lock:
            now = time.time()
            self._mem_cache[rel_path] = (now, payload)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(payload), encoding="utf-8")
                tmp.replace(path)
            except Exception as e:
                logger.debug(f"Failed writing cache {path}: {e}")

    def _sec_fetch_text(self, url: str, timeout: float = 6.0) -> Optional[str]:
        self._sec_limiter.acquire(1.0)
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.sec_ua,
                "Accept": "application/json, text/html, text/plain, */*",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read(6 * 1024 * 1024).decode("utf-8", errors="ignore")
        except Exception as e:
            logger.debug(f"SEC fetch failed ({url}): {e}")
            return None

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

        if self._ticker_to_cik_map is None:
            cached = self._read_json_cache("sec_company_tickers.json", self._TICKER_MAP_TTL_SECS)
            if cached is None and not enable_live_fetch:
                cached = self._read_stale_json_cache("sec_company_tickers.json")
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

        if self._ticker_to_cik_map:
            return self._ticker_to_cik_map.get(sym)
        return None

    def get_yf_meta(
        self,
        ticker: str,
        enable_live_fetch: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Fetches & disk-caches real Yahoo Finance info, analyst consensus, and insider rows."""
        sym = ticker.upper().strip()
        rel_path = f"yf_meta/{sym}.json"
        cached = self._read_json_cache(rel_path, self._YF_META_TTL_SECS)
        if cached is not None:
            return cached
        if not enable_live_fetch:
            return None

        fail_key = f"yf_meta:{sym}"
        if self._is_in_cooldown(fail_key):
            return None

        try:
            import yfinance as yf

            tk = yf.Ticker(sym)
            info = tk.info or {}
            if not isinstance(info, dict) or len(info) == 0:
                self._mark_failed(fail_key)
                return None

            # Clean JSON-serializable subset of tk.info
            safe_info: Dict[str, Any] = {}
            for k in (
                "longName", "shortName", "exchange", "fullExchangeName", "quoteType",
                "sector", "industry", "marketCap", "trailingPE", "forwardPE",
                "dividendYield", "trailingAnnualDividendYield",
                "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "longBusinessSummary",
                "city", "state", "country", "fullTimeEmployees", "companyOfficers",
                "currentPrice", "regularMarketPrice", "targetMeanPrice",
                "targetMedianPrice", "targetHighPrice", "targetLowPrice",
                "recommendationMean", "recommendationKey", "numberOfAnalystOpinions",
            ):
                if k in info and info[k] is not None:
                    safe_info[k] = info[k]

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
                        text_desc = str(row.get("Text") or row.get("Transaction") or "").lower()
                        tx_code = "P" if ("purchase" in text_desc or "buy" in text_desc) else "S"
                        price_est = round(val / shares, 2) if shares > 0 and val > 0 else 0.0
                        insiders.append({
                            "filing_date": dt_str,
                            "insider_name": str(row.get("Insider") or "Insider").strip(),
                            "relationship": str(row.get("Position") or "Officer/Director").strip(),
                            "transaction_code": tx_code,
                            "shares": int(round(abs(shares))),
                            "price": price_est,
                            "value": round(abs(val), 2),
                            "is_10b5_1": ("10b5-1" in text_desc or "automatic" in text_desc or "plan" in text_desc),
                            "raw_text": str(row.get("Text") or row.get("Transaction") or "").strip(),
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
            return payload
        except Exception as e:
            logger.debug(f"Yahoo meta fetch failed for {sym}: {e}")
            self._mark_failed(fail_key)
            return None

    def get_sec_submissions(
        self,
        cik: str,
        enable_live_fetch: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Fetches & disk-caches SEC EDGAR submissions JSON for a 10-digit CIK."""
        cik10 = str(cik).strip().zfill(10)
        rel_path = f"edgar_submissions/CIK{cik10}.json"
        cached = self._read_json_cache(rel_path, self._SUBMISSIONS_TTL_SECS)
        if cached is not None:
            return cached
        if not enable_live_fetch:
            return None

        fail_key = f"sec_sub:{cik10}"
        if self._is_in_cooldown(fail_key):
            return None

        raw = self._sec_fetch_text(f"https://data.sec.gov/submissions/CIK{cik10}.json", timeout=6.0)
        if not raw:
            self._mark_failed(fail_key)
            return None
        try:
            full = json.loads(raw)
            recent = full.get("filings", {}).get("recent", {})
            # Store compact recent filings slice (~500 filings) rather than multi-MB raw JSON
            compact = {
                "cik": cik10,
                "name": full.get("name", ""),
                "sicDescription": full.get("sicDescription", ""),
                "stateOfIncorporation": full.get("stateOfIncorporation", ""),
                "exchanges": full.get("exchanges", []),
                "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "recent": {
                    k: (v[:250] if isinstance(v, list) else v)
                    for k, v in recent.items()
                },
            }
            self._write_json_cache(rel_path, compact)
            return compact
        except Exception as e:
            logger.debug(f"Failed parsing SEC submissions for {cik10}: {e}")
            self._mark_failed(fail_key)
            return None

    def get_sec_company_facts(
        self,
        cik: str,
        enable_live_fetch: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Fetches & disk-caches SEC EDGAR US-GAAP CompanyFacts JSON."""
        cik10 = str(cik).strip().zfill(10)
        rel_path = f"edgar_facts/CIK{cik10}.json"
        cached = self._read_json_cache(rel_path, self._FACTS_TTL_SECS)
        if cached is not None:
            return cached
        if not enable_live_fetch:
            return None

        fail_key = f"sec_facts:{cik10}"
        if self._is_in_cooldown(fail_key):
            return None

        raw = self._sec_fetch_text(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json", timeout=8.0)
        if not raw:
            self._mark_failed(fail_key)
            return None
        try:
            parsed = json.loads(raw)
            # Prune strictly to us-gaap concepts needed so disk cache stays compact & fast
            us_gaap = parsed.get("facts", {}).get("us-gaap", {})
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
            return None

    @staticmethod
    def _extract_annual_xbrl_series(
        us_gaap: Dict[str, Any],
        concepts: Tuple[str, ...],
        is_flow: bool,
        unit_pref: str = "USD",
        pit_strict: bool = True,
        valid_period_ends: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, Tuple[str, str]]]:
        """
        Extracts annual values keyed by fiscal period end ('YYYY-MM-DD') -> val.
        When multiple concepts exist in hierarchy, selects the highest-priority concept
        available per fiscal period_end (enforcing PIT earliest-filed vs As-Restated latest-filed).
        Returns (period_end -> val, period_end -> (filed_date, accn)).
        """
        out_vals: Dict[str, float] = {}
        out_meta: Dict[str, Tuple[str, str]] = {}

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

                obs_list.sort(key=lambda o: str(o.get("filed", "")))
                chosen = obs_list[0] if pit_strict else obs_list[-1]
                out_vals[canonical_end] = float(chosen["val"])
                out_meta[canonical_end] = (str(chosen.get("filed", ""))[:10], str(chosen.get("accn", "")))

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
        if subs and subs.get("recent"):
            rec = subs["recent"]
            forms = rec.get("form", [])
            fdates = rec.get("filingDate", [])
            rdates = rec.get("reportDate", [])
            accns = rec.get("accessionNumber", [])
            docs = rec.get("primaryDocument", [])
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
            us_gaap, self._REVENUE_CONCEPTS, True, "USD", True, valid_map
        )
        rev_rest, rev_meta_rest = self._extract_annual_xbrl_series(
            us_gaap, self._REVENUE_CONCEPTS, True, "USD", False, valid_map
        )
        rev = rev_pit if pit_strict else rev_rest
        rev_meta = rev_meta_pit if pit_strict else rev_meta_rest

        cor, _ = self._extract_annual_xbrl_series(us_gaap, self._COST_OF_REV_CONCEPTS, True, "USD", pit_strict, valid_map)
        gp, _ = self._extract_annual_xbrl_series(us_gaap, self._GROSS_PROFIT_CONCEPTS, True, "USD", pit_strict, valid_map)
        opex, _ = self._extract_annual_xbrl_series(us_gaap, self._OPEX_CONCEPTS, True, "USD", pit_strict, valid_map)
        opinc, _ = self._extract_annual_xbrl_series(us_gaap, self._OP_INC_CONCEPTS, True, "USD", pit_strict, valid_map)
        ni_pit, _ = self._extract_annual_xbrl_series(us_gaap, self._NET_INC_CONCEPTS, True, "USD", True, valid_map)
        ni_rest, _ = self._extract_annual_xbrl_series(us_gaap, self._NET_INC_CONCEPTS, True, "USD", False, valid_map)
        ni = ni_pit if pit_strict else ni_rest
        eps, _ = self._extract_annual_xbrl_series(us_gaap, self._EPS_CONCEPTS, True, "USD/shares", pit_strict, valid_map)

        cash, _ = self._extract_annual_xbrl_series(us_gaap, self._CASH_CONCEPTS, False, "USD", pit_strict, valid_map)
        assets, _ = self._extract_annual_xbrl_series(us_gaap, self._ASSETS_CONCEPTS, False, "USD", pit_strict, valid_map)
        lt_debt, _ = self._extract_annual_xbrl_series(us_gaap, self._LT_DEBT_CONCEPTS, False, "USD", pit_strict, valid_map)
        st_debt, _ = self._extract_annual_xbrl_series(us_gaap, self._ST_DEBT_CONCEPTS, False, "USD", pit_strict, valid_map)
        equity, _ = self._extract_annual_xbrl_series(us_gaap, self._EQUITY_CONCEPTS, False, "USD", pit_strict, valid_map)

        ocf, _ = self._extract_annual_xbrl_series(us_gaap, self._OCF_CONCEPTS, True, "USD", pit_strict, valid_map)
        capex, _ = self._extract_annual_xbrl_series(us_gaap, self._CAPEX_CONCEPTS, True, "USD", pit_strict, valid_map)

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
            rev_diff = abs((rev_rest.get(end_dt, 0.0) or 0.0) - (rev_pit.get(end_dt, 0.0) or 0.0))
            ni_diff = abs((ni_rest.get(end_dt, 0.0) or 0.0) - (ni_pit.get(end_dt, 0.0) or 0.0))
            is_restated = bool((rev_diff > 1e5 or ni_diff > 1e5) and (end_dt in rev_rest or end_dt in ni_rest))
            note = (
                f"Subsequent 10-K/A restatement changed Revenue by {(rev_rest.get(end_dt, 0) - rev_pit.get(end_dt, 0)) / 1e6:+.0f}M"
                if is_restated
                else None
            )
            filings.append({
                "period": label,
                "filed_date": filed_dt,
                "period_end_date": end_dt,
                "accession_number": accn,
                "sec_url": sec_url,
                "is_restated": is_restated,
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
        if subs and subs.get("recent"):
            rec = subs["recent"]
            forms = rec.get("form", [])
            fdates = rec.get("filingDate", [])
            accs = rec.get("acceptanceDateTime", [])
            accns = rec.get("accessionNumber", [])
            docs = rec.get("primaryDocument", [])
            for form, fdt, acc_dt, accn, doc in zip(forms, fdates, accs, accns, docs):
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

        form4_rows: List[Dict[str, Any]] = []
        if yf_meta and isinstance(yf_meta.get("insider_transactions"), list):
            for tx in yf_meta["insider_transactions"]:
                if tx.get("shares", 0) > 0:
                    form4_rows.append({k: v for k, v in tx.items() if k != "raw_text"})

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
        if acc_raw and len(acc_raw) >= 19:
            pit_dt = MarketDataLoader.apply_pit_cutoff(acc_raw)
            shifted_note = f"{acc_raw[:19].replace('T', ' ')} UTC -> PIT Effective {pit_dt.strftime('%Y-%m-%d')}"
        else:
            shifted_note = f"{curr_f['filing_date']} (SEC Filing Date)"

        # Cached section redline comparison for latest vs prior 10-K/10-Q
        rel_redline = f"edgar_redlines/{sym}_{curr_f['accession_number']}.json"
        cached_diff = self._read_json_cache(rel_redline, self._REDLINE_TTL_SECS)
        if cached_diff is None and enable_live_fetch and cik and prior_f is not None and not self._is_in_cooldown(rel_redline):
            curr_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{curr_f['accession_number'].replace('-', '')}/{curr_f['primary_document']}"
            prior_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{prior_f['accession_number'].replace('-', '')}/{prior_f['primary_document']}"
            html_curr = self._sec_fetch_text(curr_url, timeout=6.0)
            html_prior = self._sec_fetch_text(prior_url, timeout=6.0)
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
            else:
                self._mark_failed(rel_redline, seconds=300.0)

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
            "prior_filing_date": prior_f["filing_date"] if prior_f else None,
            "acceptance_timestamp_et": shifted_note,
            "item1a_similarity": cached_diff.get("item1a_similarity") if cached_diff else None,
            "mda_similarity": cached_diff.get("mda_similarity") if cached_diff else None,
            "item1a_sentiment_drift": cached_diff.get("item1a_sentiment_drift") if cached_diff else None,
            "mda_catalyst_score": cached_diff.get("mda_catalyst_score") if cached_diff else None,
            "redlines": redlines,
            "form4_transactions": form4_rows,
        }
