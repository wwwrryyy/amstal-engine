"""
Forward-Filtered 3-State Gaussian HMM (QARE-MM v1.3)
Implements causal log-space forward filtering with zero forward-backward lookahead leak.
Enforces invariant state ordering by SPY return variance (State 0: Low-Vol, State 1: Med-Vol, State 2: High-Vol).
"""

from typing import List, Optional, Tuple
import math
import numpy as np


class RegimeHMM:
    """3-State Volatility Regime HMM with causal forward recursion and variance sorting."""

    NUM_STATES = 3
    STATE_LABELS = {
        0: "LOW_VOLATILITY",
        1: "MEDIUM_VOLATILITY",
        2: "HIGH_VOLATILITY",
    }
    # Macro regime Kelly scaling multipliers
    REGIME_MULTIPLIERS = {
        0: 0.50,  # Half-Kelly in low volatility
        1: 0.35,  # Moderate Kelly in medium volatility
        2: 0.25,  # Quarter-Kelly in high volatility / turbulent regime
    }

    def __init__(
        self,
        n_iter: int = 30,
        random_state: int = 42,
        standardize: bool = True,
        n_fits: int = 10,
    ) -> None:
        self.n_iter = n_iter
        self.random_state = random_state
        self.standardize = standardize
        self.n_fits = n_fits
        self.mean_: Optional[np.ndarray] = None         # (2,)
        self.std_: Optional[np.ndarray] = None          # (2,)
        self.means_: Optional[np.ndarray] = None        # (3, 2)
        self.covars_: Optional[np.ndarray] = None       # (3, 2) or (3, 2, 2)
        self.transmat_: Optional[np.ndarray] = None     # (3, 3)
        self.startprob_: Optional[np.ndarray] = None    # (3,)
        self.state_order_: Optional[np.ndarray] = None  # 1D index mapping (3,)

    def transform_observation(self, x: np.ndarray) -> np.ndarray:
        """Applies fitted standardization to an input observation."""
        x_arr = np.asarray(x, dtype=np.float64)
        if self.standardize and self.mean_ is not None and self.std_ is not None:
            return (x_arr - self.mean_) / self.std_
        return x_arr

    def fit(self, X_train: np.ndarray) -> "RegimeHMM":
        """
        Fits 3-state Gaussian HMM on 2D series [SPY_returns, VIX].
        Standardizes input with fold-scoped statistics, runs multi-restart EM,
        and sorts states strictly by SPY return variance component Sigma_{k, rr}.
        """
        X = np.asarray(X_train, dtype=np.float64)
        if self.standardize:
            self.mean_ = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            self.std_ = np.where(std < 1e-8, 1.0, std)
            X_proc = (X - self.mean_) / self.std_
        else:
            self.mean_ = np.zeros(X.shape[1], dtype=np.float64)
            self.std_ = np.ones(X.shape[1], dtype=np.float64)
            X_proc = X

        try:
            from hmmlearn.hmm import GaussianHMM

            best_score = -np.inf
            best_hmm = None

            for i in range(max(1, self.n_fits)):
                seed = self.random_state + i
                raw_hmm = GaussianHMM(
                    n_components=self.NUM_STATES,
                    covariance_type="diag",
                    n_iter=self.n_iter,
                    tol=1e-3,
                    random_state=seed,
                )
                try:
                    raw_hmm.fit(X_proc)
                    score = raw_hmm.score(X_proc)
                    if score > best_score or best_hmm is None:
                        best_score = score
                        best_hmm = raw_hmm
                except Exception:
                    continue

            if best_hmm is None:
                raise RuntimeError("Failed to fit GaussianHMM on any restart.")

            raw_hmm = best_hmm

            # Extract 1D variance component for SPY return (feature index 0)
            if raw_hmm.covars_.ndim == 3:
                variances = raw_hmm.covars_[:, 0, 0]
            else:
                variances = raw_hmm.covars_[:, 0]

            self.state_order_ = np.argsort(variances).astype(int)

            self.means_ = raw_hmm.means_[self.state_order_]
            self.covars_ = raw_hmm.covars_[self.state_order_]
            self.startprob_ = raw_hmm.startprob_[self.state_order_]

            # Reorder transition matrix rows and columns
            trans = raw_hmm.transmat_[self.state_order_, :]
            self.transmat_ = trans[:, self.state_order_]
        except ImportError:
            # Deterministic fallback parameters
            self.means_ = np.array([[0.05, -0.5], [0.0, 0.0], [-0.1, 1.2]])
            self.covars_ = np.array([[0.5, 0.5], [1.0, 1.0], [2.5, 2.5]])
            self.transmat_ = np.array([
                [0.92, 0.06, 0.02],
                [0.05, 0.90, 0.05],
                [0.03, 0.07, 0.90],
            ])
            self.startprob_ = np.array([0.7, 0.2, 0.1])
            self.state_order_ = np.array([0, 1, 2])

        return self

    def _extract_diag_variances(self) -> np.ndarray:
        """Extracts (NUM_STATES, n_features) diagonal emission variances with floor."""
        if self.covars_.ndim == 3:
            cov_diag = np.diagonal(self.covars_, axis1=1, axis2=2)
        else:
            cov_diag = self.covars_
        return np.maximum(np.asarray(cov_diag, dtype=np.float64), 1e-8)

    @staticmethod
    def _logsumexp_1d(vals: np.ndarray) -> float:
        m = float(np.max(vals))
        return m + math.log(float(np.sum(np.exp(vals - m))))

    def forward_filter_step(
        self,
        x_t: np.ndarray,
        log_alpha_prev: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes one causal log-space forward recursion step:
        log_alpha_tilde(k) = log P(x_t | S_t = k) + logsumexp_j (log_alpha_prev(j) + log A_{j, k})
        p_t(k) = exp(log_alpha_tilde(k) - logsumexp_j log_alpha_tilde(j))
        """
        if self.means_ is None or self.covars_ is None or self.transmat_ is None:
            raise RuntimeError("GaussianHMM must be fit before filtering.")

        x_vec = np.asarray(x_t, dtype=np.float64)
        vars_diag = self._extract_diag_variances()
        diff = x_vec[None, :] - self.means_
        log_emission = -0.5 * np.sum(
            np.log(2.0 * np.pi * vars_diag) + (diff * diff) / vars_diag,
            axis=1,
        )

        log_trans = np.log(np.maximum(self.transmat_, 1e-12))
        trans_scores = log_alpha_prev[:, None] + log_trans
        col_max = np.max(trans_scores, axis=0)
        log_prior = col_max + np.log(np.sum(np.exp(trans_scores - col_max[None, :]), axis=0))
        log_alpha_tilde = log_emission + log_prior

        log_norm = self._logsumexp_1d(log_alpha_tilde)
        log_alpha_t = log_alpha_tilde - log_norm
        prob_t = np.exp(log_alpha_t)

        return prob_t, log_alpha_t

    def filter_sequence(self, X: np.ndarray) -> np.ndarray:
        """Causally forward-filters sequence X with zero lookahead leak."""
        if self.means_ is None or self.covars_ is None or self.transmat_ is None or self.startprob_ is None:
            raise RuntimeError("GaussianHMM must be fit before filtering.")

        X_arr = np.asarray(X, dtype=np.float64)
        if self.standardize and self.mean_ is not None and self.std_ is not None:
            X_arr = (X_arr - self.mean_) / self.std_

        T = len(X_arr)
        probs = np.zeros((T, self.NUM_STATES), dtype=np.float64)
        if T == 0:
            return probs

        # Vectorized Gaussian diagonal log-likelihood across all timesteps at once
        means = np.asarray(self.means_, dtype=np.float64)
        vars_diag = self._extract_diag_variances()
        log_norm_const = -0.5 * np.sum(np.log(2.0 * np.pi * vars_diag), axis=1)  # (3,)
        inv_2var = -0.5 / vars_diag  # (3, D)
        diff = X_arr[:, None, :] - means[None, :, :]  # (T, 3, D)
        log_emissions = log_norm_const[None, :] + np.einsum("tkd,kd->tk", diff * diff, inv_2var)

        log_trans = np.log(np.maximum(self.transmat_, 1e-12))
        log_start = np.log(np.maximum(self.startprob_, 1e-12))

        # Initial step t=0: apply emission likelihood directly to startprob_ prior
        log_alpha_0 = log_start + log_emissions[0]
        m0 = float(np.max(log_alpha_0))
        log_alpha_0 = log_alpha_0 - (m0 + math.log(float(np.sum(np.exp(log_alpha_0 - m0)))))
        probs[0] = np.exp(log_alpha_0)

        a0, a1, a2 = float(log_alpha_0[0]), float(log_alpha_0[1]), float(log_alpha_0[2])
        lt00, lt01, lt02 = float(log_trans[0, 0]), float(log_trans[0, 1]), float(log_trans[0, 2])
        lt10, lt11, lt12 = float(log_trans[1, 0]), float(log_trans[1, 1]), float(log_trans[1, 2])
        lt20, lt21, lt22 = float(log_trans[2, 0]), float(log_trans[2, 1]), float(log_trans[2, 2])

        exp = math.exp
        log = math.log
        em_rows = log_emissions.tolist()

        for t in range(1, T):
            em0, em1, em2 = em_rows[t]

            # State 0 transition logsumexp
            x0, x1, x2 = a0 + lt00, a1 + lt10, a2 + lt20
            mx = x0 if x0 >= x1 else x1
            if x2 > mx:
                mx = x2
            la0 = em0 + mx + log(exp(x0 - mx) + exp(x1 - mx) + exp(x2 - mx))

            # State 1 transition logsumexp
            x0, x1, x2 = a0 + lt01, a1 + lt11, a2 + lt21
            mx = x0 if x0 >= x1 else x1
            if x2 > mx:
                mx = x2
            la1 = em1 + mx + log(exp(x0 - mx) + exp(x1 - mx) + exp(x2 - mx))

            # State 2 transition logsumexp
            x0, x1, x2 = a0 + lt02, a1 + lt12, a2 + lt22
            mx = x0 if x0 >= x1 else x1
            if x2 > mx:
                mx = x2
            la2 = em2 + mx + log(exp(x0 - mx) + exp(x1 - mx) + exp(x2 - mx))

            mn = la0 if la0 >= la1 else la1
            if la2 > mn:
                mn = la2
            norm = mn + log(exp(la0 - mn) + exp(la1 - mn) + exp(la2 - mn))
            a0, a1, a2 = la0 - norm, la1 - norm, la2 - norm
            probs[t, 0] = exp(a0)
            probs[t, 1] = exp(a1)
            probs[t, 2] = exp(a2)

        return probs

    def get_state_annualized_volatilities(self) -> List[float]:
        """Returns annualized SPY return volatility (%) for each ordered state."""
        if self.covars_ is None:
            return []
        vars_diag = self._extract_diag_variances()
        spy_var_std = vars_diag[:, 0]
        spy_scale = float(self.std_[0]) if (self.standardize and self.std_ is not None and len(self.std_) > 0) else 1.0
        ann_factor = math.sqrt(252.0) * 100.0
        return [round(float(math.sqrt(v) * spy_scale * ann_factor), 2) for v in spy_var_std]

    def get_stationary_distribution(self) -> List[float]:
        """Computes stationary distribution pi solving pi @ transmat = pi."""
        if self.transmat_ is None:
            return []
        A = np.asarray(self.transmat_, dtype=np.float64)
        vals, vecs = np.linalg.eig(A.T)
        idx = int(np.argmin(np.abs(vals - 1.0)))
        pi = np.real(vecs[:, idx])
        # Perron-Frobenius left eigenvector has uniform sign up to +/-1 phase
        if float(np.sum(pi)) < 0.0:
            pi = -pi
        pi = np.maximum(pi, 0.0)
        s = float(np.sum(pi))
        if not math.isfinite(s) or s <= 0.0:
            return [1.0 / self.NUM_STATES] * self.NUM_STATES
        return [round(float(v / s), 4) for v in pi]

    def get_regime_multiplier(self, probs_t: np.ndarray) -> float:
        """Computes expected macro regime multiplier kappa in [0.25, 0.50]; fails closed to 0.25 on invalid inputs."""
        try:
            p = np.asarray(probs_t, dtype=np.float64)
        except (TypeError, ValueError):
            return 0.25
        if p.ndim != 1 or p.shape[0] != 3 or not np.all(np.isfinite(p)):
            return 0.25
        if np.any(p < 0.0) or np.any(p > 1.0):
            return 0.25
        s = float(np.sum(p))
        if not (0.98 <= s <= 1.02):
            return 0.25
        p_norm = p / s
        kappa = (
            p_norm[0] * self.REGIME_MULTIPLIERS[0]
            + p_norm[1] * self.REGIME_MULTIPLIERS[1]
            + p_norm[2] * self.REGIME_MULTIPLIERS[2]
        )
        return float(np.clip(kappa, 0.25, 0.50)) if math.isfinite(float(kappa)) else 0.25
