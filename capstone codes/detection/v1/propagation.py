r"""propagation.py -- Part II: iterative network-based score estimation (Algorithm 1).

    x(k) = rho * x0 + (1 - rho) * B x(k-1),   rho = 1/2 in the paper,
    stop when ||x(k) - x(k-1)||_2 / ||x(k-1)||_2 < tol (= 1e-5),

where b_ij = fraction of wallet i's volume traded with counterparty j (row-stochastic, zero
diagonal). The two properties proved in the paper's Appendix C double as correctness checks:

  Proposition 1  the iteration converges to the unique solution of (I - (1-rho) B) x = rho x0
                 -> ``stationary`` solves that directly for comparison;
  Proposition 2  v' x(k) is the same for every k, where v_i = wallet i's volume
                 -> holds only if B is built from SYMMETRIC pair volumes (v' B = v').
"""
import numpy as np


def row_normalize(V: np.ndarray) -> np.ndarray:
    """B = V / row sums (rows of isolated wallets stay zero)."""
    s = V.sum(axis=1, keepdims=True)
    return np.divide(V, s, out=np.zeros_like(V, dtype=float), where=s > 0)


def propagate(x0, B, rho: float = 0.5, tol: float = 1e-5, max_iter: int = 10_000):
    """Run Algorithm 1 Part II. Returns (x, history, n_iter); history[k] = x(k), history[0] = x0."""
    x0 = np.asarray(x0, dtype=float)
    hist = [x0.copy()]
    prev = x0.copy()
    for k in range(1, max_iter + 1):
        x = rho * x0 + (1.0 - rho) * (B @ prev)
        hist.append(x)
        denom = np.linalg.norm(prev)
        if denom == 0.0 or np.linalg.norm(x - prev) / denom < tol:
            return x, np.vstack(hist), k
        prev = x
    raise RuntimeError(f"score propagation did not converge in {max_iter} iterations")


def stationary(x0, B, rho: float = 0.5) -> np.ndarray:
    """Closed-form fixed point: solve (I - (1-rho) B) x = rho x0."""
    n = len(x0)
    return np.linalg.solve(np.eye(n) - (1.0 - rho) * B, rho * np.asarray(x0, dtype=float))
