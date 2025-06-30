import numpy as np
from numba import njit

# === Finite-Difference and Numba Kernels ===

@njit
def _fd_grad(func, x, eps):
    n = x.size
    grad = np.zeros(n)
    f0 = func(x)
    for i in range(n):
        x_step = x.copy()
        x_step[i] += eps
        grad[i] = (func(x_step) - f0) / eps
    return grad

@njit
def _central_fd_hess(func, x, args, eps):
    x = np.asarray(x, dtype=float)
    n = x.size
    hess = np.zeros((n, n), dtype=float)
    f0 = func(x, *args)
    for i in range(n):
        x_i_up = x.copy(); x_i_down = x.copy()
        x_i_up[i] += eps; x_i_down[i] -= eps
        f_up = func(x_i_up, *args); f_down = func(x_i_down, *args)
        hess[i, i] = (f_up - 2*f0 + f_down) / (eps * eps)
        for j in range(i+1, n):
            x_pp = x.copy(); x_pp[i] += eps; x_pp[j] += eps
            x_pn = x.copy(); x_pn[i] += eps; x_pn[j] -= eps
            x_np = x.copy(); x_np[i] -= eps; x_np[j] += eps
            x_nn = x.copy(); x_nn[i] -= eps; x_nn[j] -= eps
            f_pp = func(x_pp, *args); f_pn = func(x_pn, *args)
            f_np = func(x_np, *args); f_nn = func(x_nn, *args)
            val = (f_pp - f_pn - f_np + f_nn) / (4 * eps * eps)
            hess[i, j] = hess[j, i] = val
    return hess

@njit
def _numba_fd_grad(func, x, eps):
    n = x.size
    grad = np.zeros(n, dtype=np.float64)
    f0 = func(x)
    for i in range(n):
        x_step = x.copy()
        x_step[i] += eps
        grad[i] = func(x_step) - f0
    return grad / eps

@njit
def _numba_central_fd_hess(func, x, eps):
    n = x.size
    hess = np.zeros((n, n), dtype=np.float64)
    f0 = func(x)
    for i in range(n):
        x_i_up = x.copy(); x_i_down = x.copy()
        x_i_up[i] += eps; x_i_down[i] -= eps
        f_up = func(x_i_up); f_down = func(x_i_down)
        hess[i, i] = (f_up - 2*f0 + f_down) / (eps * eps)
        for j in range(i+1, n):
            x_pp = x.copy(); x_pp[i] += eps; x_pp[j] += eps
            x_pn = x.copy(); x_pn[i] += eps; x_pn[j] -= eps
            x_np = x.copy(); x_np[i] -= eps; x_np[j] += eps
            x_nn = x.copy(); x_nn[i] -= eps; x_nn[j] -= eps
            f_pp = func(x_pp); f_pn = func(x_pn)
            f_np = func(x_np); f_nn = func(x_nn)
            val = (f_pp - f_pn - f_np + f_nn) / (4 * eps * eps)
            hess[i, j] = hess[j, i] = val
    return hess

# === Gradient & Hessian Wrappers ===

class GradientWrapper:
    def __init__(self, func, args=(), fprime=None, eps=1e-8, use_numba=False):
        self.func = func
        self.args = args
        self.eps = eps
        if fprime is not None:
            self._grad = lambda x: np.asarray(fprime(x, *args), float)
        else:
            if use_numba:
                njit_func = njit(lambda x: func(x, *args))
                self._grad = lambda x: _numba_fd_grad(njit_func, np.asarray(x, float), eps)
            else:
                self._grad = lambda x: _fd_grad(func, x, eps)
    def __call__(self, x):
        return self._grad(x)

class HessianWrapper:
    def __init__(self, func, args=(), hess=None, eps=1e-5, use_numba=False):
        self.func = func
        self.args = args
        self.eps = eps
        self.hess = hess
        if hess is not None:
            self._hess = lambda x: np.asarray(hess(x, *args), float)
        else:
            if use_numba:
                njit_func = njit(lambda x: func(x, *args))
                self._hess = lambda x: _numba_central_fd_hess(njit_func, np.asarray(x, float), eps)
            else:
                self._hess = lambda x: _central_fd_hess(func, x, args, eps)
    def __call__(self, x):
        return self._hess(x)

# === Constraint Handling ===

class ConstraintHandler:
    def __init__(self, eqcons=(), ieqcons=(), args=(), jac_eqcons=None, jac_ieqcons=None, eps=1e-8):
        self.eqcons = list(eqcons) if eqcons is not None else []
        self.ieqcons = list(ieqcons) if ieqcons is not None else []
        self.args = args
        self.eps = eps
        # Build Jacobian functions
        self.jac_eqcons = []
        for i, h in enumerate(self.eqcons):
            if jac_eqcons and i < len(jac_eqcons) and jac_eqcons[i] is not None:
                self.jac_eqcons.append(lambda x, h=h: np.asarray(jac_eqcons[i](x, *args), float))
            else:
                self.jac_eqcons.append(lambda x, h=h: _fd_grad(h, x, eps))
        self.jac_ieqcons = []
        for i, g in enumerate(self.ieqcons):
            if jac_ieqcons and i < len(jac_ieqcons) and jac_ieqcons[i] is not None:
                self.jac_ieqcons.append(lambda x, g=g: np.asarray(jac_ieqcons[i](x, *args), float))
            else:
                self.jac_ieqcons.append(lambda x, g=g: _fd_grad(g, x, eps))

    def eval_eq(self, x):
        x = np.asarray(x, float)
        return np.array([h(x, *self.args) for h in self.eqcons], float)

    def jac_eq(self, x):
        if not self.eqcons:
            return np.zeros((0, len(x)))
        return np.vstack([j(x) for j in self.jac_eqcons])

    def eval_ieq(self, x):
        x = np.asarray(x, float)
        return np.array([g(x, *self.args) for g in self.ieqcons], float)

    def jac_ieq(self, x):
        if not self.ieqcons:
            return np.zeros((0, len(x)))
        return np.vstack([j(x) for j in self.jac_ieqcons])

# === QP Subproblem Solver ===

class QPSubproblemSolver:
    @staticmethod
    def solve(H, g, A_eq, c_eq, A_ieq, c_ieq, tol=1e-8, max_iter=10):
        n = H.shape[0]
        m_eq = A_eq.shape[0]
        m_ieq = A_ieq.shape[0]

        # Active-set initialization
        W_ieq = [i for i in range(m_ieq) if c_ieq[i] < 0]

        def assemble(W_ieq):
            A_W = np.vstack([A_eq] + ([A_ieq[W_ieq]] if W_ieq else []))
            c_W = np.concatenate([c_eq] + ([c_ieq[W_ieq]] if W_ieq else []))
            return A_W, c_W

        for _ in range(max_iter):
            A_W, c_W = assemble(W_ieq)
            KKT = np.block([[H, A_W.T], [A_W, np.zeros((A_W.shape[0], A_W.shape[0]))]])
            rhs = -np.concatenate([g, c_W])
            # Robust regularization loop
            reg = 1e-8
            for _ in range(5):
                try:
                    sol = np.linalg.solve(KKT, rhs)
                    break
                except np.linalg.LinAlgError:
                    H_reg = H + reg * np.eye(H.shape[0])
                    KKT = np.block([[H_reg, A_W.T], [A_W, np.zeros((A_W.shape[0], A_W.shape[0]))]])
                    reg *= 10
            else:
                sol, *_ = np.linalg.lstsq(KKT, rhs, rcond=None)
            p = sol[:n]
            multipliers = sol[n:]
            lambda_eq = multipliers[:m_eq]
            lambda_ieq_W = multipliers[m_eq:]
            # Drop violated multipliers
            to_remove = [idx for idx, lam in zip(W_ieq, lambda_ieq_W) if lam < -tol]
            if not to_remove:
                break
            worst = min(to_remove, key=lambda i: lambda_ieq_W[W_ieq.index(i)])
            W_ieq.remove(worst)
        lambda_ieq = np.zeros(m_ieq)
        for idx, lam in zip(W_ieq, lambda_ieq_W):
            lambda_ieq[idx] = lam
        return p, lambda_eq, lambda_ieq

# === Merit Function & Line Search ===

class LineSearch:
    @staticmethod
    def merit(x, func, eval_eq, eval_ieq, sigma, args):
        f_val = func(x, *args)
        c_eq = eval_eq(x)
        c_ieq = eval_ieq(x)
        penalty = np.sum(np.abs(c_eq)) + np.sum(np.maximum(0, -c_ieq))
        return f_val + sigma * penalty

    @staticmethod
    def backtracking(x, p, func, grad, eval_eq, eval_ieq, jac_eq, jac_ieq,
                     lambda_eq, lambda_ieq, args=(), alpha0=1.0, rho=0.5, c1=1e-4, max_iter=20):
        sigma = max(1.0, np.max(np.abs(np.concatenate([lambda_eq, lambda_ieq]))))
        phi0 = LineSearch.merit(x, func, eval_eq, eval_ieq, sigma, args)
        g0 = grad(x)
        h0 = eval_eq(x); je = jac_eq(x)
        gi = eval_ieq(x); ji = jac_ieq(x)
        d_pen_eq = np.dot(np.sign(h0), je.dot(p))
        d_pen_ieq = -np.dot((gi < 0).astype(float), ji.dot(p))
        dphi0 = np.dot(g0, p) + sigma * (d_pen_eq + d_pen_ieq)
        alpha = alpha0
        for _ in range(max_iter):
            x_new = x + alpha * p
            phi = LineSearch.merit(x_new, func, eval_eq, eval_ieq, sigma, args)
            if phi <= phi0 + c1 * alpha * dphi0:
                break
            alpha *= rho
        return alpha

# === Main SLSQP Class ===

class SLSQP:
    def __init__(self, func, x0, args=(), bounds=None,
                 eqcons=(), ieqcons=(), fprime=None, hess=None,
                 jac_eqcons=None, jac_ieqcons=None,
                 maxiter=100, tol=1e-6, iprint=0, disp=False,
                 full_output=False, eps=1e-8, use_numba=False):
        self.func = func
        self.x0 = np.array(x0, float)
        self.args = args
        self.bounds = bounds
        self.eqcons = eqcons
        self.ieqcons = ieqcons
        self.fprime = fprime
        self.hess = hess
        self.jac_eqcons = jac_eqcons
        self.jac_ieqcons = jac_ieqcons
        self.maxiter = maxiter
        self.tol = tol
        self.iprint = iprint
        self.disp = disp
        self.full_output = full_output
        self.eps = eps
        self.use_numba = use_numba

    def optimize(self):
        x = self.x0.copy()
        n = x.size
        if self.bounds is None:
            lb = -np.inf * np.ones(n)
            ub = np.inf * np.ones(n)
        else:
            lb = np.array([b[0] if b[0] is not None else -np.inf for b in self.bounds], float)
            ub = np.array([b[1] if b[1] is not None else  np.inf for b in self.bounds], float)
        grad = GradientWrapper(self.func, self.args, self.fprime, self.eps, self.use_numba)
        hess_wrapper = HessianWrapper(self.func, self.args, self.hess, self.eps, self.use_numba)
        ch = ConstraintHandler(self.eqcons, self.ieqcons, self.args,
                               self.jac_eqcons, self.jac_ieqcons, self.eps)
        H = hess_wrapper(x) if self.hess is not None else np.eye(n)
        for k in range(1, self.maxiter + 1):
            f_val = self.func(x, *self.args)
            g = grad(x)
            c_eq = ch.eval_eq(x); c_ieq = ch.eval_ieq(x)
            A_eq = ch.jac_eq(x); A_ieq = ch.jac_ieq(x)
            p, lam_eq, lam_ieq = QPSubproblemSolver.solve(H, g, A_eq, c_eq, A_ieq, c_ieq)
            alpha = LineSearch.backtracking(x, p, self.func, grad,
                                            ch.eval_eq, ch.eval_ieq,
                                            ch.jac_eq, ch.jac_ieq,
                                            lam_eq, lam_ieq,
                                            self.args)
            x_new = np.minimum(np.maximum(x + alpha * p, lb), ub)
            g_new = grad(x_new)
            s = x_new - x
            if self.hess is None:
                y = g_new - g
                ys = np.dot(y, s)
                if ys > 1e-12:
                    H = (H + np.outer(y, y) / ys
                         - H.dot(np.outer(s, s)).dot(H) / (s.dot(H.dot(s))))
            else:
                H = hess_wrapper(x_new)
            c_eq_new = ch.eval_eq(x_new)
            c_ieq_new = ch.eval_ieq(x_new)
            grad_L = g_new + ch.jac_eq(x_new).T.dot(lam_eq) + ch.jac_ieq(x_new).T.dot(lam_ieq)
            grad_norm = np.linalg.norm(grad_L, np.inf)
            eq_viol = np.max(np.abs(c_eq_new)) if c_eq_new.size else 0.0
            ieq_viol = np.max(np.maximum(0, -c_ieq_new)) if c_ieq_new.size else 0.0
            if max(grad_norm, eq_viol, ieq_viol) < self.tol:
                status, message, success = 0, "Optimization converged.", True
                x = x_new
                break
            x = x_new
            if self.disp:
                print(f"Iter {k}: f = {f_val:.6g}, ||grad_L|| = {grad_norm:.3g}, "
                      f"eq_viol = {eq_viol:.3g}, ieq_viol = {ieq_viol:.3g}")
        else:
            status, message, success = 1, "Max iterations exceeded.", False
        res = {
            'x': x,
            'fun': float(self.func(x, *self.args)),
            'jac': grad(x),
            'nit': k,
            'status': status,
            'message': message,
            'success': success
        }
        return res if self.full_output else x
# === Benchmark Section ===

if __name__ == "__main__":
    import numpy as np
    import time
    from scipy.optimize import minimize

    # --- Deterministic 3-asset MVO problem ---
    mu = np.array([0.10, 0.12, 0.15])  # expected returns
    Sigma = np.array([
        [0.005, -0.010, 0.004],
        [-0.010, 0.040, -0.002],
        [0.004, -0.002, 0.023]
    ])
    target_return = 0.12

    def obj(w):
        return 0.5 * w @ Sigma @ w

    def obj_grad(w):
        return Sigma @ w

    def obj_hess(w):
        return Sigma

    def eq_sum_to_1(w):
        return np.sum(w) - 1

    def eq_sum_to_1_jac(w):
        return np.ones_like(w)

    def ineq_target_return(w):
        return w @ mu - target_return

    def ineq_target_return_jac(w):
        return mu

    x0 = np.array([1/3, 1/3, 1/3])
    bounds = [(0, 1) for _ in range(3)]

    optimizer = SLSQP(
        func=obj, x0=x0, fprime=obj_grad, hess=obj_hess,
        eqcons=[eq_sum_to_1], jac_eqcons=[eq_sum_to_1_jac],
        ieqcons=[ineq_target_return], jac_ieqcons=[ineq_target_return_jac],
        bounds=bounds, disp=True, full_output=True, tol=1e-5, maxiter=500
    )

    t0 = time.time()
    result = optimizer.optimize()
    t1 = time.time()
    print("Custom SLSQP Result:", result)
    print(f"Custom SLSQP Time: {t1 - t0:.6f} seconds")
    print("Portfolio return:", result['x'] @ mu)
    print("Portfolio volatility:", np.sqrt(result['x'] @ Sigma @ result['x']))

    # SciPy benchmark
    cons = [
        {'type': 'eq', 'fun': eq_sum_to_1, 'jac': eq_sum_to_1_jac},
        {'type': 'ineq', 'fun': ineq_target_return, 'jac': ineq_target_return_jac}
    ]
    t0 = time.time()
    res_scipy = minimize(
        obj, x0, jac=obj_grad, hess=obj_hess, constraints=cons, bounds=bounds, method='SLSQP', options={'disp': True}
    )
    t1 = time.time()
    print("SciPy SLSQP Result:", res_scipy)
    print(f"SciPy SLSQP Time: {t1 - t0:.6f} seconds")
    print("Portfolio return:", res_scipy.x @ mu)
    print("Portfolio volatility:", np.sqrt(res_scipy.x @ Sigma @ res_scipy.x))