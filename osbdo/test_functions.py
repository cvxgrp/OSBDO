import numpy as np
import cvxpy as cp

from osbdo.utils import *
from osbdo.quasi_newton import *
    

def project_g(b0, g, var_x):
    prob = cp.Problem(cp.Minimize(cp.sum_squares(var_x - b0)), g.domain)
    prob.solve(solver='ECOS')
    return var_x.value


def test_f_subgradient(*, grad_f_a, f_a, a, var_x, func_f, g, num_iters=None,  b=None):
    """
    Test if grad_f_a \in \partial f(a)
    """
    if b is not None:
        f_b = func_f(b)
        assert f_b  + 1e-7 >= f_a + grad_f_a.T.dot(b - a), print(f_b  + 1e-7, f_a + grad_f_a.T.dot(b - a))
    else:
        for _ in range(num_iters):
            b0 = np.random.randn(grad_f_a.size)/a.size + a
            b = project_g(b0, g, var_x)
            f_b = func_f(b)
            assert f_b  + 1e-7 >= f_a + grad_f_a.T.dot(b - a), print(f_b  + 1e-7, f_a + grad_f_a.T.dot(b - a))


def test_g_subgradient(*, grad_g_a, a, var_x, g, num_iters=None,  b=None):
    """
    Test if grad_g_a \in \partial g(a)
    """
    cp.Problem(cp.Minimize(0), [var_x == a] + g.domain).solve(solver='ECOS')
    g_a = g.function.value
    if b is not None:
        cp.Problem(cp.Minimize(0), [var_x == b] + g.domain).solve(solver='ECOS')
        g_b = g.function.value
        assert g_b  + 1e-7 >= g_a + grad_g_a.T.dot(b - a), print(g_b  + 1e-7, g_a + grad_g_a.T.dot(b - a))
    else:
        for _ in range(num_iters):
            b0 = np.random.randn(grad_g_a.size)/a.size + a
            b = project_g(b0, g, var_x)
            g_b = g.function.value
            assert g_b  + 1e-7 + np.abs(g_b)*1e-7 >= g_a + grad_g_a.T.dot(b - a), \
                print(g_b  + 1e-7 + np.abs(g_b)*1e-7, g_a + grad_g_a.T.dot(b - a))


def test_hat_f_subgradient(*, grad_hat_f_a, hat_f_a, a, var_x, exp_hat_f, g, num_iters=None, b=None):
    """
    Test if grad_hat_f_a \in \partial \hat f(a)
    """
    if b is not None:
        cp.Problem(cp.Minimize(0), [var_x == b] + g.domain).solve(solver='ECOS')
        hat_f_b = exp_hat_f.value
        assert hat_f_b  + 1e-7 >= hat_f_a + grad_hat_f_a.T.dot(b - a)
    else:
        for _ in range(num_iters):
            b0 = np.random.randn(grad_hat_f_a.size)/a.size + a
            b = project_g(b0, g, var_x)
            hat_f_b = exp_hat_f.value
            assert hat_f_b  + 1e-7 >= hat_f_a + grad_hat_f_a.T.dot(b - a), print(hat_f_b  + 1e-7, hat_f_a + grad_hat_f_a.T.dot(b - a))


def test_h_subgradient(*, grad_h_a, h_a, a, var_x, func_f, g, num_iters=None, b=None):
    """
    Test if grad_h_a \in \partial h(a)
    """
    if b is not None:
        cp.Problem(cp.Minimize(0), [var_x == b] + g.domain).solve(solver='ECOS')
        h_b = func_f(b) + g.function.value
        assert h_b  + 1e-7 >= h_a + grad_h_a.T.dot(b - a), print(h_b  + 1e-7, h_a + grad_h_a.T.dot(b - a))
    else:
        for _ in range(num_iters):
            b0 = a + np.random.randn(grad_h_a.size)/a.size
            b = project_g(b0, g, var_x)
            h_b = func_f(b) + g.function.value
            assert h_b  + 1e-7 >= h_a + grad_h_a.T.dot(b - a), print(h_b  + 1e-7, h_a + grad_h_a.T.dot(b - a))


def test_hat_h_subgradient(*, grad_hat_h_a, hat_h_a, a, var_x, hat_h, g, num_iters=None, b=None):
    """
    Test if grad_hat_h_a \in \partial \hat h(a)
    """
    if b is not None:
        cp.Problem(cp.Minimize(0), [var_x == b] + g.domain).solve(solver='ECOS')
        hat_h_b = hat_h.value
        assert hat_h_b  + 1e-7 >= hat_h_a + grad_hat_h_a.T.dot(b - a), print(hat_h_b  + 1e-7, hat_h_a + grad_hat_h_a.T.dot(b - a))
    else:
        for _ in range(num_iters):
            b0 = np.random.randn(grad_hat_h_a.size)/a.size + a
            b = project_g(b0, g, var_x)
            hat_h_b = hat_h.value
            assert hat_h_b  + 1e-7 >= hat_h_a + grad_hat_h_a.T.dot(b - a), print(hat_h_b  + 1e-7, hat_h_a + grad_hat_h_a.T.dot(b - a))


def test_curve_search(delta_t, e_t, hat_e_t, solver, h_x, h_tilde_x, var_x, x, mu_k, q_t, tilde_x,\
                    hat_h_tilde_x, hat_q_t, hat_fis, grad_g, tL, t, tR, grad_hat_f, f_tilde_x,\
                    coupling_g, hat_h, func_sum_hat_fi, func_get_f_x, grad_f):
    try:
        # test_g_subgradient(grad_g_a=grad_g, a=tilde_x, var_x=var_x, g=coupling_g, num_iters=20)
        assert delta_t >= -1e-6 and e_t >= -1e-6 and hat_e_t >= -1e-6,\
            print(delta_t, e_t, hat_e_t)
        func_f = lambda z: func_get_f_x(x=z, solver=solver)[0]
        test_h_subgradient(grad_h_a=q_t, h_a=h_tilde_x, a=tilde_x, var_x=var_x, func_f=func_f, \
                        g=coupling_g, num_iters=20)
    except:
        print(f"{h_x=}, {h_tilde_x + q_t.T.dot(x - tilde_x)=}, {h_tilde_x=}")
        print(f"     {tL=}, {t=}, {tR=}, {mu_k=}, {hat_h_tilde_x + hat_q_t.T.dot(x - tilde_x)}") 
        
        exp_hat_f = func_sum_hat_fi(hat_fis)
        cp.Problem(cp.Minimize(0), [var_x == tilde_x]+coupling_g.domain).solve(solver='ECOS')
        hat_f_tilde_x = exp_hat_f.value 
        func_f = lambda z: func_get_f_x(x=z, solver=solver)[0]
        assert np.allclose(hat_h_tilde_x, hat_f_tilde_x + coupling_g.function.value)
        assert np.allclose(h_tilde_x, func_f(tilde_x) + coupling_g.function.value)
        test_hat_f_subgradient(grad_hat_f_a=grad_hat_f, hat_f_a=hat_f_tilde_x, a=tilde_x, var_x=var_x, \
                        exp_hat_f=exp_hat_f, g=coupling_g, num_iters=30)
        test_hat_f_subgradient(grad_hat_f_a=grad_hat_f, hat_f_a=hat_f_tilde_x, a=tilde_x, var_x=var_x, \
                        exp_hat_f=exp_hat_f, g=coupling_g, b=x)
        # test subgradient hat_q_t \in \partial \hat h(tilde_x)
        test_hat_h_subgradient(grad_hat_h_a=hat_q_t, hat_h_a=hat_h_tilde_x, a=tilde_x, var_x=var_x, \
                            hat_h=hat_h, g=coupling_g, num_iters=30)
        test_hat_h_subgradient(grad_hat_h_a=hat_q_t, hat_h_a=hat_h_tilde_x, a=tilde_x, var_x=var_x, \
                            hat_h=hat_h, g=coupling_g, b=x)
        # test subgradient q_t \in \partial h(tilde_x)
        test_f_subgradient(grad_f_a=grad_f, f_a=f_tilde_x, a=tilde_x, var_x=var_x, func_f=func_f, \
                        g=coupling_g, num_iters=30)
        test_f_subgradient(grad_f_a=grad_f, f_a=f_tilde_x, a=tilde_x, var_x=var_x, func_f=func_f, \
                        g=coupling_g, b=x)
        cp.Problem(cp.Minimize(0), [var_x == x]+coupling_g.domain).solve(solver='ECOS')
        assert np.allclose(h_x, func_f(x) + coupling_g.function.value)
        test_g_subgradient(grad_g_a=grad_g, a=tilde_x, var_x=var_x, g=coupling_g, num_iters=30)
        test_g_subgradient(grad_g_a=grad_g, a=tilde_x, var_x=var_x, g=coupling_g, num_iters=None, b=x)
        test_h_subgradient(grad_h_a=q_t, h_a=h_tilde_x, a=tilde_x, var_x=var_x, func_f=func_f, \
                        g=coupling_g, num_iters=100)
        # test_h_subgradient(grad_h_a=q_t, h_a=h_tilde_x, a=tilde_x, var_x=var_x, func_f=func_f, \
        #                 g=coupling_g, b=x)
        
        assert delta_t >= -1e-8 and e_t >= -1e-8 and hat_e_t >= -1e-8,\
            print(delta_t, e_t, hat_e_t)