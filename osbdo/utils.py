import numpy as np


class Point:
    def __init__(self, *, x, q, f):
        self.x = x
        self.q = q
        self.f = f


def verify_bounds(f_true, Lk, Uk):
    rdiff_low = (f_true - Lk[-1]) / min(np.abs(f_true), np.abs(Lk[-1]))
    rdiff_up  = (Uk[-1] - f_true) / min(np.abs(f_true), np.abs(Uk[-1]))
    assert rdiff_low > -5 * 10**(-5), \
            print(f"{rdiff_low =}, {Lk[-1] = }, {f_true = }, {Uk[-1] = }")
    assert rdiff_up > -5 * 10**(-5),\
            print(f"{rdiff_up = }, {Lk[-1] = }, {f_true = }, {Uk[-1] = }")
    print(f"{rdiff_up = }, {rdiff_low = }")
    print("L <= h^\star <= U")


def get_mus(alpha, params, N):
    assert N ==  len(params), print(f"{N=}, {len(params)}, {alpha=}")
    Ds = [0]* N
    for i  in  range(N):
        m = params[i]["dimension"]
        Ds[i] = alpha * np.ones(m)
    return Ds


def sign(z):
    z[z>=0] = 1
    z[z<0] = -1
    return z