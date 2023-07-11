import numpy as np


def quasi_newton_update(*, G_k, x_cur, x_prev, grad_cur, grad_prev, hess_rank, tol_1=1e-10, tol_2=1e-2, ep=1e-15):
    """
    Implementation is taken from CurvatureUpdate.low_rank_quasi_Newton_update
    https://github.com/cvxgrp/osmm/blob/main/osmm/curvature_updates.py
    @article{oracle_struc_composite,
        author  = {Shen, Xinyue and Ali, Alnur and Boyd, Stephen},
        title   = {Minimizing Oracle-Structured Composite Functions},
        journal = {arXiv},
        year    = {2021}}
    """
    n = G_k.shape[0]
    s = x_cur - x_prev
    y = grad_cur - grad_prev
    y_abs_too_small_idxes = np.where(np.abs(y) <= ep)[0]
    y[y_abs_too_small_idxes] = 0
    yTs = y.T.dot(s)
    r1 = hess_rank
    w = G_k.T.dot(s)
    if yTs < max(tol_1, np.linalg.norm(y) * np.linalg.norm(s) * tol_2):
        if np.linalg.norm(s) < tol_2 * np.linalg.norm(x_prev) or np.linalg.norm(w) < tol_1:
            # print("U_k not updated", np.linalg.norm(s), tol_2 * np.linalg.norm(x_prev), np.linalg.norm(v))
            return G_k
        r1 = -1
    while r1 > 0 and yTs - w[0:r1].T.dot(w[0:r1]) < np.linalg.norm(y - G_k[:, 0:r1].dot(w[0:r1])) \
            * np.linalg.norm(s) * tol_2:
        r1 -= 1
    if r1 >= 0:
        w1 = w[0:r1]
        G1 = G_k[:, 0:r1]
        alpha_k = np.sqrt(yTs - w1.T.dot(w1))
        P = np.fliplr(np.eye(r1 + 1))
        tmp = np.eye(r1 + 1)
        tmp[0, 0] = 1.0 / alpha_k
        tmp[1:r1 + 1, 0] = - w1 / alpha_k
        _, R1_tilde = np.linalg.qr(np.transpose(P.dot(tmp)))
        R1 = P.dot(R1_tilde.T.dot(P))
        new_G1 = np.concatenate([y.reshape((n, 1)), G1], axis=1).dot(R1)  # n by r1+1
    else:
        new_G1 = None
    if r1 == hess_rank:
        G_k_plus_one = new_G1[:, 0:hess_rank]
    else:
        r2 = hess_rank - max(0, r1)
        w2 = w[hess_rank - r2:hess_rank]
        G2 = G_k[:, hess_rank - r2:hess_rank]
        basis, _, _ = np.linalg.svd(w2.reshape((r2, 1)))
        Q2 = np.array(basis)
        Q2[:, r2 - 1] = basis[:, 0]
        Q2[:, 0:r2 - 1] = basis[:, 1::]
        G2Q2 = G2.dot(Q2)
        new_G2 = G2Q2[:, 0:r2 - 1]  # n by r2-1
        if r1 >= 0:
            G_k_plus_one = np.concatenate([new_G1, new_G2], axis=1)
        else:
            G_k_plus_one = G_k
            G_k_plus_one[:, 0:hess_rank - 1] = new_G2
            G_k_plus_one[:, hess_rank - 1] = np.zeros(n)
    return G_k_plus_one


def matvec_inv_Hk(G_k, a):
    res = np.zeros(G_k.shape[0])
    for i in range(G_k.shape[1]):
        norm_i = np.linalg.norm(G_k[:, i])
        if not np.allclose(norm_i, 0):
            res += (1./norm_i**4) * (G_k[:, i].dot(a)) * G_k[:, i]
    return res
