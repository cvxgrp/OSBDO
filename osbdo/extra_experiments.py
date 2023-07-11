import numpy as np
import cvxpy as cp

import copy, types

from osbdo.utils import *
from osbdo.quasi_newton import *
from osbdo.test_functions import *


def matvec_slicing(self, *, Q, x, i):
    Qi = Q[:, self.agent_varidx[i][0]:self.agent_varidx[i][1]]
    return Qi @ self.slicing(x=self.x, i=i)


def operator_prox_minorant(self, *, t, x, mat_hat_f, hat_h, solver, mu_k=None):
    if mu_k is not None:
        hess_term = (1./(2*t)) * mu_k * cp.sum_squares(self.x - x)
        # hess_term = (1./(2*t)) * cp.quad_form(self.x - x, np.diag(mu_k))
    # prob = cp.Problem(cp.Minimize(hat_h + hess_term), self.g.domain)
    Q, shifts = mat_hat_f["Q"], mat_hat_f["shifts"]
    z = cp.Variable(self.M)
    lwbnd = []
    for i in range(self.M):
        lwbnd += [self.matvec_slicing(Q=Q, x=self.x, i=i) + shifts[:, i] <= z[i]]
    prob = cp.Problem(cp.Minimize(cp.sum(z) + self.g.function + hess_term), lwbnd + self.g.domain)

    try:
        prob.solve(solver=solver)
    except:
        try:
            print('proxmin ECOS fail, increase #iter in ECOS')
            prob.solve(solver='ECOS', max_iters = 200000)
        except:  
            print('proxmin ECOS fail again, switch to OSQP')
            prob.solve(solver = 'OSQP', max_iter = 500000)
            
    hat_f_tilde_x = cp.sum(z).value
    hat_h_tilde_x = hat_f_tilde_x + self.g.function.value
    thetas = np.zeros(self.M * Q.shape[0])
    for i in range(self.M):
        # store coefficients for active constraints
        # total sum is 1 for a given agent over all minorant coefficients
        thetas[i * Q.shape[0] : (i + 1) * Q.shape[0]] = lwbnd[i].dual_value
    return prob, hat_h_tilde_x, hat_f_tilde_x, thetas


def subgrad_h_tilde_x(self, *, affine_minorants, mat_hat_f, thetas, x, tilde_x, t, k, mu_k=None):
    # get grad_f \in \partial f(tilde_x)
    grad_f = np.zeros(self.x.size)
    idx = 0
    for i in range(self.M):
        grad_f[idx:idx + affine_minorants[i].q.size] = affine_minorants[i].q.reshape(-1)
        idx += affine_minorants[i].x.size
    if mu_k is not None:
        grad_hat_h = (1./t) * mu_k * (x - tilde_x)
        assert grad_hat_h.shape == grad_hat_h.flatten().shape
    
    if k == 1:
        return None, grad_f, None, None, grad_hat_h
    Q = mat_hat_f["Q"]
    # get grad_hat_f \in \partial \hat f(tilde_x)
    grad_hat_f = np.zeros(self.x.size)
    for i in range(self.M):
        grad_hat_f[self.agent_varidx[i][0]:self.agent_varidx[i][1]] = \
                Q[:, self.agent_varidx[i][0]:self.agent_varidx[i][1]].T @ thetas[i*Q.shape[0]:(i + 1)*Q.shape[0]]
    try:
        assert np.allclose(thetas.sum(), self.M) and np.allclose(np.ones(self.M), thetas.reshape(self.M, Q.shape[0]).sum(axis=1))
    except:
        print(thetas.reshape(self.M, Q.shape[0]).sum(axis=1))
    grad_g = grad_hat_h - grad_hat_f
    grad_h = grad_f + grad_g
    return grad_h, grad_f, grad_hat_f, grad_g, grad_hat_h


def curve_search(self, *, t, tL, tR, x, hat_h, h_x, mat_hat_f, k, hat_fis, solver, mu_k=None, \
                    eta=0.0005, eps=0.00005, m1=0.0005, m2=0.001, m3=1, m4=5, t_min=1e-6):
    assert 0 < m1 < m2 < 1
    ## step 1. Solve proximal minorant problem
    prob, hat_h_tilde_x, hat_f_tilde_x, thetas = self.operator_prox_minorant(t=t, mu_k=mu_k, x=x, \
                                            hat_h=hat_h, mat_hat_f=mat_hat_f, solver=solver)
    tilde_x = self.x.value
    hat_h_prox = prob.value
    g_tilde_x = self.g.function.value
    f_tilde_x, affine_minorants = self.get_f_x(x=tilde_x, solver=solver)
    h_tilde_x = f_tilde_x + g_tilde_x

    try:
        assert np.allclose(hat_h.value, hat_h_tilde_x)
    except:
        print(f"{hat_h.value=}, {hat_h_tilde_x=}")

    # q_t \in \partial h(tilde_x);  hat_q_t \in \partial \hat h(tilde_x)
    q_t, grad_f, grad_hat_f, grad_g, hat_q_t = self.subgrad_h_tilde_x(affine_minorants=affine_minorants, mat_hat_f=mat_hat_f, x=x, \
                                            thetas=thetas, tilde_x=tilde_x, mu_k=mu_k, t=t, k=k)
    
    vals = (tilde_x, h_tilde_x, hat_h_prox, grad_f, q_t, hat_q_t, affine_minorants)
    if k == 1:
        return "init", t, tL, tR, vals
    hat_grad_T_step = (hat_q_t.T.dot(tilde_x - x)).sum()
    grad_T_step = (q_t.T.dot(tilde_x - x)).sum()

    # nominal decrease
    delta_t = h_x - hat_h_tilde_x + 0.5 * hat_grad_T_step
    # linearization error
    e_t     = h_x - h_tilde_x     + grad_T_step
    hat_e_t = h_x - hat_h_tilde_x + hat_grad_T_step

    # test_curve_search(delta_t, e_t, hat_e_t, solver, h_x, h_tilde_x, self.x, x, mu_k, q_t, tilde_x,\
    #             hat_h_tilde_x, hat_q_t, hat_fis, grad_g, tL, t, tR, grad_hat_f, f_tilde_x,\
    #             self.g, hat_h, self.sum_hat_fi, self.get_f_x, grad_f)
        
    # check stopping criteria
    if np.linalg.norm(hat_q_t) <= eta and hat_e_t <= eps:
        return "stop", t, tL, tR, vals
    ## step 2
    descent = (h_tilde_x <= h_x - m1 * delta_t)
    if descent:
        tL = t
    else:
        ## step 3
        tR = t
        if tL == 0 and (e_t <= m3 * delta_t):
            return "null", t, tL, tR, vals
    if descent:
        ## step 4
        if grad_T_step >= - m2 * delta_t:
                return "descent", t, tL, tR, vals
        ## step 5
        if tR == np.inf and (np.linalg.norm(hat_q_t) <= eta or hat_grad_T_step >= - m4 * hat_e_t):
            return "cutting-plane", t, tL, tR, vals
    ## step 6
    if tR == np.inf:
        t = t * 1.5
    else:
        t = max(t/1.5, tL + (t-tL)/2)
    if t < t_min:
        return "null", t, tL, tR, vals
    # print(f"    {tL=}, {t=}, {tR=}, {k=}") 
    return "continue", t, tL, tR, vals


def vars_memory_to_matrix(self, *, k, memory, vars_memory, q0, shift0):
    size = min(k, memory)
    Q = np.zeros((size + 1, self.x.size))
    shifts = np.zeros((size + 1, self.M))
    for i in range(self.M):
        shifts[:size, i] = vars_memory[i]["shifts"][:size]
        Q[:size, self.agent_varidx[i][0]:self.agent_varidx[i][1]] = vars_memory[i]["normals"][:size]
    Q[-1, :] = q0
    shifts[-1, :] = shift0
    return Q, shifts


def solve_var_metric(self, *, memory=np.inf, rel_gap=10**(-2), abs_gap=10**(-3), max_iter=100, solver='ECOS', \
            agent_reply_pattern=None, minorant_update='agg_lin', print_freq=1, poorman=True):
    
    self.lower_bnd = [-np.inf]
    self.upper_bnd = [np.inf]
    U = np.inf; L = -np.inf
    agents = self.agents 
    g = self.g 
    if minorant_update == 'agg_lin':
        func_update_minorant = self.update_minorant_agg_linearization
    elif minorant_update == 'drop_constr':
        func_update_minorant = self.update_minorant_dropping_constraints
    
    # construct parameters
    k = 0
    eta = 0.01
    M = self.M
    
    # Initialization: x_0, init_minorant hat_fis,initialize hat_f, hat_h, h_x
    x =  self.get_init_feasible(solver=solver)
    
    hat_fis = [0] * M
    vars_memory = [0] * M
    agent_update_time = [0]*M
    self.agent_fail = [True]*M
    # store m of subgradients and shifts for each agent (i=1, ..., M) 
    mat_hat_f = {"Q":np.zeros((1, self.x.size)), "shifts":np.zeros((1, self.M))}
    
    for i in range(M): 
        hat_fis[i] = agents[i].get_init_minorant()
        if memory < np.inf:
            # create minorants and variables for finite memory
            vars_memory[i] = {}
            vars_memory[i]["init"]    = agents[i].get_init_minorant()
            vars_memory[i]["lwb_const"] = agents[i].lwb_const
            vars_memory[i]["shifts"]  = np.zeros((memory))
            vars_memory[i]["normals"] = np.zeros((memory, agents[i].dim)) 
        # store initial minorants in the first row of Q, shifts
        mat_hat_f["shifts"][0, i] = agents[i].lwb_const  

    hat_h = self.update_hat_h(hat_fis)  
    
    # evaluate h(x_0)
    f_x, _ = self.get_f_x(x = x, solver=solver)
    h_x = f_x + g.function.value

    num_descents = 0
    mu_k = 1
    cs_max_iter = 60
    
    # iterative solve
    while  (((U-L) >= abs_gap) and ((U-L)>=rel_gap * min(np.abs(U), np.abs(L)) or U*L<=0) and (k < max_iter)):
        
        ## update k, U, L  
        L, U, hat_x_star = self.record_optim_gap(hat_h, h_x, k, U=U, solver=solver, printing=(k % print_freq == 0))
        k += 1

        ## step 1 Curve search
        t = 1; tL = 0; tR = np.inf
        status = "continue"
        iters = 0
        k_init = copy.copy(k) 
        
        while status == "continue" and not(iters == 1 and k == 1) and (iters <= cs_max_iter):
            status, t, tL, tR, vals = self.curve_search(t=t, tL=tL, tR=tR, mu_k=mu_k, x=x, hat_h=hat_h,\
                                    h_x=h_x, mat_hat_f=mat_hat_f, k=k, solver=solver, hat_fis=hat_fis)
            iters = iters + 1
        assert k == k_init
        tilde_x, h_tilde_x, hat_h_prox, grad_f, grad_h, grad_hat_h, affine_minorants = vals
        if ((k - 1)% print_freq == 0):
            print(f"   CS {status=}, {iters=}")
        if k == 1:
            delta = h_x - hat_h_prox
            ## step 2 Update iterate
            if  h_x - h_tilde_x >= eta * delta: 
                x = tilde_x
                h_x = h_tilde_x
        else:
            if status == "stop":
                print(f"   Terminating loop")
                break

            ## step 2 Update current iterate
            if status != "null":
                x = tilde_x
                h_x = h_tilde_x
                x_cur = tilde_x.reshape(-1)
                grad_cur = grad_h 
        
        ## step 2 update H_k = G_k G_K^T (or H_k = \mu_k I)
        if status == "descent" and k >= 2:
            num_descents += 1
            if num_descents >= 2:
                v = grad_cur - grad_prev
                u = x_cur - x_prev + (t/mu_k) * v
                mu_k = v.dot(v) / v.dot(u)

        if status != 'null' and k >= 2:
            x_prev = np.copy(x_cur)
            grad_prev = np.copy(grad_cur)

        ## step 3 Update agent minorants \hat f_i = max(\hat f_i, f_i + qi^T (x_i - tilde_x_i))
        for i in range(M):
            if affine_minorants[i] is None:
                continue
            agent_update_time[i] += 1
            hat_fis[i], vars_memory_i = func_update_minorant(hat_fi = hat_fis[i], \
                                                affine_minorant = affine_minorants[i],\
                                                i = i, memory = memory, vars_memory = vars_memory[i],\
                                                k = agent_update_time[i])
            vars_memory[i] = vars_memory_i
        Q, shifts = self.vars_memory_to_matrix(k=k, memory=memory, vars_memory=vars_memory, \
                                                shift0=mat_hat_f["shifts"][-1], q0=mat_hat_f["Q"][-1])
        mat_hat_f["Q"] = Q
        mat_hat_f["shifts"] = shifts

        ## step 3 Update \hat h
        hat_h = self.update_hat_h(hat_fis)

    agent_x = []
    for i in range(M):
        agent_x.append(self.cond.solver_to_agent_np(x=self.slicing(x=x, i=i), i=i))
    x_global = np.hstack(agent_x)
    self.hat_h = hat_h
    return agent_x, x_global


def feasible_L_k(self, *, hat_h, l_k, solver='ECOS'):
    s = cp.Variable()
    prob = cp.Problem(cp.Minimize(s), self.g.domain + [hat_h - s <= l_k])
    try:
        prob.solve(solver=solver)
    except:
        try:
            # print('proxmin ECOS fail, increase #iter in ECOS')
            prob.solve(solver='ECOS', max_iters = 200000)
        except:  
            # print('proxmin ECOS fail again, switch to OSQP')
            prob.solve(solver = 'OSQP', max_iter = 500000)
    return s.value <= 0
    

def operator_doubly_stabilized(self, *, hat_h, x_k, tau_k, l_k, solver):
    r = cp.Variable()
    prox_term = (1./(2*tau_k)) * cp.sum_squares(self.x - x_k)
    constraints = [hat_h <= r, r <= l_k] + self.g.domain
    prob = cp.Problem(cp.Minimize(r + prox_term), constraints)

    try:
        prob.solve(solver=solver)
    except:
        try:
            # print('proxmin ECOS fail, increase #iter in ECOS')
            prob.solve(solver='ECOS', max_iters = 200000)
        except:  
            # print('proxmin ECOS fail again, switch to OSQP')
            prob.solve(solver = 'OSQP', max_iter = 500000)

    assert np.allclose(constraints[0].dual_value, 1 + constraints[1].dual_value)
    return self.x.value, r.value, constraints[1].dual_value


def solve_doubly_stabilized(self, *, ds_params, memory=np.inf, rel_gap=10**(-2), abs_gap=10**(-3), max_iter=100, solver='ECOS', \
            agent_reply_pattern=None, minorant_update='agg_lin', print_freq=1, tol=1e-6, printing=True):
    
    self.lower_bnd = [-np.inf]
    self.upper_bnd = [np.inf]
    U = np.inf; L = -np.inf
    agents = self.agents 
    g = self.g 
    if minorant_update == 'agg_lin':
        func_update_minorant = self.update_minorant_agg_linearization
    elif minorant_update == 'drop_constr':
        func_update_minorant = self.update_minorant_dropping_constraints
    
    # construct parameters
    k = 0
    M = self.M
    tau_k, m_l, m_f, mu_max = ds_params
    # tau_k = 10.    # [1, 5, 10]; tau_k >= tau_min
    tau_min = 1e-6    # [1e-6, 1e-5, 1e-3]
    # m_l = 0.2    # [0.2, 0.5, 0.7]
    # m_f = 0.5    # [0.1, 0.5, 0.7]
    # mu_max = 5.    # [1, 5, 10]
    
    # Initialization: x_0, init_minorant hat_fis,initialize hat_f, hat_h, h_x
    x_k =  self.get_init_feasible(solver=solver)
    
    hat_fis = [0] * M
    vars_memory = [0] * M
    agent_update_time = [0]*M
    self.agent_fail = [True]*M
    
    for i in  range(M): 
        hat_fis[i] = agents[i].get_init_minorant()
        if memory < np.inf:
            # create minorants and variables for finite memory
            vars_memory[i] = {}
            vars_memory[i]["init"]      = agents[i].get_init_minorant()
            vars_memory[i]["lwb_const"] = agents[i].lwb_const
            vars_memory[i]["shifts"]  = np.zeros((memory))
            vars_memory[i]["normals"] = np.zeros((memory, agents[i].dim))   
            
    hat_h = self.update_hat_h(hat_fis)  
    
    # evaluate h(x_0)
    f_x_k, _ = self.get_f_x(x = x_k, solver=solver)
    h_x_k = f_x_k + g.function.value

    # iterative solve
    while (k < max_iter):
        
        ## update k, U, L  
        L, U, hat_x_star = self.record_optim_gap(hat_h, h_x_k, k, U=U, solver=solver, printing=(k % print_freq == 0)and printing)
        if k == 0:
            v_l_k = (1 - m_l) * (h_x_k - L)
            L_k = L
        k += 1

        ## step 1: stopping test
        if (h_x_k - L_k <= abs_gap) or ((h_x_k - L_k) <= rel_gap * min(np.abs(h_x_k), np.abs(L_k))):
            print(f"{L_k=}, {h_x_k=}, {rel_gap * min(np.abs(h_x_k), np.abs(L_k))=}")
            print("stopping: small gap")
            break

        ## step 2: trial point finding
        l_k = h_x_k - v_l_k
        # step 2.1: feasibility detection
        if not self.feasible_L_k(hat_h=hat_h, l_k=l_k):
            k -= 1
            L_k = l_k # increase lower bound on h_star
            v_l_k = (1 - m_l) * (h_x_k - L_k)
            continue
        # step 2.2: next iterate
        tilde_x_kp1, r_kp1, lambda_kp1 = self.operator_doubly_stabilized(hat_h=hat_h, x_k=x_k, tau_k=tau_k,\
                                                                            l_k=l_k, solver=solver)
        g_tilde_x_kp1 = g.function.value
        mu_k = lambda_kp1 + 1
        v_tau_k = h_x_k - r_kp1
        hat_q_k = (x_k - tilde_x_kp1) / (tau_k * mu_k)
        # func_f = lambda z: self.func_get_f_x(x=z, solver=solver)[0]
        # hat_h_tilde_x_kp1 = hat_h.value
        # test_hat_h_subgradient(grad_hat_h_a=hat_q_k, hat_h_a=hat_h_tilde_x_kp1, a=tilde_x_kp1, var_x=self.x, \
        #                 hat_h=hat_h, g=self.g, num_iters=20)
        hat_e_k = v_tau_k - tau_k * mu_k * (hat_q_k.T.dot(hat_q_k)).sum()

        ## step 3: stopping criteria
        if (hat_e_k <= tol) and ((hat_q_k.T.dot(hat_q_k)).sum() <= tol**2):
            print("stopping: eps-subgradient")
            break
        
        ## step 4: oracle call
        f_tilde_x_kp1, affine_minorants = self.get_f_x(x=tilde_x_kp1, solver=solver, agent_reply_pattern=agent_reply_pattern, k=k)
        h_tilde_x_kp1 = f_tilde_x_kp1 + g_tilde_x_kp1
        assert h_tilde_x_kp1 >= max(L, L_k), print(f"{h_tilde_x_kp1=},  {max(L, L_k)=}")

        ## step 5: descent test
        if  h_tilde_x_kp1 <= h_x_k - m_f * v_tau_k: 
            # step 5.1: descent step
            x_k = tilde_x_kp1
            h_x_k = h_tilde_x_kp1
            tau_k = tau_k * mu_k
            v_l_k = min(v_l_k, (1 - m_l) * (h_x_k - L_k))
            if ((k - 1)% print_freq == 0) and printing:
                print(f"   descent step: {tau_k=}, {mu_k=}, {L_k=}, {l_k=}")
        else:
            # step 5.2: null step
            tau_k = (tau_min + tau_k) / 2.
            if mu_k > mu_max:
                v_l_k = m_l * v_l_k
            if ((k - 1)% print_freq == 0) and printing:
                print(f"   null step: {tau_k=}, {mu_k=}, {L_k=}, {l_k=}")
        # Update agent minorants \hat f_i = max(\hat f_i, f_i + qi^T (x_i - tilde_x_i))
        for i in range(M):
            if affine_minorants[i] is None:
                continue
            agent_update_time[i] += 1
            hat_fis[i], vars_memory_i = func_update_minorant(hat_fi = hat_fis[i], \
                                                affine_minorant = affine_minorants[i],\
                                                i = i, memory = memory, vars_memory = vars_memory[i],\
                                                k = agent_update_time[i])
            vars_memory[i] = vars_memory_i
        # Update \hat h
        hat_h = self.update_hat_h(hat_fis)
    if printing:
        print(f"rel_gap={self.get_rel_gap(U=U, L=L)}, {L=}, {L_k=}, {U=}, {l_k=}")
    agent_x = []
    for i in range(M):
        agent_x.append(self.cond.solver_to_agent_np(x=self.slicing(x=x_k, i=i), i=i))
    x_global = np.hstack(agent_x)
    self.hat_h = hat_h
    return agent_x, x_global


def add_extra_methods_to_problem(prob):
    prob.matvec_slicing = types.MethodType(matvec_slicing, prob)
    prob.operator_prox_minorant = types.MethodType(operator_prox_minorant, prob)
    prob.subgrad_h_tilde_x = types.MethodType(subgrad_h_tilde_x, prob)
    prob.curve_search = types.MethodType(curve_search, prob)
    prob.vars_memory_to_matrix = types.MethodType(vars_memory_to_matrix, prob)
    prob.solve_var_metric = types.MethodType(solve_var_metric, prob)
    prob.feasible_L_k = types.MethodType(feasible_L_k, prob)
    prob.operator_doubly_stabilized = types.MethodType(operator_doubly_stabilized, prob)
    prob.solve_doubly_stabilized = types.MethodType(solve_doubly_stabilized, prob)
