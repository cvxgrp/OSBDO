import numpy as np
import cvxpy as cp

from osbdo.utils import *


############################# Find true solution for Supply chain problem #########################

def cvx_supply_chain(*, params, norm="l1", x_val=None):
    """
    params: parameters for OT agents

    Returns the true solution to the distributed optimization problem
    of  Optimal Transport using CVXPY and  centralised formulation
        min. \sum_i  f_i(xi)
        s.t. Ax = b,
             Cx <= d   
    """
    N = len(params)
    x_size = sum([params[i]["dimension"] for  i in range(1, N-1)])
    A = [] 
    cur_sum = 0; idx = []; l_all = []; u_all = []
    for i in range(1, N-1):
        m, n = params[i]["m"], params[i]["n"]
        size = params[i]["dimension"]
        idx += [(cur_sum, cur_sum + size - 1)]
        if i < N-2:
            A += [np.concatenate([np.zeros((n, cur_sum + m)), np.eye(n), -np.eye(n), \
                                            np.zeros((n, x_size - cur_sum - m - 2*n)) ], axis=1)]
        A += [np.concatenate([np.zeros((1, cur_sum)), np.ones((1,m)), -np.ones((1, n)),\
                                            np.zeros((1, x_size - cur_sum - m - n)) ], axis=1)]
        l = np.zeros((size, ))
        u = params[i]["upper_bound"]
        l_all += [ l ]
        u_all += [ u ]
        cur_sum += size
    A = np.concatenate(A, axis=0)
    b = np.zeros((A.shape[0], ))
    l = np.concatenate(l_all, axis=0)
    u = np.concatenate(u_all, axis=0)
    idx = [params[0]['dimension'] - 1] + idx + [params[-1]['dimension']]
    
    mu = get_mus(50, params[1:-1], N-2)
    z = cp.Variable((A.shape[1], ))
    f = 0
    constraints = []
    for i in range(1, N-1):
        xi = z[idx[i][0] : idx[i][1]+1]
        size = params[i]["dimension"]
        m, n = params[i]["m"], params[i]["n"]
        X = cp.Variable((n,m), nonneg=True)
        tilde_x = cp.Variable((size, ), nonneg=True)
        lin, quad, cap = params[i]["lin"], params[i]["quad"], params[i]["cap"]
        constraints += [ cp.hstack([cp.sum(X, axis=0), cp.sum(X, axis=1)]) == tilde_x]
        if cap.sum() < np.inf: constraints += [X <= cap]

        f += cp.vec(X) @ cp.vec(lin) + 0.5 * cp.vec(cp.power(X,2)) @ cp.vec(quad)
        r = cp.Variable((size, ))
        constraints += [ r == tilde_x - xi ]
        if norm == "l2":
            f += 0.5 * cp.sum_squares(np.diag(mu[i-1]**(0.5)) @ r)
        elif norm == "l1":
            f += 0.5 * cp.norm(np.diag(mu[i-1]) @ r, 1)
    obj = f + cp.sum(params[0]["sale"].T @ z[:params[0]['dimension']])-cp.sum(params[-1]["retail"].T @ z[- params[-1]['dimension']:])
    if x_val is not None:
        constraints += [z ==  x_val] 
    constraints += [A@z==b, z<=u, z>=l]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver='ECOS')
    return obj.value, z.value, prob 

############################# Find true solution for network of Multi Commodity Flow Agent #########################

def cvx_multi_commodity_flow(*, params, solver='ECOS'):
    """
    Returns the true solution to the distributed optimization problem
    of  Multi Commodity Flow using CVXPY and  centralised formulation
        min. \sum_i  f_i(xi)
        s.t. x1 + ... + xM = c   
    """
    M = len(params)
    constraints = []
    f = 0
    A = params[0]['incidence']
    num_vertices, num_edges = A.shape[0], A.shape[1]
    z = cp.Variable((M, num_edges))
    for i in range(M):
        xi = z[i:i+1].T
        fi = params[i]["f"].reshape(params[i]["f"].shape[0],1)
        t = cp.Variable(nonneg=True)
        si = t * fi
        zi = cp.Variable((num_edges, 1), nonneg=True)
        f += -t * params[i]['b'] + cp.sum(cp.sum_squares(A @ zi  + si))
        constraints += [ A @ zi  + si == 0, 
                         zi <= xi, 
                         xi <= params[i]['upper_bound'].reshape((num_edges,1)),
                         xi >= params[i]['lower_bound'].reshape((num_edges,1))]
    constraints += [cp.sum(z, axis=0, keepdims=True).T<=params[0]['upper_bound'].reshape((num_edges,1))]
    prob = cp.Problem(cp.Minimize(f), constraints) 
    prob.solve(solver=solver)
    return z.value, prob, f.value

############################# Find true solution for network of Resouce Allocaton Agents #########################

def cvx_resource_allocation(*, params, R):
    """
    Returns the true solution to the distributed optimization problem
    of  Resource Allocation using CVXPY and  centralised formulation
        min. \sum_i  f_i(xi)
        s.t. Ax = b,
             Cx <= d   
    """
    z = cp.Variable(len(params)*params[0]['dimension'])
    num_agents = len(params)
    constraints = []
    f = 0
    xi_list = []
    for i in range(num_agents):
        xi = z[(i*params[0]['dimension']):((i+1)*params[0]['dimension'])]
        X = cp.Variable((len(params[i]['A_list']), params[i]['dimension']))
        for j in range(len(params[i]['A_list'])):
            f += -cp.geo_mean(params[i]['A_list'][j] @ X[j].T + params[i]['b_list'][j])
        constraints += [cp.sum(X,axis=0) <= xi, X>=0, xi<=params[i]['upper_bound'], 
                        xi>=params[i]['lower_bound']]
        xi_list.append(xi)
    prob = cp.Problem(cp.Minimize(f), constraints+[sum(xi_list)<=R])
    prob.solve(solver='ECOS')
    return f.value, z.value

############################# Find true solution for Federated Learning Agents #########################

def cvx_federated_learning(*, params):
    M = len(params)
    z = cp.Variable((M,params[0]['dimension']))
    f = 0
    constraints = []
    for i in range(M):
        f += cp.sum(cp.logistic(cp.multiply(params[i]['y'],- params[i]['x'] @ z[i])))
        constraints += [z[i]==z[0]]
    prob = cp.Problem(cp.Minimize(f+5*cp.norm(z[0],1)),constraints)
    prob.solve(solver='ECOS')
    return z.value, prob, (f+5*cp.norm(z[0],1)).value

############################# Find true solution for Intersection of Convex Sets Problem #########################

def cvx_intersection_cvx_sets(*, params):
    num_agents = len(params)
    f = 0
    constraints = []
    z = cp.Variable((num_agents, params[0]['dimension']))
    for i in range(num_agents):
        xi = z[i]
        vi = cp.Variable(xi.shape)
        f += cp.sum_squares(xi-vi)
        constraints += [params[i]['A']@vi<=params[i]['b'], z[i]==z[0]]
    prob = cp.Problem(cp.Minimize(f),constraints) 
    prob.solve(solver='ECOS')
    return z.value, prob, f.value