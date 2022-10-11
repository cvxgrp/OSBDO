import numpy as np
import cvxpy as cp

from osbdo.utils import *


"""
Agent class for distributed optimization
"""
class Agent:
    def  __init__(self, params):
        self.params = params
        self.dim = params["dimension"]
        self.lwb = params["lower_bound"]
        self.upb = params["upper_bound"]
        self.x = cp.Variable(self.dim)
        self._construct_params()
    
    def query(self, *, v, solver='ECOS'):
        raise NotImplementedError
    
    def get_init_minorant(self):
        raise NotImplementedError

    def _construct_params(self):
        raise NotImplementedError




"""
Coupling class for distributed optimization
"""
class Coupling:
    def __init__(self, *, agents, function, domain):
        self.function = function
        self.domain = domain
        for i in range(len(agents)):
            if agents[i].lwb is not None:
                self.domain = self.domain + [agents[i].x>=agents[i].lwb]
            if agents[i].upb is not None:
                self.domain = self.domain + [agents[i].x<=agents[i].upb]
        self.prob = cp.Problem(cp.Minimize(self.function), self.domain)




"""
Preconditioner class
Interface between our method and agent query
"""
class Preconditioner:
    def __init__(self, *, agents):
        self.upb = []
        self.lwb = []
        for i in range(len(agents)):
            self.upb.append(agents[i].upb)
            self.lwb.append(agents[i].lwb)
            
    def solver_to_agent_cp(self, *, x, i):
        if self.upb[i] is None or self.lwb[i] is None:
            return x
        return cp.multiply(self.upb[i]-self.lwb[i] , x)
    
    def solver_to_agent_np(self, *, x, i):
        if self.upb[i] is None or self.lwb[i] is None:
            return x
        return np.multiply(self.upb[i]-self.lwb[i] , x)
    
    def agent_to_solver(self, *, x, i):
        if self.upb[i] is None or self.lwb[i] is None:
            return x
        return np.multiply(1/(self.upb[i]-self.lwb[i]) , x)




"""
Distributed optimization problem class
"""
class Problem:
    def  __init__(self, *, agents, g):
        self.agents = agents
        self.M = len(agents)
        self.g = g
        # create normalized global var
        var_dim = [0]*self.M
        for i in range(self.M):
            var_dim[i] = self.agents[i].dim
        self.x = cp.Variable(sum(var_dim))
        # create global to agent variable index
        self.agent_varidx = []
        num_var = 0
        for i in range(self.M):
            self.agent_varidx.append([num_var, (num_var+var_dim[i])])
            num_var = num_var+var_dim[i]
        # create preconditioner
        self.cond = Preconditioner(agents=self.agents)
        # assign unnormalized agent var
        for i in range(self.M):
            self.g.domain += [self.agents[i].x == self.cond.solver_to_agent_cp(x=self.slicing(x=self.x, i=i), i=i)]
    

    def slicing(self, *, x, i):
        return x[self.agent_varidx[i][0]:self.agent_varidx[i][1]]
               

    def get_hat_h_star(self, hat_h, solver):
        # Return minimizer of PWL lower bound of h
        prob = cp.Problem(cp.Minimize(hat_h), self.g.domain)
        try:
            prob.solve(solver=solver)
        except:
            try:
                print('get_hat_h_star ECOS fail')
                prob.solve(solver='ECOS', max_iters = 200000)
            except:
                print('get_hat_h_star ECOS fail, switch to OSQP')
                prob.solve(solver='OSQP', max_iter = 1000000)
        return prob.value, self.x.value


    def get_init_feasible(self, solver):
        mid_point = np.zeros(self.x.shape)
        for i in range(self.M):
            if self.agents[i].upb is None or self.agents[i].lwb is None:
                 mid_point[self.agent_varidx[i][0]:self.agent_varidx[i][1]] = \
                                            np.zeros(self.agent_varidx[i][1]-self.agent_varidx[i][0])
            else:
                mid_point[self.agent_varidx[i][0]:self.agent_varidx[i][1]] = \
                            self.cond.agent_to_solver(x=(self.agents[i].upb+self.agents[i].lwb)/2, i=i)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(self.x - mid_point)), self.g.domain)
        prob.solve(solver=solver)
        return self.x.value


    def update_hat_h(self, hat_fis):
        # refine minorant \hat h
        hat_f = 0
        for i in range(self.M):
            hat_f = hat_f + hat_fis[i] 
        hat_h =  hat_f + self.g.function
        return hat_h


    def get_rel_gap(self, *, U, L):
        if U * L > 0:
            return (U - L) / (min(np.abs(U), np.abs(L)))
        else:
            return np.inf


    def record_optim_gap(self, hat_h, h_x, k, solver, U=None):
        L, hat_x_star = self.get_hat_h_star(hat_h, solver=solver)
        if U is None: U = h_x
        else: U = min(U, h_x)
        rel_gap = self.get_rel_gap(U=U, L=L)
        self.lower_bnd += [ max(L, self.lower_bnd[-1]) ]
        self.upper_bnd += [ min(U, self.upper_bnd[-1]) ]
        print(f"{k=}, {rel_gap=}, {L=}, {U=}")
        return L, U, hat_x_star


    def update_minorant(self, *, affine_minorant, hat_fi, i, memory=np.inf, vars_memory={}, k=None):
        xi = cp.vec(self.slicing(x=self.x, i=i))
        xi_k = cp.vec(affine_minorant.x)
        qi_k = cp.vec(affine_minorant.q)
        fi_k = affine_minorant.f
        if memory == np.inf:
            hat_fi_exp = cp.maximum(hat_fi, cp.sum(fi_k + qi_k @ (xi - xi_k)))
        else:           
            old_idx = (k - 1) % memory 
            vars_memory["normals"][old_idx] = qi_k.value
            vars_memory["shifts"][old_idx] = (fi_k - qi_k @ xi_k).value
            if k < memory:
                minorant = cp.max(vars_memory["normals"][:k] @ xi + vars_memory["shifts"][:k])
            else:
                minorant = cp.max(vars_memory["normals"] @ xi + vars_memory["shifts"])
            hat_fi_exp = cp.maximum(vars_memory["init"], minorant)
        return hat_fi_exp, vars_memory


    def get_f_x(self, *, x, solver, agent_reply_pattern=None, k=None):
        # get value f(x) and affine minorant for each fi at given x
        f_x = 0; affine_minorants = [0] * self.M
        for i in range(self.M):
            query_ans = None
            if agent_reply_pattern is None or agent_reply_pattern[i][k]:
                query_ans = self.agents[i].query(v=self.cond.solver_to_agent_np(x=self.slicing(x=x, i=i), i=i),solver=solver)
            if query_ans is None:
                affine_minorants[i] = None
                self.agent_fail[i] = True
            else:  
                affine_minorants[i]   = query_ans
                affine_minorants[i].x = self.slicing(x=x, i=i)
                affine_minorants[i].q = self.cond.solver_to_agent_np(x=affine_minorants[i].q, i=i)
                self.agent_fail[i] = False
                f_x += affine_minorants[i].f
        if np.array(self.agent_fail).any():
            f_x = np.inf
        return f_x, affine_minorants

    
    def get_proj_sublev_val(self, *, hat_h, x_k, L, U, solver, ratio = 0.5, verbose=False):
        prox_term = cp.sum(cp.sum_squares(self.x  - x_k))
        if L > -np.inf:
            sublev_set = [ hat_h <= ratio * U + (1 - ratio) * L ] 
        else:
            sublev_set = [ hat_h <= U ]
        status = None
        while status != 'optimal':
            try:
                prob = cp.Problem(cp.Minimize(prox_term), sublev_set + self.g.domain)
                prob.solve(solver=solver)
                status = prob.status
                if  status != 'optimal':
                    raise Exception("solver status not optimal")
            except:   
                try:
                    prob.solve(solver='OSQP', max_iter=500000)
                    status = prob.status
                    if status != 'optimal':
                        raise Exception("OSQP status not optimal")
                except:
                    ratio = ratio * 1.1 
                    if ratio>1e10:
                        raise Exception("ratio large, solve fail")
                    sublev_set = [ hat_h <= ratio * U + (1 - ratio) * L ]

        lambd = max(10**(-12), sublev_set[0].dual_value)
        rho = 2 / lambd
        return hat_h.value, rho


    def solve(self, *, memory=np.inf, rel_gap=10**(-2), abs_gap=10**(-3), max_iter=100, \
                                                            solver='ECOS', agent_reply_pattern=None):
        rhos = []
        self.lower_bnd = [-np.inf]
        self.upper_bnd = [np.inf]
        U = np.inf; L = -np.inf
        agents = self.agents 
        g = self.g 
        
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
        
        for i in  range(M): 
            hat_fis[i] = agents[i].get_init_minorant()
            if memory < np.inf:
                # create minorants and variables for finite memory
                vars_memory[i] = {}
                vars_memory[i]["init"]    = agents[i].get_init_minorant()
                vars_memory[i]["shifts"]  = np.zeros((memory))
                vars_memory[i]["normals"] = np.zeros((memory, agents[i].dim))   

                
        hat_h = self.update_hat_h(hat_fis)  
        
        # evaluate h(x_0)
        f_x, _ = self.get_f_x(x = x, solver=solver)
        h_x = f_x + g.function.value
      
        # iterative solve
        while  (((U-L) >= abs_gap) and ((U-L)>=rel_gap * min(np.abs(U),np.abs(L)) or U*L<=0) and (k < max_iter)):
            
            ## additional: update k, U, L  

            L, U, hat_x_star = self.record_optim_gap(hat_h, h_x, k, U=U, solver=solver)
            k += 1

            ## step 1 Tentative update with rho using projected sublevel set
            if k <= 20:
                hat_h_val, rho = self.get_proj_sublev_val(hat_h=hat_h, x_k=x, L=L, U=U, solver=solver)
                tilde_x = self.x.value
                prox_term_val = (rho / 2) * np.linalg.norm(tilde_x - x)**2
                hat_h_prox = hat_h_val + prox_term_val
                    
            ## step 1 Tentative update with rho using geometric mean over last 5 steps
            if k ==  21:
                rho = np.product(rhos[-5:])**(1/5)
            rho = min(max(rho, 1e-6), 1e6)
            rhos.append(rho)

            ## step 1 Tentative update with fixed rho

            if k > 20: 
                prox_term = (rho / 2) * cp.sum_squares(self.x - x)
                prob = cp.Problem(cp.Minimize(hat_h+prox_term), g.domain)
                
                try:
                    prob.solve(solver=solver)
                except:
                    try:
                        print('proxmin ECOS fail, increase #iter in ECOS')
                        prob.solve(solver='ECOS', max_iters = 200000)
                    except:  
                        print('proxmin ECOS fail again, switch to OSQP')
                        prob.solve(solver = 'OSQP', max_iter = 500000)
                tilde_x = self.x.value
                hat_h_prox = prob.value    
            
            ## step 2 Query agents, update affine_minorants and h_tilde_x
            f_tilde_x, affine_minorants = self.get_f_x(x=tilde_x, solver=solver, agent_reply_pattern=agent_reply_pattern, k=k)
            if f_tilde_x != np.inf:
                h_tilde_x = f_tilde_x + g.function.value
                ## step 3 Evaluate delta
                delta = h_x - hat_h_prox

                ## step 4 Update iterate
                if  h_x - h_tilde_x >= eta * delta: 
                    x = tilde_x
                    h_x = h_tilde_x
                
            ## step 5 Update agent minorants \hat f_i = max(\hat f_i, f_i + qi^T (x_i - tilde_x_i))
            for i in range(M):
                if affine_minorants[i] is None:
                    continue
                agent_update_time[i] += 1
                hat_fis[i], vars_memory_i = self.update_minorant(hat_fi = hat_fis[i], \
                                                   affine_minorant = affine_minorants[i],\
                                                   i = i, memory = memory, vars_memory = vars_memory[i],\
                                                   k = agent_update_time[i])
                vars_memory[i] = vars_memory_i

            ## step 6 Update \hat h
            hat_h = self.update_hat_h(hat_fis)

        agent_x = []
        for i in range(M):
            agent_x.append(self.cond.solver_to_agent_np(x=self.slicing(x=x, i=i), i=i))
        x_global = np.hstack(agent_x)
        self.hat_h = hat_h
        return agent_x, x_global
