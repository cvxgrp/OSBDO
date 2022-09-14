import numpy as np
import cvxpy as cp
from scipy import stats

from osbdo.problem import *
from osbdo.utils import *

###################  Optimal Transport Agent for Supply Chain Problem ###################

"""
Optimal Transport Agent
Defined by m input  nodes and n output nodes.
The task is to choose the flows on edges, respecting given capacities
and minimizing the cost of sending the flow.
"""
class OptimalTransport(Agent):

    def _construct_params(self):
        """
        Method for constructing parameters
        Input: 
            none
        Output: 
            none
        Change: 
            self.lin: np.array(size=(n,m)) 
                      linear cost term
            self.quad: np.array(size=(n,m)) 
                       quadratic cost term
            self.cap: np.array(size=(n,m)) 
                      capacity limit on capacity matrix
        """
        self.m = self.params["m"]
        self.n = self.params["n"]
        self.lin = self.params["lin"]
        self.quad = self.params["quad"]
        self.cap = self.params["cap"]
        self.norm = self.params["norm"]
        self.mu = self.params["mu"]

    def query(self, *, v, solver='ECOS'):
        X = cp.Variable((self.n, self.m), nonneg=True)
        u = cp.Variable((self.dim, ), nonneg=True)
        r = cp.Variable((self.dim, ))
        constraints = [ u - r == v, \
                        cp.hstack([cp.sum(X, axis=0), cp.sum(X, axis=1)]) == u,\
                        u <= self.upb, u >= self.lwb, \
                        cp.sum(u[:self.m]) == cp.sum(u[self.m:]) ]
        if self.cap.sum() < np.inf: constraints += [X <= self.cap]
        if self.norm == "l2":
            penalty = 0.5 * cp.sum_squares(np.diag(self.mu**(0.5)) @ r)
        elif self.norm == "l1":
            penalty = 0.5 * cp.norm(np.diag(self.mu) @ r, 1)
        F = cp.vec(X) @ cp.vec(self.lin) + 0.5 * cp.vec(cp.power(X,2)) @ cp.vec(self.quad)
        f = F + penalty
        prob = cp.Problem(cp.Minimize(f), constraints)
        try:
            prob.solve(solver=solver)
        except:
            print('solver fails, change to OSQP')
            prob.solve(solver='OSQP')
        x = v; f = f.value
        q = -constraints[0].dual_value
        return Point(x=x, q=q, f=f)

    def get_init_minorant(self):
        return cp.Constant(0)


#############################  Commodity Flow Agent #########################

class CommodityFlowRoute(Agent):
    def _construct_params(self):
        self.f = self.params['f']
        self.b = self.params['b']
        self.num_vert = self.f.shape[0]
        self.A = self.params["incidence"]
        self.init_lwb = self.params["init_lwb"]

    def query(self, *, v, solver, verbose=False):
        v = np.fmax(v,0)
        t = cp.Variable(nonneg=True)
        s = t * self.f
        z = cp.Variable((self.dim, ), nonneg=True)
        f = - t * self.b + cp.sum(cp.sum_squares(self.A @ z + s))
        constraints = [ z <= v,\
                        self.A @ z + s == 0]
        prob = cp.Problem(cp.Minimize(f), constraints)
        prob.solve(solver=solver, verbose=verbose)
        if prob.status == "infeasible": 
            print("# MCF infeasible" )
            prob.solve(solver='OSQP', verbose=verbose)
        f = f.value
        q = -constraints[0].dual_value
        return Point(x=v, q=q, f=f)

    def get_init_minorant(self):                                
        return cp.Constant(self.init_lwb)

   

 #############################  Resouce Allocaton Agent #########################


class ResourceAllocation(Agent):
    """
    minimize \sum_{i=1}^num_work -geomean(A_ir_i+bi)
    subject to r_i>=0, \sum_{i=1}^num_work r_i<=agent.x
    """
    def _construct_params(self):
        self.A_list = self.params['A_list']
        self.b_list = self.params['b_list']
        self.num_worker = len(self.A_list)

    def query(self, *, v, solver='ECOS'):
        v = v.reshape((self.dim, 1))
        if not (v >= 0).all():
            v = np.fmax(v,0)
        X = cp.Variable((self.num_worker, self.dim))
        f = 0
        for i in range(len(self.A_list)):
            f += -cp.geo_mean(self.A_list[i] @ X[i].T + self.b_list[i])
        constraints = [ cp.sum(X, axis=0, keepdims=True) <= v.T,  X >= 0 ]
        prob = cp.Problem(cp.Minimize(f), constraints)
        prob.solve(solver=solver)
        f = f.value
        q = -constraints[0].dual_value
        return Point(x=v.squeeze(), q=q, f=f)
    
    def get_init_minorant(self):
        lwb = 0
        for i in range(len(self.b_list)):
            lwb += - stats.mstats.gmean(self.A_list[i] @ self.upb + self.b_list[i], axis=None)
        return cp.Constant(lwb)
        
        
############################### Federated Learning Agent ############################

class FederatedLearning(Agent):
    def _construct_params(self):
        self.X = self.params['x']
        self.y = self.params['y']
        
    def query(self, *, v, solver='ECOS'):
        v = v
        q = np.sum(np.diag((1/(1+np.exp(-self.X@v))-(self.y/2+1/2)))@self.X,axis=0)
        f = np.log(1+np.exp(np.multiply(self.y,-self.X@v)))
        
        return Point(x = v, q = q, f = f.sum())
        
    def get_init_minorant(self):
        return cp.Constant(0)
       
############### Agent for Intersection of Convex Sets Problem ##################

class DistanceToSet(Agent):
    """
    min   \|xi-vi\|_2^2
    s.t.  Ai vi <= bi
    """
    def _construct_params(self):
        self.A = self.params['A']
        self.b = self.params['b']
        
    def query(self, *, v, solver='ECOS'):
        v = v
        x = cp.Variable(v.shape)
        y = cp.Variable(v.shape)
        obj = cp.sum_squares(x-y)
        constraints = [x == v, self.A@y <= self.b]
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=solver)
        q = -constraints[0].dual_value
        f = prob.value
        return Point(x = v, q = q, f = f)
    
    def get_init_minorant(self):
        return cp.Constant(0)


############################# Hello World Agents #########################

class AffineAgent(Agent):
    """
       objective: c^T x + d
    """    
    def _construct_params(self):
        self.c = self.params['c']
        self.d = self.params['d']
        
    def query(self, *, v, solver='ECOS'):
        return Point(x=v, q=self.c, f=(self.c.T@v+self.d).sum())
    
    def get_init_minorant(self):
        var = cp.Variable(self.dim)
        prob = cp.Problem(cp.Minimize(self.c.T@var+self.d),[var<=self.upb, var>=self.lwb])
        prob.solve()
        return cp.Constant(prob.value)

            
class QuadraticAgent(Agent):
    """
       objective:(1/2) x^T P x + q^T x + r
    """
    def _construct_params(self):
        self.P = self.params['P']
        self.q = self.params['q']
        self.r = self.params['r']

    def query(self, *, v, solver='ECOS'):
        return Point(x=v, q=self.P@v+self.q, f=(0.5*v.T@self.P@v+self.q.T@v+self.r).sum())
    
    def get_init_minorant(self):
        var = cp.Variable(self.dim)
        prob = cp.Problem(cp.Minimize(0.5*cp.quad_form(var,self.P)+self.q.T@var+self.r),\
                                                                    [var<=self.upb, var>=self.lwb])
        prob.solve()
        return cp.Constant(prob.value)
