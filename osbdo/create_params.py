import numpy as np
import random, copy
import cvxpy as cp

from osbdo.utils import *
from osbdo.agents import *
from osbdo.problem import *

############################# Create parameters for network of Optimal Transport Agents #########################

def sc_params(ms, ns):
    def generate_costs(n, m):
        lin  = np.exp(np.random.normal(loc = 0.07, scale = 0.7, size = (n, m))) 
        cap  = np.exp(np.random.normal(loc = 0,   scale = 1, size = (n, m)))
        quad = 0.5 * np.divide(lin, cap)
        return [lin, quad, cap]

    N = len(ms) + 2
    params = [0] * N
    sale   = np.random.uniform(low = 8,  high = 10, size = (ms[0], 1))   
    retail = np.random.uniform(low = 10, high = 12, size = (ns[-1], 1))   
    params[0]   = {"m":0,      "n":ms[0],  "sale":sale, "dimension" : ms[0] }
    params[N-1] = {"m":ns[-1], "n":0, "retail":retail, "dimension" : ns[-1]}

    ranges = [0] * N
    for i, (m, n) in enumerate(zip(ms, ns)):
        lin, quad, cap = generate_costs(n, m)
        idx = i + 1
        params[idx] = {"cap":cap, "lin":lin, "quad":quad, "dimension": m+n, "m":m, "n":n, "norm":"l1"}
    for i, (m, n) in enumerate(zip(ms, ns)):
        idx = i + 1
        if N-1 > idx > 0:
            # input nodes ai and output nodes bi
            a = params[idx]["cap"].sum(axis=0).T
            b = params[idx]["cap"].sum(axis=1)
            if idx != N-2:
                b = np.maximum(b, params[idx+1]["cap"].sum(axis=0).T)
            if idx != 1:
                a = np.maximum(a, params[idx-1]["cap"].sum(axis=1))
            assert a.shape[0] == m and b.shape[0] == n
            # range for each node of  agent i is maximum over the capacities of
            # incoming and outgoing edges
            ranges[idx] = np.concatenate([a, b], axis=0)
    ranges[0] = params[1]["cap"].sum(axis=0).T
    ranges[N-1] = params[N-2]["cap"].sum(axis=1)
    for i in range(N):
        assert np.zeros((params[i]["dimension"], )).shape == ranges[i].shape
        params[i]["lower_bound"] = np.zeros((params[i]["dimension"], ))
        params[i]["upper_bound"] = copy.deepcopy(ranges[i])
    
    return params
   
def sc_agents(params):
    N = len(params) - 2
    agents  = [0]*N
    mus = get_mus(50, params[1:-1], N)
    for i  in  range(N):
        params[1:-1][i]['mu'] = mus[i]
        agents[i] = OptimalTransport(params[1:-1][i])
    return agents

def sc_coupling(params, agents):
    obj = 0
    obj += cp.sum(params[0]["sale"].T @ agents[0].x[:params[0]['dimension']])
    obj +=  - cp.sum(params[-1]["retail"].T @ agents[-1].x[- params[-1]['dimension']:])
    
    domain = []
    N = len(params)-2
    params = params[1:-1]
    for i in range(N):
        m, n = params[i]["m"], params[i]["n"]
        if i<N-1:
#             print(f'{i=},{m=},{agents[i].x[m:].shape=},{agents[i+1].x[:m].shape=}')
            domain += [agents[i].x[-n:] == agents[i+1].x[:n]]
        domain += [cp.sum(agents[i].x[:m]) == cp.sum(agents[i].x[m:])]
    
    g = Coupling(agents=agents, function= obj, domain = domain)
    return g

############################# Create parameters for network of Multi-Commodity flow Agent #########################

def mcf_params(*, num_vertices, num_edges, M):
    """
    Generate parameters for resource management of Multi-Commodity flow Agent
    Arguments:
        num_vertices: int
        num_edges: int
        M: int
            number of commodities
    Returns:
        params
    """
    n = num_vertices
    # create  incidence matrix A for directed graph
    # to make graph strongly connected -- add a cycle
    A = np.zeros((num_vertices, num_edges))
    unsampled_edges = list(range(n*(n-1)))
    cur_edge = 0; global_idx = -1
    for i in range(num_vertices):
        for j in range(num_vertices):
            if i == j: continue
            global_idx += 1
            idx = (i+1) % (num_vertices)
            if j != idx: continue
            A[i, cur_edge] =  1
            A[j, cur_edge] = -1
            unsampled_edges.remove(global_idx)
            cur_edge += 1
    assert (len(unsampled_edges) == n*(n-1) - num_vertices) and (cur_edge == num_vertices)
    chosen_edges = random.sample(unsampled_edges, num_edges - num_vertices)
    chosen_pairs = random.sample(list(range(n*(n-1))), M)
    idx = 0; cur_edge = num_vertices
    F = np.zeros((M, num_vertices))
    cur_pair = 0

    for i in range(num_vertices):
        for j in range(num_vertices):
            if i == j: continue
            if idx in chosen_edges:
                # direction of the edge is coming from (i,j) or (j,i)
                # direction = (-1)**(int(np.random.rand() > 0.5))
                A[i, cur_edge] =  1
                A[j, cur_edge] = -1
                cur_edge += 1
            if idx in chosen_pairs:
                # source  sink pair for  i-th commodity
                F[cur_pair, i] =  1
                F[cur_pair, j] = -1
                cur_pair += 1
            idx += 1
    assert idx == (n*(n-1)) and (cur_edge  == len(chosen_edges) + num_vertices)
    assert cur_pair == M and (A.sum(axis=0) == 0).all()

    # create capacities on edges
    #low  = min(100, max(10, min(num_edges)/10 ))
    cap_min = 0.2; cap_max = 2
    cap = np.random.uniform(low=cap_min, high=cap_max, size=(num_edges, 1))  
    init_lwb = - num_edges * cap_max
    b_min =  0.5; b_max = 1.5
    params = [0]*M
    for i in range(M):
        params[i] = { 'f': F[i],
                                'b': np.random.uniform(low=b_min, high=b_max),
                                'dimension': num_edges, \
                                'lower_bound': np.zeros((num_edges,)),\
                                'upper_bound': cap.squeeze(), \
                                "incidence": A,\
                                "init_lwb": init_lwb}
    return params

def mcf_agents(params, probs=None, alpha=1):
    N = len(params)
    agents  = [0]*N
    for i in range(N):
        agents[i] = CommodityFlowRoute(params[i])
    return agents

def mcf_coupling(agents, params):
    domain = [sum([agents[i].x for i in range(len(agents))])<=params[0]["upper_bound"]]
    g = Coupling(function = cp.Constant(0), agents=agents, domain=domain)
    return g

############################# Create parameters for network of Resouce Allocaton Agent #########################

def ra_params(num_resources, num_agents, num_participants):
    """
    Generate parameters for resource management of Resource Allocation (RA) agents
    Arguments:
        num_resources: int
            number of total accessible resources 
        num_agents: int
            number of agents 
        num_participants: int
            number of RA participants
    Returns:
        params: list of parameters for each of N agents
            "A_list": list of utility relation matrix
                each A has 5 rows, num_resources columns, uniform random 0-1 valued entries. 
            "b_list": list of utility constant vector
                each b has 5 rows, uniform random (0-1)*int(num_resources/10) valued entries if num_resources>=0, 
                otherwise uniform random (0-1)*num_resources valued entries
            "range":  Range object
                range of each agent's public variables
            "size": scalar
                size of public variables for each agents
        g_params: parameters on global variable x
            "C", "d": np.ndarray
                inequality constraint Cx <= d
            "var_size": tuple
                size of global variable x
            "idx": list
                linkage between  agents' public variable to global variable x
            "N": scalar
                num_agents
            "n": scalar
                num_resources
            "cap": np.ndarray
                maximum amount of each resource
            "range": Range object
                range of global variable x                             
    """
    num_resource_per_participant = int(num_resources/10)
    cap = np.exp(np.random.normal(loc = np.log(num_participants/10),   scale = 1, size = (num_resources, )))
    resource_list = []
    for i in range(num_participants):
        res = random.sample(range(num_resources), num_resource_per_participant)
        resource_list.append(res)
    participants_params = [0]*num_participants
    for i in range(num_participants):
        A = np.random.uniform(size=(5,num_resources))
        A[:,resource_list[i]] = 0
        participants_params[i] = {'A': A,
                     'b': (num_resource_per_participant)*np.random.uniform(size=(5,)),
                      }
        
    params = [0]*num_agents
    num_participants_per_agent = int(num_participants/num_agents)
    for i in range(num_agents):
        A_list = []
        b_list = []
        for j in range(i*num_participants_per_agent, (i+1)*num_participants_per_agent):
            A_list.append(participants_params[j]['A'])
            b_list.append(participants_params[j]['b'])
        params[i] = {'dimension': num_resources, 'lower_bound': 0, 'upper_bound': cap,
                     'A_list':A_list,'b_list':b_list}

    return params, cap
            
def ra_agents(params):
    M = len(params)
    agents  = [0]*M
    for i  in  range(M):
        agents[i] = ResourceAllocation(params[i])
    return agents

def ra_coupling(agents, R):
    domain = [sum([agents[i].x for i in range(len(agents))])<=R]
    g = Coupling(agents = agents, function = cp.Constant(0), domain = domain)
    return g

############################# Create parameters for Federated Learning Agents #########################

def fl_params(num_samples, num_agents, size):
    assert(num_samples >= num_agents)
    cols = random.sample(range(size), int(size/10))
    theta_true = np.zeros((size,))
    theta_true[cols] = np.random.normal()
    X = np.random.normal(size=(num_samples, size))
    Y = sign(X @ theta_true + 0.1*np.random.normal())
    num_samples_per_agent = int(num_samples/num_agents)
    params = [0]*num_agents
    for i in range(num_agents):
        params[i] = {'x':X[(i*num_samples_per_agent):((i+1)*num_samples_per_agent),:],  
                         'y':Y[(i*num_samples_per_agent):((i+1)*num_samples_per_agent)],    
                         'lower_bound': None,
                         'upper_bound': None,
                         'dimension':size}
    
    return params

def fl_agents(params):
    M = len(params)
    agents  = [0]*M
    for i  in  range(M):
        agents[i] = FederatedLearning(params[i])
    return agents   

def fl_coupling(agents, params):    
    lbd = 5
    obj = lbd*cp.norm(agents[0].x,1)
    domain = [agent.x==agents[0].x for agent in agents[1:]]
    g = Coupling(agents=agents, function = obj, domain = domain)
    return g

############################# Create parameters for Intersection of Convex Sets #########################

def ics_params(num_row, num_col, num_agents):
    size = num_col
    A = np.random.random((num_row,num_col))-0.5
    x_feas = np.random.random(num_col)-0.5
    b = A@x_feas + 0.1*np.random.random(num_row)
    params = [0]*num_agents
    row_per_agent = int(num_row/num_agents)
    for i in range(num_agents):
        params[i] = {'A': A[(i*row_per_agent):((i+1)*row_per_agent),:], 'b': b[(i*row_per_agent):((i+1)*row_per_agent)],
                     'lower_bound':None, 'upper_bound': None, 'dimension':size}

    return params

def ics_agents(params):
    num_agents = len(params)
    agents = [0]*num_agents
    for i in range(num_agents):
        agents[i] = DistanceToSet(params[i])
    return agents

def ics_coupling(agents, params):
    domain = [agent.x==agents[0].x for agent in agents[1:]]
    g = Coupling(agents=agents, function = cp.Constant(0), domain = domain)
    return g