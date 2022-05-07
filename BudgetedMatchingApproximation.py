#!/usr/bin/env python
# coding: utf-8

# In[4]:


# import libraries needed
import time
import networkx as nx
from sortedcontainers import SortedList, SortedSet, SortedDict
import random
import numpy as np
import math
from itertools import chain, combinations
from networkx.generators.random_graphs import erdos_renyi_graph, fast_gnp_random_graph
from networkx.algorithms.matching import max_weight_matching
from networkx.algorithms.bipartite import random_graph
from subprocess import check_output
from scipy.optimize import fsolve
from gurobipy.gurobipy import Model, quicksum, GRB


# In[5]:


# Create new edge and set edge weight and cost
def setEdgeData(G, edge, weight, cost):
    G.add_edge(edge[0], edge[1])
    G[edge[0]][edge[1]]['weight'] = weight
    G[edge[0]][edge[1]]['cost'] = cost

def getEdgeCost(G, edge):
    return G[edge[0]][edge[1]]['cost']

def getEdgeWeight(G, edge):
    return G[edge[0]][edge[1]]['weight']

def zerotruncatedPoisson(p):
    #source: https://web.archive.org/web/20180826164029/http://giocc.com/zero_truncated_poisson_sampling_algorithm.html
    k = 1
    t = math.e**(-p) / (1 - math.e**(-p)) * p
    s = t
    u = random.random()
    while s < u:
        k += 1
        t *= p / k
        s += t
    return k

def isMatching(edges):
    # Check if set of edges is matching
    seen = []
    for e in edges:
        if e[0] in seen:
            return False
        seen.append(e[0])
        if e[1] in seen:
            return False
        seen.append(e[1])
    return True

def powerset(iterable, maxSize):
    "Source: https://docs.python.org/3/library/itertools.html#itertools-recipes"
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(maxSize+1))
def powersetAll(iterable):
    "Source: https://docs.python.org/3/library/itertools.html#itertools-recipes"
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def generateRandomGraph(n, p, Ew, Ec):
    """
    Generates random graph using the Erdös-Rényi model.
    params
    n: number of vertices
    p: probability parameter Erdös-Rényi
    Ew: poisson weight parameter
    Ec: poisson cost parameter
    """
    graph = erdos_renyi_graph(n,p)
    for edge in graph.edges:
        weight = 0
        cost = 0
        weight = zerotruncatedPoisson(Ew)
        cost = np.random.poisson(Ec)
        setEdgeData(graph, edge, weight, cost)
    return graph

def computeCost(g, matching):
    return sum(getEdgeCost(g, e) for e in matching)
def computeWeight(g, matching):
    return sum(getEdgeWeight(g, e) for e in matching)
def budgetedMatchingProblemBruteforce(g, B):
    maximum = 0
    cost = 0
    matching = None
    for e in powerset(g.edges):
        if len(e) <= len(g.nodes) and isMatching(e):
            c = computeCost(g, e)
            m = computeWeight(g, e)
            if c <= B and m > maximum:
                maximum = m
                matching = e
                cost = c
    return cost, maximum, matching

def generateRandomUnrestricted(n, p, Ec, Ew):
    """
    Generates random graph using the Erdös-Rényi model.
    This version puts costs on "weight" label
    and weights on "realWeight" label. This is for simpler code later on
    params
    n: number of vertices
    p: probability parameter Erdös-Rényi
    Ew: poisson weight parameter
    Ec: poisson cost parameter
    """
    graph = fast_gnp_random_graph(n,p) # does same as erdos_renyi_graph
    for edge in graph.edges:
        graph[edge[0]][edge[1]]['weight'] = np.random.poisson(Ec) # These are the costs!
        graph[edge[0]][edge[1]]['realWeight'] = zerotruncatedPoisson(Ew) # These are the weights!
    return graph

def generateRandomUnrestrictedBipartite(n, p, Ec, Ew):
    """
    Generates random bipartite graph using the Erdös-Rényi model.
    params
    n: number of vertices
    p: probability parameter Erdös-Rényi
    Ew: poisson weight parameter
    Ec: poisson cost parameter
    """
    graph = random_graph(math.floor(n/2), math.ceil(n/2), p)
    for edge in graph.edges:
        graph[edge[0]][edge[1]]['weight'] = np.random.poisson(Ec)
        graph[edge[0]][edge[1]]['realWeight'] = zerotruncatedPoisson(Ew)
    return graph

def restrictedToGeneral(G):
    """
    Transforms restricted version that has weight = 1 everywhere to
    generic instance of graph
    """
    graph = nx.Graph()
    for e in G.edges():
        setEdgeData(graph, e, 1, G[e[0]][e[1]]['weight'])
    return graph

def transform_graph(G, weight="weight"):
    """
    Transforms min weight to max weight problem
    """
    if len(G.edges) == 0:
        return max_weight_matching(G, maxcardinality, weight)
    G_edges = G.edges(data=weight, default=1)
    min_weight = min([w for _, _, w in G_edges])
    InvG = nx.Graph()
    edges = ((u, v, 1 / (1 + w - min_weight)) for u, v, w in G_edges)
    InvG.add_weighted_edges_from(edges, weight=weight)
    return InvG
    

def transform_graph_max_size_k(G, k, weight="weight"):
    """
    Transforms max weight matching to max weight matching on k vertices
    """
    sumOfWeights = sum(G[e[0]][e[1]][weight] for e in G.edges)
    transG = nx.Graph()
    G_edges = G.edges(data=weight, default=1)
    edges = ((u, v, w + sumOfWeights) for u, v, w in G_edges)
    transG.add_weighted_edges_from(edges, weight=weight)
    old_nodes = list(transG.nodes).copy()
    for i in range(len(transG.nodes), 2*len(transG.nodes)-2*k):
        for j in old_nodes:
            transG.add_edge(j, i, weight=2*sumOfWeights)
    return transG


def transform_graph_min_size_k(G, k, weight="weight"):
    """
    Transforms min weight perfect matching to min weight matching on k vertices
    """
    transG = nx.Graph()
    G_edges = G.edges(data=weight, default=1)
    edges = ((u, v, w) for u, v, w in G_edges)
    transG.add_weighted_edges_from(edges, weight=weight)
    old_nodes = list(G.nodes).copy()
    for i in range(len(G.nodes), 2*len(G.nodes)-2*k):
        for j in old_nodes:
            transG.add_edge(j, i, weight=0)
    return transG

def graphToAdjacency(G):
    """
    Transform graph to adjacency list
    """
    lines = []
    lines.append(str(len(G.nodes)))
    lines.append(str(len(G.edges)))
    for e in G.edges:
        e0 = e[0]
        e1 = e[1]
        lines.append(f"{e0} {e1} {G[e0][e1]['weight']}")
    return lines

def min_cost_matching(G):
    """
    Returns minimum cost perfect matching, if exists else false
    this uses the C# implementation
    source: https://github.com/dilsonpereira/Minimum-Cost-Perfect-Matching
    """
    adjacency = graphToAdjacency(G)
    f = open("input.txt", "w")
    for line in adjacency:
        f.write(line)
        f.write("\n")
    f.close()
    try:
        raw_data = check_output('MinimumCostMatching\example -f input.txt --minweight', shell=True).decode('utf-8').split(":")[-1].split("\r\n")
    except:
        return False
    return set(tuple(int(e) for e in x.split(" ")) for x in raw_data if x!="")

def min_cost_matching_size_k(G, k):
    """
    Compute minimum cost matching of size k
    """
    graph = transform_graph_min_size_k(G, k)
    matching = min_cost_matching(graph)
    if matching == False: return False
    nodesG = len(G.nodes)
    return set(x for x in matching if (x[0] < nodesG and x[1] < nodesG))
    
def budgetedMatchingPolynomial(G, B):
    """
    BudgetedMatching on graph G with weights restricted to 1
    and buget B
    """
    matching = min_cost_matching(G)
    maximumWeight = len(G.nodes)//2
    cost = sum(G[e[0]][e[1]]["weight"] for e in matching) if matching != False else B
    
    while (cost > B or not(matching)) and maximumWeight > 0:
        maximumWeight -= 1
        matching = min_cost_matching_size_k(G, maximumWeight)
        if (matching != False): cost = sum(G[e[0]][e[1]]["weight"] for e in matching)
    return (cost, maximumWeight, matching)

def penalizedGraph(G, lamb):
    """
    Tranform graph weights to weights in Lagrangian relaxation
    """
    graph = nx.Graph()
    for e in G.edges:
        graph.add_edge(*e)
        graph[e[0]][e[1]]["weight"] = G[e[0]][e[1]]["realWeight"] - G[e[0]][e[1]]["weight"] * lamb
    return graph

def budgetedMatchingPenalized(G, B, lamb = 1):
    """
    Compute the Lagrangian relaxation of the budgeted matching problem on graph G with budget B and Lambda value lamb
    """
    graph = penalizedGraph(G, lamb)
    matching = nx.max_weight_matching(graph)
    cost = sum(G[e[0]][e[1]]["weight"] for e in matching) if matching != False else False
    weight = lamb * B + sum(graph[e[0]][e[1]]["weight"] for e in matching) if matching != False else False
    return (cost, weight, matching)
        

def graph_to_gurobi(graph, budget):
    """
    Transform instance of BMP graph to ILP in Gurobi
    """
    model = Model("Budgeted Matching Constaint")
    x = {}
    edges = list(graph.edges)
    nodes = list(graph.nodes)
    for e in edges:
        x[e] = model.addVar(vtype=GRB.INTEGER, name=f'edge[{e}]', lb = 0, ub=1)
    model.addConstrs(
        (quicksum(x[e] for e in edges if v in e) <= 1 for v in nodes),
        name='Satisfy Matching'
    )
    model.addConstr(
        quicksum(x[e] * graph[e[0]][e[1]]['weight'] for e in edges) <= budget,
        name='Satisfy Budget'
    )
    model.setObjective(
        quicksum(x[e] * graph[e[0]][e[1]]['realWeight'] for e in edges),
        sense=GRB.MAXIMIZE
    )
    return model

def linear_relaxation(G, B):
    """
    Compute linear relaxation of BMP instance
    """
    model = Model("Linear relaxation")
    x = {}
    edges = list(G.edges)
    nodes = list(G.nodes)
    for e in edges:
        x[e] = model.addVar(vtype=GRB.CONTINUOUS, name=f'edge[{e}]', lb = 0, ub=1)
    model.addConstrs(
        (quicksum(x[e] for e in edges if v in e) <= 1 for v in nodes),
        name='Satisfy Matching'
    )
    model.addConstr(
        quicksum(x[e] * G[e[0]][e[1]]['weight'] for e in edges) <= B,
        name='Satisfy Budget'
    )
    model.setObjective(
        quicksum(x[e] * G[e[0]][e[1]]['realWeight'] for e in edges),
        sense=GRB.MAXIMIZE
    )
    return model

    


# In[53]:


def ternary_search(G, B, f, a, b, eps):
    """
    Approximate lambda that minimses Langrangian relaxation using ternary search
    """
    LB = False
    while (b - a) > eps:
        c = a + (b-a)/3
        d = a + 2/3*(b-a)
        fc, fd = f(c), f(d)
        sol_fc, sol_fd = sum(G[e[0]][e[1]]['realWeight'] for e in fc[2]), sum(G[e[0]][e[1]]['realWeight'] for e in fd[2])
        if (not(LB) or sol_fc > LB) and fc[0] <= B:
            LB = sol_fc
        if (not(LB) or sol_fd > LB) and fd[0] <= B:
            LB = sol_fd
        if fc[1] < fd[1]:
            b = d
        else:
            a = c
    return (a if f(a)[1] < f(b)[1] else b, LB)

def binary_search(G, B, f, a, b, eps):
    """
    Approximate lambda that minimses Langrangian relaxation using binary search
    """
    LB = False
    while (b - a) > eps:
        c = (a + b)/2
        fc = f(c)
        sol_fc = sum(G[e[0]][e[1]]['realWeight'] for e in fc[2])
        if (not(LB) or sol_fc > LB) and fc[0] <= B:
            LB = sol_fc
        if fc[0] > B:
            a = c
        else:
            b = c
    return (a if f(a)[1] < f(b)[1] else b, LB)

def minimize_LD(G, B):
    """
    Approximate Lagrangian dual using ternary search
    """
    maxWeight = max(G[e[0]][e[1]]["realWeight"] for e in G.edges)
    minCost = min(G[e[0]][e[1]]["weight"] for e in G.edges)
    maxLam = maxWeight/max(1, minCost)
    def f(x):
        return budgetedMatchingPenalized(G, B, x)
    x = ternary_search(G, B, f, 0, maxLam, 0.01)
    return (x[0], f(x[0]), x[1])

def minimize_LD_bin(G, B):
    """
    Approximate Lagrangian dual using binary search
    """
    maxWeight = max(G[e[0]][e[1]]["realWeight"] for e in G.edges)
    minCost = min(G[e[0]][e[1]]["weight"] for e in G.edges)
    maxLam = maxWeight/max(1, minCost)
    def f(x):
        return budgetedMatchingPenalized(G, B, x)
    x = binary_search(G, B, f, 0, maxLam, 0.0001)
    return (x[0], f(x[0]), x[1])

def bounds_LD(G, B):
    """
    Approximate lower and upper bound using Lagrangian relaxation using ternary search
    """
    ans = minimize_LD(G,B)
    return (math.ceil(ans[-1]), math.floor(ans[1][1]))

def bounds_LD_bin(G, B):
    """
    Approximate lower and upper bound using Lagrangian relaxation using binary search
    """
    ans = minimize_LD_bin(G,B)
    return (math.ceil(ans[-1]), math.floor(ans[1][1]))

def bounds_LR(G, B):
    """
    Approximate lower and upper bound using linear relaxation and rounding down method
    """
    model = linear_relaxation(G, B)
    model.optimize()
    e = list(G.edges)
    x = model.x
    LB = sum(G[e[i][0]][e[i][1]]['realWeight'] * math.floor(x[i]) for i in range(len(x)))
    UB = model.objval
    return(LB, math.floor(UB))

def compute_lambda3(G, B, l1, l2):
    """
    Given lambda1 and lambda2, compute lambda3 such that
    sum (x1 (w(e) - lambda3 c(e))) = sum (x2 (w(e) - lambda3 c(e)))
    """
    sol_1 = budgetedMatchingPenalized(G, B, l1)
    sol_2 = budgetedMatchingPenalized(G, B, l2)
    
    func1 = lambda x: sum(G[e[0]][e[1]]['realWeight'] - x * G[e[0]][e[1]]['weight'] for e in sol_1[2])
    func2 = lambda x: sum(G[e[0]][e[1]]['realWeight'] - x * G[e[0]][e[1]]['weight'] for e in sol_2[2])
    func3 = lambda x: func1(x) - func2(x)
    
    lambda3 = fsolve(func3, 0)[0]
    
    return (lambda3, budgetedMatchingPenalized(G, B, lambda3), sol_1, sol_2)

def compute_LD(G, B):
    LB = 0
    maxWeight = max(G[e[0]][e[1]]["realWeight"] for e in G.edges)
    minCost = min(G[e[0]][e[1]]["weight"] for e in G.edges)
    UB = maxWeight/max(1, minCost)
    while True:
        lamb3, sol3, solLB, solUB = compute_lambda3(G, B, LB, UB)
        if sol3[1] < solLB[1] or sol3[1] < solUB[1]:
            if sum(G[e[0]][e[1]]['weight'] for e in sol3[2]) > B:
                LB = lamb3
            else:
                UB = lamb3
        else:
            return LB, budgetedMatchingPenalized(G, B, LB)


# In[21]:


'''
Test of budgeted matching approximation on sparse graphs
LRs = Linear Relaxation bounds
LDs_bin = Lagrangian relaxation bounds
sols = Exact solutions
'''
# Sparse Graphs
LRs = {}
LDs_bin = {}
sols = {}
for n in range(100, 1001, 50):
    print("iterationnumber", n)
    running_time_bruteforce = 0
    running_time_gurobi = 0
    LRs[n] = []
    LDs_bin[n] = []
    sols[n] = []
    for i in range(5):
        g = generateRandomUnrestricted(n, 2/n, 10, 10)
        LR = bounds_LR(g, 3*n)
        LRs[n].append(LR)
        model = graph_to_gurobi(g, 3*n)
        model.optimize()
        sols[n].append(model.objval)
        LDbin = bounds_LD_bin(g, 3*n)
        LDs_bin[n].append(LDbin)


# In[22]:


LDs


# In[23]:


LRs


# In[24]:


sols


# In[25]:


LDs_bin


# In[50]:





# In[22]:


'''
Test of budgeted matching approximation on dense graphs
LRs_dense = Linear Relaxation bounds
LDs_bin_dense = Lagrangian relaxation bounds
sols_dense = Exact solutions
'''
import time
# dense Graphs
LRs_dense = {}
LDs_bin_dense = {}
sols_dense = {}
for n in range(100, 1001, 50):
    print("iterationnumber", n)
    running_time_bruteforce = 0
    running_time_gurobi = 0
    LRs_dense[n] = []
    LDs_bin_dense[n] = []
    sols_dense[n] = []
    for i in range(5):
        g = generateRandomUnrestricted(n, 0.8, 10, 10)
        LR = bounds_LR(g, 3*n)
        LRs_dense[n].append(LR)
        model = graph_to_gurobi(g, 3*n)
        model.optimize()
        sols_dense[n].append(model.objval)
        LDbin = bounds_LD_bin(g, 3*n)
        LDs_bin_dense[n].append(LDbin)


# In[23]:


LRs_dense


# In[24]:


LDs_bin_dense


# In[25]:


sols_dense


# In[25]:


'''
Test of budgeted matching approximation on sparse bipartite graphs
LRs = Linear Relaxation bounds
LDs_bin = Lagrangian relaxation bounds
sols = Exact solutions
'''
# Sparse Graphs
LRs = {}
LDs_bin = {}
sols = {}
for n in range(100, 1001, 50):
    print("iterationnumber", n)
    running_time_bruteforce = 0
    running_time_gurobi = 0
    LRs[n] = []
    LDs_bin[n] = []
    sols[n] = []
    for i in range(5):
        g = generateRandomUnrestrictedBipartite(n, 2/n, 10, 10)
        LR = bounds_LR(g, 2*n)
        LRs[n].append(LR)
        model = graph_to_gurobi(g, 2*n)
        model.optimize()
        sols[n].append(model.objval)


# In[26]:


sols


# In[27]:


LRs


# In[28]:


'''
Test of budgeted matching approximation on dense bipartite graphs
LRs = Linear Relaxation bounds
LDs_bin = Lagrangian relaxation bounds
sols = Exact solutions
'''
# dense Graphs
LRs = {}
LDs_bin = {}
sols = {}
for n in range(100, 1001, 50):
    print("iterationnumber", n)
    running_time_bruteforce = 0
    running_time_gurobi = 0
    LRs[n] = []
    LDs_bin[n] = []
    sols[n] = []
    for i in range(5):
        g = generateRandomUnrestrictedBipartite(n, 0.8, 10, 10)
        LR = bounds_LR(g, 2*n)
        LRs[n].append(LR)
        model = graph_to_gurobi(g, 2*n)
        model.optimize()
        sols[n].append(model.objval)


# In[29]:


sols


# In[30]:


LRs


# In[8]:


graph = generateRandomUnrestricted(10, 0.8, 10,10)


# In[9]:


minimize_LD_bin(graph, 30)


# In[48]:


compute_lambda3(graph, 30, 0.1, 0.62)


# In[42]:


budgetedMatchingPenalized(graph, 30, 0.62)


# In[45]:


budgetedMatchingPenalized(graph, 30, 0.0)


# In[54]:


compute_LD(graph, 30)


# In[ ]:




