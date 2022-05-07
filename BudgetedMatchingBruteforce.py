#!/usr/bin/env python
# coding: utf-8

# In[3]:


# import libraries needed
import networkx as nx
import random
import numpy as np
import math
from itertools import chain, combinations
from networkx.generators.random_graphs import erdos_renyi_graph
import matplotlib.pylab as plt
import time


# In[4]:


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
    Ew: zero truncated poisson weight parameter
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
    


# In[71]:


def computeCost(g, matching):
    # compute cost of matching
    return sum(getEdgeCost(g, e) for e in matching)
def computeWeight(g, matching):
    # compute weight of matching
    return sum(getEdgeWeight(g, e) for e in matching)
def budgetedMatchingProblemBruteforce(g, B):
    # solve BMP by bruteforce
    maximum = 0
    cost = 0
    matching = None
    for e in powerset(g.edges, len(g.nodes)//2):
        if isMatching(e):
            c = computeCost(g, e)
            m = computeWeight(g, e)
            if c <= B and m > maximum:
                maximum = m
                matching = e
                cost = c
    return cost, maximum, matching


# In[1]:


from gurobipy.gurobipy import Model, quicksum, GRB
def graph_to_gurobi(graph, budget):
    # Turn budgeted matching instance into linear program
    model = Model("Budgeted Matching Problem")
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
        quicksum(x[e] * getEdgeCost(graph, e) for e in edges) <= budget,
        name='Satisfy Budget'
    )
    model.setObjective(
        quicksum(x[e] * getEdgeWeight(graph, e) for e in edges),
        sense=GRB.MAXIMIZE
    )
    return model


# In[87]:


'''
Test of budgeted matching bruteforce on dense graphs
Results get saved in dict times_brute_dense and times_gurobi_dense
'''
times_brute_dense = {}
times_gurobi_dense = {}
for n in range(2,13):
    running_time_bruteforce = 0
    running_time_gurobi = 0
    for i in range(10):
        g = generateRandomGraph(n, 0.8, 10, 10)
        gurobi = graph_to_gurobi(g, 3*n)
        start = time.time()
        budgetedMatchingProblemBruteforce(g, 3*n)
        end = time.time()
        running_time_bruteforce += end - start
        
        start = time.time()
        gurobi.optimize()
        end = time.time()
        running_time_gurobi += end - start
    times_brute_dense[n] = running_time_bruteforce / 10
    times_gurobi_dense[n] = running_time_gurobi / 10


# In[88]:


# Plot results
# Source: https://stackoverflow.com/a/37266356/18307616
plt.figure(figsize=(15,9))
lists = sorted(times_brute_dense.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(x, y)

lists = sorted(times_gurobi_dense.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(x, y)

plt.legend(['Brute Force', 'Gurobi'], prop={'size':20})
plt.title("Running time of Brute Force Algorithm on dense graphs", size=28, weight='bold')
plt.xlabel("Number of vertices", size=20)
plt.ylabel("Running time (seconds)", size=20)
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.show()


# In[60]:


'''
Test of budgeted matching bruteforce on sparse graphs
Results get saved in dict times_brute and times_gurobi
'''
# Sparse Graphs
times_brute = {}
times_gurobi = {}
for n in range(2,22):
    running_time_bruteforce = 0
    running_time_gurobi = 0
    for i in range(10):
        g = generateRandomGraph(n, 2 / n, 10, 10)
        gurobi = graph_to_gurobi(g, 10*n)
        start = time.time()
        budgetedMatchingProblemBruteforce(g, 10*n)
        end = time.time()
        running_time_bruteforce += end - start
        
        start = time.time()
        gurobi.optimize()
        end = time.time()
        running_time_gurobi += end - start
    times_brute[n] = running_time_bruteforce / 10
    times_gurobi[n] = running_time_gurobi / 10


# In[61]:


# Plot results
# Source: https://stackoverflow.com/a/37266356/18307616
plt.figure(figsize=(15,9))

lists = sorted(times_brute.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(x, y)

lists = sorted(times_gurobi.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(x, y)

plt.legend(['Brute Force', 'Gurobi'], prop={'size':20})
plt.title("Running time of Brute Force Algorithm on sparse graphs", size=28, weight='bold')
plt.xlabel("Number of vertices", size=20)
plt.ylabel("Running time (seconds)", size=20)
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.show()


# In[ ]:




