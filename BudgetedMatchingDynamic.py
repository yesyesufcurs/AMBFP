#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries needed
import time
import networkx as nx
from sortedcontainers import SortedList, SortedSet, SortedDict
import random
import numpy as np
import math
from itertools import chain, combinations
from networkx.generators.random_graphs import erdos_renyi_graph
import matplotlib.pylab as plt


# In[2]:


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

def isSubset(a, b):
    """
    Checks if a subset b
    """
    return a.intersection(b)==a


# In[3]:


# w[i,j,k] = (maximumWeight using first i edges with budget j using vertices in k)  
# w : {0,...,n} x {0,...,B} x P(V) -> Z+
# w[0,j,k] = 0
# w[i,j,{}] = 0
# w[i,j,k] = max(w[i-1,j,k], w[i-1,j-c(i),k\{v : v in e(i)}] + w(i)) if c(i) <= j and {v : v in e(i)} \in k
# w[i,j,k] = w[i-1,j,k] otherwise

def verticesIn(edge):
    return SortedSet([edge[0], edge[1]])

def maximumMatchingDynamic(G, B):
    """
    Solve Budgeted Matching Problem using dynamic programming
    """
    w = [[dict() for _ in range(B + 1)] for _ in range(len(G.edges) + 1)]
    edges = list(G.edges)
    nodes = list(G.nodes)
    for i in range(len(G.edges) + 1):
        for j in range(B + 1):
            for el in powerset(nodes):
                k = SortedSet(el)
                if i == 0: w[i][j][str(k)] = 0
                elif k == SortedSet(set()): w[i][j][str(k)] = 0
                elif getEdgeCost(G,edges[i - 1]) <= j and isSubset(verticesIn(edges[i-1]), k): 
                    w[i][j][str(k)] = max(w[i-1][j][str(k)], w[i-1][j-getEdgeCost(G,edges[i-1])][str(k.difference(verticesIn(edges[i-1])))] + getEdgeWeight(G, edges[i-1]))
                else:
                    w[i][j][str(k)] = w[i-1][j][str(k)]
    return w[len(G.edges)][B][str(SortedSet(nodes))]

def maximumMatchingDynamicAlt(G, B):
    """
    Solve Budgeted Matching Problem using dynamic programming
    filling in w[i,j,k] by iterating through P(V) first
    """
    w = dict()
    edges = list(G.edges)
    nodes = list(G.nodes)
    for el in powerset(nodes):
        k = SortedSet(el)
        w[str(k)] = [[dict() for _ in range(B + 1)] for _ in range(len(G.edges) + 1)]
        for i in range(len(G.edges) + 1):
            for j in range(B + 1):
                if i == 0: w[str(k)][i][j] = 0
                elif k == SortedSet(set()): w[str(k)][i][j] = 0
                elif getEdgeCost(G,edges[i - 1]) <= j and isSubset(verticesIn(edges[i-1]), k):
                    w[str(k)][i][j] = max(w[str(k)][i-1][j], w[str(k.difference(verticesIn(edges[i-1])))][i-1][j-getEdgeCost(G,edges[i-1])] + getEdgeWeight(G, edges[i-1]))
                else:
                    w[str(k)][i][j] = w[str(k)][i-1][j]
    return w[str(SortedSet(nodes))][len(G.edges)][B]


def setToBitstring(inp, size):
    "Convert set to bitstring"
    string = bytearray(size)
    for i in inp:
        string[i] = 1
    return string

def bitsToInt(bitstring):
    "Convert bitstring to integer"
    return int.from_bytes(bitstring, "little")

def edgeIsSubset(edge, bitstring):
    "Given an edge and a bitstring return if edge in bitstring"
    return bitstring[edge[0]] == 1 and bitstring[edge[1]] == 1

def bitStringDifference(a, b):
    "Return the bitstring for the set a setminus b"
    string = bytearray(len(a))
    string[:] = a
    for i in b:
        string[i] = 0
    return string
    

def maximumMatchingDynamicBitstring(G, B):
    """
    Solve budgeted matching problem using Dynamic programming
    by using bitstrings
    """
    w = dict()
    edges = list(G.edges)
    nodes = list(G.nodes)
    size = len(nodes)
    e_size = len(edges)
    for el in powerset(nodes):
        k = setToBitstring(el, size)
        int_k = bitsToInt(k)
        w[int_k] = [[dict() for _ in range(B + 1)] for _ in range(e_size + 1)]
        for i in range(e_size + 1):
            for j in range(B + 1):
                if i == 0: w[int_k][i][j] = 0
                elif int_k == 0: w[int_k][i][j] = 0
                elif getEdgeCost(G,edges[i - 1]) <= j and edgeIsSubset(edges[i-1], k):
                    w[int_k][i][j] = max(w[int_k][i-1][j], w[bitsToInt(bitStringDifference(k, edges[i - 1]))][i-1][j-getEdgeCost(G,edges[i-1])] + getEdgeWeight(G, edges[i-1]))
                else:
                    w[int_k][i][j] = w[int_k][i-1][j]
    return w[bitsToInt(setToBitstring(nodes, size))][e_size][B]


# In[9]:


from gurobipy.gurobipy import Model, quicksum, GRB
def graph_to_gurobi(graph, budget):
    # Turn budgeted matching instance into linear program
    model = Model("Budgeted Matching Constaint")
    x = {}
    edges = list(g.edges)
    nodes = list(g.nodes)
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


# In[29]:


'''
Test of budgeted matching dynamic program on dense graphs using bitstrings
Results get saved in dict times_brute_dense and times_gurobi_dense
'''
# Dense Graphs
times_brute_dense = {}
times_gurobi_dense = {}
for n in range(2,11):
    running_time_bruteforce = 0
    running_time_gurobi = 0
    for i in range(10):
        g = generateRandomGraph(n, 0.8, 10, 10)
        gurobi = graph_to_gurobi(g, 3*n)
        start = time.time()
        maximumMatchingDynamicBitstring(g, 3*n)
        end = time.time()
        running_time_bruteforce += end - start
        
        start = time.time()
        gurobi.optimize()
        end = time.time()
        running_time_gurobi += end - start
    times_brute_dense[n] = running_time_bruteforce / 10
    times_gurobi_dense[n] = running_time_gurobi / 10


# In[30]:


'''
Test of budgeted matching dynamic program on sparse graphs using bitstrings
Results get saved in dict times_brute and times_gurobi
'''
# Sparse Graphs
times_brute = {}
times_gurobi = {}
for n in range(2,11):
    running_time_bruteforce = 0
    running_time_gurobi = 0
    for i in range(10):
        g = generateRandomGraph(n, 2 / n, 10, 10)
        gurobi = graph_to_gurobi(g, 3*n)
        start = time.time()
        maximumMatchingDynamicBitstring(g, 3*n)
        end = time.time()
        running_time_bruteforce += end - start
        
        start = time.time()
        gurobi.optimize()
        end = time.time()
        running_time_gurobi += end - start
    times_brute[n] = running_time_bruteforce / 10
    times_gurobi[n] = running_time_gurobi / 10


# In[31]:


# Plot results
# Source: https://stackoverflow.com/a/37266356/18307616
plt.figure(figsize=(15,9))
lists = sorted(times_brute_dense.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(x, y)

lists = sorted(times_gurobi_dense.items())

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(x, y)

lists = sorted(times_brute.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(x, y)

lists = sorted(times_gurobi.items())

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(x, y, 'x')

plt.legend(['Dynamic Dense', 'Gurobi Dense', 'Dynamic Sparse', 'Gurobi Sparse'], prop={'size':20})
plt.title("Running time of Dynamic Programming Algorithm, B=3n", size=28, weight='bold')
plt.xlabel("Number of vertices", size=20)
plt.ylabel("Running time (seconds)", size=20)
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.show()


# In[ ]:




