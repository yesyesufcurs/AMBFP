#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries needed
import networkx as nx
from sortedcontainers import SortedList, SortedSet, SortedDict
import random
import numpy as np
import math
from itertools import chain, combinations
from networkx.generators.random_graphs import erdos_renyi_graph, fast_gnp_random_graph
from networkx.algorithms.matching import max_weight_matching
from subprocess import check_output
from gurobipy.gurobipy import Model, quicksum, GRB
import matplotlib.pylab as plt
import time


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

def generateRandomRestricted(n, p, Ec):
    """
    Generates random graph using the Erdös-Rényi model.
    The weights of all edges are 1
    The costs are following a poisson parameter
    Here the 'weight' variable of each edge denotes the edge cost not edge weight
    params
    n: number of vertices
    p: probability parameter Erdös-Rényi
    Ec: poisson cost parameter
    """
    graph = fast_gnp_random_graph(n,p)
    for edge in graph.edges:
        graph[edge[0]][edge[1]]['weight'] = np.random.poisson(Ec) # Edge cost!
    return graph

def restrictedToGeneral(G):
    """
    Transform graph instance with weights 1
    to general instance of budgeted matching problem
    """
    graph = nx.Graph()
    for e in G.edges():
        setEdgeData(graph, e, 1, G[e[0]][e[1]]['weight'])
    return graph


# In[3]:


### Using Min Weight Max Cardinality Matching

G = generateRandomRestricted(40,0.8,5)


# In[4]:


nx.draw(G)


# In[5]:


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

def transform_graph_min_size_k(G, k, weight="weight"):
    """
    Transforms max weight matching to max weight matching on k vertices
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
    and buget B in polynomial time
    """
    matching = min_cost_matching(G)
    maximumWeight = len(G.nodes)//2
    cost = sum(G[e[0]][e[1]]["weight"] for e in matching) if matching != False else B
    
    while (cost > B or not(matching)) and maximumWeight > 0:
        maximumWeight -= 1
        matching = min_cost_matching_size_k(G, maximumWeight)
        if (matching != False): cost = sum(G[e[0]][e[1]]["weight"] for e in matching)
    return (cost, maximumWeight, matching)


# In[6]:


def computeCost(g, matching):
    # Compute cost of matching
    return sum(getEdgeCost(g, e) for e in matching)
def computeWeight(g, matching):
    # Compute weight of matching
    return sum(getEdgeWeight(g, e) for e in matching)
def budgetedMatchingProblemBruteforce(g, B):
    # solve BMP by bruteforce
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


# In[7]:


from gurobipy.gurobipy import Model, quicksum, GRB
# ============================================ GUROBI ====================================================
def restricted_graph_to_gurobi(graph, budget):
    # Transform graph with weights = 1 to gurobi instance
    model = Model("Budgeted Matching Restricted")
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
        quicksum(x[e] * getEdgeWeight(graph, e) for e in edges) <= budget,
        name='Satisfy Budget'
    )
    model.setObjective(
        quicksum(x[e] for e in edges),
        sense=GRB.MAXIMIZE
    )
    return model


# In[8]:


def check_model(graph, budget):
    # Check that budgeted matching done in polynomial time the same value gives as
    # Gurobi linear program
    m = restricted_graph_to_gurobi(graph, budget)
    m.optimize()
    num1 = budgetedMatchingPolynomial(graph, budget)[1]
    num2 = int(m.objVal)
    return num1 == num2


# In[38]:


'''
Test of budgeted matching polynomial program on dense graphs
Results get saved in dict times_brute_dense and times_gurobi_dense
'''
# Dense Graphs
times_brute_dense = {}
times_gurobi_dense = {}
for n in range(2,101):
    print("iterationnumber", n)
    running_time_bruteforce = 0
    running_time_gurobi = 0
    for i in range(5):
        g = generateRandomRestricted(n, 0.8, 10)
        gurobi = restricted_graph_to_gurobi(g, 3*n)
        start = time.time()
        budgetedMatchingPolynomial(g, 3*n)
        end = time.time()
        running_time_bruteforce += end - start
        
        start = time.time()
        gurobi.optimize()
        end = time.time()
        running_time_gurobi += end - start
    times_brute_dense[n] = running_time_bruteforce / 5
    times_gurobi_dense[n] = running_time_gurobi / 5


# In[42]:


# Plot results
# Source: https://stackoverflow.com/a/37266356/18307616
plt.figure(figsize=(15,9))
lists = sorted(times_brute_dense.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(x, y)

lists = sorted(times_gurobi_dense.items())

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(x, y)


plt.legend(['MCBMP Solver', 'Gurobi'], prop={'size':20})
plt.title("Running time of MCBMP solver on dense graphs", size=28, weight='bold')
plt.xlabel("Number of vertices", size=20)
plt.ylabel("Running time (seconds)", size=20)
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.show()


# In[40]:


'''
Test of budgeted matching polynomial program on sparse graphs
Results get saved in dict times_brute and times_gurobi
'''
# Sparse Graphs
times_brute = {}
times_gurobi = {}
for n in range(2,101):
    print("iterationnumber", n)
    running_time_bruteforce = 0
    running_time_gurobi = 0
    for i in range(5):
        g = generateRandomRestricted(n, 2/n, 10)
        gurobi = restricted_graph_to_gurobi(g, 3*n)
        start = time.time()
        budgetedMatchingPolynomial(g, 3*n)
        end = time.time()
        running_time_bruteforce += end - start
        
        start = time.time()
        gurobi.optimize()
        end = time.time()
        running_time_gurobi += end - start
    times_brute[n] = running_time_bruteforce / 5
    times_gurobi[n] = running_time_gurobi / 5


# In[43]:


# Plot results
# Source: https://stackoverflow.com/a/37266356/18307616
plt.figure(figsize=(15,9))
lists = sorted(times_brute.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples
plt.plot(x, y)

lists = sorted(times_gurobi.items())

x, y = zip(*lists) # unpack a list of pairs into two tuples
plt.plot(x, y)


plt.legend(['MCBMP Solver', 'Gurobi'], prop={'size':20})
plt.title("Running time of MCBMP solver on sparse graphs", size=28, weight='bold')
plt.xlabel("Number of vertices", size=20)
plt.ylabel("Running time (seconds)", size=20)
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.show()


# In[36]:


'''
Test of budgeted matching polynomial program on dense graphs
Results get saved in dict times_brute_dense and times_gurobi_dense
'''
# Dense Graphs
times_brute_dense = {}
times_gurobi_dense = {}
for n in range(2,101,2):
    print("iterationnumber", n)
    running_time_bruteforce = 0
    running_time_gurobi = 0
    for i in range(5):
        g = generateRandomRestricted(n, 0.8, 10)
        gurobi = restricted_graph_to_gurobi(g, 3*n)
        start = time.time()
        budgetedMatchingPolynomial(g, 3*n)
        end = time.time()
        running_time_bruteforce += end - start
        
        start = time.time()
        gurobi.optimize()
        end = time.time()
        running_time_gurobi += end - start
    times_brute_dense[n] = running_time_bruteforce / 5
    times_gurobi_dense[n] = running_time_gurobi / 5
    


# In[37]:


# Plot results
# Source: https://stackoverflow.com/a/37266356/18307616
plt.figure(figsize=(15,9))
lists = sorted(times_brute_dense.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(x, y)

lists = sorted(times_gurobi_dense.items())

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(x, y)


plt.legend(['MCBMP Solver', 'Gurobi'], prop={'size':20})
plt.title("Running time of MCBMP solver on dense graphs of even number of vertices", size=28, weight='bold')
plt.xlabel("Number of vertices", size=20)
plt.ylabel("Running time (seconds)", size=20)
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.show()


# In[ ]:




