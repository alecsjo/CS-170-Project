import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance_fast
import sys
import os
import math
import heapq

def solve(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """

    # TODO: your code here!

    T = nx.create_empty_copy(G)

    d = [0] * G.number_of_nodes()
    s = [0] * G.number_of_nodes()
    m = [0] * G.number_of_nodes()
    sumWeights = 0

    for (i, j) in G.edges():
        weight = G[i][j]['weight']
        d[i] = d[i] + 1
        s[i] = s[i] + weight
        m[i] = max(m[i], weight)

        d[j] = d[j] + 1
        s[j] = s[j] + weight
        m[j] = max(m[j], weight)

        sumWeights = sumWeights + weight

    mean = sumWeights / G.number_of_edges()
    sums = 0

    for (i, j) in G.edges():
        weight = G[i][j]['weight']
        sums = sums + (weight - mean) ** 2

    stdDev = math.sqrt(sums / (G.number_of_edges() - 1))
    ratio = stdDev / mean

    w = [0] * G.number_of_nodes()
    wd = [float("inf")] * G.number_of_nodes()
    jsp = [0] * G.number_of_nodes()

    color = [0] * G.number_of_nodes()
    sp = [0] * G.number_of_nodes()
    sp_max = 0
    cf = [0] * G.number_of_nodes()

    p = [None] * G.number_of_nodes()
    pd = [None] * G.number_of_nodes()
    ps = [None] * G.number_of_nodes()

    for v in G.nodes():
        w[v] = float("inf")
        cf[v] = float("inf")
        color[v] = 'WHITE'
        sp[v] = 0.2 * d[v] + 0.6 * (d[v] / s[v]) + 0.2 * (1 / m[v])
        if (sp[v] > sp_max):
            sp_max = sp[v]
            f = v

    w[f] = 0
    cf[f] = 0
    p[f] = None
    pd[f] = 0
    ps[f] = 1
    color[f] = 'GRAY'

    # change heap
    L = [(1, 1, f)]
    heapq.heapify(L)

    C_4 = .9
    C_5 = .1

    spanned_vertices = 0

    while len(L) > 0:
        u = heapq.heappop(L)[2]
        if p[u]:
            cf[u] = cf[p[u]] + G[u][p[u]]['weight']
        else:
            cf[u] = 0

        for v in G.neighbors(u):
            if color[v] == 'BLACK':
                continue
            # remember to come back for t (replaced t with v)
            # t = v.copy()

            wdt = C_4 * G[u][v]['weight'] + C_5 * (cf[u] + G[u][v]['weight'])
            jspt = (d[v] + d[u]) + (d[v] + d[u]) / (s[v] + s[u])

            # add heapify compare here
            if (wdt < wd[v] or (wdt == wd[v] and jsp[v] > jspt)):
                if (color[v] == 'WHITE'):
                    wd[v] = wdt
                    jsp[v] = jspt
                    p[v] = u

                    # do heapify thing!!!
                    # L.append((wd[v],jsp[v], v))
                    heapq.heappush(L, (wd[v], jsp[v], v))
                    color[v] = 'GRAY'

                elif (color[v] == 'GRAY'):
                    # do heapify thing!!!
                    heapq.heappush(L, (wd[v], jsp[v], v))
                    # L.update(v)  # COME BACK
        color[u] = 'BLACK'
        spanned_vertices += 1

        if p[u] != None:
            T.add_edge(u, p[u])
            T.add_weighted_edges_from([(u, p[u], G[u][p[u]]['weight'])])


    for v in T.nodes():
        if T.degree(v) == 1:
            pruned_tree = T.copy()
            pruned_tree.remove_node(v)
            if average_pairwise_distance_fast(pruned_tree) < average_pairwise_distance_fast(T):
                T = pruned_tree

    return T

# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in
#
# if __name__ == '__main__':
#     assert len(sys.argv) == 2
#     path = sys.argv[1]
#     G = read_input_file(path)
#     T = solve(G)
#     assert is_valid_network(G, T)
#     print("Average  pairwise distance: {}".format(average_pairwise_distance_fast(T)))
#     write_output_file(T, 'out/test.out')

if __name__ == "__main__":
    output_dir = "submission"
    input_dir = "inputs"
    for input_path in os.listdir(input_dir):
        graph_name = input_path.split(".")[0]
        G = read_input_file(f"{input_dir}/{input_path}")
        T = solve(G)
        write_output_file(T, f"{output_dir}/{graph_name}.out")
