#!/usr/bin/env python
from __future__ import print_function

import argparse
import networkx as nx
import numpy as np
import numpy.linalg as npl


def choicerank(digraph, c_in, c_out, alpha=1.0, beta=0.0, epsilon=1e-6,
               max_iter=1000, debug=False):
    """Compute the MAP estimate of the network choice model.
    
    Args:
        digraph (nx.DiGraph): A directed graph representing the network.
        c_in (np.ndarray): The aggregate number of transitions arriving in
            each node.
        c_out (np.ndarray): The aggregate number of transitions originating
            from each node.
        alpha (Optional[float]): The shape parameter of the Gamma prior.
        beta (Optional[float]): The rate parameter of the Gamma prior.
        epsilon (Optional[float]): The convergence criterion.
        max_iter (Optional[int]): The maximum number of iterations.
        debug (Optional[bool]: If true, prints some debugging information.

    Returns:
        lambdas (np.ndarray): The MAP parameters of the network choice model.
    """
    assert len(digraph) == len(c_in) == len(c_out)
    n = len(digraph)
    # Regularization vectors.
    reg1 = (alpha - 1.0) * np.ones(n)
    reg2 = beta * np.ones(n)
    # Initialize the parameters.
    lambdas = np.ones(n, dtype=float)
    # Precompute the adjacency matrix and its transpose.
    adj = nx.to_scipy_sparse_matrix(digraph)
    adj_t = adj.T.tocsr()
    for itr in range(max_iter):
        prev = lambdas
        # First phase of message passing.
        zs = adj.dot(lambdas)
        # Second phase of message passing.
        zs = adj_t.dot(c_out / zs) + reg2
        # Computing the next MM iterate.
        lambdas = (c_in + reg1) / zs
        # Normalizing the lambdas can accelerate convergence.
        lambdas = lambdas / (lambdas.sum() / n)
        diff = npl.norm(lambdas - prev, ord=1) / n
        if debug:
            print("iteration {}, diff: {:.8f}".format(itr, diff))
        if diff < epsilon:
            return lambdas
    print("WARNING: did not converge after {} iterations".format(max_iter))
    return lambdas


def main(n, p, nb_samples):
    np.set_printoptions(precision=3, suppress=True)
    graph = nx.gnp_random_graph(n, p, directed=True)
    print("Erdos-Renyi random graph G({}, {:.3f})".format(n, p))
    if nx.is_strongly_connected(graph):
        print("  The graph is strongly connected")
    else:
        print("  The graph is NOT strongly connected")
    # Generate strengths uniformly at random.
    lambdas = np.random.uniform(low=0.1, high=2.0, size=n)
    lambdas = lambdas / (lambdas.sum() / n)
    # Generate random transitions in the network.
    c_in = np.zeros(n, dtype=int)
    c_out = np.zeros(n, dtype=int)
    print("Generating {} choices...".format(nb_samples))
    for _ in range(nb_samples):
        src = np.random.choice(n)
        neighbors = graph.successors(src)
        if len(neighbors) == 0:
            continue
        probs = np.array([lambdas[x] for x in neighbors])
        probs = probs / probs.sum()
        dst = np.random.choice(neighbors, p=probs)
        # Update the data.
        c_out[src] += 1
        c_in[dst] += 1
    print("Estimating model parameters with ChoiceRank... ", end='')
    est = choicerank(graph, c_in, c_out, alpha=1.1, beta=0.1)
    print("Done.")
    print("\nMAP estimate of 10 first parameters:")
    print(est[:10])
    print("\nGround truth:")
    print(lambdas[:10])
    #print(np.linalg.norm(lambdas - est))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=100, type=int)
    parser.add_argument('--p', default=0.15, type=float)
    parser.add_argument('--nb-samples', default=20000, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(args.n, args.p, args.nb_samples)
