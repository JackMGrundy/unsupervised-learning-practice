import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from clusters import uniformRandomSum, gaussianClusterParams, gaussianClusters
import os
import pandas as pd
import random 


def nextGen(parents, numChildren, mutateRate, surivorLimit):
    """
        nextGen: produces numChildren offspring sequences from a collection of parents DNA sequences 

    Args:
        parents (n by k numpy array): n DNA sequencies of length k encoded as 
        { 1: 'A',
          2: 'B',
          3: 'C',
          4: 'D'}

        numChildren (int): number of children that each DNA sequence produces
        mutateRate (double in [0, 1]): probability that any given nucleotide mutates. It could mutate but remain the same. 
        survivorLimit (int): the max number of children that survive

    Returns:
        children (survivorLimit by k numpy array): children spawned from the parent sequences
    """
    children = np.tile(parents, reps=(numChildren, 1))
    n, k = children.shape

    # Identify bases to mutate
    mutations = np.random.rand(n, k)
    mutations[mutations < mutateRate] = 1
    mutations[mutations < 1] = 0
    # Zero out old bases that are now being mutated
    children[mutations==1] = 0 
    # Randomly mutate selected bases to new nucleotides
    children += mutations * (np.floor(np.random.rand(n, k)*4)+1) 

    # Random select survivorLimit children to return
    np.random.shuffle(children)
    return(children[0:surivorLimit, :])


def simpleEvolutionSimulation(numAncestors, k, numChildren, numGenerations, mutateRate, survivorLimit):
    """
    simpleEvolutionSimulation: Randomly, uniformly generates a specified number of ancestor DNA sequences of k length. 
                                Generates a specified number of successor generations according to specified mutation rate.
                                Returns children sequences of final generation. 

    Args:
        numAncestors (int): number of original ancestor sequences of DNA
        k (int): length of each DNA sequence
        numChildren (int): number of children each sequences produces in each generation
        numGenerations (int): number of generations of children DNA to simulate
        mutateRate (double): rate at which any particular nucelotide mutates
        survivorLimit (int): max number of children that survive each generation
    
    Returns:
        ancestors (numAncestors by k numpy array): array of original ancestor DNA sequences
        children (survivorLimit by k numpy array): array of children DNA sequences in the final generation produced.
                                            Nucleotides are encoded as:
                                            { 1: 'A',
                                            2: 'B',
                                            3: 'C',
                                            4: 'D'}
    """
    # nucleotides = ['A', 'C', 'G', 'T']
    # nucleotideCodes = dict(zip(range(1, 5), nucleotides))

    # Original ancestors
    ancestors = np.floor(np.random.rand(numAncestors, k)*4)+1

    for g in range(numGenerations):
        print("generation: " + str(g))
        if g==0: 
            children = nextGen(parents=ancestors, numChildren=numChildren, mutateRate=mutateRate, surivorLimit=survivorLimit)
        else:
            children = nextGen(parents=children, numChildren=numChildren, mutateRate=mutateRate, surivorLimit=survivorLimit)

    return(ancestors, children)



def randomGaussianClusters():
    """
    Simple test of running agglomerative clustering on random collection of multivariate gaussians
    """
    M, V, N = gaussianClusterParams(c=3, k=3, n=100, minMean=0, maxMean=50, minVar=0, maxVar=1, stretch="circle")
    print(M)
    print(V)
    print(N)
    X = gaussianClusters(M, V, N)
    H = linkage(X, 'ward')
    dendrogram(H)
    plt.show()


if __name__ == '__main__':
    # Run simple evolution simulation
    ancestors, children = simpleEvolutionSimulation(numAncestors=4, k=1000, numChildren=40, numGenerations=5, mutateRate=.04, survivorLimit=1000)

    # Test agglomerative clustering's ability to recover ancestor clusters
    # Highly sensitive to mutation rate
    # Survivor rate must be sufficiently large
    # Appears reasonably robust in response to changes in numAncestors
    H = linkage(children, 'complete')
    dendrogram(H)
    plt.show()