import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from clusters import uniformRandomSum, gaussianClusterParams, gaussianClusters
import os
import pandas as pd

def randomGaussianClusters():
    M, V, N = gaussianClusterParams(c=3, k=3, n=100, minMean=0, maxMean=50, minVar=0, maxVar=1, stretch="circle")
    print(M)
    print(V)
    print(N)
    X = gaussianClusters(M, V, N)
    H = linkage(X, 'ward')
    dendrogram(H)
    plt.show()


def bballClusters():
    cwd = os.path.dirname(os.path.realpath(__file__))
    player_stats = os.path.join(cwd, "data", "nba_player_stats.csv")
    df = pd.read_csv(player_stats)
    X = df.loc[df['Year']==2017]
    # X['AST'] = X['AST'] / X['G']
    # X['TRB'] = X['TRB'] / X['G']
    X = X[['OWS', 'DWS', 'Player']].values
    X = X[0:50, :]

    fig, ax = plt.subplots()
    ax.scatter(X[:,0], X[:,1])

    labels = list(X[:,2])
    labels = [s.replace(" ", "_") for s in labels]
    X = X[:,0:2]

    for i, txt in enumerate(labels):
        ax.annotate(txt, (X[i,0], X[i,1]))

    plt.show()   

    H = linkage(X, 'complete')
    dendrogram(H)
    plt.show()


if __name__ == '__main__':
    bballClusters()