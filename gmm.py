import numpy as np
from clusters import uniformRandomSum, gaussianClusterParams, gaussianClusters
from sklearn import mixture
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
from scipy import linalg
import sys
import random 
from scipy.stats import multivariate_normal

def gmm(c, X, maxits=20):
    """
    gmm:
        Runs EM algo with full covariance matrices to fit gaussian mixture model on X

    Args:
        c (int): number of clusters
        X (n by k numpy array): array of n data points of dimensionality k

    Returns:
        means_ (c by k numpy array): specifies the k dimensional means of each cluster
        covariances_ (c by k by k numpy array): covariance matrices of each cluster
    """
    # Initialize
    pi = np.ones(c)/c
    n, k = X.shape
    initial_clusters = random.sample(range(n), c)
    means_ = X[initial_clusters, :]
    covariances_ = np.tile(np.eye(k), reps=(c, 1, 1)) #Initialize to Identity
    R = np.zeros((n, c))

    for i in range(maxits):
        # Expectation step
        temp = np.zeros((n, c))
        for i in range(c):
            u = means_[i, :]
            s = covariances_[i, :, :]
            temp[:, i] = pi[i] * multivariate_normal.pdf(x=X, mean=u, cov=s)

        # Normalize
        R = temp / np.tile(np.sum(temp, axis=1), reps=(c, 1)).T

        # Maximiation step

        # means
        Nk = np.sum(R, axis=0)
        R_expanded = np.tile(R, reps=(k, 1, 1))
        X_expanded = np.swapaxes(np.tile(X, reps=(c, 1, 1)), 0, 2)
        means_ = (np.sum(R_expanded*X_expanded, axis=1) / np.tile(Nk, reps=(k, 1))).T

        # covariances
        for i in range(c):
            delta = X - means_[i]
            Rdelta = np.expand_dims(R[:,i], -1) * delta
            covariances_[i] = Rdelta.T.dot(delta) / Nk[i] + np.eye(k)*1e-2

        # pi
        pi = Nk/n

    # Todo: stopping conditions

    return means_, covariances_


# From: https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html
color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])

def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-9., 5.)
    plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
#############################


if __name__=='__main__':
    M, V, N = gaussianClusterParams(c=2, k=2, n=1000, minMean=0, maxMean=5, minVar=-1, maxVar=1, stretch="ellipse")
    X = gaussianClusters(M, V, N)

    # Simple implementation
    means_, covariances_ = gmm(c=2, X=X)

    print("Means: " + "\nactual:\n" + str(M) + "\npredicted:\n\n" + str(means_) + "\n")
    print("Variances: " + "\nactual:\n" + str(V) + "\npredicted:\n\n" + str(covariances_) + "\n")
    print("\nCounts: " + str(N))


    # Scit-kit learn
    gmm = mixture.GaussianMixture(n_components=2, covariance_type='full')
    gmm.fit(X)

    print("Means: " + "\nactual:\n" + str(M) + "\npredicted:\n\n" + str(gmm.means_) + "\n")
    print("Variances: " + "\nactual:\n" + str(V) + "\npredicted:\n\n" + str(gmm.covariances_) + "\n")
    print("\nCounts: " + str(N))

    plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
             'Gaussian Mixture')
    plt.show()