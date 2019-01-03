import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D

def genTestData(M, V, N, random_colors):
    """
    genTestData: takes in parameters specifying multivariate Gaussian distributions. Draws the specified
                number of observations from each distribution and returns the combined, shuffled data.

                c: num clusters
                k: num features
                n: number of data points
    Args:
        M (numpy array): c by k matrix specifying the cluster means
        V (numpy array): c by k by k matrix specifying the covariance matrix for each Gaussian
        N (numpy array): c by 1 vector specifying number of data points generate by each distribution

    Returns:
        X: (numpy array): n by k matrix of shuffled data points
    """
    # Housekeeping
    n = np.sum(N)
    c, k = np.shape(M)
    X = np.zeros((n, k))
    
    colors = np.zeros((n, 3))
    
    # Draw from each distribution
    index = 0
    for i in range(len(N)):
        group_n = N[i]
        nextData = np.random.multivariate_normal(M[i], V[i], N[i])
        X[index:(index+group_n)] = nextData
        colors[index:(index+group_n), :] = np.tile(random_colors[i, :], reps=(group_n, 1))
        index += group_n
    
    # Display
    if k in [1, 2]:
        plt.scatter(X[:,0], X[:,1], c=colors)
        plt.show()            

    elif k==3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:,0], X[:,1], X[:,2], c=colors)
        plt.show()
    
    else:
        print("Does not display " + str(k) + " dimensional data") 

    # Mix and return
    np.random.shuffle(X)
    return(X)



def softKmeans(X, c, random_colors, maxIters=50, b=1.0, plot=True):
    """
    softKmeans: given input data X, calculates cluster centers M, and responsibility matrix R. Plots it plot=True.

    Args: 
        X (numpy array): n by k for n observations each with k features
        c (int): number of clusters
        maxIters (int): maximum number of iterations of kmeans to complete
        b (double): variance parameter


    Returns:
        (R, M): (n by c responsibility matrix, c by k clusters matrix)
    """
    n, k = np.shape(X)
    R = np.zeros((n, c))
    M = np.zeros((c, k))

    # Randomly initialize clusters to datapoints
    indices = np.random.randint(n, size=c)
    M = X[indices, :]

    # while True:
    oldCost = float("inf")
    cost = float("inf")
    its = 0
    while True:

        # Step 1: calculate cluster responsibilities
        D = np.exp(-b*cdist(X, M))
        denom = np.tile(np.sum(D, axis=1), reps=(3, 1)).T
        R = D/denom

        # Step 2: calculate cluster means
        X_c = np.repeat(X[:, :, np.newaxis], c, axis=2)
        R_k = np.repeat(R[:, np.newaxis, :], k, axis=1)
        X_R = np.sum(X_c * R_k, axis=0)
        R_sum = np.tile(np.sum(R, axis=0), reps=(k, 1))
        M = np.transpose(X_R / R_sum)

        # Stop early if minor change in cost or hit max its
        oldCost = cost
        cost = np.sum(cdist(X, M) * R)
        if (its == maxIters or abs(cost-oldCost)<0.05): break

    if plot:
        # random_colors = np.random.random((c, 3))
        colors = R.dot(random_colors)
        if k in [1, 2]:
            plt.scatter(X[:,0], X[:,1], c=colors)
            plt.show()            

        elif k==3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X[:,0], X[:,1], X[:,2], c=colors)
            plt.show()
        
        else:
            print("Does not display " + str(k) + " dimensional data")

    return(R, M)

if __name__ == '__main__':
    
    # 2D
    M = np.array([ 
                [0, 0],
                [10, 0],
                [10, 10]
                ])
    
    V = np.array([
                [ [1, 0], [0, 1] ],
                [ [1, 0], [0, 1] ],
                [ [1, 0], [0, 1] ]
                ])

    N = np.array([50, 50, 50])
    c, _ = M.shape
    random_colors = np.random.random((c, 3))
    
    # Generate data
    X = genTestData(M, V, N, random_colors=random_colors)
    #  Soft k means
    R, M = softKmeans(X, c=3, random_colors=random_colors)

    # 3D
    M = np.array([ 
                [0, 0, 0],
                [10, 0, 0],
                [0, 10, 0]
                ])
    
    V = np.array([
                [ [10, 0, 0], [0, 1, 0], [0, 0, 10] ],
                [ [1, 0, 0], [0, 1, 0], [0, 0, 1] ],
                [ [1, 0, 0], [0, 1, 0], [0, 0, 1] ]
                ])

    N = np.array([50, 50, 50])
    c, _ = M.shape
    random_colors = np.random.random((c, 3))

    # Generate data
    X = genTestData(M, V, N, random_colors=random_colors)
    #  Soft k means
    R, M = softKmeans(X, c=3, random_colors=random_colors)
# EOF