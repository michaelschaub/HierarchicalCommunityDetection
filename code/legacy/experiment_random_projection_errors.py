from __future__ import division
import numpy as np
import scipy
import scipy.linalg
import GHRGmodel
import GHRGbuild
import spectral_algorithms as spectral
import metrics
from matplotlib import pyplot as plt
import metrics
from scipy.stats import ortho_group
from sklearn.cluster import KMeans
from sklearn import preprocessing

"""Given a set of eigenvectors find the clusters of the SBM"""
def clusterEVwithQR(EV, randomized=False, gamma=4):
    if randomized is True:
        Z, Q = orthogonalizeQR_randomized(EV,gamma)
    else:
        Z, Q = orthogonalizeQR(EV)

    cluster_ = scipy.absolute(Z).argmax(axis=1)

    return cluster_

def orthogonalizeQR(EV):
    """Given a set of eigenvectors V coming from a operator associated to the SBM,
    use QR decomposition as described in Damle et al 2017, to compute new coordinate
    vectors aligned with clustering vectors
    Input EV is an N x k matrix where each column corresponds to an eigenvector
    """
    k = EV.shape[1]
    Q, R, P = scipy.linalg.qr(EV.T, mode='economic', pivoting=True)
    # get indices of k representative points
    P = P[:k]

    # polar decomposition to find nearest orthogonal matrix
    U, S, V = scipy.linalg.svd(EV[P,:].T,full_matrices=False)

    #TODO: check this part!
    # Z = EV.dot(U.dot(V.T))
    Z = EV.dot(EV[P,:].T)

    return Z, Q

def orthogonalizeQR_randomized(EV,gamma=4):
    """Given a set of eigenvectors V coming from a operator associated to the SBM,
    use randomized QR decomposition as described in Damle et al 2017, to compute new
    coordinate vectors aligned with clustering vectors.

    Input EV is an N x k matrix where each column corresponds to an eigenvector
    gamma is the oversampling factor
    """
    n, k = EV.shape

    # sample EV according to leverage scores and the build QR from those vectors
    count = scipy.minimum(scipy.ceil(gamma*k*scipy.log(k)),n)
    elements = np.arange(n)
    prob = (EV.T**2).sum(axis=0)
    probabilities = prob / prob.sum()
    elements = np.random.choice(elements, count, p=probabilities)   #changed to np.random (since scipy.random is not a real module)
    ellemens = scipy.unique(elements)


    Q, R, P = scipy.linalg.qr(EV[elements,:].T, mode='economic', pivoting=True)
    # get indices of k representative points
    P = P[:k]

    # polar decomposition to find nearest orthogonal matrix
    U, S, V = scipy.linalg.svd(EV[P,:].T,full_matrices=False)

    Z = EV.dot(U.dot(V.T))

    return Z, Q

def pseudoinverse_partition_indicator(H):
    """ function to compute the pseudoinverse H^+ of the partition indicator matrix H"""
    if not scipy.sparse.issparse(H):
        print("Partition Indicator matrix should be sparse matrix")

    # sparse matrices enjoy more pleasant syntax, so below is matrix mult.
    D = H.T*H
    Hplus = scipy.sparse.linalg.spsolve(D,H.T)
    return Hplus

def project_to(subspace_basis,vectors_to_project):
    """
    Subspace basis: linearly independent (not necessarily orthogonal or normalized)
    vectors that span the space to which we want to project
    vectors_to_project: project these vectors into the specified subspace
    """

    if not scipy.sparse.issparse(vectors_to_project):
        V = np.matrix(vectors_to_project)
    else:
        V = vectors_to_project

    if not scipy.sparse.issparse(subspace_basis):
        S = np.matrix(subspace_basis)
    else:
        S = subspace_basis

    # compute S*(S^T*S)^{-1}*S'*V
    X1 = S.T*V
    X2 = S.T*S
    projected = S*scipy.sparse.linalg.spsolve(X2,X1)

    return projected

def calculate_proj_error(U,V,norm):
    proj1 = project_to(U,V)

    if norm == 'F':
        error = scipy.linalg.norm(proj1)
    elif norm == '2':
        error = scipy.linalg.norm(proj1,2)

    return error

def project_orthogonal_to(subspace_basis,vectors_to_project):
    """
    Subspace basis: linearly independent (not necessarily orthogonal or normalized)
    vectors that span the space orthogonal to which we want to project
    vectors_to_project: project these vectors into the orthogonal complement of the
    specified subspace
    """

    if not scipy.sparse.issparse(vectors_to_project):
        V = np.matrix(vectors_to_project)
    else:
        V = vectors_to_project

    if not scipy.sparse.issparse(subspace_basis):
        S = np.matrix(subspace_basis)
    else:
        S = subspace_basis

    # compute S*(S^T*S)^{-1}*S'*V
    X1 = S.T*V
    # print X1
    X2 = S.T*S
    # print X2
    projected = S*scipy.sparse.linalg.spsolve(X2,X1)
    # print projected

    orthogonal_proj = V - projected
    return orthogonal_proj

def calculate_proj_error2(evecs,H,norm):
    n, k = np.shape(H)
    if n == k:
        error =0
        return error
    V = evecs[:,:k]
    proj1 = project_orthogonal_to(H,V)
    proj2 = project_orthogonal_to(V,H)


    if norm == 'F':
        norm1 = scipy.linalg.norm(proj1)/np.sqrt(k-k**2/n)
        norm2 = scipy.linalg.norm(proj2)/np.sqrt(k-k**2/n)
        error = 0.5*(norm1+norm2)
    elif norm == '2':
        norm1 = scipy.linalg.norm(proj1,2)
        norm2 = scipy.linalg.norm(proj2,2)
        error = .5*(norm1+norm2)

    return error

def test_random_projection(n=24,k=3,p=21):
    """Within an n dimensional space, project an orthogonal set of k vectors (a k-dimensional subspace) into a space of dimension p"""

    nsamples = 250
    # create another set of p random vectors
    U = ortho_group.rvs(dim=n)
    U = U[:,:p]

    error = np.zeros(nsamples)
    expected_error = np.sqrt(k*p/n)

    for i in range(nsamples):
        # create k orthogonal vectors
        V = ortho_group.rvs(dim=n)
        V = V[:,:k]
        error[i] = calculate_proj_error(U,V,'F')
        print i, error[i]/expected_error

    plt.figure()
    plt.plot(error/expected_error)

    diff  = np.mean(error) - expected_error
    print diff
    print np.std(error)

    # second test
    error2 = np.zeros(nsamples)
    expected_error2 = np.sqrt(1 - k**2/n)
    for i in range(nsamples):
        # create k orthogonal vectors
        V = ortho_group.rvs(dim=n)
        V = V[:,:k]
        clust = KMeans(n_clusters = k)
        clust.fit(V)
        partition_vec = clust.labels_
        partition_vec = spectral.relabel_partition_vec(partition_vec)
        H = spectral.create_partition_matrix_from_vector(partition_vec)
        H = preprocessing.normalize(H,axis=0,norm='l2')
        error2[i] = scipy.linalg.norm(V - project_to(H,V))


    plt.figure()
    plt.plot(error2/expected_error2)

    diff  = np.mean(error2) - expected_error2
    print diff
    print np.std(error2)


    return V, U

def test_projection_diagonal_blocks(k=4,q=3):

    nsamples = 250
    p =0.8
    noise = 0.0001
    error = np.zeros(nsamples)
    qr_error = np.zeros(nsamples)
    norm = "F"
    mode = "SBM"

    for ii in range(nsamples):
        A = np.diag(np.ones(k)*p)
        # A[0,1]=p*0.5
        # A[1,0]=p*0.5
        # A[2,3]=p*0.5
        # A[3,2]=p*0.5
        noise = np.triu(noise*np.random.rand(k,k))
        noise = noise + noise.T - np.diag(np.diag(noise))
        A = A+ noise

        reg= False
        # normalized Laplacian is D^-1/2 A D^-1/2
        L, Dtau_sqrt_inv = spectral.construct_normalised_Laplacian(A,reg)
        if reg:
            # set tau to average degree
            tau = A.sum()/A.shape[0]
        else:
            tau = 0

        ev, evecs = scipy.linalg.eigh(L)
        index = np.argsort(np.abs(ev))
        evecs = evecs[:,index[::-1]]
        sigma = np.abs(ev[index[::-1]])


        # plt.figure()
        # plt.plot(evecs)
        # plt.title("eigenvectors")


        partition_vec, Hnorm = spectral.find_partition(evecs, q, tau, norm, mode, Dtau_sqrt_inv)
        H = spectral.create_partition_matrix_from_vector(partition_vec)
        error[ii] = calculate_proj_error2(evecs, Hnorm, norm)
        print(error[ii])

        partition_vec = clusterEVwithQR(evecs[:,:q])
        H = spectral.create_partition_matrix_from_vector(partition_vec)
        Hnorm = preprocessing.normalize(H, axis=0, norm='l2')
        qr_error[ii] = calculate_proj_error2(evecs, Hnorm, norm)
        print("QR, error / exp rand error")
        print(qr_error[ii])

    plt.figure()
    plt.plot(error)
    plt.figure()
    plt.plot(qr_error)
