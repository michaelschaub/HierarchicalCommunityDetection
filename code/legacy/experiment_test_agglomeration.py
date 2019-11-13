from __future__ import division
import numpy as np
import scipy
import scipy.linalg
import GHRGmodel
import GHRGbuild
import spectral_algorithms as spectral
import metrics
import sample_networks
from matplotlib import pyplot as plt
from scipy.signal import argrelmin
from scipy.stats import ortho_group
from sklearn import preprocessing

"""Add some small random noise to a (dense) small matrix as a perturbation"""
def add_noise_to_small_matrix(M,snr=0.001,noise_type="gaussian"):

    #noise level is taken relative to the Froebenius norm
    normM = scipy.linalg.norm(M)

    if noise_type == "uniform":
        #TODO -- should we have uniform noise?
        pass
    elif noise_type == "gaussian":
        n,m = M.shape
        noise = scipy.triu(scipy.random.randn(n,m))
        noise = noise + noise.T -scipy.diag(scipy.diag(noise))
        normNoise = scipy.linalg.norm(noise)
        Mp = M + snr*normM/normNoise*noise

    return Mp

def pseudoinverse_partition_indicator(H):
    """ function to compute the pseudoinverse H^+ of the partition indicator matrix H"""
    if not scipy.sparse.issparse(H):
        print("Partition Indicator matrix should be sparse matrix")

    # sparse matrices enjoy more pleasant syntax, so below is matrix mult.
    D = H.T*H
    Hplus = scipy.sparse.linalg.spsolve(D,H.T)
    return Hplus


def xlogy(x,y):
    """ compute x log(y) elementwise, with the convention that 0log0 = 0"""
    logy = np.ones_like(y)
    logy[y==0] = 0
    logy[y!=0] = np.log(y[y!=0])
    xlogy = x*logy
    return xlogy


def compute_likelihood_SBM(pvec,A,omega=None):
    H = spectral.create_partition_matrix_from_vector(pvec)
    # self-loops and directedness is not allowed here
    Emat, Nmat = spectral.compute_number_links_between_groups(A,pvec,directed=False)
    if omega is None:
        # note the 1* is important -- otherwise no deep copy is made!
        omega = 1*Emat
        omega[omega!=0] = omega[omega!=0]/Nmat[omega!=0]

    logPmat = xlogy(Emat,omega) + xlogy(Nmat-Emat,1 - omega)
    likelihood = logPmat.sum()
    return likelihood

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

    # print "HERE"
    # compute S*(S^T*S)^{-1}*S'*V
    X1 = S.T*V
    # print X1
    X2 = S.T*S
    # print X2
    projected = S*scipy.sparse.linalg.spsolve(X2,X1)
    # print projected

    orthogonal_proj = V - projected
    return orthogonal_proj

def calculate_proj_error(evecs,H,norm):
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


def test_subspace_angle():
    U = ortho_group.rvs(dim=20)
    U = U[:,1:10]
    V = ortho_group.rvs(dim=20)
    V = V[:,1:10]
    angle = find_subspace_angle_between_ev_bases(U,V)
    return angle

def find_subspace_angle_between_ev_bases(U,V):
    Q1 = U.T.dot(V)
    sigma = scipy.linalg.svd(Q1,compute_uv=False)
    angle = -np.ones(sigma.size)
    cos_index = sigma**2 <=0.5
    if np.all(cos_index):
        angle = np.arccos(sigma)
    else:
        Q2 = V - U.dot(Q1)
        sigma2 = scipy.linalg.svd(Q2,compute_uv=False)
        angle[cos_index] = np.arccos(sigma[cos_index])
        sin_index = np.bitwise_not(cos_index)
        angle[sin_index] = np.arcsin(sigma2[sin_index])

    return angle, sigma

def clusterEVwithQR(EV, randomized=False, gamma=4):
    """Given a set of eigenvectors find the clusters of the SBM"""
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



def createTiagoHierNetwork():
    a = 0.95
    b = 0.75
    c = 0.4
    d = 0.1
    Omega = np.array([[a, b, b, 0, 0, 0, d, 0, 0, 0, 0, 0],
                      [0, a, b, 0, 0, 0, 0, d, 0, 0, 0, 0],
                      [0, 0, a, c, 0, 0, 0, 0, d, 0, 0, 0],
                      [0, 0, 0, a, b, b, 0, 0, 0, d, 0, 0],
                      [0, 0, 0, 0, a, b, 0, 0, 0, 0, d, 0],
                      [0, 0, 0, 0, 0, a, 0, 0, 0, 0, 0, d],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    Omega = Omega + Omega.T -np.diagflat(np.diag(Omega))
    plt.imshow(Omega)

    # get some eigenvectors
    nc = np.ones(12,dtype=int)*100
    A = sample_networks.sample_block_model(Omega,nc)
    Ln,_ = spectral.construct_normalised_Laplacian(A,False)
    ev, evecs = scipy.sparse.linalg.eigsh(Ln,12,which='LM')
    print ev
    plt.figure()
    plt.plot(evecs)

    p0 = spectral.cluster_with_BetheHessian(A,num_groups=-1,mode='unweighted', regularizer='BHa',clustermode='kmeans')
    p0 = spectral.relabel_partition_vec(p0)
    plt.figure()
    plt.plot(p0)
    # for i in range(12):
        # plt.figure()
        # plt.plot(evecs[:,i])

    # aggregate matrix
    # TODO: perhaps we should have the EEP normalization here?!
    Eagg, Nagg = spectral.compute_number_links_between_groups(A,p0)
    Aagg = Eagg / Nagg
    # plt.figure()
    # plt.imshow(Aagg,interpolation='nearest')

    reg= False
    # normalized Laplacian is D^-1/2 A D^-1/2
    L, Dtau_sqrt_inv = spectral.construct_normalised_Laplacian(Aagg,reg)
    # D = scipy.sparse.diags(np.array(Aagg.sum(1)).flatten(),0)
    # L = D - Aagg
    # plt.figure()
    # plt.imshow(L,interpolation='none')
    if reg:
        # set tau to average degree
        tau = Aagg.sum()/Aagg.shape[0]
    else:
        tau = 0

    ev, evecs = scipy.linalg.eigh(L)
    index = np.argsort(np.abs(ev))
    evecs = evecs[:,index[::-1]]
    sigma = np.abs(ev[index[::-1]])
    # evecs = evecs[:,index]
    plt.figure()
    plt.plot(sigma)

    total_energy = sigma.sum()
    sigma_gap = np.abs(np.diff(sigma))
    approx_energy = np.cumsum(sigma)
    plt.figure()
    plt.plot(sigma_gap)
    plt.figure()
    plt.plot(np.diff(approx_energy/total_energy))

    max_k = np.max(p0)+1
    norm = 'F'
    mode = 'SBM'
    thres = 0.2
    error = np.zeros(max_k)
    likelihood = np.zeros(max_k)


    for k in range(max_k):
        partition_vec, Hnorm = spectral.find_partition(evecs, k+1, tau, norm, mode, Dtau_sqrt_inv)
        H = spectral.create_partition_matrix_from_vector(partition_vec)
        error[k] = calculate_proj_error(evecs, Hnorm, norm)
        likelihood[k] = compute_likelihood_SBM(partition_vec[p0],A)
        print("K, error / exp rand error, likelihood")
        print(k+1, error[k], likelihood[k])

    plt.figure()
    plt.plot(1+np.arange(max_k),error)
    plt.figure()
    plt.plot(1+np.arange(max_k),likelihood)
    return A

def createCliqueNetwork():
    a = 1
    Omega = np.eye(12)*a
    plt.imshow(Omega)

    # get some eigenvectors
    nc = np.ones(12,dtype=int)*5
    A = sample_networks.sample_block_model(Omega,nc)
    # Ln,_ = spectral.construct_normalised_Laplacian(A,False)
    # ev, evecs = scipy.sparse.linalg.eigsh(Ln,12,which='LM')

    p0 = spectral.cluster_with_BetheHessian(A,num_groups=-1,mode='unweighted', regularizer='BHa',clustermode='kmeans')
    p0 = spectral.relabel_partition_vec(p0)
    plt.figure()
    plt.plot(p0)
    # for i in range(12):
        # plt.figure()
        # plt.plot(evecs[:,i])

    # aggregate matrix
    # TODO: perhaps we should have the EEP normalization here?!
    Eagg, Nagg = spectral.compute_number_links_between_groups(A,p0)
    Aagg = Eagg / Nagg
    plt.figure()
    plt.imshow(Aagg,interpolation='nearest')

    reg= False
    # normalized Laplacian is D^-1/2 A D^-1/2
    L, Dtau_sqrt_inv = spectral.construct_normalised_Laplacian(Aagg,reg)
    # D = scipy.sparse.diags(np.array(Aagg.sum(1)).flatten(),0)
    # L = D - Aagg
    # plt.figure()
    # plt.imshow(L,interpolation='none')
    if reg:
        # set tau to average degree
        tau = Aagg.sum()/Aagg.shape[0]
    else:
        tau = 0

    ev, evecs = scipy.linalg.eigh(L)
    index = np.argsort(np.abs(ev))
    evecs = evecs[:,index[::-1]]
    sigma = np.abs(ev[index[::-1]])
    # evecs = evecs[:,index]
    plt.figure()
    plt.plot(sigma)

    total_energy = sigma.sum()
    sigma_gap = np.abs(np.diff(sigma))
    approx_energy = np.cumsum(sigma)
    plt.figure()
    plt.plot(sigma_gap)
    plt.figure()
    plt.plot(np.diff(approx_energy/total_energy))

    max_k = np.max(p0)+1
    norm = 'F'
    mode = 'SBM'
    thres = 0.2
    error = np.zeros(max_k)
    likelihood = np.zeros(max_k)


    for k in range(max_k):
        partition_vec, Hnorm = spectral.find_partition(evecs, k+1, tau, norm, mode, Dtau_sqrt_inv)
        H = spectral.create_partition_matrix_from_vector(partition_vec)
        error[k] = calculate_proj_error(evecs, Hnorm, norm)
        likelihood[k] = compute_likelihood_SBM(partition_vec[p0],A)
        print("K, error / exp rand error, likelihood")
        print(k+1, error[k], likelihood[k])

    plt.figure()
    plt.plot(1+np.arange(max_k),error)
    plt.figure()
    plt.plot(1+np.arange(max_k),likelihood)

def test_agglomeration_ideas(groups_per_level=3):
    n=2*2700
    snr=3
    c_bar=100
    n_levels=1


    #generate
    D_actual=GHRGbuild.create2paramGHRG(n,snr,c_bar,n_levels,groups_per_level)
    ptrue, _ = D_actual.get_partition_at_level(-1) # true partition lowest level
    G=D_actual.generateNetworkExactProb()
    A=D_actual.to_scipy_sparse_matrix(G)
    # display A
    plt.figure
    # plt.spy(A,markersize=0.01)
    plt.imshow(A.A,cmap='Greys')

    # do a first round of clustering with the Bethe Hessian
    # p0 = spectral.cluster_with_BetheHessian(A,num_groups=groups_per_level**n_levels,mode='unweighted', regularizer='BHa',clustermode='kmeans')
    # p0 = spectral.cluster_with_BetheHessian(A,num_groups=-1,mode='unweighted', regularizer='BHa',clustermode='kmeans')
    p0=ptrue.astype(int)
    p0 = spectral.relabel_partition_vec(p0)
    plt.figure()
    plt.plot(p0)
    plt.title("partition found / used")


    max_k = np.max(p0)+1
    norm = 'F'
    mode = 'SBM'
    thres = 0.2
    error = np.zeros(max_k)
    likelihood = np.zeros(max_k)

    # aggregate matrix
    # TODO: perhaps we should have the EEP normalization here?!
    Eagg, Nagg = spectral.compute_number_links_between_groups(A,p0)
    Aagg = Eagg / Nagg
    plt.figure()
    plt.imshow(Aagg,interpolation='nearest')
    plt.title("aggregated A matrix")

    reg= False
    # normalized Laplacian is D^-1/2 A D^-1/2
    L, Dtau_sqrt_inv = spectral.construct_normalised_Laplacian(Aagg,reg)
    # D = scipy.sparse.diags(np.array(Aagg.sum(1)).flatten(),0)
    # L = D - Aagg
    # plt.figure()
    # plt.imshow(L,interpolation='none')
    if reg:
        # set tau to average degree
        tau = Aagg.sum()/Aagg.shape[0]
    else:
        tau = 0

    ev, evecs = scipy.linalg.eigh(L)
    index = np.argsort(np.abs(ev))
    evecs = evecs[:,index[::-1]]
    sigma = np.abs(ev[index[::-1]])
    # evecs = evecs[:,index]
    plt.figure()
    plt.plot(sigma)
    plt.title("Singular values")

    total_energy = sigma.sum()
    sigma_gap = np.abs(np.diff(sigma))
    approx_energy = np.cumsum(sigma)
    plt.figure()
    plt.plot(1+np.arange(1,max_k),sigma_gap)
    plt.title("Singular values gap")


    plt.figure()
    plt.plot(1+np.arange(1,max_k),np.diff(approx_energy/total_energy))
    plt.title("Approximation quality / energy")

    plt.figure()
    plt.plot(evecs)
    plt.title("eigenvectors")


    for k in range(max_k):

        partition_vec, Hnorm = spectral.find_partition(evecs, k+1, tau, norm, mode, Dtau_sqrt_inv)
        H = spectral.create_partition_matrix_from_vector(partition_vec)
        error[k] = calculate_proj_error(evecs, Hnorm, norm)
        likelihood[k] = compute_likelihood_SBM(partition_vec[p0],A)
        print("K, error / exp rand error, likelihood")
        print(k+1, error[k], likelihood[k])

        partition_vec = clusterEVwithQR(evecs[:,:k+1])
        H = spectral.create_partition_matrix_from_vector(partition_vec)
        Hnorm = preprocessing.normalize(H, axis=0, norm='l2')
        error[k] = calculate_proj_error(evecs, Hnorm, norm)
        likelihood[k] = compute_likelihood_SBM(partition_vec[p0],A)
        print("K, error / exp rand error, likelihood")
        print(k+1, error[k], likelihood[k])

    plt.figure()
    plt.plot(1+np.arange(max_k),error)
    plt.title("projection error")

    plt.figure()
    plt.plot(1+np.arange(max_k),likelihood)
    plt.title("likelihood")
    return D_actual

def test_agglomeration_ideas_noise_pert(groups_per_level=2):
    # n=2**13
    n=3**9
    snr=7
    c_bar=50
    n_levels=2

    max_k = groups_per_level**n_levels
    norm = 'F'
    mode = 'SBM'
    thres = 1/3

    #generate
    D_actual=GHRGbuild.create2paramGHRG(n,snr,c_bar,n_levels,groups_per_level)
    ptrue, _ = D_actual.get_partition_at_level(-1) # true partition lowest level
    G=D_actual.generateNetworkExactProb()
    A=D_actual.to_scipy_sparse_matrix(G)

    # do a first round of clustering with the Bethe Hessian
    # p0 = spectral.cluster_with_BetheHessian(A,num_groups=groups_per_level**n_levels,mode='unweighted', regularizer='BHa',clustermode='kmeans')
    p0 = spectral.cluster_with_BetheHessian(A,num_groups=-1,mode='unweighted', regularizer='BHa',clustermode='kmeans')
    # p0=ptrue.astype(int)
    # k0 = p0.max() + 1
    # p0, _ = spectral.regularized_laplacian_spectral_clustering(A,num_groups=k0,clustermode='qr')
    p0 = spectral.relabel_partition_vec(p0)
    plt.figure(1)
    plt.plot(p0,'x')


    # aggregate matrix
    Eagg, Nagg = spectral.compute_number_links_between_groups(A,p0)
    Aagg = Eagg / np.sqrt(Nagg)
    aggregate = True

    while aggregate:
        max_k = np.max(p0)+1
        error = np.zeros(max_k)
        angle_min = np.zeros(max_k)
        angle_max = np.zeros(max_k)
        likelihood = np.zeros(max_k)
        print "\n\nAggregation round started"
        print("maximal k ", max_k)
        reg= False
        # normalized Laplacian is D^-1/2 A D^-1/2
        L, Dtau_sqrt_inv = spectral.construct_normalised_Laplacian(Aagg,reg)
        D = scipy.sparse.diags(np.array(Aagg.sum(1)).flatten(),0)
        L = D - Aagg
        tau = 0
        ev, evecs = scipy.linalg.eigh(L)

        index = np.argsort(np.abs(ev))
        evecs = evecs[:,index[::1]]
        sigma = np.abs(ev[index[::1]])

        # evecs = evecs[:,index]
        # plt.figure()
        # plt.plot(sigma)
        # plt.title("Singular values")
        # total_energy = sigma.sum()
        # sigma_gap = np.abs(np.diff(sigma))
        # approx_energy = np.cumsum(sigma)
        # plt.figure()
        # plt.plot(1+np.arange(1,max_k),sigma_gap)
        # plt.title("Singular values gap")
        # plt.figure()
        # plt.plot(1+np.arange(1,max_k),np.diff(approx_energy/total_energy))
        # plt.title("Approximation quality / energy")
        # plt.figure()
        # plt.plot(evecs)
        # plt.title("eigenvectors")


        candidates_for_hier = np.zeros(max_k)
        for k in range(max_k):

            partition_vec, Hnorm = spectral.find_partition(evecs, k+1, tau, norm, mode, Dtau_sqrt_inv)
            H = spectral.create_partition_matrix_from_vector(partition_vec)
            error[k] = calculate_proj_error(evecs, Hnorm, norm)
            angles,_ = find_subspace_angle_between_ev_bases(Hnorm,evecs[:,:k+1])
            angle_min[k] = np.min(angles)
            angle_max[k] = np.max(angles)
            likelihood[k] = compute_likelihood_SBM(partition_vec[p0],A)
            print("K, error / exp rand error, likelihood")
            print(k+1, error[k], likelihood[k])

            # partition_vec = clusterEVwithQR(evecs[:,:k+1])
            # H = spectral.create_partition_matrix_from_vector(partition_vec)
            # Hnorm = preprocessing.normalize(H, axis=0, norm='l2')
            # error[k] = calculate_proj_error(evecs, Hnorm, norm)
            # likelihood[k] = compute_likelihood_SBM(partition_vec[p0],A)
            # print("K, error / exp rand error, likelihood")
            # print(k+1, error[k], likelihood[k])
            if error[k] - thres < 0:
                candidates_for_hier[k] = 1

        plt.figure(4)
        plt.plot(1+np.arange(max_k),angle_max)
        plt.plot(1+np.arange(max_k),angle_min)

        candidate_list = np.nonzero(candidates_for_hier)[0]+1
        print "\ninitial candidate_list: "
        print candidate_list
        print "\n\ncreating perturbed samples"
        num_pert = 10
        error_rand = np.ones((num_pert,max_k))
        angle_max_rand = np.zeros((num_pert,max_k))
        angle_min_rand = np.zeros((num_pert,max_k))
        likelihood_rand = np.zeros((num_pert,max_k))
        for pp in range(num_pert):
            Anew = add_noise_to_small_matrix(Aagg)
            reg= False
            # normalized Laplacian is D^-1/2 A D^-1/2
            L, Dtau_sqrt_inv = spectral.construct_normalised_Laplacian(Anew,reg)
            if reg:
                # set tau to average degree
                tau = Anew.sum()/Anew.shape[0]
            else:
                tau = 0

            ev, evecs = scipy.linalg.eigh(L)
            index = np.argsort(np.abs(ev))
            evecs = evecs[:,index[::-1]]
            sigma = np.abs(ev[index[::-1]])

            for k in candidate_list:

                # partition_vec, Hnorm = spectral.find_partition(evecs, k, tau, norm, mode, Dtau_sqrt_inv)
                # error_rand[pp,k] = calculate_proj_error(evecs, Hnorm, norm)
                # likelihood_rand[pp,k] = compute_likelihood_SBM(partition_vec[p0],A)
                # print("K, error / exp rand error, likelihood")
                # print(k, error_rand[pp,k], likelihood_rand[pp,k])

                partition_vec = clusterEVwithQR(evecs[:,:k])
                H = spectral.create_partition_matrix_from_vector(partition_vec)
                Hnorm = preprocessing.normalize(H, axis=0, norm='l2')
                error_rand[pp,k-1] = calculate_proj_error(evecs, Hnorm, norm)
                angle_min_rand[pp,k-1] = np.min(find_subspace_angle_between_ev_bases(Hnorm,evecs[:,:k])[0])
                angle_max_rand[pp,k-1] = np.max(find_subspace_angle_between_ev_bases(Hnorm,evecs[:,:k])[0])
                likelihood_rand[pp,k-1] = compute_likelihood_SBM(partition_vec[p0],A)
                print("K, error / exp rand error, likelihood")
                print(k, error_rand[pp,k-1], likelihood_rand[pp,k-1])

        error_av = np.mean(error_rand,0)
        error_std = np.std(error_rand,0)
        plt.figure(2)
        plt.plot(1+np.arange(max_k),error_av)
        plt.plot(np.array([3, 9, 27]),0.1*np.ones(3),'o')
        plt.figure(3)
        plt.plot(1+np.arange(max_k),error_std)
        plt.plot(np.array([3, 9, 27]),0.1*np.ones(3),'o')

        angle_min_rand_av = np.mean(angle_min_rand,0)
        angle_max_rand_av = np.mean(angle_max_rand,0)
        angle_min_rand_std = np.std(angle_min_rand,0)
        angle_max_rand_std = np.std(angle_max_rand,0)
        plt.figure(5)
        plt.errorbar(1+np.arange(max_k),angle_min_rand_av,yerr=angle_min_rand_std)
        plt.plot(np.array([3, 9, 27]),0.1*np.ones(3),'o')
        plt.figure(6)
        plt.errorbar(1+np.arange(max_k),angle_max_rand_av,yerr=angle_max_rand_std)
        plt.plot(np.array([3, 9, 27]),0.1*np.ones(3),'o')


        relative_minima = argrelmin(error_av)[0] + 1
        print "Relative minima"
        print relative_minima
        filtered_candidates_local_minima = np.intersect1d(relative_minima, candidate_list)
        filter_start = 0*np.nonzero(np.diff(candidates_for_hier)==-1)[0]+1
        print "Filter start"
        print filter_start
        filtered_candidates = np.union1d(filtered_candidates_local_minima,filter_start)
        # filtered_candidates = filtered_candidates_local_minima
        filtered_candidates = np.setdiff1d(filtered_candidates,np.ones(1))
        print "Candidate levels for merging"
        print filtered_candidates

        found_partition = False

        for k in filtered_candidates[::-1]:
            print k
            std = scipy.std(error_rand[:,k-1])
            if std  < 0.01 :
                print "std: ", std
                print "\nAgglomeration Test passed, at level\n", k
                found_partition = True
                partition_vec = clusterEVwithQR(evecs[:,:k])
                p0 = partition_vec[p0]
                p0 = spectral.relabel_partition_vec(p0)
                plt.figure(1)
                plt.plot(p0,'-')
                break


        if found_partition:
            Eagg, Nagg = spectral.compute_number_links_between_groups(A,p0)
            Aagg = Eagg / np.sqrt(Nagg)
            aggregate = True
        else:
            aggregate = False

    return D_actual

def agglomeration_loop_SNR():
    n = 2 * 2700
    SNR = np.arange(0.5, 10.5, 0.5)  # start, stop (exclusive), spacing
    # SNR = np.arange(5, 10.5, 5)  # start, stop (exclusive), spacing
    c_bar = 30
    n_levels = 3
    groups_per_level = 3
    nsamples = 10

    norm = 'F'
    mode = 'SBM'
    thres = 0.25
    reg = False
    tau = 0
    kmax = groups_per_level**n_levels
    error = np.zeros((SNR.size,nsamples,kmax))
    likelihood = np.zeros((SNR.size,nsamples,kmax))
    candidates_for_hier = np.zeros((SNR.size,nsamples,kmax))
    filtered_candidates_hier = np.zeros((SNR.size,nsamples,kmax))
    for s, snr in enumerate(SNR):
        print("SNR level: ", snr)
        for ni in range(nsamples):
            print("Sample: ", ni)
            # generate
            D_actual = GHRGbuild.create2paramGHRG(n, snr, c_bar, n_levels, groups_per_level)
            ptrue, _ = D_actual.get_partition_at_level(-1)  # true partition lowest level
            G = D_actual.generateNetworkExactProb()
            A = D_actual.to_scipy_sparse_matrix(G)

            # Use true solution for initial testing
            # p0 = ptrue.astype(int)
            # do a first round of clustering with the Bethe Hessian
            # p0 = spectral.cluster_with_BetheHessian(A,num_groups=groups_per_level**n_levels,mode='unweighted', regularizer='BHa',clustermode='kmeans')
            p0 = spectral.cluster_with_BetheHessian(A,num_groups=-1,mode='unweighted', regularizer='BHa',clustermode='kmeans')
            # p0 = spectral.relabel_partition_vec(p0)
            # plt.figure()
            # plt.plot(p0)

            # aggregate matrix
            Eagg, Nagg = spectral.compute_number_links_between_groups(A, p0)
            Aagg = Eagg / Nagg

            # normalized Laplacian is D^-1/2 A D^-1/2
            L, Dtau_sqrt_inv = spectral.construct_normalised_Laplacian(Aagg, reg)

            ev, evecs = scipy.linalg.eigh(L)
            index = np.argsort(np.abs(ev))
            evecs = evecs[:, index[::-1]]
            sigma = np.abs(ev[index[::-1]])

            total_energy = sigma.sum()
            sigma_gap = np.abs(np.diff(sigma))
            approx_energy = np.cumsum(sigma)

            max_k = int(np.max(p0) + 1)
            for k in range(max_k):
                partition_vec, Hnorm = spectral.find_partition(evecs, k + 1, tau, norm, mode, Dtau_sqrt_inv)
                H = spectral.create_partition_matrix_from_vector(partition_vec)
                error[s,ni,k] = calculate_proj_error(evecs, Hnorm, norm)
                likelihood[s,ni,k] = compute_likelihood_SBM(partition_vec[p0], A)

                print("K, error / exp rand error - threshold, likelihood")
                print(k + 1, error[s,ni,k] - thres, likelihood[s,ni,k])

                if error[s,ni,k] - thres < 0:
                    candidates_for_hier[s,ni,k] = 1

            relative_minima = argrelmin(error[s,ni,:])[0] + 1
            candidate_list = np.nonzero(candidates_for_hier[s,ni,:])[0] + 1
            filtered_candidates = np.intersect1d(relative_minima, candidate_list)
            print filtered_candidates
            filtered_candidates_hier[s,ni,filtered_candidates-1] = 1

    for s, snr in enumerate(SNR):
        plt.figure()
        plt.plot(np.arange(1,kmax+1),np.mean(error[s,:,:],axis=0))
        plt.plot(np.arange(1,kmax+1),np.mean(candidates_for_hier[s,:,:],axis=0))
        plt.plot(np.arange(1,kmax+1),np.mean(filtered_candidates_hier[s,:,:],axis=0))

        for i in range(kmax):
            plt.plot(i*np.ones(nsamples)+1,error[s,:,i],'x')
        name = "AgglomerationTest_snr" + str(snr) + ".pdf"
        plt.savefig(name)

def findNumberOfCommunitiesByInvariantSubspace():
    """Given a matrix M, find the dominant 'low-rank' components by perturbing the matrix slightly and checking, what number of eigenvectors keep spanning the same subspace"""

    num_pert = 20
    kguess = 15

    a = 0.95
    b = 0.7
    c = 0.35
    d = 0.2
    Omega = np.array([[a, b, b, 0, 0, 0, d, 0, 0, 0, 0, 0],
                      [0, a, b, 0, 0, 0, 0, d, 0, 0, 0, 0],
                      [0, 0, a, c, 0, 0, 0, 0, d, 0, 0, 0],
                      [0, 0, 0, a, b, b, 0, 0, 0, d, 0, 0],
                      [0, 0, 0, 0, a, b, 0, 0, 0, 0, d, 0],
                      [0, 0, 0, 0, 0, a, 0, 0, 0, 0, 0, d],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    Omega = Omega + Omega.T -np.diagflat(np.diag(Omega))

    # get some eigenvectors
    nc = np.ones(12,dtype=int)*100
    # A = 1.0*sample_networks.sample_block_model(Omega,nc)

    n = 2700
    c_bar = 30
    n_levels = 2
    groups_per_level = 3
    snr = 6

    D_actual = GHRGbuild.create2paramGHRG(n, snr, c_bar, n_levels, groups_per_level)
    ptrue, _ = D_actual.get_partition_at_level(-1)  # true partition lowest level
    G = D_actual.generateNetworkExactProb()
    A = D_actual.to_scipy_sparse_matrix(G)
    regularize = False
    LnA, _ = spectral.construct_normalised_Laplacian(A.astype(float),regularize)
    U, S, Vt = scipy.sparse.linalg.svds(LnA,k=kguess)
    index = scipy.argsort(S)
    V = Vt.T
    V = V[:,index]
    V = V[:,::-1]

    plt.figure()
    plt.plot(V[:,:groups_per_level**n_levels])

    angle_compare = np.zeros(kguess)
    std_compare = np.zeros(kguess)
    for ktest in xrange(kguess):
        sangles = np.zeros((ktest+1,num_pert))
        for i in xrange(num_pert):
            Apert = perturbNetworkWithSparseNoise(A)
            LnApert, _ = spectral.construct_normalised_Laplacian(Apert.astype(float),regularize)
            print "Norm difference"
            print scipy.sparse.linalg.norm(LnApert-LnA)
            U, S, Vt2 = scipy.sparse.linalg.svds(LnApert,k=ktest+1)
            Vpert =  Vt2.T
            index = scipy.argsort(S)
            Vpert = Vpert[:,index]
            Vpert = Vpert[:,::-1]
            sangles[:,i], _ = find_subspace_angle_between_ev_bases(V[:,0:ktest+1],Vpert)
        plt.figure()
        plt.plot(Vpert,V[:,:ktest+1],'x')

        print sangles.mean(axis=1)
        print sangles.std(axis=1)
        std_compare[ktest] = sangles.max(axis=1).std()
        angle_compare[ktest] = sangles.max(axis=1).mean()

    plt.figure()
    plt.plot(angle_compare,'x')
    plt.figure()
    plt.plot(std_compare,'x')

def findNumberOfCommunitiesByInvariantSubspace2():
    """Given a matrix M, find the dominant 'low-rank' components by perturbing the matrix slightly and checking, what number of eigenvectors keep spanning the same subspace"""

    num_pert = 10
    kguess = 35

    n = 3**9
    c_bar = 50
    n_levels = 3
    groups_per_level = 3
    snr = 5

    D_actual = GHRGbuild.create2paramGHRG(n, snr, c_bar, n_levels, groups_per_level)
    ptrue, _ = D_actual.get_partition_at_level(-1)  # true partition lowest level
    G = D_actual.generateNetworkExactProb()
    A = D_actual.to_scipy_sparse_matrix(G)

    regularize = True
    LnA, _ = spectral.construct_normalised_Laplacian(A.astype(float),regularize)
    U, S, Vt = scipy.sparse.linalg.svds(LnA,k=kguess)
    index = scipy.argsort(S)
    V = Vt.T
    V = V[:,index]
    V = V[:,::-1]

    rho = A.mean()
    ERgraph = ER(A.shape[0],rho)
    U, S, Vt = scipy.sparse.linalg.svds(LnA,k=kguess)
    index = scipy.argsort(S)
    VEr = Vt.T
    VEr = VEr[:,index]
    VEr = VEr[:,::-1]

    angle_compare = np.zeros((kguess,num_pert))
    sigmas_compare = np.zeros((kguess,num_pert))
    angle_compareER = np.zeros((kguess,num_pert))
    sigmas_compareER = np.zeros((kguess,num_pert))
    for i in xrange(num_pert):
        Apert = perturbNetworkWithSparseNoise(A)
        LnApert, _ = spectral.construct_normalised_Laplacian(Apert.astype(float),regularize)
        print "Norm difference"
        print scipy.sparse.linalg.norm(LnApert-LnA)
        U, S, Vt2 = scipy.sparse.linalg.svds(LnApert,k=kguess)
        Vpert =  Vt2.T
        index = scipy.argsort(S)
        Vpert = Vpert[:,index]
        Vpert = Vpert[:,::-1]
        angle_compare[:,i], sigmas_compare[:,i] = find_subspace_angle_between_ev_bases(V,Vpert)

        #same with ER
        Apert = perturbNetworkWithSparseNoise(ERgraph)
        LnApert, _ = spectral.construct_normalised_Laplacian(Apert.astype(float),regularize)
        print "Norm difference"
        print scipy.sparse.linalg.norm(LnApert-LnA)
        U, S, Vt2 = scipy.sparse.linalg.svds(LnApert,k=kguess)
        VErp = Vt2.T
        index = scipy.argsort(S)
        VErp = VErp[:,index]
        VErp = VErp[:,::-1]
        angle_compareER[:,i], sigmas_compareER[:,i] = find_subspace_angle_between_ev_bases(VEr,VErp)


    plt.figure()
    plt.errorbar(np.arange(1,kguess+1),sigmas_compare.mean(axis=1),yerr=sigmas_compare.std(axis=1))
    plt.plot(np.arange(1,kguess+1),sigmas_compare.min(axis=1))
    plt.figure()
    plt.plot(np.arange(1,kguess),np.diff(sigmas_compare.mean(axis=1)))
    # plt.figure()
    # plt.errorbar(np.arange(1,kguess+1),sigmas_compareER.mean(axis=1),yerr=sigmas_compareER.std(axis=1))

    plt.figure()
    plt.errorbar(np.arange(1,kguess+1),angle_compare.mean(axis=1),yerr=angle_compare.std(axis=1))
    plt.plot(np.arange(1,kguess+1),angle_compare.min(axis=1))
    plt.figure()
    plt.errorbar(np.arange(1,kguess+1),angle_compareER.mean(axis=1),yerr=angle_compareER.std(axis=1))

def perturbNetworkWithSparseNoise(A):
    """ Add sparse Gaussian noise to matrix"""
    snr = 0.2
    rho = A.mean()
    P = sprandsym(A.shape[0],rho)
    normNoise = scipy.sparse.linalg.norm(P)
    normA = scipy.sparse.linalg.norm(A)
    print "NORMS"
    print normNoise, normA
    Ap = A + snr*normA/normNoise*P

    return Ap


import scipy.stats as stats
import scipy.sparse as sparse
def sprandsym(n, density):
    rvs = stats.norm().rvs
    X = sparse.random(n, n, density=density, data_rvs=rvs)
    # X = sparse.random(n, n, density=density)
    upper_X = sparse.triu(X)
    result = upper_X + upper_X.T - sparse.diags(X.diagonal())
    return result

def ER(n, density):
    X = sparse.random(n, n, density=density)
    upper_X = sparse.triu(X)
    result = upper_X + upper_X.T - sparse.diags(X.diagonal())
    result = 1*(result > 0)
    return result

def aggressiveEVModelSelection():
    n = 2700
    c_bar = 30
    n_levels = 2
    groups_per_level = 3
    snr = 5

    D_actual = GHRGbuild.create2paramGHRG(n, snr, c_bar, n_levels, groups_per_level)
    ptrue, _ = D_actual.get_partition_at_level(-1)  # true partition lowest level
    G = D_actual.generateNetworkExactProb()
    A = D_actual.to_scipy_sparse_matrix(G)

    Ln,_ = spectral.construct_normalised_Laplacian(A,True)

    rho = A.mean()
    ERgraph = ER(A.shape[0],rho)
    LnNull,_ = spectral.construct_normalised_Laplacian(ERgraph,True)
    print rho

    Kest_pos = 50
    if Kest_pos > A.shape[0]:
        Kest_pos = A.shape[0]
    ev_BH_pos, evecs_BH_pos = scipy.sparse.linalg.eigsh(Ln,Kest_pos,which='LM')
    ev_BH_neg, evecs_BH_neg = scipy.sparse.linalg.eigsh(LnNull,Kest_pos,which='LM')
    print ev_BH_pos
    print ev_BH_neg

    return evecs_BH_pos, evecs_BH_neg
