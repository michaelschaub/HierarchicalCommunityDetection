from GHRGmodel import GHRG
import spectral_algorithms as spectral
import inference
import metrics
import numpy as np
import scipy
from matplotlib import pyplot as plt

np.set_printoptions(precision=4,linewidth=200)

# Within ipython shell:
# %load_ext autoreload
# %autoreload 2
# %pylab
# import experiments_michael

def leto_experiment():
    # mean degree and number of nodes etc.
    n=1000
    n_levels = 1
    n_groups = 4
    ratio = 0.4

    snr = 2

    # create GHRG object with specified parameters and create a sample network from it
    D_gen=create2paramGHRG(n,snr,ratio,n_levels,n_groups)
    G= D_gen.generateNetworkExactProb()
    A= D_gen.to_scipy_sparse_matrix(G)

    D_inferred = inference.split_network_spectral_partition(A,mode='Bethe',num_groups=n_groups)
    Gnew= D_inferred.generateNetworkExactProb()

    return snr, D_inferred, D_gen


"""
Experiment: Test Spectral inference algorithm on hierarchical test graph

Create a sequence of test graphs (realizations of a specified hier. random model) and try
to infer the true partition using spectral methods
"""
def run_spectral_algorithms_hier(n_levels=2,groups_per_level=4):
    # mean degree number of nodes etc.
    SNR = 9
    n=2**10
    K=groups_per_level**n_levels
    ratio = 0.1

    D_gen=create2paramGHRG(n,SNR,ratio,n_levels,groups_per_level)
    G=D_gen.generateNetworkExactProb()
    A = D_gen.to_scipy_sparse_matrix(G)
    # plt.figure()
    # plt.spy(A,markersize=0.5)

    pvec = spectral.spectral_partition(A,'Bethe',-1)
    # plt.figure()
    # plt.plot(pvec,marker='s')

    partition_true = D_gen.get_lowest_partition()
    partition_high = D_gen.partition_level(0)

    ol_score = metrics.overlap_score(pvec,partition_true)
    print "Partition into "+ str(np.max(pvec)+1) +" groups"
    print "OVERLAP SCORE (SINGLE LAYER INFERENCE): ", ol_score, "\n\n"


    pvecs = spectral.hier_spectral_partition(A)
    print pvecs

    ol_score = metrics.overlap_score(pvecs[0],partition_true)
    print "Partition into "+ str(np.max(pvecs[0])+1) +" groups"
    print "OVERLAP SCORE Finest: ", ol_score, "\n\n"


    ol_score = metrics.overlap_score(pvecs[-1],partition_high)
    print "Partition into "+ str(np.max(pvecs[-1])+1) +" groups"
    print "OVERLAP SCORE Coarsest: ", ol_score, "\n\n"
    return pvecs


def test_spectral_algorithms_non_hier():
    SNR, overlap_Bethe, overlap_Rohe, overlap_Seidel = run_spectral_algorithms_non_hier()
    plot_results_overlap(SNR, overlap_Bethe, overlap_Rohe, overlap_Seidel)

def test_spectral_algorithms_multiple_networks():
    SNR, overlap_Bethe, overlap_Rohe, overlap_Seidel = run_spectral_algorithms_n_networks()
    plot_results_overlap(SNR, overlap_Bethe, overlap_Rohe, overlap_Seidel)

def run_spectral_algorithms_n_networks(n_groups=4):
    # mean degree and number of nodes etc.
    n=1000
    n_levels = 1
    K=n_groups**n_levels
    ratio = 0.4

    SNR = np.arange(0.5,3,0.5)
    nsamples = 20
    overlap_Bethe = np.zeros((SNR.size,nsamples))
    overlap_Rohe = np.zeros((SNR.size,nsamples))
    overlap_Seidel = np.zeros((SNR.size,nsamples))
    nr_network_samples = 5

    for ii, snr in enumerate(SNR):

        # create GHRG object with specified parameters and create a sample network from it
        D_gen=create2paramGHRG(n,snr,ratio,n_levels,n_groups)
        partition_true = D_gen.get_lowest_partition()

        for jj in np.arange(nsamples):
            A = 0;
            for kk in xrange(nr_network_samples):
                G= D_gen.generateNetworkExactProb()
                A1= D_gen.to_scipy_sparse_matrix(G)
                A = A + A1

            A = A/float(nr_network_samples)


            pvec = spectral.spectral_partition(A,'Bethe',n_groups)
            ol_score = metrics.overlap_score(pvec,partition_true)
            overlap_Bethe[ii,jj] = ol_score

            pvec = spectral.spectral_partition(A,'Lap',n_groups)
            ol_score = metrics.overlap_score(pvec,partition_true)
            overlap_Rohe[ii,jj] = ol_score

            pvec = spectral.spectral_partition(A,'SeidelLap',n_groups)
            ol_score = metrics.overlap_score(pvec,partition_true)
            overlap_Seidel[ii,jj] = ol_score

    return SNR, overlap_Bethe, overlap_Rohe, overlap_Seidel

def run_spectral_algorithms_non_hier(n_groups=4):
    # mean degree and number of nodes etc.
    n=2000
    n_levels = 1
    K=n_groups**n_levels
    ratio = 0.4

    SNR = np.arange(0.5,3,0.25)
    nsamples = 20
    overlap_Bethe = np.zeros((SNR.size,nsamples))
    overlap_Rohe = np.zeros((SNR.size,nsamples))
    overlap_Seidel = np.zeros((SNR.size,nsamples))

    for ii, snr in enumerate(SNR):

        # create GHRG object with specified parameters and create a sample network from it
        D_gen=create2paramGHRG(n,snr,ratio,n_levels,n_groups)
        partition_true = D_gen.get_lowest_partition()

        for jj in np.arange(nsamples):
            G= D_gen.generateNetworkExactProb()
            A= D_gen.to_scipy_sparse_matrix(G)


            pvec = spectral.spectral_partition(A,'Bethe',n_groups)
            ol_score = metrics.overlap_score(pvec,partition_true)
            overlap_Bethe[ii,jj] = ol_score

            pvec = spectral.spectral_partition(A,'Lap',n_groups)
            ol_score = metrics.overlap_score(pvec,partition_true)
            overlap_Rohe[ii,jj] = ol_score

            pvec = spectral.spectral_partition(A,'SeidelLap',n_groups)
            ol_score = metrics.overlap_score(pvec,partition_true)
            overlap_Seidel[ii,jj] = ol_score

    return SNR, overlap_Bethe, overlap_Rohe, overlap_Seidel

def plot_results_overlap(SNR,overlap_Bethe,overlap_Rohe,overlap_Seidel):
    plt.figure()
    plt.errorbar(SNR, overlap_Bethe.mean(axis=1), overlap_Bethe.std(axis=1),label="BH")
    plt.errorbar(SNR, overlap_Rohe.mean(axis=1), overlap_Rohe.std(axis=1),label="regL")
    plt.errorbar(SNR, overlap_Seidel.mean(axis=1), overlap_Seidel.std(axis=1),label="SL")
    plt.legend()
    plt.xlabel("SNR")
    plt.ylabel("overlap score")


def calculateDegreesFromSNR(snr,ratio=0.5,num_cluster=2):
    # SNR a= in-weight, b = out-weight
    # SNR = (a-b)^2 / (ka + k(k-1)*b)
    # fix SNR and b =r*a
    # SNR = a^2 *(1-r)^2 / (ka + k(k-1)*ra)
    # SNR = a * (1-r)^2 / (k + k(k-1)*r)
    # a = SNR * (k + k(k-1)*r) / (1-r)^2
    a = snr * (num_cluster + num_cluster*(num_cluster-1)*ratio) / float((1-ratio)**2);
    b = ratio*a;

    return a, b

def expand_partitions_to_full_graph(pvecs):
    pvec_new = []
    pvec_new.append(pvecs[0])

    for i in xrange(len(pvecs)-1):
        pold = pvecs[i]
        pnew = pvecs[i+1]
        partition = pnew[pold]
        pvec_new.append(partition)

    return pvec_new

"""
Function to create a test GHRG for simulations
parameters:
    n   : number of nodes
    n_levels    : depth of GHRG
    groups_per_level     : number of groups at each level
"""
def create2paramGHRG(n,snr,ratio,n_levels,groups_per_level):

    #interaction probabilities
    omega={}
    n_this_level = n
    for level in xrange(n_levels):
        # cin, cout = calculateDegrees(cm,ratio,groups_per_level)
        cin, cout = calculateDegreesFromSNR(snr,ratio,groups_per_level)
        print "Hierarchy Level: ", level, '| KS Detectable: ', snr >=1, "| Link Probabilities in / out per block: ", cin/n_this_level,cout/n_this_level

        # Omega is assigned on a block level, i.e. for each level we have one omega array
        # this assumes a perfect hierarchy with equal depth everywhere
        omega[level] = np.ones((groups_per_level,groups_per_level))*cout/n_this_level + np.eye(groups_per_level)*(cin/n_this_level-cout/n_this_level)
        n_this_level = n_this_level / float(groups_per_level)
        if np.floor(n_this_level) != n_this_level:
            print "Rounding number of nodes"


    D=GHRG()

    #network_nodes contains an ordered list of the network nodes
    # order is important so that we can efficiently create views at each
    # internal dendrogram node
    D.network_nodes = np.arange(n)

    # create root node and store attribues of graph in it
    # this corresponds to an unclustered graph
    D.root_node = 0
    D.add_node(D.root_node, Er=np.zeros((groups_per_level,groups_per_level)), Nr=np.zeros((groups_per_level,groups_per_level)))
    D.node[D.root_node]['nnodes'] = D.network_nodes[:]
    D.node[D.root_node]['n'] = n

    # split network into groups -- add children in dendrogram
    nodes_this_level = D.add_children(D.root_node, groups_per_level)
    for ci, child in enumerate(nodes_this_level):
        D.node[child]['nnodes'] = D.node[D.root_node]['nnodes'][ci*n/groups_per_level:(ci+1)*n/groups_per_level]
        D.node[child]['n'] = len(D.node[child]['nnodes'])

    #construct dendrogram breadth first
    for nl in xrange(n_levels-1):
        nodes_last_level=list(nodes_this_level)
        nodes_this_level=[]
        for parent in nodes_last_level:
            children=D.add_children(parent, groups_per_level)
            nodes_this_level.extend(children)

            #create local view of network node assignment
            level_n=len(D.node[parent]['nnodes'])
            for ci,child in enumerate(children):
                D.node[child]['nnodes'] = D.node[D.predecessors(child)[0]]['nnodes'][ci*level_n/groups_per_level:(ci+1)*level_n/groups_per_level]
                D.node[child]['n'] = len(D.node[child]['nnodes'])

    D.setLeafNodeOrder()
    D.setParameters(omega)


    return D
