from GHRGmodel import GHRG
import spectral_algorithms as spectral
import inference
import metrics
import numpy as np
import scipy
from matplotlib import pyplot as plt

np.set_printoptions(precision=4,linewidth=200)

def testModelSelection(max_num_groups=20):
    ratio=0.1
    n_levels=3 #number of levels generated in GHRG

    groups_per_level=2 # number of groups at each level

    for n in 2**np.arange(8,14):
        for cm in 2**np.arange(2,6):
            for rep in xrange(100):
                print n,cm, rep
                #~ cm=20 # degree parameter
                #~ n=1000 #nodes
                failed=True
                attempts=0
                while failed:
                    try:
                        D_gen=create2paramGHRG(n,cm,ratio,n_levels,groups_per_level)
                        G=D_gen.generateNetwork()
                        A = D_gen.to_scipy_sparse_matrix(G)
                        looxv=inference.infer_spectral_blockmodel(A, max_num_groups=max_num_groups)/float(n)
                        failed=False
                    except:
                        attempts+=1
                        print attempts


                #~ plt.figure()
                #~ plt.plot(np.arange(1,max_num_groups),looxv)
                #~ print looxv
                diff = looxv[1:]-looxv[:-1]
                #~ print diff
                #~ plt.plot(np.arange(2,max_num_groups),diff)
                try:
                    belowzero=((looxv[1:]-looxv[:-1])<0).nonzero()[0][0]+2
                except IndexError:
                    belowzero=20
                print (looxv[7]-looxv[6]), (looxv[8]-looxv[7]),(belowzero>7)

                with open('res_tms01.txt','a') as f:
                    f.write('%i\t%i\t%f\t%f\t%i\n' % (n,cm,(looxv[7]-looxv[6]), (looxv[8]-looxv[7]),(belowzero>7) ))



"""
Experiment: Test Spectral inference algorithm on hierarchical test graph

Create a sequence of test graphs (realizations of a specified hier. random model) and try
to infer the true partition using spectral methods
"""
def exp1(runs=10,n_levels=3,groups_per_level=2):
    cm=20
    n=1000
    K=groups_per_level**n_levels

    ratios=np.arange(0.1,1.,0.1)

    bb_mean=np.zeros(len(ratios))
    tt_mean=np.zeros(len(ratios))
    tb_mean=np.zeros(len(ratios))

    run_count=np.ones(len(ratios))*runs

    for ri,ratio in enumerate(ratios):

        for run in xrange(runs):
            print ratio, run
            D_gen=create2paramGHRG(n,cm,ratio,n_levels,groups_per_level)
            G=D_gen.generateNetwork()
            print G
            A = D_gen.to_scipy_sparse_matrix(G)
            if ~scipy.sparse.isspmatrix(A):
                print "TEST2"

            #~ try:
            D_inferred = inference.split_network_by_recursive_spectral_partition(A,mode='Bethe',max_depth=-1,num_groups=-1)

            partitions=np.empty((2,n))
            partitions[0,:] = D_gen.get_lowest_partition()
            partitions[1,:] = D_inferred.get_lowest_partition()
            bb_mean[ri]+=metrics.calcVI(partitions)[0,1]

            partitions[1,:] = D_inferred.partition_level(0)
            tb_mean[ri]+= metrics.calcVI(partitions)[0,1]
            partitions[0,:] = D_gen.partition_level(0)
            tt_mean[ri]+= metrics.calcVI(partitions)[0,1]
            #~ except:
                #~ print 'FAIL'
                #~ run_count[ri]-=1

    tt_mean/=run_count
    tb_mean/=run_count
    bb_mean/=run_count

    plt.figure()
    plt.plot(ratios,bb_mean)
    plt.plot(ratios,tb_mean)
    plt.plot(ratios,tt_mean)

    plt.legend(['low-low','high-low', 'high-high'])

    return bb_mean, tb_mean, tt_mean

def exp2(n_levels=2,groups_per_level=3):
    # mean degree number of nodes etc.
    cm=80
    n=1000
    K=groups_per_level**n_levels
    ratio = 0.3



    D_gen=create2paramGHRG(n,cm,ratio,n_levels,groups_per_level)
    G=D_gen.generateNetworkExactProb()

    A = D_gen.to_scipy_sparse_matrix(G)
    # plt.spy(A)

    pvec = spectral.spectral_partition(A,'Bethe',-1)
    plt.figure()
    plt.plot(pvec,marker='s')

    links_between_groups, possible_links_between_groups = inference.compute_number_links_between_groups(A,pvec)

    print "Aggregated network:"
    print links_between_groups

    k, pvec2, H, error = spectral.identify_hierarchy_in_affinity_matrix(links_between_groups)
    plt.figure()
    plt.plot(pvec2)

    links_between_groups, possible_links_between_groups = inference.compute_number_links_between_groups(links_between_groups,pvec2)


    k, pvec2, H, error = spectral.identify_hierarchy_in_affinity_matrix(links_between_groups)
    plt.figure()
    plt.plot(pvec2)


    #~ try:
    D_inferred = inference.split_network_by_recursive_spectral_partition(A,mode='Bethe',max_depth=-1,num_groups=-1)

    partitions=np.empty((2,n))
    partitions[0,:] = D_gen.get_lowest_partition()
    partitions[1,:] = D_inferred.get_lowest_partition()

    return D_gen, D_inferred

def test_spectral_algorithms_non_hier(n_groups=4):
    # mean degree and number of nodes etc.
    n=1000
    n_levels = 1
    K=n_groups**n_levels
    ratio = 0.3

    SNR = np.arange(0.5,3,0.5)
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

def plot_results_overlap(SNR,overlap):
    plt.figure()
    plt.errorbar(SNR, overlap.mean(axis=1), overlap.std(axis=1))



"""
Calculate in and out block degree parameters for a given mean degree and ratio
parameters:
    cm  : mean degree
    ratio   : cout/cin
    K   : number of groups
    ncin    : number of cin blocks per row

Note that cin (cout) is not the expected degree inside the block, but rather defined via
cm = ncin *cin / K + (K-ncin)*cout / K
"""
def calculateDegrees(cm,ratio,K,ncin=1.):
    cin = (K*cm) / (ncin+(K-ncin)*ratio)
    cout = cin * ratio
    return cin, cout

def calculateDegreesFromSNR(snr,ratio=0.5,num_cluster=2):
    # SNR a= in-weight, b = out-weight
    # SNR = (a-b)^2 / (ka + k(k-1)*b)
    # fix SNR and b =r*a
    # SNR = a^2 *(1-r)^2 / (ka + k(k-1)*ra)
    # SNR = a * (1-r)^2 / (k + k(k-1)*r)
    # a = SNR * (k + k(k-1)*r) / (1-r)^2
    a = snr * (num_cluster + num_cluster*(num_cluster-1)*ratio) / ((1-ratio)**2);
    b = ratio*a;

    return a, b


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
    for level in xrange(n_levels):
        # cin, cout = calculateDegrees(cm,ratio,groups_per_level)
        cin, cout = calculateDegreesFromSNR(snr,ratio,groups_per_level)
        #TODO
        # This test here is wrong, cin and cout have to be referred to the total network (cm has to come from the total) -- it would be detectable, if we were to zoom inm but not on first shot!
        print "Hierarchy Level: ", level, '| Detectable: ', snr >=1, "| Link Probabilities in / out per block: ", cin/n,cout/n

        # Omega is assigned on a block level, i.e. for each level we have one omega array
        # this assumes a perfect hierarchy with equal depth everywhere
        omega[level] = np.ones((groups_per_level,groups_per_level))*cout/n + np.eye(groups_per_level)*(cin/n-cout/n)
        cm=cin


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
