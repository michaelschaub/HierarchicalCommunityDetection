import numpy as np
from GHRGmodel import GHRG
import spectral_algorithms as spectral
import metrics
from matplotlib import pyplot as plt
import partialpooling as ppool

plt.ion()

"""
Test partial pooling
 - input: ratio - ratio of probabilities between on- and off-diagonals
 
 returns:
 -  D_gen - Dendro for generating example
 - D_inferred - inferred Dendro
 - mergeList - list of triples (pairs of blocks to merge and p-value)
"""
def testpp(ratio=0.1):
    cm=20 # degree parameter
    n=1000 #nodes
    n_levels=3 #number of levels generated in GHRG
    
    level_k=2 # number of groups at each level
    
    D_gen=create2paramGHRG(n,cm,ratio,n_levels,level_k)
    G=D_gen.generateNetwork()
    A = D_gen.to_scipy_sparse_matrix(G)
    
    D_inferred = spectral.split_network_by_recursive_spectral_partition(A,mode='Bethe',max_depth=-1,num_groups=-1)
    partitions=np.empty((2,n))
    partitions[0,:] = D_gen.get_lowest_partition()
    partitions[1,:] = D_inferred.get_lowest_partition()
    print "VI", metrics.calcVI(partitions)[0,1]
    Di_nodes, Di_edges = D_inferred.construct_full_block_params()
    mergeList=ppool.createMergeList(Di_edges.flatten(),Di_nodes.flatten())
    #~ ppool.plotComparison(mcmc)
    #~ ppool.compare(mcmc)
    return D_gen, D_inferred, mergeList

"""
Experiment 1
"""
def exp1(runs=10):
    cm=20
    n=1000
    n_levels=3
    level_k=2
    K=level_k**n_levels

    ratios=np.arange(0.1,1.,0.1)

    bb_mean=np.zeros(len(ratios))
    tt_mean=np.zeros(len(ratios))
    tb_mean=np.zeros(len(ratios))

    run_count=np.ones(len(ratios))*runs

    for ri,ratio in enumerate(ratios):

        for run in xrange(runs):
            print ratio, run
            D_gen=create2paramGHRG(n,cm,ratio,n_levels,level_k)
            G=D_gen.generateNetwork()
            A = D_gen.to_scipy_sparse_matrix(G)

            #~ try:
            D_inferred = spectral.split_network_by_recursive_spectral_partition(A,mode='Bethe',max_depth=-1,num_groups=-1)

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

"""
Experiment 2
"""
def exp2(runs=10):
    cm=20
    n=1000
    n_levels=3
    level_k=2
    K=level_k**n_levels

    ratios=np.arange(0.1,1,0.1)

    bb_mean=np.zeros(len(ratios))
    tt_mean=np.zeros(len(ratios))
    tb_mean=np.zeros(len(ratios))

    run_count=np.ones(len(ratios))*runs

    for ri,ratio in enumerate(ratios):

        for run in xrange(runs):
            print ratio, run
            D_gen=create2paramGHRG(n,cm,ratio,n_levels,level_k)
            G=D_gen.generateNetwork()
            A = D_gen.to_scipy_sparse_matrix(G)

            #~ try:
            D_inferred = spectral.split_network_hierarchical_by_spectral_partition(A,mode='Bethe',num_groups=-1)

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

"""
Calculate in and out block degree parameters for a given mean degree and ratio
parameters:
    cm  : mean degree
    ratio   : cout/cin
    K   : number of groups
    ncin    : number of cin blocks per row
"""
def calculateDegrees(cm,ratio,K,ncin=1.):
    cin = (K*cm) / (ncin+(K-ncin)*ratio)
    cout = cin * ratio
    return cin,cout


"""
Function to create a test GHRG for simulations
parameters:
    n   : number of nodes
    p_in    : within community prob
    p_out   : across community prob
    n_levels    : depth of GHRG
    level_k     : number of groups at each level
"""
def create2paramGHRG(n,cm,ratio,n_levels,level_k):

    #interaction probabilities
    omega={}
    for level in xrange(n_levels):
        cin,cout=calculateDegrees(cm,ratio,level_k)
        print level, 'Detectable:',cin-cout>2*np.sqrt(cm), cin/n,cout/n
        omega[level] = np.ones((level_k,level_k))*cout/n + np.eye(level_k)*(cin/n-cout/n)
        cm=cin

    D=GHRG()

    #network_nodes contains an ordered list of the network nodes
    # order is important so that we can efficiently create views at each
    # internal dendrogram node
    D.network_nodes = np.arange(n)
    #~ D.add_nodes_from(D.network_nodes, leaf=True)

    # create root node and store attribues of graph in it
    # TODO --- len(D) will evaluate to zero here, why write it like this?
    D.root_node = len(D)
    D.add_node(D.root_node, Er=np.zeros((level_k,level_k)), Nr=np.zeros((level_k,level_k)))
    D.node[D.root_node]['nnodes'] = D.network_nodes[:]
    D.node[D.root_node]['n'] = n

    # add root's children
    nodes_this_level = D.add_children(D.root_node, level_k)
    #create local view of network node assignment
    for ci,child in enumerate(nodes_this_level):
        #~ print child, D.predecessors(child), D.node[D.predecessors(child)[0]]['nnodes'][ci*n/level_k:(ci+1)*n/level_k]
        D.node[child]['nnodes'] = D.node[D.root_node]['nnodes'][ci*n/level_k:(ci+1)*n/level_k]
        D.node[child]['n'] = len(D.node[child]['nnodes'])

    #construct dendro breadth first
    for nl in xrange(n_levels-1):
        nodes_last_level=list(nodes_this_level)
        nodes_this_level=[]
        for parent in nodes_last_level:
            children=D.add_children(parent, level_k)
            nodes_this_level.extend(children)

            #create local view of network node assignment
            level_n=len(D.node[parent]['nnodes'])
            for ci,child in enumerate(children):

                #~ print child, D.predecessors(child), level_n, D.node[D.predecessors(child)[0]]['nnodes'][ci*level_n/level_k:(ci+1)*level_n/level_k]
                D.node[child]['nnodes'] = D.node[D.predecessors(child)[0]]['nnodes'][ci*level_n/level_k:(ci+1)*level_n/level_k]
                D.node[child]['n'] = len(D.node[child]['nnodes'])

    D.setLeafNodeOrder()
    D.setParameters(omega)


    return D
