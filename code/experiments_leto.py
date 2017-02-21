from __future__ import division
import numpy as np
from GHRGmodel import GHRG
import spectral_algorithms as spectral
import inference
import metrics
from matplotlib import pyplot as plt
#~ import partialpooling as ppool
import model_selection as ppool
import change_points as cp
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence
plt.ion()



"""
Test change point detection
"""
def test_cp():
    #sliding window
    w=4
    #degree
    cm=20
    #before change model
    D1=create2paramGHRG(100,cm,0.5,1,2)
    #~ D1=create2paramGHRG(100,cm,1,1,2)
    #after change model
    D2=create2paramGHRG(100,cm,1,1,2)
    
    #create sequence of graphs
    Gs=[D1.generateNetworkExactProb() for i in xrange(w+1)]
    Gs.extend([D2.generateNetworkExactProb() for i in xrange(w+1)])
    
    print [len(G.edges()) for G in Gs]
    
    return cp.detectChanges_flat(Gs,w,mode='Lap')
    
    

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

    D_inferred = inf.split_network_by_recursive_spectral_partition(A,mode='Bethe',max_depth=-1,num_groups=-1)
    partitions=np.empty((2,n))
    partitions[0,:] = D_gen.get_lowest_partition()
    partitions[1,:] = D_inferred.get_lowest_partition()
    print "VI", metrics.calcVI(partitions)[0,1]
    K = partitions[1,:].max().astype('int')
    Di_nodes, Di_edges = D_inferred.construct_full_block_params()
    mergeList=ppool.createMergeList(Di_edges.flatten(),Di_nodes.flatten(),K)
    #~ ppool.plotComparison(mcmc)
    #~ ppool.compare(mcmc)
    return D_gen, D_inferred, mergeList


def testModelSelection(max_num_groups=20,ratio=0.1):
    n_levels=3 #number of levels generated in GHRG

    level_k=2 # number of groups at each level
    
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
                        D_gen=create2paramGHRG(n,cm,ratio,n_levels,level_k)
                        G=D_gen.generateNetwork()
                        A = D_gen.to_scipy_sparse_matrix(G)
                        print (n*cm), A.nnz/2
                #~ looxv=inference.infer_spectral_blockmodel(A, max_num_groups=max_num_groups)/(n*cm)
                        looxv=2*inference.infer_spectral_blockmodel(A, max_num_groups=max_num_groups)/(A.nnz)
                        failed=False
                    except ArpackNoConvergence:
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
                try:
                    below05=((looxv[1:]-looxv[:-1])<0.05).nonzero()[0][0]+2
                except IndexError:
                    below05=20
                print (looxv[7]-looxv[6]), (looxv[8]-looxv[7]),(belowzero>7)
                
                with open('out/res_tms%sm.txt' % (str(ratio).replace('.','')),'a') as f:
                    f.write('%i\t%i\t%f\t%f\t%i\t%i\n' % (n,cm,(looxv[7]-looxv[6]), (looxv[8]-looxv[7]),(belowzero>7), (below05-1) )) 
    

def plot_tms():
    #~ with open('out/res_tms01c.txt') as f:
        #~ results=np.float64([row.strip().split() for row in f.readlines()])
    #~ plt.figure()
    #~ plt.plot(np.arange(len(results)),results[:,2])
    #~ plt.plot(np.arange(len(results)),results[:,3])
    
    with open('out/res_tms01m.txt') as f:
        results=np.float64([row.strip().split() for row in f.readlines()])
    plt.figure()
    plt.plot(np.arange(len(results)),results[:,2])
    plt.plot(np.arange(len(results)),results[:,3])
    
    plt.axhline(0.05,color='k')
    plt.axhline(0.0,color='y')
    plt.title('ratio=0.1')
    plt.ylabel('looxv difference')
    plt.xlabel('network index')
    
    with open('out/res_tms03m.txt') as f:
        results=np.float64([row.strip().split() for row in f.readlines()])
    plt.figure()
    plt.plot(np.arange(len(results)),results[:,2])
    plt.plot(np.arange(len(results)),results[:,3])
    plt.title('ratio=0.3')
    plt.ylabel('looxv difference')
    plt.xlabel('network index')

def tms(n=1000,cm=20,max_num_groups=20):
    ratio=0.1
    n_levels=3 #number of levels generated in GHRG

    level_k=2 # number of groups at each level
    
    
     # degree parameter
     #nodes
    #~ failed=True
    #~ attempts=0
    #~ while failed:
        #~ try:
    D_gen=create2paramGHRG(n,cm,ratio,n_levels,level_k)
    G=D_gen.generateNetwork()
    A = D_gen.to_scipy_sparse_matrix(G)
    looxv=inference.infer_spectral_blockmodel(A, max_num_groups=max_num_groups)/float(n)
            #~ failed=False
        #~ except:
            #~ attempts+=1
            #~ print attempts
            
        
    #~ plt.figure()
    #~ plt.plot(np.arange(1,max_num_groups),looxv)
    print looxv
    diff = looxv[1:]-looxv[:-1]
    print diff
    plt.plot(np.arange(2,max_num_groups),diff)
    try:
        belowzero=((looxv[1:]-looxv[:-1])<0).nonzero()[0][0]+2
    except IndexError:
        belowzero=20
    print (looxv[7]-looxv[6]), (looxv[8]-looxv[7]),(belowzero>7)


"""
Experiment: Test Spectral inference algorithm on hierarchical test graph

Create a sequence of test graphs (realizations of a specified hier. random model) and try
to infer the true partition using spectral methods
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

# Still in use somewhere?
# """
# Experiment 2
# """
# def exp2(runs=10):
    # cm=20
    # n=1000
    # n_levels=3
    # level_k=2
    # K=level_k**n_levels

    # ratios=np.arange(0.1,1,0.1)

    # bb_mean=np.zeros(len(ratios))
    # tt_mean=np.zeros(len(ratios))
    # tb_mean=np.zeros(len(ratios))

    # run_count=np.ones(len(ratios))*runs

    # for ri,ratio in enumerate(ratios):

        # for run in xrange(runs):
            # print ratio, run
            # D_gen=create2paramGHRG(n,cm,ratio,n_levels,level_k)
            # G=D_gen.generateNetwork()
            # A = D_gen.to_scipy_sparse_matrix(G)

            # #~ try:
            # D_inferred = spectral.split_network_hierarchical_by_spectral_partition(A,mode='Bethe',num_groups=-1)

            # partitions=np.empty((2,n))
            # partitions[0,:] = D_gen.get_lowest_partition()
            # partitions[1,:] = D_inferred.get_lowest_partition()
            # bb_mean[ri]+=metrics.calcVI(partitions)[0,1]

            # partitions[1,:] = D_inferred.partition_level(0)
            # tb_mean[ri]+= metrics.calcVI(partitions)[0,1]
            # partitions[0,:] = D_gen.partition_level(0)
            # tt_mean[ri]+= metrics.calcVI(partitions)[0,1]
            # #~ except:
                # #~ print 'FAIL'
                # #~ run_count[ri]-=1

    # tt_mean/=run_count
    # tb_mean/=run_count
    # bb_mean/=run_count

    # plt.figure()
    # plt.plot(ratios,bb_mean)
    # plt.plot(ratios,tb_mean)
    # plt.plot(ratios,tt_mean)

    # plt.legend(['low-low','high-low', 'high-high'])

    # return bb_mean, tb_mean, tt_mean

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
