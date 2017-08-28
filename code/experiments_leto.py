from __future__ import division
import numpy as np
from GHRGmodel import GHRG
import GHRGmodel
import GHRGbuild
import spectral_algorithms as spectral
#~ import inference
import metrics
from matplotlib import pyplot as plt
#~ import partialpooling as ppool
import model_selection as ppool
import change_points as cp
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence
plt.ion()
import metrics
from random import sample
import sample_networks
import networkx as nx
import scipy.sparse as sparse
from scipy.stats import norm


def test_overlap(snr=0.5,c_bar=5):
    n=2048
    
    n_levels=2
    groups_per_level=2
    D=GHRGbuild.create2paramGHRG(n,snr,c_bar,n_levels,groups_per_level)
    pvec = D.get_partition_all()
    
    G=D.generateNetworkExactProb()
    A=D.to_scipy_sparse_matrix(G)
    
    D_inf=GHRGmodel.GHRG()
    D_inf.infer_spectral_partition_flat(A)
    pvec_inf = D_inf.get_partition_all()
    
    return metrics.calculate_level_comparison_matrix(pvec_inf,pvec)
    
########## BOOTSTRAP TEST ################

def boot():
    #params
    n=999
    snr=10
    c_bar=20
    n_levels=3
    groups_per_level=3
    
    #generate
    D_actual=GHRGbuild.create2paramGHRG(n,snr,c_bar,n_levels,groups_per_level)
    G=D_actual.generateNetworkExactProb()
    A=D_actual.to_scipy_sparse_matrix(G)
    
    D=GHRGmodel.GHRG()
    D.infer_spectral_partition_hier(A,thresh_method='bootstrap')
    pvec = D.get_partition_all()
    print [len(np.unique(p)) for p in pvec]
    return pvec



########## ZOOM TEST ################

def zoom_exp(n_cliques=64, noise=1e-5):
    A=construct_cliques(n_cliques=n_cliques, clique_size=10,noise=noise)
    D=GHRGmodel.GHRG()
    D.infer_spectral_partition_hier(A)
    pvec = D.get_partition_all()
    print [len(np.unique(p)) for p in pvec]
    return pvec


########## RESOLUTION TEST ################

def clique_test(n_cliques=64, clique_size=10,noise=0.01,A=None,K_known=False,regularizer='BHa'):
    if A is None:
        A = construct_cliques(n_cliques=64, clique_size=10,noise=0.01)
    print "infer"
    D=GHRGmodel.GHRG()
    if K_known:
        D.infer_spectral_partition_flat(A,num_groups=n_cliques, regularizer=regularizer)
    else:
        D.infer_spectral_partition_flat(A)
    print D.nodes()
    return D

def construct_cliques(n_cliques=64, clique_size=10,noise=0.01):
    np.random.seed(np.random.randint(2**31))
    
    block=sparse.coo_matrix(np.ones((clique_size,clique_size))-np.diag(np.ones(clique_size)))
    blocks=[]
    print "construct"
    for nc in xrange(n_cliques):
        blocks.append(block)
    
    A=sparse.block_diag(blocks).astype('bool')
    noise_matrix=sparse.random(n_cliques*clique_size,n_cliques*clique_size,noise/2).astype('bool')
    noise_matrix=(noise_matrix+noise_matrix.T).astype('bool')
    print noise, noise_matrix.sum(), "noise elements added", (n_cliques*clique_size)**2
    A_noisy=((A+noise_matrix)-(A*noise_matrix)).astype('float')
    
    rvs = norm(0, 0.01).rvs
    
    nnoise=sparse.random(n_cliques*clique_size,n_cliques*clique_size,5e-3,data_rvs=rvs)
    print nnoise.sum(), nnoise.astype('bool').sum(), "nnoise elements added", (n_cliques*clique_size)**2
    A_noisy=A_noisy+nnoise
    return A_noisy

def noise_levels(n_cliques):
    min_noise = {128:-5.5,64:-5,32:-4.5,16:-4}[n_cliques]
    return 10**np.arange(min_noise,-.5,0.1)

def clique_test_batch(n_cliques=64, regularizer='BHa'):
    
    runs=50
    
    clique_size=10
    file='out/resolution%i.txt' % n_cliques
    
    for i in xrange(runs):
        for ni,noise in enumerate(noise_levels(n_cliques)):
        
            try:
                A=construct_cliques(n_cliques, clique_size, noise)
                D=clique_test(n_cliques=64, clique_size=10,noise=0.01,A=A,K_known=False, regularizer=regularizer)
                
            except:
                A=construct_cliques(n_cliques, clique_size, noise)
                D=clique_test(n_cliques=64, clique_size=10,noise=0.01,A=A,K_known=False, regularizer=regularizer)
                
            with open(file,'a') as f:
                f.write('%i ' % (len(D.nodes())-1))
        with open(file,'a') as f:
            f.write('\n')

def plot_res(n_cliques=64, regularizer=None):
    if regularizer is None:
        file='out/resolution%i.txt' % n_cliques
    else:
        file='out/resolution_BHm.txt'
    
    with open(file) as f:
        results = np.float64([row.strip().split() for row in f.readlines()[:-1]])
    print len(results)
    results = np.mean(results,0)
    plt.figure()
    plt.semilogx(noise_levels(n_cliques),results,lw=2)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.xlabel('noise',size=24)
    plt.ylabel('number groups detected',size=24)
    plt.axhline(n_cliques)
    plt.tight_layout()

##########################################


"""
Test overlap precision and recall
"""
def test_pr1():
    reps=20
    n=800
    
    true_pvecs=np.zeros((2,n))
    true_pvecs[0,n/2:]=1
    for i in xrange(8):
        true_pvecs[1,(i*100):(i+1)*(100)]=i
    
    precision=np.zeros(40)
    recall=np.zeros(40)
    for rep in xrange(reps):
        for li,l in enumerate(xrange(0,n,20)):
            print "l",l
            pvecs=true_pvecs.copy()
            inds=sample(xrange(n),l)
            _,new_values=np.random.multinomial(1,np.ones(8)/8,size=l).nonzero()
            pvecs[1,inds]=new_values
            pvecs[0,inds]=np.int32(new_values>3)
        
            sm=metrics.calculate_level_comparison_matrix(pvecs,true_pvecs)
            p,r=metrics.calculate_precision_recall(sm)
            precision[li]+=p/reps
            recall[li]+=r/reps
    
    plt.figure()
    plt.plot(np.arange(0,n,20),precision,'b-.')
    plt.plot(np.arange(0,n,20),recall,'r--')
    

def test_pr2():
    reps=20
    n=800
    
    pred_pvecs=np.zeros((2,n))
    pred_pvecs[0,n/2:]=1
    for i in xrange(8):
        pred_pvecs[1,(i*100):(i+1)*(100)]=i
        
    true_pvecs=np.zeros((4,n))
    true_pvecs[0,n/2:]=1
    for i in xrange(4):
        true_pvecs[1,(i*200):(i+1)*(200)]=i
    for i in xrange(8):
        true_pvecs[2,(i*100):(i+1)*(100)]=i
    for i in xrange(16):
        true_pvecs[3,(i*50):(i+1)*(50)]=i
    
    
    precision=np.zeros(40)
    recall=np.zeros(40)
    for rep in xrange(reps):
        for li,l in enumerate(xrange(0,n,20)):
            print "l",l
            pvecs=pred_pvecs.copy()
            inds=sample(xrange(n),l)
            _,new_values=np.random.multinomial(1,np.ones(8)/8,size=l).nonzero()
            pvecs[1,inds]=new_values
            pvecs[0,inds]=new_values//4
            
            sm=metrics.calculate_level_comparison_matrix(pvecs,true_pvecs)
            p,r=metrics.calculate_precision_recall(sm)
            precision[li]+=p/reps
            recall[li]+=r/reps
    
    plt.figure()
    plt.plot(np.arange(0,n,20),precision,'b-.')
    plt.plot(np.arange(0,n,20),recall,'r--')
    
    precision=np.zeros(40)
    recall=np.zeros(40)
    for rep in xrange(reps):
        for li,l in enumerate(xrange(0,n,20)):
            print "l",l
            pvecs=true_pvecs.copy()
            inds=sample(xrange(n),l)
            _,new_values=np.random.multinomial(1,np.ones(16)/16,size=l).nonzero()
            pvecs[3,inds]=new_values
            pvecs[2,inds]=new_values//2
            pvecs[1,inds]=new_values//4
            pvecs[0,inds]=new_values//8
        
            sm=metrics.calculate_level_comparison_matrix(pvecs,pred_pvecs)
            p,r=metrics.calculate_precision_recall(sm)
            precision[li]+=p/reps
            recall[li]+=r/reps
    
    plt.figure()
    plt.plot(np.arange(0,n,20),precision,'b-.')
    plt.plot(np.arange(0,n,20),recall,'r--')




"""
Test change point detection
"""
def test_cp(snr_before = 1, snr_after = 1, n_groups=2):
    
    # mean degree and number of nodes etc.
    n=1000
    n_levels = 1
    K=n_groups**n_levels
    ratio_before = 0.5
    ratio_after = 1
    #~ snr_before = 1
    #~ snr_after = 1
    
    #before change model
    print "before change model"
    D1=create2paramGHRG(n,snr_before,ratio_before,n_levels,n_groups)
    #after change model
    print "after change model"
    D2=create2paramGHRG(n,snr_after,ratio_after,n_levels,n_groups)
    
    
    #sliding window
    w=4
    #degree
    cm=20
    #before change model
    #~ D1=create2paramGHRG(100,cm,0.5,1,2)
    #~ D1=create2paramGHRG(100,cm,1,1,2)
    #after change model
    #~ D2=create2paramGHRG(100,cm,1,1,2)
    
    #create sequence of graphs
    Gs=[D1.generateNetworkExactProb() for i in xrange(w+1)]
    Gs.extend([D2.generateNetworkExactProb() for i in xrange(w+1)])
    
    print [len(G.edges()) for G in Gs]
    
    return cp.detectChanges_flat(Gs,w)
    
    
def runEnron(w=4):
    #get networks
    print "Constructing networks..."
    path_to_data = '../../../Dropbox/Projects/data/enron_min/'
    with open(path_to_data + 'filelist.txt') as f:
        netFiles=[file.strip() for file in f.readlines()]
    
    Gs=[]
    
    for netFile in netFiles:
        G=nx.Graph()
        G.add_nodes_from(range(151))
        
        with open(path_to_data + netFile) as f:
            edgeList=np.int32([row.strip().split() for row in f.readlines()])
        
        G.add_edges_from(edgeList)
        
        Gs.append(G)
    
    return cp.detectChanges_flat(Gs,w)



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
        cin, cout = sample_networks.calculateDegreesFromSNR(snr,ratio,groups_per_level)
        print "Hierarchy Level: ", level, '| KS Detectable: ', snr >=1, "| Link Probabilities in / out per block: ", cin/n_this_level,cout/n_this_level
        # Omega is assigned on a block level, i.e. for each level we have one omega array
        # this assumes a perfect hierarchy with equal depth everywhere
        omega[level] = np.ones((groups_per_level,groups_per_level))*cout/n_this_level + np.eye(groups_per_level)*(cin/n_this_level-cout/n_this_level)
        if np.any(omega[level]>=1):
            print "no probability > 1 not allowed"
            raise ValueError("Something wrong")
        n_this_level = n_this_level / float(groups_per_level)
        if np.floor(n_this_level) != n_this_level:
            print "Rounding number of nodes"


    D=GHRG()

    #network_nodes contains an ordered list of the network nodes
    # order is important so that we can efficiently create views at each
    # internal dendrogram node
    D.network_nodes = np.arange(n)
    D.directed = False
    D.self_loops = False

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
