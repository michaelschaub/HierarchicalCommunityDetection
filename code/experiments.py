import numpy as np
from GHRGmodel import GHRG
import spectral_algorithms as spectral

"""
Experiment 1
"""
def exp1():
    cm=20
    n=1000
    n_levels=3
    level_k=2
    K=level_k**n_levels
    
    
    for ratio in [0.1]:
    
        cin,cout=calculateDegrees(cm,ratio,K)
        D_gen=create2paramGHRG(n,cin/n,cout/n,n_levels,level_k)
        G=D_gen.generateNetwork()
        A = D_gen.to_scipy_sparse_matrix(G)
        
        D_inferred = spectral.split_network_by_recursive_spectral_partition(A,mode='Bethe',max_depth=-1,num_groups=-1)
    
    #~ return A
    
    return D_gen, D_inferred
    


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
def create2paramGHRG(n,p_in,p_out,n_levels,level_k):

    #interaction probabilities
    omega = np.ones((level_k,level_k))*p_out + np.eye(level_k)*(p_in-p_out)

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