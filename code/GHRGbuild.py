from __future__ import division

import numpy as np

import GHRGmodel


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

def calculateDegreesFromAvDegAndSNR(SNR,av_degree,num_cluster=2):
    # SNR, a= in-weight, b = out-weight
    # SNR = (a-b)^2 / (ka + k(k-1)*b) = (a-b)^2 / [k^2 *av_degree]
    # av_degree = a/k + (k-1)*b/k = a-b /k + b
    amb = num_cluster * np.sqrt(av_degree*SNR)
    b = av_degree - amb/float(num_cluster)
    a = amb + b

    return a, b




def create2paramGHRG(n,snr,c_bar,n_levels,groups_per_level):
    """
    Function to create a test GHRG for simulations
    parameters:
        n   : number of nodes
        snr : signal to noise ratio (1 represents theoretical detectability threshold)
        c_bar : average degree
        n_levels    : depth of GHRG
        groups_per_level     : number of groups at each level
    """

    #interaction probabilities
    omega={}
    n_this_level = n
    for level in xrange(n_levels):
        # cin, cout = calculateDegrees(cm,ratio,groups_per_level)
        #~ cin, cout = sample_networks.calculateDegreesFromSNR(snr,ratio,groups_per_level)
        cin, cout = calculateDegreesFromAvDegAndSNR(snr,c_bar,groups_per_level)
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
        c_bar=(cin/n_this_level)*(n_this_level / float(groups_per_level))


    D=GHRGmodel.GHRG()

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
        D.node[child]['nnodes'] = D.node[D.root_node]['nnodes'][int(ci*n/groups_per_level):int((ci+1)*n/groups_per_level)]
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
                D.node[child]['nnodes'] = D.node[D.predecessors(child)[0]]['nnodes'][int(ci*level_n/groups_per_level):int((ci+1)*level_n/groups_per_level)]
                D.node[child]['n'] = len(D.node[child]['nnodes'])

    D.setLeafNodeOrder()
    D.setParameters(omega)

    return D


def createNonSymmetric2paramGHRG(n,snr,c_bar,n_levels,groups_per_level,combine_groups):
    """Create a non-symmetic version of a hierarchical model.
       Do this by first creating a balanced hierarchical model and then aggregating and
       averaging the groups defined in the vector 'combine_groups' which refers to the
       groups at the lowest level.

       This should (approximately) preserve the detectability of the model as specified
       via the snr in the orginal symmetric and homogeneous model.

       Example:
       Consider a hier model with 3x3 groups, and combine_groups=[[0,1,2],[3,4]].
       This would result in a model in which the first group is now 'flat', and within
       the second group with heterogeneous group sizes.
    """
    D = create2paramGHRG(n,snr,c_bar,n_levels,groups_per_level,combine_groups)

