from GHRGmodel import GHRG
import spectral_algorithms as spectral
import model_selection
import numpy as np
from itertools import izip


def split_network_by_recursive_spectral_partition(A, mode='Lap', num_groups=2, max_depth=3):
    """ Recursively split graph into pieces by employing a spectral clustering strategy.

    Inputs: A          -- input adjacency matrix
            mode       -- variant of spectral clustering to use (reg. Laplacian, Bethe Hessian, Non-Backtracking)
            num_groups -- in how many groups do we want to split the graph at each step
                          (default: 2; set to -1 to infer number of groups from spectrum)
            max_depth  -- how many times do we want to recursively split the graph (default:3)
                          Set to -1 for partitioning graph completely


            Output: networkx dendrogram
    """

    nr_nodes = A.shape[0]
    current_partition = spectral.spectral_partition(A, mode=mode, num_groups=num_groups)

    # initialise networkx output dendrogram, and store some things as properties of the graph
    Dendro = GHRG()
    Dendro.network_nodes = np.arange(nr_nodes)
    Dendro.root_node = 0

    # create root node and assign properties
    Emat, Nmat = compute_number_links_between_groups(A,current_partition)
    Dendro.add_node(Dendro.root_node, Er=Emat, Nr=Nmat)
    # names of nodes corresponding to node in Dendrogram
    Dendro.node[Dendro.root_node]['nnodes'] = Dendro.network_nodes
    # number of nodes corresponding to this node
    Dendro.node[Dendro.root_node]['n'] = nr_nodes

    nr_groups = current_partition.max()+1
    nodes_next_level = Dendro.add_children(Dendro.root_node, nr_groups)
    Dendro.node[Dendro.root_node]['children'] = nodes_next_level
    for i, n in enumerate(nodes_next_level):
        subpart = current_partition == i
        Dendro.node[n]['nnodes'] = subpart.nonzero()[0]
        Dendro.node[n]['n'] = len(subpart.nonzero()[0])

    hier_depth = 0
    print "\nNow running recursion"

    # as long as we have not reached the max_depth yet,
    # and there is more than one group in the partition
    while (hier_depth < max_depth or max_depth == -1) and len(nodes_next_level):

        #~ print "\nLEVEL"
        #~ print nodes_next_level
        next_level_temp = []

        for node in nodes_next_level:
            # print "\nNODE"
            # print node

            # create subgraphs
            subpart = Dendro.node[node]['nnodes']
            Asub = A[subpart,:]
            Asub = Asub[:,subpart]
            # print Asub
            # print "SUBPART"
            # print subpart


            # cluster subgraph recursively
            partition = spectral_partition(Asub, mode=mode, num_groups=num_groups)
            # print "PARTITION"
            # print partition

            Emat, Nmat = compute_number_links_between_groups(Asub,partition)
            Dendro.node[node]['Er'] = Emat
            Dendro.node[node]['Nr'] = Nmat
            # print "EMAT"
            # print Emat
            # print Dendro.node[0]['Er']
            nr_groups = np.unique(partition).size

            # print "NRG"
            # print nr_groups
            if nr_groups > 1:
                children  = Dendro.add_children(node,nr_groups)
                Dendro.node[node]['children'] = children
                next_level_temp.extend(children)
                parent_nnodes = Dendro.node[node]['nnodes']
                for i, n in enumerate(children):
                    subpart = partition == i
                    Dendro.node[n]['nnodes'] = parent_nnodes[subpart.nonzero()[0]]
                    Dendro.node[n]['n'] = len(subpart.nonzero()[0])
            else:
                Dendro.node[node]['children'] = []


        nodes_next_level = next_level_temp
        hier_depth +=1

    return Dendro



def infer_spectral_blockmodel(A, mode='Lap', max_num_groups=None):
    if max_num_groups is None:
        max_num_groups=A.shape[0]
    
    looxv=np.empty(max_num_groups-1)
    
    for k in xrange(1,max_num_groups):
        partition = spectral.spectral_partition(A, mode=mode, num_groups=k)
        
        E_rs, N_rs = compute_number_links_between_groups(A,partition)
        
        looxv[k-1] = sum(model_selection.looxv(Er,Nr) for Er, Nr in izip(E_rs.flatten(),N_rs.flatten()))
    return looxv
    

#######################################################
# HELPER FUNCTIONS
#######################################################
def compute_number_links_between_groups(A,partition_vec):
    """
    Compute the number of possible and actual links between the groups indicated in the
    partition vector
    """

    pmatrix = spectral.create_partition_matrix_from_vector(partition_vec)
    # number of columns is number of groups
    nr_groups = pmatrix.shape[1]

    links_between_groups = pmatrix.T.dot(A).dot(pmatrix).toarray()

    nodes_per_group = pmatrix.sum(0).getA()
    possible_links_between_groups = np.outer(nodes_per_group,nodes_per_group)

    return links_between_groups, possible_links_between_groups

# Still in use somewhere???
# def split_network_hierarchical_by_spectral_partition(A, mode='Lap', num_groups=2):
    # """ Recursively split graph into pieces by employing a spectral clustering strategy.

    # Inputs: A          -- input adjacency matrix
            # mode       -- variant of spectral clustering to use (reg. Laplacian, Bethe Hessian, Non-Backtracking)
            # num_groups -- in how many groups do we want to split the graph at each step
                          # (default: 2; set to -1 to infer number of groups from spectrum)

            # Output: networkx dendrogram
    # """

    # nr_nodes = A.shape[0]
    # current_partition = spectral.hier_spectral_partition(A, mode=mode, num_groups=num_groups)

    # # initialise networkx output dendrogram, and store some things as properties of the graph
    # Dendro = GHRG()
    # Dendro.network_nodes = np.arange(nr_nodes)
    # Dendro.root_node = 0

    # # create root node and assign properties
    # Emat, Nmat = compute_number_links_between_groups(A,current_partition)
    # Dendro.add_node(Dendro.root_node, Er=Emat, Nr=Nmat)
    # # names of nodes corresponding to node in Dendrogram
    # Dendro.node[Dendro.root_node]['nnodes'] = Dendro.network_nodes
    # # number of nodes corresponding to this node
    # Dendro.node[Dendro.root_node]['n'] = nr_nodes

    # nr_groups = current_partition.max()+1
    # nodes_next_level = Dendro.add_children(Dendro.root_node, nr_groups)
    # Dendro.node[Dendro.root_node]['children'] = nodes_next_level

    # for i, n in enumerate(nodes_next_level):
        # subpart = current_partition == i
        # Dendro.node[n]['nnodes'] = subpart.nonzero()[0]
        # Dendro.node[n]['n'] = len(subpart.nonzero()[0])

    # for node in nodes_next_level:
            # # create subgraphs
            # subpart = Dendro.node[node]['nnodes']
            # Asub = A[subpart,:]
            # Asub = Asub[:,subpart]

            # partition = np.zeros(Asub.shape[0],dtype='int')
            # Emat, Nmat = compute_number_links_between_groups(Asub,partition)
            # Dendro.node[node]['Er'] = Emat
            # Dendro.node[node]['Nr'] = Nmat
            # Dendro.node[node]['children'] = []

    # return Dendro

#~ """From GHRG - Do we still need this?"""
#~ def fitGHRG(input_network):
    #~ A = input_network.to_scipy_sparse_matrix(input_network)

    #~ D_inferred = spectral.split_network_by_recursive_spectral_partition(A,mode='Bethe',max_depth=-1,num_groups=-1)

    #~ partitions= D_inferred.get_lowest_partition()
    #~ K = partitions.max().astype('int')
    #~ Di_nodes, Di_edges = D_inferred.construct_full_block_params()
    #~ mergeList=ppool.createMergeList(Di_edges.flatten(),Di_nodes.flatten(),K)


    #~ for blocks_to_merge in mergeList:
        #~ D_inferred.insert_hier_merge_node(blocks_to_merge)

    #~ return D_inferred
