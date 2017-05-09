from GHRGmodel import GHRG
import spectral_algorithms as spectral
import model_selection
import numpy as np
import scipy
from itertools import izip

def split_network_spectral_partition(A, mode='Bethe', num_groups=-1):
    """ Recursively split graph into pieces by employing a spectral clustering strategy.

    Inputs: A          -- input adjacency matrix
            mode       -- variant of spectral clustering to use (reg. Laplacian, Bethe Hessian, Non-Backtracking)
            num_groups -- in how many groups do we want to split the graph at each step
                          (default: 2; set to -1 to infer number of groups from spectrum)

            Output: networkx dendrogram
    """
    # TODO 5: demo function for leto

    # The function consists of mainly two parts
    # A) call spectral partition algorithm
    # B) assemble the output into the corresponding DHRG data structure

    ##########
    # PART (A)
    nr_nodes = A.shape[0]
    partition = spectral.spectral_partition(A, mode=mode, num_groups=num_groups)


    ##########
    # PART (B)
    # initialise networkx output dendrogram, and store some things as properties of the graph
    # create root node and assign properties
    Dendro = GHRG()
    Dendro.network_nodes = np.arange(nr_nodes)
    Dendro.directed = False
    Dendro.root_node = 0

    Emat, Nmat = spectral.compute_number_links_between_groups(A,partition,directed=True)
    # print "Emat, Nmat computed directed"
    # print Emat,"\n", Nmat

    Emat, Nmat = spectral.compute_number_links_between_groups(A,partition,directed=False)
    # print "Emat, Nmat computed undirected"
    # print Emat, "\n", Nmat,"\n\n\n"

    Er_wod = Emat - np.diag(np.diag(Emat))
    Nr_wod = Nmat - np.diag(np.diag(Nmat))

    Dendro.add_node(Dendro.root_node, Er=Er_wod, Nr=Nr_wod)
    Dendro.node[Dendro.root_node]['nnodes'] = Dendro.network_nodes
    Dendro.node[Dendro.root_node]['n'] = nr_nodes

    nr_groups = partition.max()+1
    nodes_next_level = Dendro.add_children(Dendro.root_node, nr_groups)
    Dendro.node[Dendro.root_node]['children'] = nodes_next_level

    for i, n in enumerate(nodes_next_level):
        subpart = partition == i
        Dendro.node[n]['nnodes'] = subpart.nonzero()[0]
        Dendro.node[n]['n'] = len(subpart.nonzero()[0])
        Dendro.node[n]['children'] = []
        Dendro.node[n]['Er'] = Emat[i,i]
        Dendro.node[n]['Nr'] = Nmat[i,i]

    return Dendro

#TODO: this function should be adjusted/updated -- May 9
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
    Emat, Nmat = spectral.compute_number_links_between_groups(A,current_partition)
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
        Dendro.node[node]['children'] = []

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


            # cluster subgraph recursively
            partition = spectral.spectral_partition(Asub, mode=mode, num_groups=num_groups)

            Emat, Nmat = spectral.compute_number_links_between_groups(Asub,partition)
            Dendro.node[node]['Er'] = Emat
            Dendro.node[node]['Nr'] = Nmat
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

        E_rs, N_rs = spectral.compute_number_links_between_groups(A,partition)

        looxv[k-1] = sum(model_selection.looxv(Er,Nr) for Er, Nr in izip(E_rs.flatten(),N_rs.flatten()))
    return looxv
