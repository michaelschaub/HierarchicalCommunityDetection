from __future__ import division
import networkx as nx
import numpy as np
import scipy.sparse
from itertools import izip
import spectral_algorithms as spectral


"""
GHRG base class is a networkx DiGraph that stores a dendrogram of the hierarchical model.
The dendrogram is a directed tree with edges pointing outward from the root towards the leafs.

The length to the leave nodes is always kept the same, such that every level of the
dendrogram can be interpreted as a partition. In other words, if the hiearchy does not
have the same depth in each (sub)-groups, we simply repeat nodes that do not split at the
next level.

In the following description the Dendrogram graph is denoted by D.
The hierarchical graph it encodes is denoted by G.


The dengrogram D has the following properties

D.network_nodes -- list of all nodes in G that are desribed by the dendrogram
D.root_node     -- the root node of the dendrogram
D.directed      -- is the underlying graph G directed? (True / False)
D.self_loops    -- self-loops allowed in G? (True / False)
D.leaf_nodes    -- the leaf nodes of the dendrogram


Each node in the GHRG dendrogram graph D has the following fields.

'children' -- out-neighbours of the nodes in D (corresponding to sub-groupings in G)
'n'        -- the number of nodes in G that dendrogram node corresponds to
'nnodes'   -- the Ids of the nodes in G that the dendrogram node corresponds to.
              (the total number of Ids should correspond to 'n')

'Nr'       -- len(children) x len(children) sized matrix encoding the maximal possible
              number of links between the nodes in G corresponding to the children
'Er'       -- len(children) x len(children) sized matrix encoding the number of links between
              the nodes in G corresponding to the children


REMARKS
Note that depending on whether D.directed is true or not the matrices Nr and Er will be upper
triangular and include #undirected links or #directed links

If the node is not a leaf node in D, then the diagonal entry in Nr / Er should be 0 by convention; since the precise details of the connectivity pattern are to be read out at a lower level

For the moment support for directed networks is not completely/consistently implemented.

"""

class GHRG(nx.DiGraph):

    ##
    ## FIRST PART -- MANIPULATIONS OF DENDROGRAM DATA STRUCTURE
    ##

    def new_node_generator(self):
        """
        Generator for new node numbers
        NB: Assumes node label 'v' indicates that the node was the vth node added
        """
        v=len(self)
        while True:
            yield v
            v += 1

    def add_children(self, parent, n_children):
        """
        Adds n_children to parent
        Returns list of new nodes
        """
        new_nodes=[]
        new_labels=self.new_node_generator()
        for nc in xrange(n_children):

            # generate new node
            new_node = new_labels.next()
            self.add_node(new_node, Er=np.zeros((n_children,n_children)), Nr=np.zeros((n_children,n_children)))
            new_nodes.append(new_node)

            #generate new edge
            self.add_edge(parent, new_node)

        return new_nodes

    def setParameters(self, omega):
        """
        Function to update the N_r and E_r parameters using a fixed probability matrix, omega
        """
        # print "OMEGA"
        # print omega
        for node in self.nodes_iter():
            level=len(nx.shortest_path(self,self.root_node,node))-1
            #create ordered list of child nodes
            self.node[node]['children']=self.successors(node)
            self.node[node]['children'].sort()
            if not self.node[node].has_key('level'):
                self.node[node]['level'] = level

            #~ print "LEVEL, # children\n", level, node, len(self.node[node]['children']), node in self.leaf_nodes

            if self.directed:

                for ci,childi in enumerate(self.node[node]['children']):
                    for cj,childj in enumerate(self.node[node]['children']):
                        if ci != cj:
                            self.node[node]['Nr'][ci,cj] = self.node[childi]['n']*self.node[childj]['n']
                            self.node[node]['Er'][ci,cj] = self.node[node]['Nr'][ci,cj] * omega[self.node[node]['level']][ci,cj]

                if node in self.leaf_nodes:
                    #~ print ci,cj
                    self.node[node]['Nr'] = np.array([[self.node[node]['n']*self.node[node]['n']]])
                    self.node[node]['Er'] = np.array([[self.node[node]['Nr'][0,0] * omega[self.node[node]['level']-1][ci,ci]]])

            else:

                for ci,childi in enumerate(self.node[node]['children']):
                    for cj,childj in enumerate(self.node[node]['children'][ci:],ci):
                        if ci != cj:
                            self.node[node]['Nr'][ci,cj] = self.node[childi]['n']*self.node[childj]['n']
                            self.node[node]['Er'][ci,cj] = self.node[node]['Nr'][ci,cj] * omega[self.node[node]['level']][ci,cj]

                if node in self.leaf_nodes:
                    #~ print ci,cj
                    self.node[node]['Nr'] = (np.array([[self.node[node]['n']*self.node[node]['n']]])- self.node[node]['n'])/2
                    self.node[node]['Er'] = np.array([[self.node[node]['Nr'][0,0] * omega[self.node[node]['level']-1][ci,ci]]])

                    #~ parent = self.node[node]['ancestor']
                    #~ if len(self.node[parent]['children'])==1:
                        #~ print "YES"

                if len(self.node[node]['children'])==1:
                    self.node[self.node[node]['children'][0]]['level'] = self.node[node]['level']
                    #~ print "children",self.node[node]['children']
                #~ print self.node[node]['level']


    def setLeafNodeOrder(self):
        """
        Function to identify and store leaf nodes (i.e. dendro nodes with no children) in
        internal data structure
        """
        leafs = sorted([v for v in nx.dfs_preorder_nodes(self, self.root_node) if self.out_degree(v) == 0])
        self.leaf_nodes  = leafs

    ##
    ## PART 2 --- OUTPUT INFORMATION STORED IN DENDROGRAM
    ##

    def print_nodes(self,keys=['Er','Nr']):
        for node in self.nodes_iter():
            print node
            for key in keys:
                print key, self.node[node][key]

    def get_partition_at_level(self,level):
        """Return the partition at a specified level of the dendrogram

            level == 1 corresponds to coarsest (non-trivial) partition detected
            level == -1 corresponds to finest (non-trival) partition detected

        """

        # global partition at root node
        part_vector=np.zeros(self.node[self.root_node]['n'],dtype=int)

        nr_levels = nx.dag_longest_path_length(self)
        if level > nr_levels:
            raise ValueError("Level specified does not exist")
        elif level == -1:
            level = nr_levels

        current_parent_nodes = [self.root_node]
        for current_level in range(level):
            current_level_dendro_nodes = []
            for parent in current_parent_nodes:
                if self.node[parent]['children'] != []:
                    current_level_dendro_nodes = current_level_dendro_nodes + self.successors(parent)
                else:
                    current_level_dendro_nodes = current_level_dendro_nodes + [parent]
            current_parent_nodes = sorted(current_level_dendro_nodes)
            # print "2 Current level nodes", current_level_dendro_nodes, "\n"

        for ni, node in enumerate(current_parent_nodes):
            children=self.node[node]['nnodes']
            part_vector[children]=ni

        return part_vector, current_parent_nodes

    def get_number_of_levels(self):
        nr_levels = nx.dag_longest_path_length(self)
        return nr_levels

    def get_partition_all(self):
        """ Return list with all partitions in the dendrogram"""
        pvecs= []
        num_levels = self.get_number_of_levels()
        for ii in xrange(1,num_levels+1):
            partition_true, _ = self.get_partition_at_level(ii)
            pvecs.append(partition_true)

        return pvecs


    def to_scipy_sparse_matrix(self,G):
        """ Output graph as sparse matrix"""
        return nx.to_scipy_sparse_matrix(G)

    def construct_full_block_params(self):
        leafs = self.leaf_nodes
        leafs_to_ids = {}
        for index, leaf in enumerate(leafs):
            leafs_to_ids[leaf] = index
        omega_size = len(leafs)
        omega = np.zeros((omega_size,omega_size))

        # fill diagonal
        for ii, v in enumerate(self.leaf_nodes):
            Nr=self.node[v]['Nr']
            Er=self.node[v]['Er']
            omega[ii,ii] = Er/Nr
            
        for v in self.nodes_iter():
            if v in self.leaf_nodes:
                continue
            else:
                # we symmetrize as only upper triangular part is stored
                Nr=self.node[v]['Nr']
                Nr = Nr + Nr.T
                Er=self.node[v]['Er']
                Er = Er + Er.T
                # connect all children of branch i with all children of branch j with probability p_ij as stored in node
                for ii, childi in enumerate(self.node[v]['children']):
                    child_i_allchildren = [u for u in nx.dfs_preorder_nodes(self, childi) if self.out_degree(u) == 0]
                    for jj, childj in enumerate(self.node[v]['children']):
                        if ii == jj:
                            continue
                        else:
                            child_j_allchildren = [u for u in nx.dfs_preorder_nodes(self, childj) if self.out_degree(u) == 0]
                        seti = [leafs_to_ids[ci] for ci in child_i_allchildren]
                        setj = [leafs_to_ids[cj] for cj in child_j_allchildren]
                        omega[np.ix_(seti,setj)] = Er[ii,jj]/Nr[ii,jj]
        return omega

    ##
    ## PART 3 --- SAMPLING FUNCTIONS

    def generateNetworkBeta(self,mode='Undirected'):
        """
        Function to generate networks from the model using beta prior
        """
        return self.generateNetwork(edgeProb='beta')

    def generateNetworkExactProb(self,mode='Undirected'):
        """
        Function to generate networks from the model using exact probabilities
        """
        return self.generateNetwork(edgeProb='exact')

    def generateNetwork(self,edgeProb='beta'):
        """
        Network nodes at each leaf of the dendro are equivalent.  For each leaf work out the
        probability of connection with other blocks by working up to the root of the tree.
        """
        if self.directed:
            raise KeyError('directed case not defined yet')
        else:
            # create new graph and make sure that all nodes are added
            # even though the graph might be disconnected
            G=nx.Graph()
            G.add_nodes_from(np.arange(self.node[0]['n']))


        # cycle through nodes and generate edges
        for v in self.nodes_iter():

            children=self.node[v]['children']
            Nr=self.node[v]['Nr']
            Er=self.node[v]['Er']

            for ci,cj in izip(*Nr.nonzero()):
                try:
                    childi=self.node[children[ci]]
                    childj=self.node[children[cj]]
                # if it is a leaf node
                except IndexError:
                    childi=self.node[v]
                    childj=self.node[v]
                try:
                    if edgeProb=='beta':
                        alpha=np.ones(Nr[ci,cj])+Er[ci,cj]
                        beta=np.ones(Nr[ci,cj])+(Nr[ci,cj]-Er[ci,cj])
                        p = np.random.beta(alpha,beta)
                    elif edgeProb=='exact':
                        p = Er[ci,cj]/Nr[ci,cj]
                    else :
                        raise ValueError('edge probabilities undefined')
                except ValueError:
                    print "Something went wrong when sampling from the model"

                if ci == cj and len(children) != 0:
                    raise AttributeError("Only leave nodes should have a nonzero diagonal specified")

                # create edges to insert
                edges = (np.random.rand(childi['n']*childj['n'])< p).reshape((childi['n'],childj['n']))
                if ci == cj and not self.directed:
                    edges = np.triu(edges)
                edges = edges.nonzero()

                G.add_edges_from(zip(childi['nnodes'][edges[0]],childj['nnodes'][edges[1]]))

        #remove self loops
        if not self.self_loops:
            G.remove_edges_from(G.selfloop_edges())

        return G