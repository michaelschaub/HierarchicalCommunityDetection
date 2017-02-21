from __future__ import division
import networkx as nx
import numpy as np
import scipy.sparse
from itertools import izip


"""
GHRG base class is a networkx DiGraph that stores a dendrogram of the hierarchical model.
The dendrogram is a directed tree with edges pointing outward from the root towards the leafs.
"""
class GHRG(nx.DiGraph):

    """
    Function to update the N_r and E_r parameters using a graph G
    """
    def updateParams(self, G):
        pass

    """
    Function to update the N_r and E_r parameters using a fixed probability matrix, omega
    """
    def setParameters(self, omega):
        for node in self.nodes_iter():
            level=len(nx.shortest_path(self,self.root_node,node))-1
            #create ordered list of child nodes
            self.node[node]['children']=self.successors(node)
            self.node[node]['children'].sort()
            for ci,childi in enumerate(self.node[node]['children']):
                for cj,childj in enumerate(self.node[node]['children']):
                    if not (ci==cj): #  or (childi in self.leaf_nodes):
                        #~ print ci,cj,self.node[node]['Nr'], node
                        self.node[node]['Nr'][ci,cj] = self.node[childi]['n']*self.node[childj]['n']
                        self.node[node]['Er'][ci,cj] = self.node[node]['Nr'][ci,cj] * omega[level][ci,cj]
            if node in self.leaf_nodes:
                #~ print ci,cj
                self.node[node]['Nr'] = np.array([[self.node[node]['n']*self.node[node]['n']]])
                self.node[node]['Er'] = np.array([[self.node[node]['Nr'][0,0] * omega[level-1][ci,ci]]])



    """
    Generator for new node numbers
    NB: Assumes node label 'v' indicates that the node was the vth node added
    """
    def new_node_generator(self):
        v=len(self)
        while True:
            yield v
            v += 1

    """
    Adds n_children to parent
    Returns list of new nodes
    """
    def add_children(self, parent, n_children):
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

    """
    Function to merge list of nodes in dendrogram and insert a new node in the hierarchy
    """
    def insert_hier_merge_node(self,node_ids):

        ##################################################
        # 1) preallocate information stored in merged node
        joint_nnodes = np.empty(0)
        joint_n = 0
        n_blocks = len(node_ids)

        # Check that all nodes are mergeable, i.e., have the same parent node
        parent_id = self.predecessors(node_ids[0])[0]
        ids = np.empty(len(node_ids),dtype='int')
        for counter, node in enumerate(node_ids):
            # check consistency of merger
            if self.predecessors(node)[0] != parent_id:
                print "These two nodes / blocks cannot be merged!"
                return False

            # get indices for Nr and Er arrays
            ids[counter] = self.node[parent_id]['children'].index(node)

            joint_nnodes = np.append(joint_nnodes,self.node[node]['nnodes'])
            joint_n = joint_n + self.node[node]['n']

        # read out info from old nodes and create union / joint
        joint_Nr = self.node[parent_id]['Nr'][np.ix_(ids,ids)]
        joint_Er = self.node[parent_id]['Er'][np.ix_(ids,ids)]

        old_children = self.node[parent_id]['children']
        pvec = np.zeros(len(self.node[parent_id]['children']),dtype='int')

        num_new_children = len(old_children) - len(node_ids)
        k=0
        for ni, old_child in enumerate(old_children):
            if old_child in node_ids:
                pvec[ni] = num_new_children
            else:
                pvec[ni] = k
                k = k+1


        ##################################################
        # create new node and insert joint info
        new_label = self.new_node_generator()
        new_id = new_label.next()
        self.add_node(new_id)
        self.node[new_id]['nnodes'] = joint_nnodes
        self.node[new_id]['n'] = joint_n
        self.node[new_id]['Er'] = joint_Er
        self.node[new_id]['Nr'] = joint_Nr

        # let new node point to old two nodes
        self.node[new_id]['children'] = node_ids
        for node in node_ids:
            self.add_edge(new_id,node)


        ##################################################
        # update parent node info and let it point to new node
        for node in node_ids:
            self.remove_edge(parent_id,node)
        self.add_edge(parent_id,new_id)

        self.node[parent_id]['children']=self.successors(parent_id)
        self.node[parent_id]['children'].sort()

        # Check if this is doing the correct agglomeration here.
        pmatrix = create_partition_matrix_from_vector(pvec).toarray()
        A = self.node[parent_id]['Nr']
        A = pmatrix.T.dot(A).dot(pmatrix)
        self.node[parent_id]['Nr'] = A - np.diag(np.diag(A))

        A = self.node[parent_id]['Er']
        A = pmatrix.T.dot(A).dot(pmatrix)
        self.node[parent_id]['Er'] = A - np.diag(np.diag(A))



    """
    Function to identify and store leaf nodes (i.e. dendro nodes with no children) in
    internal data structure
    """
    def setLeafNodeOrder(self):
        self.leaf_nodes = [v for v in nx.dfs_preorder_nodes(self,self.root_node) if self.out_degree(v)==0]

    """
    Function to generate networks from the model
    """
    def generateNetworkBeta(self,mode='Undirected'):
        return self.generateNetwork(edgeProb='beta',directed=(mode=='Directed'))
        #~ """
        #~ Network nodes at each leaf of the dendro are equivalent.  For each leaf work out the
        #~ probability of connection with other blocks by working up to the root of the tree.
        #~ """
        #~ G=nx.Graph()

        #~ #cycle through nodes and generate edges
        #~ for v in self.nodes_iter():
            #~ children=self.node[v]['children']
            #~ Nr=self.node[v]['Nr']
            #~ for ci,cj in izip(*Nr.nonzero()):
                #~ try:
                    #~ childi=self.node[children[ci]]
                    #~ childj=self.node[children[cj]]
                #~ except IndexError:      # if it is a leaf node
                    #~ childi=self.node[v]
                    #~ childj=self.node[v]
                #~ alpha=np.ones(Nr[ci,cj])+self.node[v]['Er'][ci,cj]
                #~ beta=np.ones(Nr[ci,cj])+(Nr[ci,cj]-self.node[v]['Er'][ci,cj])
                #~ try:
                    #~ p = np.random.beta(alpha,beta)
                #~ except ValueError:
                    #~ print alpha, beta, fail
                #~ edges= (np.random.rand(int(Nr[ci,cj])) < p).reshape((childi['n'],childj['n'])).nonzero()
                #~ G.add_edges_from(zip(childi['nnodes'][edges[0]],childj['nnodes'][edges[1]]))

        #~ #remove self loops
        #~ G.remove_edges_from(G.selfloop_edges())
        #~ return G

    """
    Function to generate networks from the model
    """
    def generateNetworkExactProb(self,mode='Undirected'):
        return self.generateNetwork(edgeProb='exact',directed=(mode=='Directed'))
        
    def generateNetwork(self,edgeProb='beta',directed=False):
        """
        Network nodes at each leaf of the dendro are equivalent.  For each leaf work out the
        probability of connection with other blocks by working up to the root of the tree.
        """
        if directed:
            error('directed case not defined yet')
        else:
            G=nx.Graph()
            # make sure that all nodes are added even though the graph might be disconnected
            G.add_nodes_from(np.arange(self.node[0]['n']))
            

        #cycle through nodes and generate edges
        for v in self.nodes_iter():
            
            children=self.node[v]['children']
            Nr=self.node[v]['Nr']
            Er=self.node[v]['Er']
            if not directed:
                Nr = np.triu(Nr)
                Er = np.triu(Er)

            for ci,cj in izip(*Nr.nonzero()):
                try:
                    childi=self.node[children[ci]]
                    childj=self.node[children[cj]]
                except IndexError:      # if it is a leaf node
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
                        error('edge probabilities undefined')
                    # print "Probability"
                    # print p
                except ValueError:
                    print "Something went wrong when sampling from the model"
                edges = (np.random.rand(int(Nr[ci,cj])) < p).reshape((childi['n'],childj['n']))
                if ci == cj:
                    edges = np.triu(edges)
                edges = edges.nonzero()
                G.add_edges_from(zip(childi['nnodes'][edges[0]],childj['nnodes'][edges[1]]))

        #remove self loops
        G.remove_edges_from(G.selfloop_edges())
        return G

    def print_nodes(self,keys=['Er','Nr']):
        for node in self.nodes_iter():
            print node
            for key in keys:
                print key, self.node[node][key]

    def partition_level(self,level):
        #TODO other level except zero
        partition=np.zeros(self.node[self.root_node]['n'])
        level_nodes=self.successors(self.root_node)
        for ni,node in enumerate(level_nodes):
            children=self.node[node]['nnodes']
            partition[children]=ni
        return partition


    def lowest_partition(self):
        for node in self.nodes_iter():
            if len(self.successors(node))==0:
                children=self.node[node]['nnodes']
                print node, len(children), children
    
    def get_highest_partition(self):
        partition=np.zeros(self.node[self.root_node]['n'])
        # print len(partition)
        pi=0
        for node in self.node[0]['children']:
            children=self.node[node]['nnodes']
            #~ print node, len(children), children
            partition[children]=pi
            pi+=1
        return partition
    
    def get_lowest_partition(self):
        partition=np.zeros(self.node[self.root_node]['n'])
        # print len(partition)
        pi=0
        for node in self.nodes_iter():
            if len(self.successors(node))==0:
                children=self.node[node]['nnodes']
                #~ print node, len(children), children
                partition[children]=pi
                pi+=1
        return partition

    def to_scipy_sparse_matrix(self,G):
        return nx.to_scipy_sparse_matrix(G)


    def _get_child_params(self,v):
        #recursively obtain child node params
        Nrs=[]
        Ers=[]
        nr=[]
        for child in self.node[v]['children']:
            Nr,Er=self._get_child_params(child)
            Nrs.append(Nr)
            Ers.append(Er)
            nr.append(len(Nr))
        #current node params
        Nr=self.node[v]['Nr']
        Er=self.node[v]['Er']

        block_Nr=[]#np.empty((len(nr),len(nr)),dtype=object)
        block_Er=[]#np.empty((len(nr),len(nr)),dtype=object)

        for ci,childi in enumerate(self.node[v]['children']):
            block_Nr.append([])
            block_Er.append([])
            for cj,childi in enumerate(self.node[v]['children']):
                if ci==cj:
                    block_Er[ci].append(Ers[ci])
                    block_Nr[ci].append(Nrs[ci])
                else:
                    block_Nr[ci].append(Nr[ci,cj]*np.ones((nr[ci],nr[cj])))
                    block_Er[ci].append(Er[ci,cj]*np.ones((nr[ci],nr[cj])))

        try:
            return np.bmat(block_Nr).getA(),np.bmat(block_Er).getA()
        except ValueError:
            return Nr,Er


    def construct_full_block_params(self):
        return self._get_child_params(self.root_node)

    def detectability_report(self):
        pass

def create_partition_matrix_from_vector(partition_vec):
    """
    Create a partition indicator matrix from a given vector; -1 entries in partition vector will
    be ignored and can be used to denote unasigned nodes.
    """
    nr_nodes = partition_vec.size
    k=len(np.unique(partition_vec))

    partition_matrix = scipy.sparse.coo_matrix((np.ones(nr_nodes),(np.arange(nr_nodes), partition_vec)),shape=(nr_nodes,k)).tocsr()
    return partition_matrix


