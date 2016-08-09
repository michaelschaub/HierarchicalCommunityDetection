import numpy as np

#calculate a distance matrix based on variation of information
def calcVI(partitions):
    
    num_partitions,n=np.shape(partitions)
    nodes = np.arange(n)
    c=len(np.unique(partitions[0]))
    vi_mat=np.zeros((num_partitions,num_partitions))
    
    
    for i in xrange(num_partitions):
        A1 = sparse.coo_matrix((np.ones(n),(partitions[i,:],nodes)),shape=(c,n),dtype=np.uint).tocsc()
        n1all = np.array(A1.sum(1),dtype=float)
        
        for j in xrange(i):
            
            A2 = sparse.coo_matrix((np.ones(n),(nodes,partitions[j,:])),shape=(n,c),dtype=np.uint).tocsc()
            n2all = np.array(A2.sum(0),dtype=float)
            
            n12all = np.array(A1.dot(A2).todense(),dtype=float)
            
            rows, columns = n12all.nonzero()
            
            vi = np.sum(n12all[rows,columns]*np.log((n12all*n12all/(np.outer(n1all,n2all)))[rows,columns]))
            
            vi = -1/n*vi
            vi_mat[i,j]=vi
            vi_mat[j,i]=vi
    
    return vi_mat