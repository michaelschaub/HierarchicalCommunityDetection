from GHRGmodel import GHRG 
import spectral_algorithms as spectral






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