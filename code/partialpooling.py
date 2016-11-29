import numpy as np
import pymc
#~ import seaborn as sns
from matplotlib import pyplot as plt
plt.ion()


"""
infers parameters and returns mergelist
"""
def createMergeList(nEdges, nNodes, nSamples=1000000):
    mcmc=inferParameters(nEdges, nNodes, nSamples)
    return compare(mcmc)




"""
Infers parameters for each of the blocks using hierarchical model
inputs:
 - nEdges: array of number of edges in each block
 - nNodes: array of number of nodes in each block
"""
def inferParameters(nEdges, nNodes, nSamples=1000000):

    @pymc.stochastic(dtype=np.float64)
    def beta_priors(value=[1.0, 1.0]):
        a, b = value
        if a <= 0 or b <= 0:
            return -np.inf
        else:
            return np.log(np.power((a + b), -2.5))

    a = beta_priors[0]
    b = beta_priors[1]


    omega = pymc.Beta('omega', a, b, size=len(nEdges))

    observed_values = pymc.Binomial('observed_values', nNodes, omega, observed=True, value=nEdges)

    model = pymc.Model([a, b, omega, observed_values])
    mcmc = pymc.MCMC(model)

    mcmc.sample(nSamples, nSamples/2)

    return mcmc

"""
compares block parameters - if distributions overlap then the merge is proposed
***only column merges are proposed ***
"""
def compare(mcmc):
    samples=mcmc.trace('omega')[:]
    Ksq=np.shape(samples)[1]
    K=int(np.sqrt(Ksq))
    indices=np.arange(Ksq).reshape(K,K)
    print indices

    mergelist=[]

    for ki in xrange(K):
        for kj in xrange(ki+1,K):
            diff_dist = samples[:,indices[ki,:]] - samples[:,indices[kj,:]]
            pval = (diff_dist<0).mean(0)
            pval = np.min([pval,1-pval],0)
            eis = indices[ki,(pval>0.00)]
            ejs = indices[kj,(pval>0.00)]
            mergelist.extend([(ei,ej,p) for ei,ej,p in zip(eis,ejs,pval[(pval>0.005)])])
    for m in mergelist:
        print m
    return mergelist


def plotComparison(mcmc):
    import seaborn as sns
    samples=mcmc.trace('omega')[:]
    Ksq=np.shape(samples)[1]
    K=int(np.sqrt(Ksq))
    indices=np.arange(Ksq).reshape(K,K)
    print indices

    for ki in [0]:#xrange(K):
        for kj in xrange(ki+1,K):
            plt.figure()
            ind_ki=np.setdiff1d(indices[ki,:],indices[:,[ki,kj]])
            ind_kj=np.setdiff1d(indices[kj,:],indices[:,[ki,kj]])
            print ind_ki
            print ki,kj
            diff_dist = samples[:,ind_ki] - samples[:,ind_kj]

            sns.kdeplot(diff_dist.flatten(), shade = True, label = "Difference %i -- %i" % (ki,kj))
            plt.axvline(0.0, color = 'black')
            print (diff_dist<0).mean(0)
            for m in diff_dist.mean(0):
                plt.axvline(m, color = 'red')



