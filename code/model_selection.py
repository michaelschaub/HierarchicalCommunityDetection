import numpy as np
#~ import seaborn as sns
from matplotlib import pyplot as plt
from scipy.special import betaln,binom
from itertools import izip
plt.ion()

def looxv(Er,Nr,a=1.,b=1.):
    
    lxv=Nr*betaln(Er+a, Nr-Er+b) 
    if Er>0:
        lxv+= - Er*betaln(Er+a-1, Nr-Er+b)
    if (Nr-Er)>0:
        lxv+= - (Nr-Er)*betaln(Er+a, Nr-Er+b-1)
    
    if not np.isfinite(lxv):
        print Er,Nr,Nr*betaln(Er+a, Nr-Er+b), Er*betaln(Er+a-1, Nr-Er+b), (Nr-Er)*betaln(Er+a, Nr-Er+b-1)
        print fail
    
    return lxv


def betabinlik(Er,Nr,a=1.,b=1.):
    
    lik= - betaln(a, b) 
    lik+= betaln(Er+a, Nr-Er+b)
    #~ lik+= np.log(binom(Nr,Er))
    
    return lik


"""
Tests sets of blocks to see if they can be merged
Inputs:
- ErList : array-like of edge counts
- NrList : array-like of no. of possible edges

Returns:
- Log of the ratio of looxv likelihoods.  If positive then merge blocks.
"""
def testBlocks(ErList,NrList,a=1.,b=1.):
    looxv_separate=sum(looxv(Er,Nr,a,b) for Er,Nr in izip(ErList,NrList))
    looxv_merged=looxv(np.sum(ErList),np.sum(NrList),a,b)
    return looxv_merged - looxv_separate


def createMergeList(nEdges, nNodes, K):
    
    indices=np.arange(K*K).reshape(K,K)
    print indices
    
    mergelist=[]
    
    for ki in xrange(K):
        for kj in xrange(ki+1,K):
            diff_dist = looxv(nEdges[indices[ki,:]],nNodes[indices[ki,:]])+looxv(nEdges[indices[kj,:]],nNodes[indices[kj,:]]) - looxv(nEdges[indices[ki,:]]+nEdges[indices[kj,:]],nNodes[indices[ki,:]]+nNodes[indices[kj,:]])
            diff_dist
            pval = (diff_dist<0).mean(0)
            pval = np.min([pval,1-pval],0)
            eis = indices[ki,(diff_dist<0.00)]
            ejs = indices[kj,(diff_dist<0.00)]
            mergelist.extend([(ei,ej,p) for ei,ej,p in zip(eis,ejs,diff_dist[(diff_dist<0.00)])])
    for m in mergelist:
        print m
    return merglist




def test1(prs=np.arange(0,1,.1), diff=0, Nr=100, a=1., b=1.):
    #~ plt.figure()
    runs=10000
    
    lxvs=np.zeros(len(prs))
    liks=np.zeros(len(prs))
    
    for pi,p in enumerate(prs):
        for i in xrange(runs):
            Er1=np.random.binomial(Nr,p)
            Er2=np.random.binomial(Nr,p+diff)
            lxvs[pi] += looxv(Er1,Nr,a,b) + looxv(Er2,Nr,a,b) - looxv(Er1+Er2,Nr*2,a,b)
            #~ liks[pi] +=betabinlik(Er1,Nr,a,b) + betabinlik(Er2,Nr,a,b) - betabinlik(Er1+Er2,Nr*2,a,b)
    lxvs/=float(runs)
    plt.plot(prs,lxvs)
    #~ plt.plot(prs,liks,'--')
    plt.axhline(0., color = 'black')
    #~ return liks