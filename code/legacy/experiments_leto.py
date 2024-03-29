from __future__ import division
import numpy as np
from GHRGmodel import GHRG
import GHRGmodel
import GHRGbuild
import spectral_algorithms as spectral
#~ import inference
import metrics
from matplotlib import pyplot as plt
#~ import partialpooling as ppool
import model_selection as ppool
import change_points as cp
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence
plt.ion()
import metrics
from random import sample
import sample_networks
import networkx as nx
import scipy.sparse as sparse
from scipy.stats import norm


def test_overlap(snr=0.5,c_bar=5):
    n=2048
    
    n_levels=2
    groups_per_level=2
    D=GHRGbuild.create2paramGHRG(n,snr,c_bar,n_levels,groups_per_level)
    pvec = D.get_partition_all()
    
    G=D.generateNetworkExactProb()
    A=D.to_scipy_sparse_matrix(G)
    
    D_inf=GHRGmodel.GHRG()
    D_inf.infer_spectral_partition_flat(A)
    pvec_inf = D_inf.get_partition_all()
    
    return metrics.calculate_level_comparison_matrix(pvec_inf,pvec)
    
########## Model Select TEST ################

def boot(groups_per_level=3,n_levels=3,snr=20):
    #params
    n=10000
    
    c_bar=30
    
    max_k=30
    
    plt.figure()
    mean_err=0
    for i in xrange(20):
        print "I",i
        #generate
        D_actual=GHRGbuild.create2paramGHRG(n,snr,c_bar,n_levels,groups_per_level)
        G=D_actual.generateNetworkExactProb()
        A=D_actual.to_scipy_sparse_matrix(G)
        k , partition_vec, H, errors = spectral.identify_hierarchy(A,max_k,mode='SBM',reg=False, norm='F',method='analytic', threshold=0)
        
        
        #D=GHRGmodel.GHRG()
        #D.infer_spectral_partition_hier(A,thresh_method='bootstrap')
        #~ D.infer_spectral_partition_hier(A,thresh_method='ttest')
        #pvec = D.get_partition_all()
        #~ print [len(np.unique(p)) for p in pvec]
        
        Ks,errs= zip(*errors)
        Ks=np.array(Ks)
        errs=np.array(errs)
        mean_err+=errs
    
        plt.plot(Ks,errs)
    
    plt.plot(Ks,mean_err/(i+1),'k',lw=3)
    
    #~ plt.plot(Ks,n*Ks-Ks**2,'k:')
    #~ plt.plot(Ks[1:],errs[1:]-errs[:-1])
    #~ errs/=np.power(Ks,1/2)
    #~ plt.plot(Ks[1:],errs[1:]-errs[:-1])
    #~ plt.axhline(0,color='k',ls=':')
    
    return partition_vec, errors


def run_real(network='agblog-pol'):
    with open('data/{}_edges.txt'.format(network)) as file:
        edges = np.int32([row.strip().split() for row in file.readlines()])
    
    G=nx.Graph()
    G.add_edges_from(edges)
    A = nx.to_scipy_sparse_matrix(G)
    pvec = spectral.hier_spectral_partition(A, reps=10)
    
    with open('data/{}_labels.txt'.format(network)) as file:
        labels = np.int32([row.strip().split() for row in file.readlines()])
    
    print "labels", len(labels[:,1]), len(pvec[0])
    
    score_matrix = metrics.calculate_level_comparison_matrix(pvec, [labels[:,1]])
    precision, recall = metrics.calculate_precision_recall(score_matrix)
    print score_matrix
    print "precision, recall"
    print precision, recall
    print "levels", [len(np.unique(pv)) for pv in pvec]
    return pvec


def run_model_select_exp(reps=20):
    for snr in [15,10,5,3,2]:
        for rep in xrange(reps):
            model_select(3,3,snr)

def model_select(groups_per_level=3,n_levels=3,snr=20):
    n=3**9
    
    c_bar=50
    
    D_actual=GHRGbuild.create2paramGHRG(n,snr,c_bar,n_levels,groups_per_level)
    G=D_actual.generateNetworkExactProb()
    A=D_actual.to_scipy_sparse_matrix(G)
    
    #get true hierarchy
    true_pvec = D_actual.get_partition_all()
    
    for ri in xrange(5):
        for use_likelihood in [False]:
            #infer partitions with no noise
            inf_pvec = spectral.hier_spectral_partition(A, reps=20)
            
            score_matrix = metrics.calculate_level_comparison_matrix(inf_pvec, true_pvec)
            precision, recall = metrics.calculate_precision_recall(score_matrix)
            diff_levels = metrics.compare_levels(true_pvec,inf_pvec)
            print len(inf_pvec), len(true_pvec)
            print diff_levels
            print "precision, recall"
            print precision, recall
            noise=0
            
            print [len(np.unique(pv)) for pv in true_pvec]
            print [len(np.unique(pv)) for pv in inf_pvec]
            
            print asdf
            
            with open('results/model_select.txt','a') as file:
                file.write('{} {} {} {:.3f} {:.3f} {} '.format(snr,noise,int(use_likelihood),precision,recall,len(inf_pvec)))
                file.write('{d[0]} {d[1]} {d[2]}\n'.format(d=diff_levels))
        
    for noise in [1e-3,1e-2,1e-1,.5]:
        for ri in xrange(5):
            for use_likelihood in [True, False]:
                inf_pvec = spectral.hier_spectral_partition(A, reps=10, noise=noise,use_likelihood=use_likelihood)
            
                score_matrix = metrics.calculate_level_comparison_matrix(inf_pvec, true_pvec)
                precision, recall = metrics.calculate_precision_recall(score_matrix)
                diff_levels = metrics.compare_levels(true_pvec,inf_pvec)
                print len(inf_pvec), len(true_pvec)
                print diff_levels
                print "precision, recall"
                print precision, recall
                
                with open('results/model_select.txt','a') as file:
                    file.write('{} {} {} {:.3f} {:.3f} {} '.format(snr,noise,int(use_likelihood),precision,recall,len(inf_pvec)))
                    file.write('{d[0]} {d[1]} {d[2]}\n'.format(d=diff_levels))
    
    return inf_pvec
    

"""
plot as a function of noise added, col=
2 : precision
3 : recall
4 : number of inferred levels
5+ : difference in number of groups at level
"""
def plot_mse(col=3):
    
    with open('results/model_select.txt') as file:
        data = np.float64([row.strip().split() for row in file.readlines()])
    
    for uselik in [0,1]:
    
        snrs = np.unique(data[:,0])
        for snr in snrs:
            snr_data = data[data[:,0]==snr,:]
            snr_data = snr_data[snr_data[:,2]==uselik,:]
            print snr_data.shape
            noise_vals = np.unique(snr_data[:,1])
            mean_score = [np.mean(snr_data[snr_data[:,1]==nv,col]) for nv in noise_vals]
            noise_vals[noise_vals==0] = 1e-6
            if uselik:
                plt.semilogx(noise_vals,mean_score,":",label="{} {}".format(snr,uselik))
            else:
                plt.semilogx(noise_vals,mean_score,label="{} {}".format(snr,uselik))
    plt.legend()
    

# also plot variance
# use k=27 as a threshold
def noise(groups_per_level=3,n_levels=3,snr=20,plot=True, max_k=30, reps=10, full_adj=True):
    #params
    n=10000
    
    c_bar=30
    
    D_actual=GHRGbuild.create2paramGHRG(n,snr,c_bar,n_levels,groups_per_level)
    G=D_actual.generateNetworkExactProb()
    A=D_actual.to_scipy_sparse_matrix(G)
    
    edges = A.nonzero()
    m=len(edges[0])
    
    if not full_adj:
        p0 = spectral.spectral_partition(A,mode='Bethe',num_groups=54)
        max_k = np.max(p0)+1
        links_between_groups, possible_links_between_groups = spectral.compute_number_links_between_groups(A,p0)
        A = links_between_groups
    
    all_errors=np.empty((max_k-1,reps))
    mag=1e-3
    if plot:
        plt.figure()
    mean_err=0
    for i in xrange(reps):
        print "I",i
        #generate noise
        if full_adj:
            #~ nnoise = sparse.coo_matrix((np.random.randn(m),(edges[0],edges[1])),shape=(n,n)).tocsr()
            rvs = norm(0, mag).rvs
            nnoise=sparse.random(n,n,m/(n*n),data_rvs=rvs)
            #~ nnoise=sparse.random(n,n,m/(n*n))
            print nnoise.sum(), nnoise.astype('bool').sum(), "nnoise elements added", n*n
            A_noisy=A+nnoise
            print A_noisy.astype('bool').sum()
        else:
            # noise is generated by unit variance normal * mag param * possible edges * noise density 
            #mag=1e-1
            nnoise = np.random.randn(max_k,max_k)*mag*possible_links_between_groups*(m/(n*n))
            print nnoise.sum(),"nnoise added"
            A_noisy=A+nnoise
        
        k , partition_vec, H, errors = spectral.identify_hierarchy(A_noisy,max_k,mode='SBM',reg=False, norm='F',method='analytic', threshold=0)
        
        Ks,errs= zip(*errors)
        Ks=np.array(Ks)
        errs=np.array(errs)
        all_errors[:,i] = errs
        mean_err+=errs
        
        if plot:
            plt.plot(Ks,errs, 'm', lw=0.5, alpha=0.2)
    
    mean_err/=reps
    
    if plot:
        plt.plot(Ks,mean_err,'k',lw=1)
        #~ plt.plot(Ks,1./np.power(Ks,1/2),'b')
        #~ plt.plot(Ks,1./np.power(Ks,1/3),'r')
        plt.xlabel('Number of groups')
        plt.ylabel('Projection error')
        plt.tight_layout()
    
    k , partition_vec, H, errors = spectral.identify_hierarchy(A,max_k,mode='SBM',reg=False, norm='F',method='analytic', threshold=0)
        
    Ks,errs= zip(*errors)
    Ks=np.array(Ks)
    errs=np.array(errs)
    
    
    if plot:        
        plt.figure()
        plt.plot(Ks[1:],errs[1:]-errs[:-1],'m')
        plt.plot(Ks[1:],mean_err[1:]-mean_err[:-1],'b')
        plt.axhline(0,color='k',ls=':')
        plt.figure()
        errs/=np.power(Ks,1/2)
        mean_err/=np.power(Ks,1/2)
        plt.plot(Ks[1:],errs[1:]-errs[:-1],'m')
        plt.plot(Ks[1:],mean_err[1:]-mean_err[:-1],'b')
        plt.axhline(0,color='k',ls=':')
       
        plt.figure()
        plt.plot(Ks,np.var(all_errors,1))
        plt.axhline(0,color='k',ls=':')
        print np.var(all_errors,1)
    
    return Ks,all_errors


def test_noise_snr(groups_per_level=3,n_levels=3):
    reps=30
    #~ fig1,ax1 = plt.subplots(1,1)
    #~ fig2,ax2 = plt.subplots(1,1)
    #~ fig3,ax3 = plt.subplots(1,1)
    #~ fig4,ax4 = plt.subplots(1,1)
    #~ fig5,ax5 = plt.subplots(1,1)
    
    max_k=28
    #~ Ks=np.arange(max_k-1,1,-1)
    
    true_vec = np.zeros(max_k-1,dtype=bool)
    true_vec[[-2,-8,-26]]=True
    snrs=np.arange(7,16,2)
    
    
    for snr in snrs:
        for i in xrange(reps):
            print i,"SNR:",snr
            Ks,errors=noise(groups_per_level=groups_per_level,n_levels=n_levels,snr=snr,plot=False,max_k=max_k,reps=10)
            print "Ks true",Ks[true_vec]
            print Ks
            mean_error=np.mean(errors,1)
            var_error=np.var(errors,1)
        
            with open('results/error_results_A.txt','a') as f:
                f.write('{} '.format(snr))
                for me in mean_error:
                    f.write('{} '.format(me))
                f.write('\n')
            with open('results/var_results_A.txt','a') as f:
                f.write('{} '.format(snr))
                for ve in var_error:
                    f.write('{} '.format(ve))
                f.write('\n')
        
def plot_err_var():
    with open('results/error_results_A.txt') as f:
        errors = np.float64([row.strip().split() for row in f.readlines()])
    
    with open('results/var_results_A.txt') as f:
        vars = np.float64([row.strip().split() for row in f.readlines()])
    
    snrs = np.unique(errors[:,0])
    Ks=np.arange(28,1,-1)
    
    fig1,ax1 = plt.subplots(1,1)
    fig2,ax2 = plt.subplots(1,1)
    fig3,ax3 = plt.subplots(1,1)
    #~ fig4,ax4 = plt.subplots(1,1)
    #~ fig5,ax5 = plt.subplots(1,1)
    
    norm=plt.Normalize(vmin=snrs.min(),vmax=snrs.max())
    cmap=plt.cm.Spectral
    
    for snr in snrs:
        snr_error = errors[errors[:,0]==snr,1:]
        mean_error = np.mean(errors[errors[:,0]==snr,1:],0)
        mean_var = np.mean(vars[vars[:,0]==snr,1:],0)
        
        #normalised error - mean
        mean_error_sqrt=mean_error/np.power(Ks,1/2)
        print np.mean(mean_error_sqrt)
        mean_error_sqrt-=np.mean(mean_error_sqrt)
        ax1.plot(Ks,mean_error_sqrt,color=cmap(norm(snr)), label=snr)
        ax1.legend()
        ax1.axhline(0,color='k',ls=':')
        
        #normalised error - diff
        mean_error_sqrt[1:]-=mean_error_sqrt[:-1]
        ax2.plot(Ks[1:],mean_error_sqrt[1:],color=cmap(norm(snr)), label=snr)
        ax2.legend()
        ax2.axhline(0,color='k',ls=':')
        
        snr_error_sqrt = snr_error/np.power(Ks,1/2)[None,:]
        snr_error_sqrt[:,1:] -= snr_error_sqrt[:,:-1]
        mean_err = np.mean(snr_error_sqrt,0)
        ax3.plot(Ks[1:],mean_err[1:],color=cmap(norm(snr)), label=snr)
        ax3.axhline(0,color='k',ls=':')
        
        #~ ax3.plot(Ks,mean_var,color=cmap(norm(snr)), label=snr)
        #~ ax3.legend()
        #~ ax3.axhline(0,color='k',ls=':')
        
    
        #~ mean_error_sqrt=mean_error/np.power(Ks,1/2)
        #~ mean_error_sqrt[1:]-=mean_error_sqrt[:-1]
        #~ ax1.plot(np.ones(sum(true_vec==False)-1)*snr,mean_error_sqrt[true_vec==False][1:],'kx')
        #~ ax1.plot(snr,mean_error_sqrt[-2],'mo',alpha=0.5)
        #~ ax1.plot(snr,mean_error_sqrt[-8],'ms',alpha=0.5)
        #~ ax1.plot(snr,mean_error_sqrt[-26],'m^',alpha=0.5)
        
        #~ ax2.plot(np.ones(sum(true_vec==False)-1)*snr,var_error[true_vec==False][1:],'kx')
        #~ ax2.plot(snr,var_error[-2],'mo',alpha=0.5)
        #~ ax2.plot(snr,var_error[-8],'ms',alpha=0.5)
        #~ ax2.plot(snr,var_error[-26],'m^',alpha=0.5)
        
        #~ norm=plt.Normalize(vmin=snrs.min(),vmax=snrs.max())
        #~ cmap=plt.cm.Spectral
        #~ ax3.plot(Ks[1:],mean_error_sqrt[1:],color=cmap(norm(snr)), label=snr)
        #~ ax3.legend()
        #~ ax3.axhline(0,color='k',ls=':')
        
        #~ mean_error_k=mean_error/np.power(Ks,1/100)
        #~ mean_error_k[1:]-=mean_error_k[:-1]
        #~ ax4.plot(Ks[1:],mean_error_k[1:],color=cmap(norm(snr)), label=snr)
        #~ ax4.legend()
        #~ ax4.axhline(0,color='k',ls=':')
        
        #~ mean_error_sq=mean_error/np.power(Ks,1/100)
        #~ mean_error_sq-=np.mean(mean_error_sq)
        #~ ax5.plot(Ks,mean_error_sq,color=cmap(norm(snr)), label=snr)
        #~ ax5.legend()
        #~ ax5.axhline(0,color='k',ls=':')

########## ZOOM TEST ################

def zoom_exp(n_cliques=64, noise=1e-5):
    A=construct_cliques(n_cliques=n_cliques, clique_size=10,noise=noise)
    D=GHRGmodel.GHRG()
    D.infer_spectral_partition_hier(A)
    pvec = D.get_partition_all()
    print [len(np.unique(p)) for p in pvec]
    return pvec


########## RESOLUTION TEST ################

def clique_test(n_cliques=64, clique_size=10,noise=0.01,A=None,K_known=False,regularizer='BHa'):
    if A is None:
        A = construct_cliques(n_cliques=64, clique_size=10,noise=0.01)
    print "infer"
    D=GHRGmodel.GHRG()
    if K_known:
        D.infer_spectral_partition_flat(A,num_groups=n_cliques, regularizer=regularizer)
    else:
        D.infer_spectral_partition_flat(A)
    print D.nodes()
    return D

def construct_cliques(n_cliques=64, clique_size=10,noise=0.01):
    np.random.seed(np.random.randint(2**31))
    
    block=sparse.coo_matrix(np.ones((clique_size,clique_size))-np.diag(np.ones(clique_size)))
    blocks=[]
    print "construct"
    for nc in xrange(n_cliques):
        blocks.append(block)
    
    A=sparse.block_diag(blocks).astype('bool')
    noise_matrix=sparse.random(n_cliques*clique_size,n_cliques*clique_size,noise/2).astype('bool')
    noise_matrix=(noise_matrix+noise_matrix.T).astype('bool')
    print noise, noise_matrix.sum(), "noise elements added", (n_cliques*clique_size)**2
    A_noisy=((A+noise_matrix)-(A*noise_matrix)).astype('float')
    
    rvs = norm(0, 0.01).rvs
    
    nnoise=sparse.random(n_cliques*clique_size,n_cliques*clique_size,5e-3,data_rvs=rvs)
    print nnoise.sum(), nnoise.astype('bool').sum(), "nnoise elements added", (n_cliques*clique_size)**2
    A_noisy=A_noisy+nnoise
    return A_noisy

def noise_levels(n_cliques):
    min_noise = {128:-5.5,64:-5,32:-4.5,16:-4}[n_cliques]
    return 10**np.arange(min_noise,-.5,0.1)

def clique_test_batch(n_cliques=64, regularizer='BHa'):
    
    runs=50
    
    clique_size=10
    file='out/resolution%i.txt' % n_cliques
    
    for i in xrange(runs):
        for ni,noise in enumerate(noise_levels(n_cliques)):
        
            try:
                A=construct_cliques(n_cliques, clique_size, noise)
                D=clique_test(n_cliques=64, clique_size=10,noise=0.01,A=A,K_known=False, regularizer=regularizer)
                
            except:
                A=construct_cliques(n_cliques, clique_size, noise)
                D=clique_test(n_cliques=64, clique_size=10,noise=0.01,A=A,K_known=False, regularizer=regularizer)
                
            with open(file,'a') as f:
                f.write('%i ' % (len(D.nodes())-1))
        with open(file,'a') as f:
            f.write('\n')

def plot_res(n_cliques=64, regularizer=None):
    if regularizer is None:
        file='out/resolution%i.txt' % n_cliques
    else:
        file='out/resolution_BHm.txt'
    
    with open(file) as f:
        results = np.float64([row.strip().split() for row in f.readlines()[:-1]])
    print len(results)
    results = np.mean(results,0)
    plt.figure()
    plt.semilogx(noise_levels(n_cliques),results,lw=2)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.xlabel('noise',size=24)
    plt.ylabel('number groups detected',size=24)
    plt.axhline(n_cliques)
    plt.tight_layout()

##########################################


"""
Test overlap precision and recall
"""
def test_pr1():
    reps=20
    n=800
    
    true_pvecs=np.zeros((2,n))
    true_pvecs[0,n/2:]=1
    for i in xrange(8):
        true_pvecs[1,(i*100):(i+1)*(100)]=i
    
    precision=np.zeros(40)
    recall=np.zeros(40)
    for rep in xrange(reps):
        for li,l in enumerate(xrange(0,n,20)):
            print "l",l
            pvecs=true_pvecs.copy()
            inds=sample(xrange(n),l)
            _,new_values=np.random.multinomial(1,np.ones(8)/8,size=l).nonzero()
            pvecs[1,inds]=new_values
            pvecs[0,inds]=np.int32(new_values>3)
        
            sm=metrics.calculate_level_comparison_matrix(pvecs,true_pvecs)
            p,r=metrics.calculate_precision_recall(sm)
            precision[li]+=p/reps
            recall[li]+=r/reps
    
    plt.figure()
    plt.plot(np.arange(0,n,20),precision,'b-.')
    plt.plot(np.arange(0,n,20),recall,'r--')
    

def test_pr2():
    reps=20
    n=800
    
    pred_pvecs=np.zeros((2,n))
    pred_pvecs[0,n/2:]=1
    for i in xrange(8):
        pred_pvecs[1,(i*100):(i+1)*(100)]=i
        
    true_pvecs=np.zeros((4,n))
    true_pvecs[0,n/2:]=1
    for i in xrange(4):
        true_pvecs[1,(i*200):(i+1)*(200)]=i
    for i in xrange(8):
        true_pvecs[2,(i*100):(i+1)*(100)]=i
    for i in xrange(16):
        true_pvecs[3,(i*50):(i+1)*(50)]=i
    
    
    precision=np.zeros(40)
    recall=np.zeros(40)
    for rep in xrange(reps):
        for li,l in enumerate(xrange(0,n,20)):
            print "l",l
            pvecs=pred_pvecs.copy()
            inds=sample(xrange(n),l)
            _,new_values=np.random.multinomial(1,np.ones(8)/8,size=l).nonzero()
            pvecs[1,inds]=new_values
            pvecs[0,inds]=new_values//4
            
            sm=metrics.calculate_level_comparison_matrix(pvecs,true_pvecs)
            p,r=metrics.calculate_precision_recall(sm)
            precision[li]+=p/reps
            recall[li]+=r/reps
    
    plt.figure()
    plt.plot(np.arange(0,n,20),precision,'b-.')
    plt.plot(np.arange(0,n,20),recall,'r--')
    
    precision=np.zeros(40)
    recall=np.zeros(40)
    for rep in xrange(reps):
        for li,l in enumerate(xrange(0,n,20)):
            print "l",l
            pvecs=true_pvecs.copy()
            inds=sample(xrange(n),l)
            _,new_values=np.random.multinomial(1,np.ones(16)/16,size=l).nonzero()
            pvecs[3,inds]=new_values
            pvecs[2,inds]=new_values//2
            pvecs[1,inds]=new_values//4
            pvecs[0,inds]=new_values//8
        
            sm=metrics.calculate_level_comparison_matrix(pvecs,pred_pvecs)
            p,r=metrics.calculate_precision_recall(sm)
            precision[li]+=p/reps
            recall[li]+=r/reps
    
    plt.figure()
    plt.plot(np.arange(0,n,20),precision,'b-.')
    plt.plot(np.arange(0,n,20),recall,'r--')




"""
Test change point detection
"""
def test_cp(snr_before = 1, snr_after = 1, n_groups=2):
    
    # mean degree and number of nodes etc.
    n=1000
    n_levels = 1
    K=n_groups**n_levels
    ratio_before = 0.5
    ratio_after = 1
    #~ snr_before = 1
    #~ snr_after = 1
    
    #before change model
    print "before change model"
    D1=create2paramGHRG(n,snr_before,ratio_before,n_levels,n_groups)
    #after change model
    print "after change model"
    D2=create2paramGHRG(n,snr_after,ratio_after,n_levels,n_groups)
    
    
    #sliding window
    w=4
    #degree
    cm=20
    #before change model
    #~ D1=create2paramGHRG(100,cm,0.5,1,2)
    #~ D1=create2paramGHRG(100,cm,1,1,2)
    #after change model
    #~ D2=create2paramGHRG(100,cm,1,1,2)
    
    #create sequence of graphs
    Gs=[D1.generateNetworkExactProb() for i in xrange(w+1)]
    Gs.extend([D2.generateNetworkExactProb() for i in xrange(w+1)])
    
    print [len(G.edges()) for G in Gs]
    
    return cp.detectChanges_flat(Gs,w)
    
    
def runEnron(w=4):
    #get networks
    print "Constructing networks..."
    path_to_data = '../../../Dropbox/Projects/data/enron_min/'
    with open(path_to_data + 'filelist.txt') as f:
        netFiles=[file.strip() for file in f.readlines()]
    
    Gs=[]
    
    for netFile in netFiles:
        G=nx.Graph()
        G.add_nodes_from(range(151))
        
        with open(path_to_data + netFile) as f:
            edgeList=np.int32([row.strip().split() for row in f.readlines()])
        
        G.add_edges_from(edgeList)
        
        Gs.append(G)
    
    return cp.detectChanges_flat(Gs,w)



"""
Test partial pooling
 - input: ratio - ratio of probabilities between on- and off-diagonals

 returns:
 -  D_gen - Dendro for generating example
 - D_inferred - inferred Dendro
 - mergeList - list of triples (pairs of blocks to merge and p-value)
"""
def testpp(ratio=0.1):
    cm=20 # degree parameter
    n=1000 #nodes
    n_levels=3 #number of levels generated in GHRG

    level_k=2 # number of groups at each level

    D_gen=create2paramGHRG(n,cm,ratio,n_levels,level_k)
    G=D_gen.generateNetwork()
    A = D_gen.to_scipy_sparse_matrix(G)

    D_inferred = inf.split_network_by_recursive_spectral_partition(A,mode='Bethe',max_depth=-1,num_groups=-1)
    partitions=np.empty((2,n))
    partitions[0,:] = D_gen.get_lowest_partition()
    partitions[1,:] = D_inferred.get_lowest_partition()
    print "VI", metrics.calcVI(partitions)[0,1]
    K = partitions[1,:].max().astype('int')
    Di_nodes, Di_edges = D_inferred.construct_full_block_params()
    mergeList=ppool.createMergeList(Di_edges.flatten(),Di_nodes.flatten(),K)
    #~ ppool.plotComparison(mcmc)
    #~ ppool.compare(mcmc)
    return D_gen, D_inferred, mergeList


def testModelSelection(max_num_groups=20,ratio=0.1):
    n_levels=3 #number of levels generated in GHRG

    level_k=2 # number of groups at each level
    
    for n in 2**np.arange(8,14):
        for cm in 2**np.arange(2,6):
            for rep in xrange(100):
                print n,cm, rep
                #~ cm=20 # degree parameter
                #~ n=1000 #nodes
                failed=True
                attempts=0
                while failed:
                    try:
                        D_gen=create2paramGHRG(n,cm,ratio,n_levels,level_k)
                        G=D_gen.generateNetwork()
                        A = D_gen.to_scipy_sparse_matrix(G)
                        print (n*cm), A.nnz/2
                #~ looxv=inference.infer_spectral_blockmodel(A, max_num_groups=max_num_groups)/(n*cm)
                        looxv=2*inference.infer_spectral_blockmodel(A, max_num_groups=max_num_groups)/(A.nnz)
                        failed=False
                    except ArpackNoConvergence:
                        attempts+=1
                        print attempts
                        
                    
                #~ plt.figure()
                #~ plt.plot(np.arange(1,max_num_groups),looxv)
                #~ print looxv
                diff = looxv[1:]-looxv[:-1]
                #~ print diff
                #~ plt.plot(np.arange(2,max_num_groups),diff)
                try:
                    belowzero=((looxv[1:]-looxv[:-1])<0).nonzero()[0][0]+2
                except IndexError:
                    belowzero=20
                try:
                    below05=((looxv[1:]-looxv[:-1])<0.05).nonzero()[0][0]+2
                except IndexError:
                    below05=20
                print (looxv[7]-looxv[6]), (looxv[8]-looxv[7]),(belowzero>7)
                
                with open('out/res_tms%sm.txt' % (str(ratio).replace('.','')),'a') as f:
                    f.write('%i\t%i\t%f\t%f\t%i\t%i\n' % (n,cm,(looxv[7]-looxv[6]), (looxv[8]-looxv[7]),(belowzero>7), (below05-1) )) 
    

def plot_tms():
    #~ with open('out/res_tms01c.txt') as f:
        #~ results=np.float64([row.strip().split() for row in f.readlines()])
    #~ plt.figure()
    #~ plt.plot(np.arange(len(results)),results[:,2])
    #~ plt.plot(np.arange(len(results)),results[:,3])
    
    with open('out/res_tms01m.txt') as f:
        results=np.float64([row.strip().split() for row in f.readlines()])
    plt.figure()
    plt.plot(np.arange(len(results)),results[:,2])
    plt.plot(np.arange(len(results)),results[:,3])
    
    plt.axhline(0.05,color='k')
    plt.axhline(0.0,color='y')
    plt.title('ratio=0.1')
    plt.ylabel('looxv difference')
    plt.xlabel('network index')
    
    with open('out/res_tms03m.txt') as f:
        results=np.float64([row.strip().split() for row in f.readlines()])
    plt.figure()
    plt.plot(np.arange(len(results)),results[:,2])
    plt.plot(np.arange(len(results)),results[:,3])
    plt.title('ratio=0.3')
    plt.ylabel('looxv difference')
    plt.xlabel('network index')

def tms(n=1000,cm=20,max_num_groups=20):
    ratio=0.1
    n_levels=3 #number of levels generated in GHRG

    level_k=2 # number of groups at each level
    
    
     # degree parameter
     #nodes
    #~ failed=True
    #~ attempts=0
    #~ while failed:
        #~ try:
    D_gen=create2paramGHRG(n,cm,ratio,n_levels,level_k)
    G=D_gen.generateNetwork()
    A = D_gen.to_scipy_sparse_matrix(G)
    looxv=inference.infer_spectral_blockmodel(A, max_num_groups=max_num_groups)/float(n)
            #~ failed=False
        #~ except:
            #~ attempts+=1
            #~ print attempts
            
        
    #~ plt.figure()
    #~ plt.plot(np.arange(1,max_num_groups),looxv)
    print looxv
    diff = looxv[1:]-looxv[:-1]
    print diff
    plt.plot(np.arange(2,max_num_groups),diff)
    try:
        belowzero=((looxv[1:]-looxv[:-1])<0).nonzero()[0][0]+2
    except IndexError:
        belowzero=20
    print (looxv[7]-looxv[6]), (looxv[8]-looxv[7]),(belowzero>7)


"""
Experiment: Test Spectral inference algorithm on hierarchical test graph

Create a sequence of test graphs (realizations of a specified hier. random model) and try
to infer the true partition using spectral methods
"""
def exp1(runs=10):
    cm=20
    n=1000
    n_levels=3
    level_k=2
    K=level_k**n_levels

    ratios=np.arange(0.1,1.,0.1)

    bb_mean=np.zeros(len(ratios))
    tt_mean=np.zeros(len(ratios))
    tb_mean=np.zeros(len(ratios))

    run_count=np.ones(len(ratios))*runs

    for ri,ratio in enumerate(ratios):

        for run in xrange(runs):
            print ratio, run
            D_gen=create2paramGHRG(n,cm,ratio,n_levels,level_k)
            G=D_gen.generateNetwork()
            A = D_gen.to_scipy_sparse_matrix(G)

            #~ try:
            D_inferred = spectral.split_network_by_recursive_spectral_partition(A,mode='Bethe',max_depth=-1,num_groups=-1)

            partitions=np.empty((2,n))
            partitions[0,:] = D_gen.get_lowest_partition()
            partitions[1,:] = D_inferred.get_lowest_partition()
            bb_mean[ri]+=metrics.calcVI(partitions)[0,1]

            partitions[1,:] = D_inferred.partition_level(0)
            tb_mean[ri]+= metrics.calcVI(partitions)[0,1]
            partitions[0,:] = D_gen.partition_level(0)
            tt_mean[ri]+= metrics.calcVI(partitions)[0,1]
            #~ except:
                #~ print 'FAIL'
                #~ run_count[ri]-=1

    tt_mean/=run_count
    tb_mean/=run_count
    bb_mean/=run_count

    plt.figure()
    plt.plot(ratios,bb_mean)
    plt.plot(ratios,tb_mean)
    plt.plot(ratios,tt_mean)

    plt.legend(['low-low','high-low', 'high-high'])

    return bb_mean, tb_mean, tt_mean

# Still in use somewhere?
# """
# Experiment 2
# """
# def exp2(runs=10):
    # cm=20
    # n=1000
    # n_levels=3
    # level_k=2
    # K=level_k**n_levels

    # ratios=np.arange(0.1,1,0.1)

    # bb_mean=np.zeros(len(ratios))
    # tt_mean=np.zeros(len(ratios))
    # tb_mean=np.zeros(len(ratios))

    # run_count=np.ones(len(ratios))*runs

    # for ri,ratio in enumerate(ratios):

        # for run in xrange(runs):
            # print ratio, run
            # D_gen=create2paramGHRG(n,cm,ratio,n_levels,level_k)
            # G=D_gen.generateNetwork()
            # A = D_gen.to_scipy_sparse_matrix(G)

            # #~ try:
            # D_inferred = spectral.split_network_hierarchical_by_spectral_partition(A,mode='Bethe',num_groups=-1)

            # partitions=np.empty((2,n))
            # partitions[0,:] = D_gen.get_lowest_partition()
            # partitions[1,:] = D_inferred.get_lowest_partition()
            # bb_mean[ri]+=metrics.calcVI(partitions)[0,1]

            # partitions[1,:] = D_inferred.partition_level(0)
            # tb_mean[ri]+= metrics.calcVI(partitions)[0,1]
            # partitions[0,:] = D_gen.partition_level(0)
            # tt_mean[ri]+= metrics.calcVI(partitions)[0,1]
            # #~ except:
                # #~ print 'FAIL'
                # #~ run_count[ri]-=1

    # tt_mean/=run_count
    # tb_mean/=run_count
    # bb_mean/=run_count

    # plt.figure()
    # plt.plot(ratios,bb_mean)
    # plt.plot(ratios,tb_mean)
    # plt.plot(ratios,tt_mean)

    # plt.legend(['low-low','high-low', 'high-high'])

    # return bb_mean, tb_mean, tt_mean

"""
Function to create a test GHRG for simulations
parameters:
    n   : number of nodes
    n_levels    : depth of GHRG
    groups_per_level     : number of groups at each level
"""
def create2paramGHRG(n,snr,ratio,n_levels,groups_per_level):

    #interaction probabilities
    omega={}
    n_this_level = n
    for level in xrange(n_levels):
        # cin, cout = calculateDegrees(cm,ratio,groups_per_level)
        cin, cout = sample_networks.calculateDegreesFromSNR(snr,ratio,groups_per_level)
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


    D=GHRG()

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
        D.node[child]['nnodes'] = D.node[D.root_node]['nnodes'][ci*n/groups_per_level:(ci+1)*n/groups_per_level]
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
                D.node[child]['nnodes'] = D.node[D.predecessors(child)[0]]['nnodes'][ci*level_n/groups_per_level:(ci+1)*level_n/groups_per_level]
                D.node[child]['n'] = len(D.node[child]['nnodes'])

    D.setLeafNodeOrder()
    D.setParameters(omega)

    return D
