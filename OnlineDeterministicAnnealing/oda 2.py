"""
Tree-Structured (Multi-Resolution) Online Deterministic Annealing for Classification and Clustering
Christos Mavridis & John Baras,
Department of Electrical and Computer Engineering, University of Maryland
<mavridis@umd.edu>
"""

#%% Import Modules

import time
import numpy as np
from numba import njit
import copy
import matplotlib.pyplot as plt
import pylab as pl
# import concurrent.futures
plt.ioff()
    
np.random.seed(13)

#%%

'''
ODA Parameters


### Data

- train_data 
    # Single layer: [[np.array], [np.array], [np.array], ...]
    # Multiple Layers/Resolutions: [[np.array, np.array, ...], [np.array, np.array, ...], [np.array, np.array, ...], ...]
- train_labels
    # [ int, int , int, ... ] (int values for classification with numba.jit)
    # [ 0, 0, 0, ... ] (int zero values for clustering)
    # [ float, float, float, ... ] (and regression=True for piece-wise constant regression)
    
### Bregman divergence

- Bregman_phi = ['phi_KL']
    # Defines Bregman divergence d_phi. 
    # Values in {'phi_Eucl', 'phi_KL'} (Squared Euclidean distance, KL divergence)


### Termination Criteria

- Kmax = [100]
    # Limit in node's children. After that stop growing
- timeline_limit = 1e6
    # Limit in the number of convergent representations. (Developer Mode) 
- error_threshold = [0.01] 
    # Desired training error. 
- error_threshold_count = [3] 
    # Stop when reached 'error_threshold_count' times


### Temperature Schedule

- Tmax = [0.9] 
- Tmin = [1e-4]
    # lambda max min values in [0,1]. T = (1-lambda)/lambda
- gamma_steady = [0.8] 
    # T' = gamma * T
- gamma_schedule = [[0.1,0.5]] 
    # Initial updates can be set to reduce faster


### Tree Structure

- node_id = [0] 
    # Tree/branch parent node
- parent = None 
    # Pointer used to create tree-structured linked list


### Regularization: Perturbation and Merging

- lvq = [0] 
    # Values in {0,1,2,3}
    # 0:ODA update
    # 1:ODA until Kmax. Then switch to 2:soft clustering with no perturbation/merging 
    # 2:soft clustering with no perturbation/merging 
    # 3: LVQ update (hard-clustering) with no perturbation/merging
- regression = False
    # if regression==True, run 2nd stochastic approximation for regression: data_label = f(x) in R
- py_cut = [1e-5] 
    # Parameter e_r: threshold to find idle codevectors
- perturb_param = [1e-1] 
    # Perturb (dublicate) existing codevectors 
    # Parameter delta = d_phi(mu, mu+'delta')/T: 
- effective_neighborhood = [1e-0] 
    # Threshold to find merged (effective) codevectors
    # Parameter e_n = d_phi(mu, mu+'effective_neighborhood')/T


### Convergence 

- em_convergence = [1e-1]
- convergence_counter_threshold = [5]
    # Convergece when d_phi(mu',mu) < e_c * (1+bb_init)/(1+bb) for 'convergence_counter_threshold' times
    # Parameter e_c =  d_phi(mu, mu+'em_convergence')/T
- convergence_loops = [0]
    # Custom number of loops until convergence is considered true (overwrites e_c) (Developer mode)
- stop_separation = [1e9-1]
    # After 'stop_separation' loops, gibbs probabilities consider all codevectors regardless of class 
- bb_init = [0.9]
    # Initial bb value for stochastic approximation stepsize: 1/(bb+1)
- bb_step = [0.9]
    # bb+=bb_step

### Verbose

- keepscore = 3 
    # Values in {0,1,2,3}    
    # 0: don't compute or show score
    # 1: compute and show score only on tree node splits 
    # 2: compute score after every SA convergence and use it as a stopping criterion
    # 3: compute and show score after every SA convergence and use it as a stopping criterion

### Numba Jit

- jit = True
    # Using numba jit instead of python python for Bregman divergences

'''


'''
ODA MODEL Parameters

# Tree Structure

- self.parent
- self.children

# Status

- self.K
- self.T

- self.perturbed = False
- self.converged = False
- self.trained = False

# Variables

- self.y = []
- self.ylabels = []
- self.yext = []

- self.py = []
- self.sxpy= []
- self.slpy= []

# History

- self.myK 
- self.myT 
- self.myY 
- self.myYlabels 
- self.myYext 

- self.myTrainError 
- self.myTestError 
- self.myLoops 
- self.myTime 
- self.myTreeK 
- self.myTreeLoops 

'''

#%% The Class
    
class ODA:
    
    ###########################################################################
    # Init Function
    ###########################################################################
    
    def __init__(self,
                 # Data
                 train_data, 
                 train_labels, 
                 # Bregman divergence
                 Bregman_phi=['phi_Eucl'], # {'phi_Eucl', 'phi_KL'}
                 # Termination
                 Kmax=[100], 
                 timeline_limit = 1e6, 
                 error_threshold=[0.01], 
                 error_threshold_count=[2], 
                 # Temperature
                 Tmax=[0.9], 
                 Tmin=[1e-4],
                 gamma_schedule=[[0.1,0.5]], 
                 gamma_steady=[0.8], 
                 # Tree Structure
                 node_id=[0], 
                 parent=None, 
                 # Regularization
                 lvq=[0], # {0,1,2,3} 
                 regression=False,
                 py_cut=[1e-5],  
                 perturb_param=[1e-1], 
                 effective_neighborhood=[1e-0],
                 # Convergence
                 em_convergence=[1e-1], 
                 convergence_counter_threshold=[5],
                 convergence_loops=[0],
                 stop_separation=[1e9-1],
                 bb_init=[0.9],
                 bb_step=[0.9],
                 # Verbose
                 keepscore=3, # {0,1,2} 
                 # Python or Jit
                 jit = True
                 ):
        
        ### Tree-Structure Parameters
        
        self.id = node_id.copy()
        self.resolution = len(node_id)-1
        self.depth = len(train_data[0]) 
        self.children = []
        self.parent = parent
        self.timeline = [self.id]
        self.keepscore = keepscore
        self.regression = regression
        
        
        ### Keep archive to pass multi-resolution parameters to children
        # !!! Revisit copy vs deepcopy here
        
        # Data
        self.train_data_arxiv = train_data.copy()
        self.train_labels_arxiv = train_labels.copy()
        # Bregman divergence
        self.Bregman_phi_arxiv = [Bregman_phi[0] for d in range(self.depth)] if len(Bregman_phi)==1 else Bregman_phi.copy()
        # Termination
        self.Kmax_arxiv = [Kmax[0] for d in range(self.depth)] if len(Kmax)==1 else Kmax.copy() 
        self.error_threshold_arxiv = [error_threshold[0] for d in range(self.depth)] if len(error_threshold)==1 else error_threshold.copy()
        self.error_threshold_count_arxiv = [error_threshold_count[0] for d in range(self.depth)] if len(error_threshold_count)==1 else error_threshold_count.copy() 
        # Temperature
        self.Tmax_arxiv = [Tmax[0] for d in range(self.depth)] if len(Tmax)==1 else Tmax.copy()
        self.Tmin_arxiv = [Tmin[0] for d in range(self.depth)] if len(Tmin)==1 else Tmin.copy()
        self.gamma_schedule_arxiv = [gamma_schedule[0].copy() for d in range(self.depth)] if len(gamma_schedule)==1 else gamma_schedule.copy()
        self.gamma_steady_arxiv = [gamma_steady[0] for d in range(self.depth)] if len(gamma_steady)==1 else gamma_steady.copy()
        # Regularization
        self.lvq_arxiv = [lvq[0] for d in range(self.depth)] if len(lvq)==1 else lvq.copy()
        self.py_cut_arxiv = [py_cut[0] for d in range(self.depth)] if len(py_cut)==1 else py_cut.copy()
        self.perturb_param_arxiv = [perturb_param[0] for d in range(self.depth)] if len(perturb_param)==1 else perturb_param.copy()
        self.effective_neighborhood_arxiv = [effective_neighborhood[0] for d in range(self.depth)] if len(effective_neighborhood)==1 else effective_neighborhood.copy()
        # Convergence
        self.em_convergence_arxiv = [em_convergence[0] for d in range(self.depth)] if len(em_convergence)==1 else em_convergence.copy()
        self.convergence_counter_threshold_arxiv = [convergence_counter_threshold[0] for d in range(self.depth)] if len(convergence_counter_threshold)==1 else convergence_counter_threshold.copy()
        self.convergence_loops_arxiv = [convergence_loops[0] for d in range(self.depth)] if len(convergence_loops)==1 else convergence_loops.copy()
        self.stop_separation_arxiv = [stop_separation[0] for d in range(self.depth)] if len(stop_separation)==1 else stop_separation.copy()
        self.bb_init_arxiv = [bb_init[0] for d in range(self.depth)] if len(bb_init)==1 else bb_init.copy()
        self.bb_step_arxiv = [bb_step[0] for d in range(self.depth)] if len(bb_step)==1 else bb_step.copy()
        
        
        ### State Parameters
        
        # Codevectors        
        self.y = []
        self.ylabels = []
        self.yext = []
        self.py = []
        self.sxpy= []
        self.slpy= []
        self.old_y = []
        self.K = len(self.y)
        
        # self.classes = list(np.unique(train_labels))
        # # Create one codevector for each class known
        # for c in self.classes:
        #     cdata = [train_data[idx][self.resolution] for idx in range(len(train_data)) if np.all(train_labels[idx]==c)]
        idx = np.random.choice(range(len(train_data)))
        y_init = train_data[idx][self.resolution]
        c_init = train_labels[idx]
        self.insert_codevector(y_init, c_init)
        self.classes = [c_init]

        # Termination
        self.Kmax = Kmax[self.resolution] 
        self.Tmax = Tmax[self.resolution]
        self.Tmin = Tmin[self.resolution]
        self.gamma_schedule = gamma_schedule[self.resolution]
        self.gamma_steady = gamma_steady[self.resolution]
        if len(self.gamma_schedule)>0:
            self.gamma = self.gamma_schedule[0] 
        else:
            self.gamma = self.gamma_steady# T'=gamma*T
        self.T = self.Tmax
        self.perturbed = False
        self.converged = False
        self.convergence_counter = 1
        self.true_convergence_counter = -1
        self.trained = False
        self.error_threshold_reached = 0
        self.timeline_limit = timeline_limit
        
        # Bregman Divergence 
        self.Bregman_phi = Bregman_phi[self.resolution] # 'phi_Eucl', 'phi_KL', 'phi_IS'
        
        # Regularization
        self.py_cut = py_cut[self.resolution]
        self.lvq = lvq[self.resolution]
        self.e_p = _BregmanD(np.array(y_init),np.array(y_init+perturb_param[self.resolution]),self.Bregman_phi)            
        self.e_n = _BregmanD(np.array(y_init),np.array(y_init+effective_neighborhood[self.resolution]),self.Bregman_phi)            
        
        # Convergence parameters
        self.e_c = _BregmanD(np.array(y_init),np.array(y_init+em_convergence[self.resolution]),self.Bregman_phi)            
        self.convergence_loops = convergence_loops[self.resolution]
        self.error_threshold = error_threshold[self.resolution]
        self.error_threshold_count = error_threshold_count[self.resolution]
        self.convergence_counter_threshold = convergence_counter_threshold[self.resolution]
        self.bb_init= bb_init[self.resolution] # initial stepsize of stochastic approximation: 1/(bb+1)
        self.bb_step = bb_step[self.resolution] # bb+=bb_step
        self.separate = True
        self.stop_separation = stop_separation[self.resolution]
        self.bb = self.bb_init
        self.sa_steps = 0

        # Keep record for each temperature level
        self.myK = [self.K]
        self.myT = [self.T]
        self.myY = [self.y.copy()]
        self.myYlabels = [self.ylabels.copy()]
        self.myYext = [copy.deepcopy(self.yext)]
        self.myTrainError = [1]
        self.myTestError = [1]
        self.myLoops = [0]    
        self.myTime = [time.perf_counter()]
        self.myTreeK = [self.K]
        self.myTreeLoops = [0]
        
        # Counters
        self.last_sample = 0 # sample at which last convergence occured
        self.current_sample = 0
        self.self_regulate_counter = 0
        self.self_regulate_threshold = 5
        
        # Timestamps
        self.tik = time.perf_counter()
        self.tok = time.perf_counter()
    
        # Other
        self.plt_counter = 0 # for plots
        self.low_p_warnings = 0 # for warnings regarding probability estimates
        self.practical_zero = 1e-9 # for the log inside KL divergence
        self.jit = jit
    
    
    
    ###########################################################################
    # Training Functions
    ###########################################################################
    
    
    # Fit ODA to a Dataset (Until Stopping Criteria Reached)
    ###########################################################################
    def fit(self,train_data=[],train_labels=[],test_data=[],test_labels=[]):
        
        if len(train_data)==0 and len(self.train_data_arxiv)>0:
            train_data = self.train_data_arxiv.copy()
            train_labels = self.train_labels_arxiv.copy()
        fit_sample = 0
        datalen = len(train_data)
        
        self.myTime[-1] = time.perf_counter()
        
        ## Whlie the entire tree is not trained
        while not self.trained: 
            
            idx = fit_sample % datalen
            datum = train_data[idx].copy()
            datum_label = train_labels[idx]
            
            ## Train this or children. train_step() is recursive.
            self.train_step(datum, datum_label, 
                            train_data=train_data, train_labels=train_labels, 
                            test_data=test_data, test_labels=test_labels)
    
            fit_sample += 1
    
    
    # Train ODA given a set of data (Use Only Once) 
    # May Terminate Before Stopping Criteria Reached
    # Can be used for Online Training (One Data Point per Call)
    ###########################################################################
    def train(self,train_data,train_labels,test_data=[],test_labels=[]):
        
        self.myTime[-1]  = time.perf_counter()
        
        ## Whlie the entire tree is not trained
        for idx in range(len(train_data)): 
            
            datum = train_data[idx].copy()
            datum_label = train_labels[idx]
            
            ## Train this or children. train_step() is recursive.
            self.train_step(datum, datum_label, 
                            train_data=self.train_data_arxiv, train_labels=self.train_labels_arxiv, 
                            test_data=test_data, test_labels=test_labels)
    
    # Training Step
    ###########################################################################
    def train_step(self, datum, datum_label=0, 
                   train_data=[],train_labels=[], test_data=[],test_labels=[]):
        
        ## For Debugging
        stop_timeline,len_timeline = self.check_timeline_limit()
        if stop_timeline:
            self.trained=True
            
        if not self.trained:
            
            if len(self.children)>0:
                ## check if all children trained
                self.check_trained()
                
                if not self.trained:
                    
                    ## find the winner child
                    j,_ = self.winner(np.array(self.y), np.array(datum[self.resolution]))
                    
                    ## Recursively call train_step(). Each child will use its own resolution of the data.
                    self.children[j].train_step(datum,datum_label,
                                                train_data,train_labels, 
                                                test_data,test_labels)
                else:
                    ## if just trained, report time
                    self.tok = time.perf_counter()
                    self.myTime.append(self.tok)
            
            else: ## Otherwise, train this cell
                    
                ## Insert class if not known so far
                if (not self.regression) and (datum_label not in self.classes):
                    self.classes.append(datum_label)
                    self.insert_codevector(datum[self.resolution], datum_label)
                
                self.current_sample += 1
                ## insert perturbations and initialize SA stepsizes
                if not self.perturbed:
                    if self.lvq<2:
                        self.perturb()
                    else: ## set it true without perturbing
                        self.perturbed = True
                    self.converged = False     
                    self.bb = self.bb_init
                    self.sa_steps = 0
                
                ## Stochastic Approximation Step
                self.sa_step(datum, datum_label)
                                
                ## Check Convergence
                self.check_convergence()
                
                if self.converged:
                    self.tok = time.perf_counter()
                    
                    ## Find effective codevectors
                    if self.lvq<2:
                        self.find_effective_clusters() 
                    if self.lvq<2 and not self.regression:
                        self.pop_idle_clusters() 
                    self.prune_siblings()  
                    
                    ## If Kmax reached, keep the last set of codevectors
                    stop_K = self.K>self.Kmax
                    if stop_K:
                        
                        self.overwrite_codevectors(self.myY[-1].copy(), self.myYlabels[-1].copy(),copy.deepcopy(self.myYext[-1]))
                        self.T = self.myT[-1]
                        
                        # self.current_sample = self.myLoops[-1]
                        if self.lvq == 1:
                            # self.find_effective_clusters() 
                            # self.pop_idle_clusters() 
                            self.lvq = 2
                            print(f'--- Keeping K={self.K} codevectors.')
                    
                    self.myT.append(self.T)
                    self.myK.append(self.K) 
                    self.myY.append(self.y.copy())
                    self.myYlabels.append(self.ylabels.copy())
                    self.myYext.append(copy.deepcopy(self.yext))
                    self.myLoops.append(self.current_sample)
                    self.myTime.append(self.tok)
                    self.put_in_timeline(self.id.copy())
                    
                    ## Check criteria to stop training self and split, if possible
                    stop_T = self.myT[-1]<=self.Tmin
                    stop_timeline,len_timeline = self.check_timeline_limit()
                    # Keep score
                    stop_error=False
                    # Compute score to be used as stopping criterion
                    if self.keepscore >1:
                        d_train = self.score(train_data,train_labels)
                        self.myTrainError.append(d_train)
                        if d_train<self.error_threshold:
                            self.confirm_error_threshold()
                        self.error_threshold_reached = self.check_error_threshold()
                        stop_error = self.error_threshold_reached>self.error_threshold_count
                        if len(test_data)>0:
                            d_test = self.score(test_data,test_labels)
                            self.myTestError.append(d_test)
                    
                    # Show score
                    if self.keepscore>2:
                        
                        tK = self.treeK()
                        tL = self.treeLoops()
                        print(f'{len_timeline} -- ID: {self.id}: '+ 
                              f'Samples: {tL}(+{self.current_sample-self.last_sample}): ' +
                              f'T = {self.myT[-1]:.4f}, K = {self.myK[-1]}, treeK = {tK}, t = {self.myTime[-1]-self.myTime[0]:.1f} [+{self.myTime[-1]-self.myTime[-2]:.1f}s]')
                        if not stop_K:
                            print(f'Train Error: {d_train:.4f}')
                            if len(test_data)>0:
                                print(f'Test Error: {d_test:.4f}')
                            if d_train<self.error_threshold:
                                print('*** Training Error threshold reached. ***')
                            
                    
                    ## if reached minimum temperature or desired score or maximum tree nodes
                    if (stop_K and self.lvq==0) or stop_T or stop_error or stop_timeline:
                        
                        ## Print State
                        if self.keepscore==1:
                            tK = self.treeK()
                            tL = self.treeLoops()
                            print(f'{len_timeline} -- ID: {self.id}: '+
                                    f'Samples: {tL}(+{self.current_sample-self.last_sample}): ' +
                                    f'T = {self.myT[-1]:.4f}, K = {self.myK[-1]}, treeK = {tK}, t = {self.myTime[-1]-self.myTime[0]:.1f} [+{self.myTime[-1]-self.myTime[-2]:.1f}s]')
                            
                            d_train = self.score(train_data,train_labels)
                            self.myTrainError.append(d_train)
                            print(f'Train Error: {d_train:.3f}')
                            if len(test_data)>0:
                                d_test = self.score(test_data,test_labels)
                                self.myTestError.append(d_test)
                                print(f'Test Error: {d_test:.3f}')
                        
                        if stop_K and self.lvq==0:
                            print('--- Maximum number of codevectors reached. ')
                        if stop_T:
                            print('--- Minimum temperature reached. ---')
                        if stop_error:
                            print('--- Minimum error reached. ---')
                        if stop_timeline:
                            print('--- Maximum number of nodes reached. ---')
                        
                        self.check_untrained_siblings()
                        
                        ## split (if possible)
                        if self.resolution+1<self.depth and len(self.myY[-1])>1: 
                            self.split(datum,datum_label)
                            self.reset_error_threshold()
                            if self.keepscore>0:
                                print(f'ID: {self.id}: Trained. Splitting..')
                        ## or declare trained
                        else:
                            self.trained = True
                            # if len(self.myY)>1:
                            if self.keepscore>0:
                                print(f'ID: {self.id}: Trained')
                            # self.tok = time.perf_counter()
                            # self.myTime.append(self.tok-self.tik)
                    
                    self.last_sample = self.current_sample
    
                    self.update_T()
                    self.perturbed=False
                    
                    # self.tik = time.perf_counter()
        
    # Stochastic Approximation Step
    ###########################################################################
    def sa_step(self, datum, datum_label):
        
        self.old_y = self.y.copy()
        self.low_p_warnings = 0
        datum = datum[self.resolution]
        
        if self.lvq < 3:
            
            for i in range(self.K): ## for all centroids
                
                if self.jit:

                    self.py[i], self.sxpy[i], self.y[i], self.slpy[i], self.ylabels[i] = _sa_update(i, 
                        np.array(self.y), np.array(self.ylabels), 
                        np.array(self.py), np.array(self.sxpy), np.array(self.slpy), 
                        np.array(datum), np.array(datum_label), self.regression, 
                        self.separate, self.T, 
                        self.bb, phi=self.Bregman_phi)

                else:
                    
                    el_inv = (1-self.T)/self.T
                    d = [self.BregmanD(datum,self.y[k]) for k in range(len(self.y))]
                    py = self.py.copy()
                    if self.separate and not self.regression:
                        py = [self.py[i] if datum_label==self.ylabels[i] else 0 for i in range(len(self.py))]
                    pyx_sum = np.dot(py,[np.exp(-dj*el_inv) for dj in d])
                    if pyx_sum == 0: # e.g. if no codevectors of the same class as observation
                        pyx = 0 # or break
                    else:    
                        pyx = py[i]*np.exp(-d[i]*el_inv)/pyx_sum
                    # SA update
                    self.py[i] = self.py[i] + 1/(self.bb+1)*(pyx - self.py[i])
                    self.sxpy[i] = self.sxpy[i] + 1/(self.bb+1)*(pyx*datum - self.sxpy[i])
                    self.y[i] = self.sxpy[i]/self.py[i]  
                    if self.regression:
                        self.slpy[i] = self.slpy[i] + 1/(self.bb+1)*(pyx*datum_label - self.slpy[i])
                        self.ylabels[i] = self.slpy[i]/self.py[i]
                                        
                # Warning sign
                # if pyx!=pyx: # or pyx<1e-32:
                #     self.low_p_warnings+=1

            # Warning
            # if self.low_p_warnings == len(self.y):
            #     # print(f'Number of Idle codevectors: {self.low_p_warnings}/{self.K}')
            #     self.low_p_warnings = 0
            #     print(f'WARNING: Conditional Probabilities too small.')
            #     # +
            #     #               f'You may need to consider scaling the input space,'+
            #     #               f'or using different initial values.')
            
        else:
                
            if self.jit:
                
                self.py[i], self.sxpy[i], self.y[i], self.slpy[i], self.ylabels[i] = _lvq_update(i, 
                    np.array(self.y), np.array(self.ylabels), 
                    np.array(self.py), np.array(self.sxpy), np.array(self.slpy), 
                    np.array(datum), np.array(datum_label), self.regression,
                    self.separate, self.T, 
                    self.bb, phi=self.Bregman_phi)
        
            else:
                
                d = [self.BregmanD(datum,self.y[k]) for k in range(len(self.y))]
                j = np.argmin(d)
                s = 1 if (self.ylabels[j]==datum_label or self.regression) else -1
                # LVQ Gradient Descent Update
                self.y[j] = self.y[j] - 1/(self.bb+1) * s * self.dBregmanD(datum,self.y[j])
                self.ylabels[j] = self.ylabels[j] - 1/(self.bb+1) * s * self.dBregmanD(datum_label,self.ylabels[j])
                
        
        self.bb += self.bb_step 
        self.sa_steps += 1
    
            
            
    # Load ODA Model
    # TODO: Extend to go back in timeline 
    ###########################################################################
    def load(self, T=[]):
        
        self.perturbed = False
        self.converged = False
        self.trained = False
        self.bb = self.bb_init
        
        if T!=[]:
            self.T = T
        
        self.tik = time.perf_counter()
        self.tok = time.perf_counter()            
        
        if len(self.children)>0:
            for child in self.children:
                child.load()
        
            
            
    ###########################################################################
    ### Low-Level ODA Functions
    ###########################################################################
    
    
    # Check Convergence
    ###########################################################################
    def check_convergence(self):
        
        ## if predefined number of loops
        if self.convergence_loops>0 and self.sa_steps>=self.convergence_loops:
            if self.convergence_counter > self.convergence_counter_threshold:
                self.converged = True
                self.convergence_counter = 1
                self.true_convergence_counter +=1
                if self.true_convergence_counter>self.stop_separation:
                    self.separate=False
            else:
                self.convergence_counter += 1
        else:
            
            conv_reached = np.all([self.BregmanD(np.array(self.old_y[i]),np.array(self.y[i])) < \
                            self.el()*self.e_c * (1+self.bb_init)/(1+self.bb)
                                                            for i in range(self.K)])  
                
            if conv_reached:
                
                if self.convergence_counter > self.convergence_counter_threshold:
                    self.converged = True
                    self.convergence_counter = 1
                    self.true_convergence_counter +=1
                    if self.true_convergence_counter>self.stop_separation:
                        self.separate=False
                else:
                    self.convergence_counter += 1
               
                
    # Perturb Codevectors
    ###########################################################################
    def perturb(self):
        ## insert perturbations of all effective yi
        for i in reversed(range(self.K)):
            # new_yi = self.y[i] + self.perturb_param*2*(np.random.rand(len(self.y[i]))-0.5)
            new_yi = self.y[i] + self.el()*self.e_p * 2 * (np.random.rand(len(self.y[i]))-0.5)
            self.py[i] = self.py[i]/2.0
            self.sxpy[i] = self.py[i]*self.y[i]
            self.slpy[i] = self.py[i]*self.ylabels[i]
            self.y.append(new_yi)
            self.ylabels.append(self.ylabels[i]) 
            self.yext.append(copy.deepcopy(self.yext[i])) 
            self.py.append(self.py[i])
            self.sxpy.append(self.py[i]*new_yi)
            self.slpy.append(self.py[i]*self.ylabels[i])
        self.K = len(self.y)
        self.perturbed = True
    
    
    # Update Temperature (lambda)
    ###########################################################################
    def update_T(self):
        
        if self.true_convergence_counter < len(self.gamma_schedule):
            self.gamma = self.gamma_schedule[self.true_convergence_counter]
        else:
            self.gamma = self.gamma_steady
        
        self.T = self.gamma * self.T
      
    
    # Compute 1/T = lambda/(1-lambda)
    ###########################################################################
    def el(self):
        return self.T/(1-self.T)
    
    
    # Find Effective Codevectors
    ###########################################################################
    def find_effective_clusters(self):
        i=0
        while i<self.K:
            for j in reversed(np.arange(i+1,self.K)):
                
                if not self.regression:
                    merged = self.BregmanD(np.array(self.y[i]),np.array(self.y[j]))< \
                            self.el()*self.e_n and self.ylabels[i]==self.ylabels[j]
                else:
                    merged = self.BregmanD(np.array(self.y[i]),np.array(self.y[j]))< \
                            self.el()*self.e_n 
                            
                if merged:
                    
                    self.py[i] = self.py[i]+self.py[j]
                    self.sxpy[i] = self.y[i]*self.py[i]
                    self.slpy[i] = self.ylabels[i]*self.py[i]
                    self.y.pop(j)
                    self.ylabels.pop(j)
                    self.yext.pop(j)
                    self.py.pop(j)
                    self.sxpy.pop(j)
                    self.slpy.pop(j)
                    self.K-=1
            
            i+=1
    
    
    # Insert Codevector
    ###########################################################################
    def insert_codevector(self,datum,datum_label,datum_py=[],datum_ext=[],norm=False): 
        self.y.append(datum)
        self.ylabels.append(datum_label)
        self.yext.append(copy.deepcopy(datum_ext))
        if datum_py==[]:
            self.py.append(1.0)
        else:
            self.py.append(datum_py)
        self.sxpy.append(self.y[-1]*self.py[-1])    
        self.slpy.append(self.ylabels[-1]*self.py[-1])    
        self.K = len(self.y)

        if norm:
            self.py = [float(p)/sum(self.py) for p in self.py]
            self.sxpy= [self.y[i]*self.py[i] for i in range(len(self.y))]    
            self.slpy= [self.ylabels[i]*self.py[i] for i in range(len(self.ylabels))]    

    # Remove Codevector
    ###########################################################################
    def pop_codevector(self,idx,norm=False): 
        self.y.pop(idx)
        self.ylabels.pop(idx)
        self.yext.pop(idx)
        self.py.pop(idx)
        self.sxpy.pop(idx)
        self.slpy.pop(idx)
        self.K = len(self.y)

        if norm:
            self.py = [float(p)/sum(self.py) for p in self.py]
            self.sxpy= [self.y[i]*self.py[i] for i in range(len(self.y))]   
            self.slpy= [self.ylabels[i]*self.py[i] for i in range(len(self.ylabels))]   
        
        
    # Discard Idle Codevectors
    ###########################################################################
    def pop_idle_clusters(self):
        i = 0
        py_cut = self.py_cut
        while i < len(self.y):
            ## if the only representatitve of its class make it harder to be pruned
            # yli = self.ylabels.copy()
            # yli.pop(i)
            # if len(yli)>0 and np.any(np.array(self.ylabels[i])==np.array(yli)):
            #     py_cut = self.py_cut**2
            # prune idle codevector
            if self.py[i]<py_cut:
                self.pop_codevector(i)
                if self.keepscore>2:
                    print('*** Idle Codevector Pruned ***')
            else:
                i+=1
    
    # Overwrite Existing Codevectors (for External Use)
    ###########################################################################
    def overwrite_codevectors(self,new_y,new_ylabels,new_ext,new_py=[]): # new_y must be a list
        self.y = new_y
        self.ylabels = new_ylabels
        if new_ext!=[]:
            self.yext = copy.deepcopy(new_ext)
        if new_py==[]:
            self.py = [1.0 / len(self.y) for i in range(len(self.y))] 
        else:
            self.py = new_py
        self.sxpy= [self.y[i]*self.py[i] for i in range(len(self.y))]    
        self.slpy= [self.ylabels[i]*self.py[i] for i in range(len(self.ylabels))]    
        self.K = len(self.y)
    
    ###########################################################################
    ### Tree-Structure Functions
    ###########################################################################
    
    
    # Split Cell: Create Children ODA nodes
    ###########################################################################
    def split(self,datum,datum_label):
        
        for i in range(len(self.myY[-1])):
        
            self.children.append(ODA(
                # Data
                train_data=[datum], 
                train_labels=[datum_label], 
                # Bregman divergence
                Bregman_phi=self.Bregman_phi_arxiv, 
                # Termination
                Kmax=self.Kmax_arxiv,
                timeline_limit = self.timeline_limit,
                error_threshold=self.error_threshold_arxiv,
                error_threshold_count=self.error_threshold_count_arxiv,
                # Temperature
                Tmax=self.Tmax_arxiv,
                Tmin=self.Tmin_arxiv,
                gamma_schedule=self.gamma_schedule_arxiv,
                gamma_steady=self.gamma_steady_arxiv,
                # Tree Structure
                node_id=self.id+[i],
                parent=self,
                # Regularization
                lvq=self.lvq_arxiv, 
                regression=self.regression,
                py_cut=self.py_cut_arxiv,
                perturb_param=self.perturb_param_arxiv, 
                effective_neighborhood=self.effective_neighborhood_arxiv, 
                # Convergence
                em_convergence=self.em_convergence_arxiv, 
                convergence_counter_threshold=self.convergence_counter_threshold_arxiv,
                convergence_loops=self.convergence_loops_arxiv,
                stop_separation=self.stop_separation_arxiv,
                bb_init=self.bb_init_arxiv,
                bb_step=self.bb_step_arxiv,
                # Verbose
                keepscore=self.keepscore,
                jit = self.jit)
                )
            
            self.children[-1].yext=[copy.deepcopy(self.yext[i])]
    
    # Calculate Nodes of the Tree
    ###########################################################################
    def treeK(self,sub=False):
        
        if sub:
            if len(self.children)>0:
                return np.sum([child.treeK(sub=True) for child in self.children])
            else:
                return self.myK[-1]
        else:
            node = self
            while node.parent:
               node = node.parent
            tK = node.treeK(sub=True)
            node.myTreeK.append(tK)
            return tK
        
    # Calculate Loops of the Tree
    ###########################################################################
    def treeLoops(self,sub=False):
        
        if sub:
            if len(self.children)>0:
                return self.myLoops[-1] + np.sum([child.treeLoops(sub=True) for child in self.children])
            else:
                return self.myLoops[-1]
        else:
            node = self
            while node.parent:
               node = node.parent
            tL = node.treeLoops(sub=True)
            node.myTreeLoops.append(tL)
            return tL
        
    # Check if Node and All Children are Trained
    ###########################################################################
    def check_trained(self):
        if len(self.children)>0:
            Ts = [child.check_trained() for child in self.children]
            if np.all(Ts):
                self.trained=True
        return self.trained
    
    
    # Put Node ID in Root's Timeline
    ###########################################################################
    def put_in_timeline(self, my_id):
        if self.parent:
            self.parent.timeline.append(my_id)
            self.parent.put_in_timeline(my_id)
        else:
            if not len(self.children)>0:
                self.timeline.append(my_id)
    
    
    # Check if Timeline Limit is Reached
    ###########################################################################
    def check_timeline_limit(self):
        if self.parent:
            return self.parent.check_timeline_limit()
        else:
            return self.timeline_limit < len(self.timeline), len(self.timeline)
    
    
    # Prune Siblings if not Trained at all
    ###########################################################################
    def prune_siblings(self):
        
        self.self_regulate_counter += 1
        
        if self.self_regulate_counter > self.self_regulate_threshold:
        
            self.self_regulate_counter = 0
            
            # Compress Redundant Codevectors (Classification Only)        
            if self.parent:
                if (not self.regression) and len(np.unique(self.parent.myYlabels[-1]))>1:
                    if len(np.unique(self.ylabels))==1:
                        self.overwrite_codevectors([self.y[0]],[self.ylabels[0]],[copy.deepcopy(self.yext[0])])
                        self.trained = True
                        if self.keepscore>2:
                            print('*** Same-Class Codevectors Pruned ***')
    
            # Find idle siblings and prune them
            if self.parent:
                for sibling in self.parent.children:
                    if sibling.current_sample == 0:
                        sibling.overwrite_codevectors([sibling.y[0]],[sibling.ylabels[0]],[copy.deepcopy(sibling.yext[0])])
                        sibling.trained = True
                        if self.keepscore>2:
                            print('*** Idle Sibling Pruned ***')
    
    # Check Untrained Siblings
    ###########################################################################
    def check_untrained_siblings(self):
        
        # Find idle siblings and prune them
        if self.parent:
            for sibling in self.parent.children:
                if sibling.current_sample == 0:
                    sibling.overwrite_codevectors([sibling.y[0]],[sibling.ylabels[0]],[copy.deepcopy(sibling.yext[0])])
                    sibling.trained = True
                    if self.keepscore>2:
                        print('*** Idle Sibling Pruned ***')
                    
                        
    # Increase Counter for Desired Error Reached 
    ###########################################################################
    def confirm_error_threshold(self):
        if self.parent:
            self.parent.confirm_error_threshold()
        else:
            self.error_threshold_reached += 1
    
    
    # Read Counter for Desired Error Reached
    ###########################################################################
    def check_error_threshold(self):
        if self.parent:
            return self.parent.check_error_threshold()
        else:
            return self.error_threshold_reached
    
    # Reset Counter for Desired Error Reached 
    ###########################################################################
    def reset_error_threshold(self):
        if self.parent:
            self.parent.reset_error_threshold()
        else:
            self.error_threshold_reached = 0
    
    
                
    
    ###########################################################################
    ### Score Functions
    ###########################################################################
    
    # Compute Score
    ###########################################################################
    def score(self, data, labels, recursive=1e3):
        if self.parent:
            return self.parent.score(data, labels)
        else:
            classification = True if len(self.classes)>1 else False
            d = 0.0
            for i in range(len(data)):
                if self.regression:
                    d += self.datum_regression_error(data[i], labels[i], recursive)         
                elif classification:
                    d += self.datum_classification_error(data[i], labels[i], recursive)
                else:
                    d += self.datum_dissimilarity(data[i], labels[i], recursive)         
            return d/len(data) #if classification else d      
    
    
    # Compute Dissimilarity between Codebook and Input Vector
    ###########################################################################
    def datum_dissimilarity(self, datum, label, recursive=1e3):
    
        y = self.myY[-1]
        j,d = self.winner(np.array(y), np.array(datum[self.resolution]))
        
        if recursive>0 and len(self.children) > 0 and len(self.children[j].myY)>1:
            return self.children[j].datum_dissimilarity(datum,label,recursive-1)
        else:
            return d   
    
    # Find best representative
    ###########################################################################
    def represent(self, datum, recursive=1e3):
    
        y = self.myY[-1]
        j,d = self.winner(np.array(y), np.array(datum[self.resolution]))
        
        if recursive>0 and len(self.children) > 0 and len(self.children[j].myY)>1:
            return self.children[j].represent(datum,recursive-1)
        else:
            return y[j]   
    
    # Return Codebook
    ###########################################################################
    def codebook(self, recursive=1e3):
    
        if recursive>0 and len(self.children) > 0:
            cb = []
            cbl = []
            for child in self.children:
                if len(child.myY)>1:
                    tcb,tcbl = child.codebook(recursive-1)
                    cb = cb + tcb
                    cbl = cbl + tcbl
            return cb, cbl
        else:
            return self.myY[-1], self.myYlabels[-1]   
    
    
    # Compute Classification Error between Codebook and Input Vector 
    ###########################################################################
    def datum_classification_error(self, datum, label, recursive=1e3):
        
        y = self.myY[-1]
        j,_ = self.winner(np.array(y), np.array(datum[self.resolution]))
        
        ## if I have children and the winner child has converged at least once
        if recursive>0 and len(self.children)>0 and len(self.children[j].myY)>1: 
            return self.children[j].datum_classification_error(datum,label,recursive-1)
        else:
            decision_label = self.myYlabels[-1][j]
            d = 0 if np.all(decision_label == label) else 1
            return d      
    
    # Compute Regression Error between Codebook and Input Vector
    ###########################################################################
    def datum_regression_error(self, datum, label, recursive=1e3):
    
        y = self.myY[-1]
        j,d = self.winner(np.array(y), np.array(datum[self.resolution]))
        
        if recursive>0 and len(self.children) > 0 and len(self.children[j].myY)>1:
            return self.children[j].datum_dissimilarity(datum,label,recursive-1)
        else:
            decision_label = self.myYlabels[-1][j]
            d = self.BregmanD(label, decision_label) 
            return d
        
    # Predict 
    ###########################################################################
    def predict(self, datum, recursive=1e3):
        
        y = self.myY[-1]
        j,_ = self.winner(np.array(y), np.array(datum[self.resolution]))
        
        ## if I have children and the winner child has converged at least once
        if recursive>0 and len(self.children)>0: 
            if len(self.children[j].myY)>1: 
                return self.children[j].predict(datum,recursive-1)
            else:
                decision_label = self.myYlabels[-1][j]
                return decision_label
        else:
            decision_label = self.myYlabels[-1][j]
            return decision_label    
    
    # Voronoi
    ###########################################################################
    def voronoi(self,ctr=-1,points=[],recursive=0,child_ctr=[],add_parent=True):
        
        def proj_back(x):
            u = np.array([1,1])
            u = u/np.linalg.norm(u)
            x = np.array(x)
            return np.array(u*x)
        
        centroids = self.myY[ctr]
        if len(centroids[0])<2:
            centroids = [proj_back(cd) for cd in centroids]

        if len(points)==0:
            delta=0.005
            xm = np.arange(0.0, 1.0, delta)
            ym = np.arange(0.0, 1.0, delta)
            points = [np.array([xi,yi]) for xi in xm for yi in ym]
        
        edges = []
        regions = [[] for i in range(len(centroids))]
        for di in points:
            dists = [self.BregmanD(di,np.array(yj)) for yj in centroids]
            idx = np.argsort(dists)
            j = idx[0]
            regions[j].append(di)
            if len(dists)>1 and np.abs(dists[idx[0]]-dists[idx[1]])<1/4*1e-0*self.myT[ctr]:
                edges.append(di)
        
        if not add_parent:
            edges = []
        
        if recursive>0 and len(self.children)>0:
            if len(child_ctr)==0:
                child_ctr = [-1 for i in range(len(self.children))]
            for i in range(len(self.children)):
                child = self.children[i]
                edges=edges + child.voronoi(ctr=child_ctr[i],points=regions[i],recursive=recursive-1)
        
        return edges
    
    # Bregman Divergences (Python Implementation)
    ###########################################################################
    def BregmanD(self,x, y):
        
        if self.jit:
            
            d = _BregmanD(x,y,self.Bregman_phi)
            
        else:
        
            pzero = self.practical_zero
            d = 0.0
            if self.Bregman_phi == 'phi_Eucl':
                d = np.dot(x-y,x-y)
                # d = (x-y).dot(np.array([[0.1,0],[0,1.0]])).dot(x-y)
            elif self.Bregman_phi == 'phi_KL':
                x[x<pzero] =pzero 
                y[y<pzero] =pzero    
                d = np.dot(x,np.log(x)-np.log(y)) - np.sum(x-y)
        
        return d

    
    # Bregman Divergence Derivatives (Python Implementation)
    ###########################################################################
    def dBregmanD(self,x, y):
        
        if self.jit:
            
            dd = _dBregmanD(x,y,self.Bregman_phi)
            
        else:
        
            pzero = self.practical_zero
            dd = 0.0
            if self.Bregman_phi == 'phi_Eucl':
                dd = -2 * (x-y)
            elif self.Bregman_phi == 'phi_KL':
                x[x<pzero] =pzero 
                y[y<pzero] =pzero 
                diag = np.diag([1/y[i] for i in range(len(y))])
                dd = - np.dot(diag,(x-y))
        
        return dd
    
    
    # Find Winner Codevector
    ###########################################################################
    def winner(self, y, datum):
        
        if self.jit:
            j,d = _winner(y, datum, phi=self.Bregman_phi)
        else:
            dists = [self.BregmanD(datum,yj) for yj in y]
            j = np.argmin(dists)
            d = dists[j]
            
        return j, d
    
    
    
    ###########################################################################    
    ### Plotting Functions
    ###########################################################################
    

    def plot_curve(self,figname,show = False,save = False,
                    # Parameters
                    fig_size=(8,6),
                    font_size = 28,
                    label_size = 20,
                    legend_size = 32,
                    line_width = 12,
                    marker_size = 8,
                    fill_size=10,
                    line_alpha = 0.6,
                    txt_size = 32,
                    txt_x = 1.0,
                    txt_y = 0.03,
                    font_weight = 'bold', 
                    ylim = 0.5
                   ):
        
        # Variables        
        
        idx = []
        myK = []
        tK=[]
        tL=[]
        myT = []
        myY = []
        myYlabels = []
        myTrainError = []
        myTestError = []
        myLoops = []    
        myTime = []
        
        ## Read results from timeline ##
        
        ## Initialize plot counters
        for i in range(len(self.timeline[1:])):
            nid = self.timeline[i+1]
            node = self
            for child_id in nid[1:]:
                node=node.children[child_id]   
            node.plt_counter = 1 # init to 1 not 0: read after first convergence
            idx.append(i+1)
        
        ## Load results from timeline
        for i in range(len(self.timeline[1:])):
            nid = self.timeline[i+1]
            node = self
            for child_id in nid[1:]:
                node=node.children[child_id]    
            
            myK.append(node.myK[node.plt_counter])
            tK.append(self.myTreeK[i+1])
            tL.append(self.myTreeLoops[i+1])
            myT.append(node.myT[node.plt_counter])
            myY.append(node.myY[node.plt_counter])
            myYlabels.append(node.myYlabels[node.plt_counter])
            myTrainError.append(node.myTrainError[node.plt_counter])
            myTestError.append(node.myTestError[node.plt_counter])
            myLoops.append(node.myLoops[node.plt_counter])   
            myTime.append(node.myTime[node.plt_counter])
            
            node.plt_counter += 1
        
        import matplotlib.colors as mcolors
        colors = mcolors.TABLEAU_COLORS
        colors = list(colors.keys())
        markers = ['s','D','o','X','P']    
        
        #######################################################################
        # Performance VS Time
        #######################################################################
        
        fig,ax = plt.subplots(figsize=fig_size,tight_layout = {'pad': 1})
        
        # Label axes
        # ax.set_ylim(-0.05,ylim+0.01)
        ax.set_xlabel('time (s)', fontsize = font_size)
        ylabel = '% error' if len(self.classes)>1 else 'distortion (ave.)'
        ax.set_ylabel(ylabel, fontsize = font_size)
        ax.tick_params(axis='both', which='major', labelsize=label_size)
        ax.tick_params(axis='both', which='minor', labelsize=8)

        x= [t - myTime[0] for t in myTime]
        y=myTrainError
        clr=colors[0]
        mrkr = markers[0]
        ax.plot(x, y, label='Train', 
          color=clr, marker=mrkr,linestyle='solid', 
          linewidth=line_width, markersize=marker_size,alpha=line_alpha)
                
        y=myTestError
        clr=colors[1]
        mrkr = markers[1]
        ax.plot(x, y, label='Test', 
          color=clr, marker=mrkr,linestyle='solid', 
          linewidth=line_width, markersize=marker_size,alpha=line_alpha)

        plt.legend(loc='lower left',prop={'size': legend_size})
        
        y=tL
        clr=colors[2]
        mrkr = markers[2]
        ax2 = ax.twinx()
        ax2.plot(x, y, label='Samples', 
          color=clr, marker=mrkr,linestyle='solid', 
          linewidth=line_width, markersize=marker_size,alpha=line_alpha/5)
        ax2.set_ylabel('Samples', fontsize = font_size)
        ax2.tick_params(axis='both', which='major', labelsize=label_size)
        ax2.tick_params(axis='both', which='minor', labelsize=8)
        
        plt.grid(color='gray', linestyle='-', linewidth=1, alpha = 0.1)
        plt.legend(loc='upper right',prop={'size': legend_size})
    
        if save:
            fig.savefig(figname+'errorVStime.png', format = 'png')
        if show:
            plt.show()
        else:
            plt.close()
        
        #######################################################################
        # Performance VS K
        #######################################################################
        
        fig,ax = plt.subplots(figsize=fig_size,tight_layout = {'pad': 1})
        
        # Label axes
        # ax.set_ylim(-0.05,ylim+0.01)
        ax.set_xlabel(r'$-\log(1/T)$', fontsize = font_size)
        ylabel = '# K'
        ax.set_ylabel(ylabel, fontsize = font_size)
        ax.tick_params(axis='both', which='major', labelsize=label_size)
        ax.tick_params(axis='both', which='minor', labelsize=8)
    
        x= [-np.log10(t) for t in myT]
        y=tK
        clr=colors[2]
        mrkr = markers[2]
        ax.plot(x, y, label='K', 
          color=clr, marker=mrkr,linestyle='solid', 
          linewidth=line_width, markersize=marker_size,alpha=line_alpha)
        
        plt.legend(loc='upper right',prop={'size': legend_size})
        
        y=myTrainError
        clr=colors[0]
        mrkr = markers[0]
        ax2 = ax.twinx()
        ax2.plot(x, y, label='Train', 
          color=clr, marker=mrkr,linestyle='solid', 
          linewidth=line_width, markersize=marker_size,alpha=line_alpha)
        y=myTestError
        clr=colors[1]
        mrkr = markers[1]
        ax2.plot(x, y, label='Test', 
          color=clr, marker=mrkr,linestyle='solid', 
          linewidth=line_width, markersize=marker_size,alpha=line_alpha)
        ylabel = '% error' if len(self.classes)>1 else 'distortion (ave.)'
        ax2.set_ylabel(ylabel, fontsize = font_size)
        ax2.tick_params(axis='both', which='major', labelsize=label_size)
        ax2.tick_params(axis='both', which='minor', labelsize=8)
        
        plt.grid(color='gray', linestyle='-', linewidth=1, alpha = 0.1)
        plt.legend(loc='lower left',prop={'size': legend_size})
    
        if save:
            fig.savefig(figname+'TvsK.png', format = 'png')
        if show:
            plt.show()
        else:
            plt.close()
        
        return ax
        
#%% ###########################################################################    
### Numba Jit Functions
###############################################################################


# Find Winner Codevector
###############################################################################
@njit(cache=True,nogil=True)
def _winner(y, datum, phi='phi_KL'):
    dists = np.zeros(len(y))
    for i in range(len(y)):
        dists[i]=_BregmanD(datum,y[i],phi)
    j = np.argmin(dists)
    return j, dists[j]


# Compute Bregman Divergence
###############################################################################
@njit(cache=True,nogil=True)
def _BregmanD(x, y, phi='phi_KL'):
    if phi == 'phi_Eucl':
        d = _dot(x-y,x-y)
    elif phi == 'phi_KL':
        pzero = 1e-9
        logx = np.zeros_like(x)
        logy = np.zeros_like(y)
        sxy=0
        for i in range(len(x)):
            if x[i]<pzero:
                x[i]=pzero
            if y[i]<pzero:
                y[i]=pzero    
            logx[i] = np.log(x[i])
            logy[i] = np.log(y[i])
            sxy += x[i]-y[i]
        d = _dot(x,logx-logy) - sxy
    return d


# Compute Bregman Divergence Derivative
###############################################################################
@njit(cache=True,nogil=True)
def _dBregmanD(x, y, phi='phi_KL'):
    if phi == 'phi_Eucl':
        dd = -2*(x-y)
    elif phi == 'phi_KL':
        pzero = 1e-9
        dd = np.zeros([len(x)])
        for i in range(len(x)):
            if x[i]<pzero:
                x[i]=pzero
            if y[i]<pzero:
                y[i]=pzero    
            dd[i] = - 1/y[i] * (x[i]-y[i])
    return dd


# Compute Dot Product
###############################################################################
@njit(cache=True,nogil=True)
def _dot(x, y):
    s = 0.0
    for i in range(len(x)):
        s += x[i]*y[i]
    return s
        
# Compute Weighted Dot Product
###############################################################################
@njit(cache=True,nogil=True)
def _wdot(x, y):
    s = 0.0
    aa = [1,1.0]
    for i in range(len(x)):
        s += aa[i]*x[i]*y[i]
    return s


# Stochastic Approximation Update
###############################################################################
@njit(cache=True)
def _sa_update(idx, y, ylabels, py, sxpy, slpy, datum, datum_label, regression, sep, T, bb, phi='phi_KL'):
    
    pzero = 1e-9
    
    selfpy = py[idx]
    selfsxpy = sxpy[idx]
    selfslpy = slpy[idx]
    
    el_inv = (1-T)/T
    
    if sep and not regression:
        for i in range(len(py)):
            if datum_label != ylabels[i]:
                py[i]=0
    
    dists = np.zeros(len(y))
    gibbs = np.zeros(len(y))
    
    for i in range(len(y)):
        dists[i]=_BregmanD(datum,y[i],phi)
        # gibbs[i] = np.exp(-dists[i]/T)
        gibbs[i] = np.exp(-dists[i]*el_inv)
    
    pyx_sum = _dot(py,gibbs)
    
    pyx = 0.0
    if pyx_sum == 0: # e.g. if no codevectors of the same class as observation
        pyx = 0 # set it zero (or break)
    else:    
        pyx = py[idx]*gibbs[idx]/pyx_sum
    
    # SA update
    pypy = selfpy + 1/(bb+1)*(pyx - selfpy)
    sxpysxpy = selfsxpy + 1/(bb+1)*(pyx*datum - selfsxpy)
    yy = sxpysxpy/pypy
    
    slpyslpy=selfslpy
    yylabel = ylabels[idx]
    if regression:
        slpyslpy = selfslpy + 1/(bb+1)*(pyx*datum_label - selfslpy)
        yylabel = slpyslpy/pypy
    
    
    if phi == 'phi_KL':
        for i in range(len(yy)):
            if yy[i]<pzero:
                yy[i] = pzero
    
    return pypy,sxpysxpy,yy,slpyslpy,yylabel


# LVQ Gradient Descent Update
###############################################################################
@njit(cache=True)
def _lvq_update(idx, y, ylabels, py, sxpy, slpy, datum, datum_label, regression, sep, T, bb, phi='phi_KL'):
    
    j,_ = _winner(y, datum, phi)
    s = 1 if (ylabels[j]==datum_label or regression) else -1
    yylabels = ylabels
    
    # LVQ Update
    yyj = y[j] - 1/(bb+1) * s * _dBregmanD(datum,y[j],phi)
    yy=y.copy()
    yy[j] = yyj
    if regression:
        yylabelj = ylabels[j] - 1/(bb+1) * s * _dBregmanD(datum_label,ylabels[j],phi)
        yylabels=ylabels.copy()
        yylabels[j] = yylabelj
        
    return 1.0,np.zeros([len(yy)]),yy,0,yylabels


#%% ###########################################################################
### Multi_Thread Implementation
###############################################################################


# Prediction Step
###############################################################################
# def _predict_threads(y, ylabels, data):
#     out = np.zeros(len(data))
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         args = ((y,ylabels,datum) for datum in data) 
#         res = executor.map(_datum_predict_threads,args)
#     for i,r in enumerate(res):
#         out[i]=r
#     return out

    
# Single Prediction Step
###############################################################################
# @njit(cache=True, nogil=True)
# def _datum_predict_threads(args):
#     y, ylabels, datum = args
#     dists = np.zeros(len(y))
#     for i in range(len(y)):
#         dists[i]=_BregmanD(datum,y[i])
#     j = np.argmin(dists)
#     return ylabels[j]

#%%

"""
Tree-Structured (Multi-Resolution) Online Deterministic Annealing for Classification and Clustering
Christos Mavridis & John Baras,
Department of Electrical and Computer Engineering, University of Maryland
<mavridis@umd.edu>
"""
