#%% Author Information

"""

Tree-Structured (Multi-Resolution) Online Deterministic Annealing for 
Clustering, Regression, Classification, and Hybrid System Identification

Christos Mavridis 

School of Electrical and Computer Engineering, University of Maryland, College Park and
Department of Electrical Engineering and Computer Science, KTH Royal Institute of Technology

< mavridis (at) kth.se >
< c.n.mavridis (at) gmail.com >

https://mavridischristos.github.io/
https://github.com/MavridisChristos

"""

#%% Dependencies

'''

dependencies = [
  "numpy>=2.2.6",
  "numba>=0.61.2",
  "matplotlib>=3.10.3",
]
requires-python = ">=3.8"

'''

#%% Tutorial 

'''

clf = ODA()
clf.fit(train_data_x, train_data_y, train_labels,
        test_data_x, test_data_y, test_labels)

hatX, hatY, hatLabel, hatMode = clf.predict(test_data_x)


'''

#%% README

'''
ODA Parameters


### Data

- train_data_x
    # Single layer: [[np.array], [np.array], [np.array], ...]
    # Multiple Layers/Resolutions: [[np.array, np.array, ...], [np.array, np.array, ...], [np.array, np.array, ...], ...]
- train_data_y
    # [ np.array, np.array, np.array, ... ] (np.atleast1d())
- train_labels
    # [ 0, 0, 0, ... ] (zero values for clustering)
    # [ int, int , int, ... ] (int values for classification with numba.jit)
- observe_xy = [0]
    # value in [0,1]. 0 considers only x values, 1 considers only y values, 0.5 considers bothe x,y equally, etc..
    
### Bregman divergence

- Bregman_phi = ['phi_Eucl']
    # Defines Bregman divergence d_phi. 
    # Values in {'phi_Eucl', 'phi_KL'} (Squared Euclidean distance, KL divergence)


### Termination Criteria

- Kmax = [32]
    # Limit in node's children. After that stop growing
- timeline_limit = 1e6
    # Limit in the number of convergent representations. (Developer Mode) 
- error_type = [0] 
    # 0:Clustering, 1:Regression, 2:Classification
- error_threshold = [0.01] 
    # Desired training error. 
- error_threshold_count = [3] 
    # Stop when reached 'error_threshold_count' times


### Temperature Schedule

- Tmax = [0.9] 
- Tmin = [1e-2]
    # lambda max min values in [0,1]. T = (1-lambda)/lambda
- gamma_steady = [0.95] 
    # T' = gamma * T
- gamma_schedule = [[0.8,0.8]] 
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
- px_cut = [1e-5] 
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

- verbose = 2 
    # Values in {0,1,2}    
    # 0: don't show score
    # 1: show score only on tree node splits 
    # 2: show score after every SA convergence 

### Numba Jit

- jit = True
    # Using jit/python for Bregman divergences


'''


'''
ODA MODEL Parameters

# Tree Structure

- self.id
- self.parent
- self.children

# Status

- self.K
- self.T

- self.perturbed = False
- self.converged = False
- self.trained = False

# Variables

- self.x 
- self.y 
- self.labels 
- self.classes
- self.parameters 
- self.model 
- self.optimizer 

- self.px 
- self.sx 

# History

- self.myK 
- self.myT 
- self.myX 
- self.myY 
- self.myLabels 
- self.myParameters
- self.myModels
- self.myOptimizers

- self.myTrainError 
- self.myTestError 
- self.myLoops 
- self.myTime 
- self.myTreeK 
- self.myTreeLoops 

# Convergence Parameters (development mode)

- self.e_p
- self.e_n
- self.e_c
- self.px_cut 
- self.lvq 
- self.convergence_loops 
- self.error_type 
- self.error_threshold 
- self.error_threshold_count 
- self.convergence_counter_threshold 
- self.bb_init
- self.bb_step 
- self.separate 
- self.stop_separation 
- self.bb 
- self.sa_steps 

'''

#%% Import Modules 

import time
import numpy as np
from numba import njit
import copy
import matplotlib.pyplot as plt
# import pylab as pl
# import concurrent.futures

#%% Global Parameters

RANDOM_SEED = 13
DTYPE = np.float64
REGULATE_THRESHOLD = 5
PRACTICAL_ZERO = 1e-9

np.random.seed(RANDOM_SEED)
# plt.ioff()

#%% The Class
    
class ODA:
    
    ###########################################################################
    # Init Function
    ###########################################################################
    
    def __init__(self,
                 # Data
                 train_data_x=None, 
                 train_data_y=None, 
                 train_labels=None,
                 # Probability domain
                 observe_xy = [0], 
                 # Layers
                 layers = [0],
                 # Bregman divergence
                 Bregman_phi=['phi_Eucl'], # {'phi_Eucl', 'phi_KL'}
                 # Termination
                 Kmax=[32], 
                 timeline_limit = 1e6, 
                 error_type = [0], # 0:Clustering, 1:Regression, 2:Classification
                 error_threshold=[0.01], 
                 error_threshold_count=[2], 
                 # Temperature
                 Tmax=[0.9], 
                 Tmin=[1e-2],
                 gamma_schedule=[[0.8,0.8]], 
                 gamma_steady=[0.95], 
                 # Tree Structure
                 node_id=[0], 
                 parent=None, 
                 # Regularization
                 lvq=[0], # {0,1,2,3} 
                 px_cut=[1e-5],  
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
                 verbose = 2, # 0,1,2
                 # Python or Jit
                 jit = True
                 ):
        
        ### Tree-Structure Parameters
        
        self.id = node_id.copy()
        self.parent = parent
        self.children = []

        self.layers = layers
        self.nlayers = len(layers)
        self.mylayer = len(self.id)-1
        self.timeline = [self.id]
        self.verbose = verbose

        ### Objective weights
        self.observe_xy = observe_xy[self.mylayer]
        self.observe_xy_arxiv = [observe_xy[0]] * self.nlayers if len(observe_xy)==1 else observe_xy.copy()
        
        ### Keep archive to pass multi-resolution parameters to children
        # !!! Revisit copy vs deepcopy here
        
        # Data
        # self.train_data_arxiv = train_data.copy()
        # self.train_labels_arxiv = train_labels.copy()
        # Bregman divergence
        self.Bregman_phi_arxiv = [Bregman_phi[0]] * self.nlayers if len(Bregman_phi)==1 else Bregman_phi.copy()
        # Termination
        self.Kmax_arxiv = [Kmax[0]] * self.nlayers if len(Kmax)==1 else Kmax.copy() 
        self.error_type_arxiv = [error_type[0]] * self.nlayers if len(error_type)==1 else error_type.copy()
        self.error_threshold_arxiv = [error_threshold[0]] * self.nlayers if len(error_threshold)==1 else error_threshold.copy()
        self.error_threshold_count_arxiv = [error_threshold_count[0]] * self.nlayers if len(error_threshold_count)==1 else error_threshold_count.copy() 
        # Temperature
        self.Tmax_arxiv = [Tmax[0]] * self.nlayers if len(Tmax)==1 else Tmax.copy()
        self.Tmin_arxiv = [Tmin[0]] * self.nlayers if len(Tmin)==1 else Tmin.copy()
        self.gamma_schedule_arxiv = [gamma_schedule[0].copy()] * self.nlayers if len(gamma_schedule)==1 else gamma_schedule.copy()
        self.gamma_steady_arxiv = [gamma_steady[0]] * self.nlayers if len(gamma_steady)==1 else gamma_steady.copy()
        # Regularization
        self.lvq_arxiv = [lvq[0]] * self.nlayers if len(lvq)==1 else lvq.copy()
        self.py_cut_arxiv = [px_cut[0]] * self.nlayers if len(px_cut)==1 else px_cut.copy()
        self.perturb_param_arxiv = [perturb_param[0]] * self.nlayers if len(perturb_param)==1 else perturb_param.copy()
        self.effective_neighborhood_arxiv = [effective_neighborhood[0]] * self.nlayers if len(effective_neighborhood)==1 else effective_neighborhood.copy()
        # Convergence
        self.em_convergence_arxiv = [em_convergence[0]] * self.nlayers if len(em_convergence)==1 else em_convergence.copy()
        self.convergence_counter_threshold_arxiv = [convergence_counter_threshold[0]] * self.nlayers if len(convergence_counter_threshold)==1 else convergence_counter_threshold.copy()
        self.convergence_loops_arxiv = [convergence_loops[0]] * self.nlayers if len(convergence_loops)==1 else convergence_loops.copy()
        self.stop_separation_arxiv = [stop_separation[0]] * self.nlayers if len(stop_separation)==1 else stop_separation.copy()
        self.bb_init_arxiv = [bb_init[0]] * self.nlayers if len(bb_init)==1 else bb_init.copy()
        self.bb_step_arxiv = [bb_step[0]] * self.nlayers if len(bb_step)==1 else bb_step.copy()
        
        ### State Parameters
        # Codevectors        
        self.x = []
        self.y = []
        self.labels = []
        self.K = 0
        self.classes = []

        self.parameters = []
        self.model = []
        self.optimizer = []
        
        self.px = []
        self.sx= []
        self.last_x = []
        self.last_y = []

        # Termination
        self.Kmax = Kmax[self.mylayer] 
        self.Tmax = Tmax[self.mylayer]
        self.Tmin = Tmin[self.mylayer]
        self.gamma_schedule = gamma_schedule[self.mylayer]
        self.gamma_steady = gamma_steady[self.mylayer]
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
        
        # Save copy for each temperature level
        self.myK = [self.K]
        self.myT = [self.T]
        self.myX = [self.x.copy()]
        self.myY = [self.y.copy()]
        self.myLabels = [self.labels.copy()]
        self.myParameters = [copy.deepcopy(self.parameters)]
        self.myModel = [copy.deepcopy(self.model)]
        self.myOptimizer = [copy.deepcopy(self.optimizer)]
        self.myTrainError = [[1,1,1]]
        self.myTestError = [[1,1,1]]
        self.myLoops = [0]    
        self.myTime = [time.perf_counter()]
        self.myTreeK = [self.K]
        self.myTreeLoops = [0]

        # Bregman Divergence 
        self.Bregman_phi = Bregman_phi[self.mylayer] # 'phi_Eucl', 'phi_KL', 'phi_IS'
        
        # Regularization
        self.px_cut = px_cut[self.mylayer]
        self.lvq = lvq[self.mylayer]
        
        # Convergence parameters
        self.convergence_loops = convergence_loops[self.mylayer]
        self.error_type = error_type[self.mylayer]
        self.error_threshold = error_threshold[self.mylayer]
        self.error_threshold_count = error_threshold_count[self.mylayer]
        self.convergence_counter_threshold = convergence_counter_threshold[self.mylayer]
        self.bb_init= bb_init[self.mylayer] # initial stepsize of stochastic approximation: 1/(bb+1)
        self.bb_step = bb_step[self.mylayer] # bb+=bb_step
        self.separate = True
        self.stop_separation = stop_separation[self.mylayer]
        self.bb = self.bb_init
        self.sa_steps = 0

        # Counters
        self.last_sample = 0 # sample at which last convergence occured
        self.current_sample = 0
        self.self_regulate_counter = 0
        self.self_regulate_threshold = REGULATE_THRESHOLD
        
        # Timestamps
        self.tik = time.perf_counter()
        self.tok = time.perf_counter()
    
        # Other
        self.plt_counter = 0 # for plots
        self.low_p_warnings = 0 # for warnings regarding probability estimates: Unused??
        self.practical_zero = PRACTICAL_ZERO # for the log inside KL divergence
        self.jit = jit

        self.e_p = self.BregmanD(np.atleast_1d(0),np.atleast_1d(0+perturb_param[self.mylayer]))            
        self.e_n = self.BregmanD(np.atleast_1d(0),np.atleast_1d(0+effective_neighborhood[self.mylayer]))            
        self.e_c = self.BregmanD(np.atleast_1d(0),np.atleast_1d(0+em_convergence[self.mylayer]))            

        # Data=dependent Initialization
        if train_data_x:

            idx = np.random.choice(range(len(train_data_x)))
            self.initialize_node(train_data_x[idx], train_data_y[idx], train_labels[idx])
            

    def initialize_node(self, datum_x, datum_y, datum_label):
        
        self.resolutions = len(datum_x)
        
        x_init = datum_x[self.layers[self.mylayer]]
        y_init = datum_y
        c_init = datum_label

        self.classes = [int(c_init)]
        self.insert_codevector(x_init, y_init, c_init)
        
        perturb_param = self.perturb_param_arxiv
        effective_neighborhood = self.effective_neighborhood_arxiv
        em_convergence = self.em_convergence_arxiv
        # self.e_p = self.BregmanD(np.array(x_init),np.array(x_init+perturb_param[self.mylayer]))            
        # self.e_n = self.BregmanD(np.array(x_init),np.array(x_init+effective_neighborhood[self.mylayer]))            
        # self.e_c = self.BregmanD(np.array(x_init),np.array(x_init+em_convergence[self.mylayer]))            
        self.e_p = self.BregmanD(np.atleast_1d(0),np.atleast_1d(0+perturb_param[self.mylayer]))            
        self.e_n = self.BregmanD(np.atleast_1d(0),np.atleast_1d(0+effective_neighborhood[self.mylayer]))            
        self.e_c = self.BregmanD(np.atleast_1d(0),np.atleast_1d(0+em_convergence[self.mylayer]))            

        self.myK = [self.K]
        self.myT = [self.T]
        self.myX = [self.x.copy()]
        self.myY = [self.y.copy()]
        self.myLabels = [self.labels.copy()]
        self.myParameters = [copy.deepcopy(self.parameters)]
        self.myModel = [copy.deepcopy(self.model)]
        self.myOptimizer = [copy.deepcopy(self.optimizer)]
        self.myTrainError = [[1,1,1]]
        self.myTestError = [[1,1,1]]
        self.myLoops = [0]    
        self.myTime = [time.perf_counter()]
        self.myTreeK = [self.K]
        self.myTreeLoops = [0]

        # print(f"*** Node Initalized x:{x_init}, y:{y_init}, c:{c_init} ***")
    
    
    ###########################################################################
    # Training Functions
    ###########################################################################
    
    
    # Fit ODA to a Dataset (Until Stopping Criteria Reached)
    ###########################################################################
    def fit(self, train_data_x=[], train_data_y=[], train_labels=[],
            test_data_x=[], test_data_y=[], test_labels=[]):

        len_data_x = len(train_data_x)

        if len_data_x==0: 
            print("Please provide data samples for training.")
            return 
        
        fit_sample = 0
        
        self.myTime[-1] = time.perf_counter()
        
        ## Whlie the entire tree is not trained
        while not self.trained: 
            
            idx = fit_sample % len_data_x
            datum_x = train_data_x[idx]
            datum_y = train_data_y[idx] 
            datum_label = train_labels[idx] 
            
            ## Train this or children. train_step() is recursive.
            self.train_step(datum_x, datum_y, datum_label,
                            train_data_x=train_data_x, train_data_y=train_data_y, train_labels=train_labels, 
                            test_data_x=test_data_x, test_data_y=test_data_y, test_labels=test_labels)

            fit_sample += 1
    
    
    # Train ODA given a set of data (data used only once) 
    # May Terminate Before Stopping Criteria Reached
    # Can be used for Online Training (One Data Point per Call)
    ###########################################################################
    def train(self, train_data_x=[], train_data_y=[], train_labels=[],
            test_data_x=[], test_data_y=[], test_labels=[]):
        
        len_data_x = len(train_data_x)

        if len_data_x==0: 
            print("Please provide data samples for training.")
            return 
        
        self.myTime[-1]  = time.perf_counter()
        
        ## Whlie the entire tree is not trained
        for datum_x, datum_y, datum_label in zip(train_data_x, train_data_y, train_labels): 
            
            ## Train this or children. train_step() is recursive.
            self.train_step(datum_x, datum_y, datum_label,
                            train_data_x=train_data_x, train_data_y=train_data_y, train_labels=train_labels, 
                            test_data_x=test_data_x, test_data_y=test_data_y, test_labels=test_labels)

    # Training Step
    ###########################################################################
    def train_step(self, datum_x, datum_y, datum_label=0,
               train_data_x=[], train_data_y=[], train_labels=[], 
               test_data_x=[], test_data_y=[], test_labels=[]):

        # Debugging Check
        stop_timeline, len_timeline = self.check_timeline_limit()
        if stop_timeline:
            self.trained = True

        if self.trained:
            return  # Exit if already trained
        
        if self.children:
            # Check if children are trained
            self.check_trained()
            if not self.trained:

                #TODO: When children are not initialized yet?

                # Find the winning child
                j, _ = self.winner(self.x, datum_x[self.layers[self.mylayer]])

                # Recursive call to the winner child
                self.children[j].train_step(datum_x, datum_y, datum_label,
                                            train_data_x, train_data_y, train_labels,
                                            test_data_x, test_data_y, test_labels)
            else:
                # If just trained, report time 
                self.tok = time.perf_counter()
                self.myTime.append(self.tok)
            
            return 

        ### Train this cell (leaf node)

        # Insert Codevector if new class is observed
        # Also initializes codevector if not initialized
        if datum_label not in self.classes:
            if self.K>0: # just add new class
                self.classes.append(datum_label)
                self.insert_codevector(datum_x[self.layers[self.mylayer]], datum_y, datum_label)
            else: # initialize codevector
                self.initialize_node(datum_x, datum_y, datum_label)

        self.current_sample += 1

        # Perturbation and SA Initialization (only if not perturbed)
        if not self.perturbed:
            if self.lvq < 2:
                self.perturb()
            else:
                self.perturbed = True  # No perturbation for lvq >= 2
            self.converged = False
            self.bb = self.bb_init
            self.sa_steps = 0

        # Stochastic Approximation Step
        self.sa_step(datum_x, datum_y, datum_label)

        # Convergence Check
        self.check_convergence()
        if not self.converged:
            return 
        
        ### SA has converged 

        # Report time
        self.tok = time.perf_counter()  

        # Find effective codevectors
        if self.lvq < 2:
            self.find_effective_clusters()
        if self.lvq < 2:
            self.pop_idle_clusters()
        self.prune_siblings()

        # If Kmax reached, keep the last set of codevectors
        stop_K = self.K > self.Kmax
        if stop_K:
            self.overwrite_codevectors(
                self.myX[-1],
                self.myY[-1],
                self.myLabels[-1],
                self.myParameters[-1])  
            self.T = self.myT[-1]
            
            # self.current_sample = self.myLoops[-1]
            if self.lvq == 1:
                # self.find_effective_clusters() 
                # self.pop_idle_clusters() 
                self.lvq = 2
                print(f'--- Keeping K={self.K} codevectors.')
                
        # Data logging - consolidated into a single block
        self.myT.append(self.T)
        self.myK.append(self.K)
        self.myX.append(self.x.copy())
        self.myY.append(self.y.copy())
        self.myLabels.append(self.labels.copy())
        self.myParameters.append(copy.deepcopy(self.parameters))
        self.myModel.append(copy.deepcopy(self.model))
        self.myOptimizer.append(copy.deepcopy(self.optimizer))
        self.myLoops.append(self.current_sample)
        self.myTime.append(self.tok)
        self.put_in_timeline(self.id.copy())

        # Check criteria to stop training self and split, if possible
        stop_T = self.myT[-1] <= self.Tmin
        stop_timeline, len_timeline = self.check_timeline_limit()
        stop_error = False

        # Compute score (if data are given)
        if train_data_x:
            # d_train = [d_train_x, d_train_y, dtrain_c]
            d_train = self.score(train_data_x, train_data_y, train_labels)
            self.myTrainError.append(d_train)
            if d_train[self.error_type] < self.error_threshold:
                self.confirm_error_threshold()
            self.error_threshold_reached = self.check_error_threshold()
            stop_error = self.error_threshold_reached > self.error_threshold_count
        if test_data_x:  
            # d_test = [d_test_x, d_test_y, dtest_c]
            d_test = self.score(test_data_x, test_data_y, test_labels)
            self.myTestError.append(d_test)

        # Verbose output after every SA convergence (if keepscore == 2)
        if self.verbose == 2:
            tK = self.treeK()
            tL = self.treeLoops()
            print(
                f'{len_timeline} -- ID: {self.id}: '
                f'Samples: {tL}(+{self.current_sample - self.last_sample}): '
                f'T = {self.myT[-1]:.4f}, K = {self.myK[-1]}, treeK = {tK}, '
                f't = {self.myTime[-1] - self.myTime[0]:.1f} '
                f'[+{self.myTime[-1] - self.myTime[-2]:.1f}s]'
                )
            if train_data_x and not stop_K:
                print(f'Train Error: [{d_train[0]:.4f}, {d_train[1]:.4f}, {d_train[2]:.4f}]')
                if d_train[self.error_type] < self.error_threshold:
                    print(f'*** Training Error threshold reached ({self.error_threshold_reached}/{self.error_threshold_count}). ***')
                if test_data_x:
                    print(f'Test Error: [{d_test[0]:.4f}, {d_test[1]:.4f}, {d_test[2]:.4f}]')

        # if termination/splitting reached minimum temperature or desired score or maximum tree nodes
        if (stop_K and self.lvq == 0) or stop_T or stop_error or stop_timeline:
            
            # Verbose output (if keepscore == 1)
            if self.verbose == 1:
                print(
                    f'{len_timeline} -- ID: {self.id}: '
                    f'Samples: {tL}(+{self.current_sample - self.last_sample}): '
                    f'T = {self.myT[-1]:.4f}, K = {self.myK[-1]}, treeK = {tK}, '
                    f't = {self.myTime[-1] - self.myTime[0]:.1f} '
                    f'[+{self.myTime[-1] - self.myTime[-2]:.1f}s]'
                    )
                if train_data_x and not stop_K:
                    print(f'Train Error: [{d_train[0]:.4f}, {d_train[1]:.4f}, {d_train[2]:.4f}]')
                    if d_train[self.error_type] < self.error_threshold:
                        print(f'*** Training Error threshold reached ({self.error_threshold_reached}/{self.error_threshold_count}). ***')
                    if test_data_x:
                        print(f'Test Error: [{d_test[0]:.4f}, {d_test[1]:.4f}, {d_test[2]:.4f}]')
            
            # Termination reason output
            if stop_K and self.lvq == 0:
                print('--- Maximum number of codevectors reached. ')
            if stop_T:
                print('--- Minimum temperature reached. ---')
            if stop_error:
                print('--- Minimum error reached. ---')
            if stop_timeline:
                print('--- Maximum number of nodes reached. ---')
            
            self.check_untrained_siblings()
            
            # Splitting or declaring trained
            if self.mylayer + 1 < self.nlayers and len(self.myX[-1]) > 1:
                self.split()
                self.reset_error_threshold()
                if self.verbose > 0:
                    print(f'ID: {self.id}: Trained. Splitting..')
            else:
                self.trained = True
                if self.verbose > 0:
                    print(f'ID: {self.id}: Trained')

        self.last_sample = self.current_sample
        self.update_T()
        self.perturbed = False

                
    # Stochastic Approximation Step
    ###########################################################################
    def sa_step(self, datum_x, datum_y, datum_label):

        # print(f'sx:{sx}')
        # print(f'self.sx:{self.sx}')

        # Prepare for update
        self.last_x = self.x.copy()
        self.last_y = self.y.copy()
        self.low_p_warnings = 0 # unused?
        datum_x = datum_x[self.layers[self.mylayer]]

        x = np.array(self.x, dtype=DTYPE)
        y = np.array(self.y, dtype=DTYPE)
        labels = np.array(self.labels, dtype=int)
        px = np.array(self.px, dtype=DTYPE)
        sx = np.array(self.sx, dtype=DTYPE)
        
        datum_x = np.array(datum_x, dtype=DTYPE)
        datum_y = np.array(datum_y, dtype=DTYPE)
        datum_label = int(datum_label)

        # Prepare Combined codevector
        mu = np.array([np.concatenate(z) for z in list(zip(x,y))], dtype=DTYPE)
        datum = np.concatenate((datum_x,datum_y))
        lenx = len(datum_x)

        ### ODA update
        if self.lvq < 3:  

            if self.jit:
                
                px, sx, mu = _sa_update(
                    mu, labels, px, sx,
                    datum, datum_label, lenx, self.observe_xy,
                    self.separate, self.T, self.bb, self.Bregman_phi)
                
                self.px = [x for x in px]
                self.sx = [x for x in sx]
                self.x = [x[:lenx] for x in mu]
                self.y = [x[lenx:] for x in mu]
                
            else:  # Python implementation

                T_inv = (1 - self.T) / self.T
                bb_step = 1 / (self.bb + 1)

                for k in range(self.K):
                    
                    if self.separate:
                        px_classification = [px[i] if datum_label==labels[i] else 0 
                                                for i in range(px.shape[0])]
                    else: 
                        px_classification = px

                    # if self.observe_xy:
                    #     d = [self.BregmanD(datum,m) for m in mu]
                    # else:
                    weight_x = 1-self.observe_xy
                    weight_y = self.observe_xy
                    d = [weight_x*self.BregmanD(datum[:lenx],m[:lenx]) 
                         + weight_y*self.BregmanD(datum[lenx:],m[lenx:]) for m in mu]

                    pxk_sum = np.dot(px_classification,[np.exp(-dj*T_inv) for dj in d])
                    if pxk_sum == 0: # e.g. if no codevectors of the same class as observation
                        pxkhat = 0 # or break
                    else:    
                        pxkhat = px_classification[k]*np.exp(-d[k]*T_inv)/pxk_sum
                    
                    # SA update
                    px[k] = px[k] + bb_step *(pxkhat - px[k])
                    sx[k] = sx[k] + bb_step *(pxkhat*datum - sx[k])
                    mu[k] = sx[k]/px[k]

                self.px = [x for x in px]
                self.sx = [x for x in sx]
                self.x = [xx[:lenx] for xx in mu]
                self.y = [xx[lenx:] for xx in mu]
            
            self.bb += self.bb_step
            self.sa_steps += 1 
            return

        ### LVQ update

        if self.jit:
            
            mu = _lvq_update(mu, labels, datum, datum_label, lenx,
                             self.bb, self.Bregman_phi)
            
            self.x = [x[:lenx] for x in mu]
            self.y = [x[lenx:] for x in mu]

        else:
            
            bb_step = 1 / (self.bb + 1)

            d = [self.BregmanD(datum[:lenx],m[:lenx]) for m in mu]
            j = np.argmin(d)
            s = 1 if (labels[j]==datum_label) else -1
            # LVQ Gradient Descent Update
            mu[j] = mu[j] - bb_step * s * self.dBregmanD(datum,mu[j])

            self.x = [x[:lenx] for x in mu]
            self.y = [x[lenx:] for x in mu]
                
        self.bb += self.bb_step
        self.sa_steps += 1 

        
            
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
            
            # conv_reached = np.all([self.BregmanD(np.array(self.last_x[i]),np.array(self.x[i])) < \
            #                 self.Temp()*self.e_c * (1+self.bb_init)/(1+self.bb)
            #                                                 for i in range(self.K)])  
            
            # last_mu = np.array([np.concatenate(z) for z in list(zip(self.last_x,self.last_y))], dtype=DTYPE)
            # mu = np.array([np.concatenate(z) for z in list(zip(self.x,self.y))], dtype=DTYPE)
            # conv_reached = np.all([self.BregmanD(np.array(last_mu[i]),np.array(mu[i])) < \
            #                 self.Temp()*self.e_c * (1+self.bb_init)/(1+self.bb)
            #                                                 for i in range(self.K)])  
            
            weight_x = 1-self.observe_xy
            weight_y = self.observe_xy
            conv_reached = np.all([weight_x*self.BregmanD(self.last_x[i],self.x[i]) 
                            + weight_y*self.BregmanD(self.last_y[i],self.y[i]) < 
                            self.Temp()*self.e_c * (1+self.bb_init)/(1+self.bb)
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
    
    # Copy pytorch optimizer
    ###########################################################################            
    def copy_opt(self,optimizer1,model):
        try: 
            optimizer = type(optimizer1)(model.parameters(), lr=optimizer1.defaults['lr'])
            optimizer.load_state_dict(optimizer1.state_dict())
        except:
            optimizer=copy.deepcopy(optimizer1)
        return optimizer
    
    # Perturb Codevectors
    ###########################################################################
    def perturb(self):
        ## insert perturbations of all effective yi
        for i in reversed(range(self.K)):
            # new_yi = self.y[i] + self.perturb_param*2*(np.random.rand(len(self.y[i]))-0.5)
            new_xi = self.x[i] + self.Temp()*self.e_p * 2 * (np.random.rand(len(self.x[i]))-0.5)
            new_yi = self.y[i] + self.Temp()*self.e_p * 2 * (np.random.rand(len(self.y[i]))-0.5)
            self.px[i] = self.px[i]/2.0
            self.sx[i] = self.px[i]*np.concatenate((self.x[i],self.y[i]))
            
            self.x.append(new_xi)
            self.y.append(new_yi)
            self.px.append(self.px[i])
            self.sx.append(self.px[i]*np.concatenate((new_xi,new_yi)))
            self.labels.append(self.labels[i]) 
            self.parameters.append(copy.deepcopy(self.parameters[i])) 
            self.model.append(copy.deepcopy(self.model[i])) 
            self.optimizer.append( self.copy_opt(self.optimizer[i],self.model[-1]) ) 
        self.K = len(self.x)
        self.perturbed = True
    
    
    # Update Temperature (lambda)
    ###########################################################################
    def update_T(self):
        
        if self.true_convergence_counter < len(self.gamma_schedule):
            self.gamma = self.gamma_schedule[self.true_convergence_counter]
        else:
            self.gamma = self.gamma_steady
        
        self.T = self.gamma * self.T
      
    
    # Compute T = lambda/(1-lambda)
    ###########################################################################
    def Temp(self):
        return self.T/(1-self.T)
    
    
    # Find Effective Codevectors
    ###########################################################################
    def find_effective_clusters(self):
        i=0
        while i<self.K:
            for j in reversed(np.arange(i+1,self.K)):
                
                # merged = self.BregmanD(np.array(self.x[i]),np.array(self.x[j]))< \
                #             self.Temp()*self.e_n and self.labels[i]==self.labels[j]
                
                weight_x = 1-self.observe_xy
                weight_y = self.observe_xy
                d = weight_x*self.BregmanD(self.x[i],self.x[j]) \
                    + weight_y*self.BregmanD(self.y[i],self.y[j])
                
                merged = d < self.Temp()*self.e_n and self.labels[i]==self.labels[j]
                            
                if merged:
                    
                    self.px[i] = self.px[i]+self.px[j]
                    self.sx[i] = self.x[i]*self.px[i]
                    self.px.pop(j)
                    self.sx.pop(j)
                    self.x.pop(j)
                    self.y.pop(j)
                    self.labels.pop(j)
                    self.parameters.pop(j)
                    self.model.pop(j)
                    self.optimizer.pop(j)
                    self.K-=1
            
            i+=1
    

    # Insert Codevector
    ###########################################################################
    def insert_codevector(self, x, y, label, px=1.0, parameters=None,
                          model=None, optimizer=None, norm=False): 
        
        x = x.astype(DTYPE)
        y = y.astype(DTYPE)
        label = int(label)
        px = np.float32(px)
        sx = np.array(np.concatenate((x,y))*px, dtype=DTYPE)

        self.x.append(x)
        self.y.append(y)
        self.labels.append(label)
        self.px.append(px)
        self.sx.append(sx)    
        self.K = len(self.x)

        if parameters is not None:
            self.parameters.append(copy.deepcopy(parameters))
        else:
            self.parameters.append([])
        self.model.append(model)
        self.optimizer.append(optimizer)

        # TODO: ensure dtypes here
        if norm:
            self.px = [float(p)/sum(self.px) for p in self.px]
            self.py = [float(p)/sum(self.py) for p in self.py]
            self.sx= [self.x[i]*self.px[i] for i in range(len(self.x))]    

    # Remove Codevector
    ###########################################################################
    def pop_codevector(self,idx,norm=False): 
        self.x.pop(idx)
        self.y.pop(idx)
        self.labels.pop(idx)
        self.parameters.pop(idx)
        self.model.pop(idx)
        self.optimizer.pop(idx)
        self.px.pop(idx)
        self.sx.pop(idx)
        self.K = len(self.x)

        # TODO: ensure dtypes here
        if norm:
            self.px = [float(p)/sum(self.px) for p in self.px]
            self.sx= [np.concatenate((self.x[i],self.y[i]))*self.px[i] for i in range(len(self.x))]   
        
    # Discard Idle Codevectors
    ###########################################################################
    def pop_idle_clusters(self):
        i = 0
        px_cut = self.px_cut
        while i < len(self.x):
            ## if the only representatitve of its class make it harder to be pruned
            # yli = self.ylabels.copy()
            # yli.pop(i)
            # if len(yli)>0 and np.any(np.array(self.ylabels[i])==np.array(yli)):
            #     py_cut = self.py_cut**2
            # prune idle codevector
            if self.px[i]<px_cut:
                self.pop_codevector(i)
                if self.verbose==2:
                    print('*** Idle Codevector Pruned (pop_idle_clusters) ***')
            else:
                i+=1
    
    # Overwrite Existing Codevectors (for External Use)
    ###########################################################################
    # TODO: check dtypes
    def overwrite_codevectors(self,new_x,new_y,new_labels,new_parameters=[],
                              new_model=[],new_optimizer=[],new_px=[]): # new_y must be a list
        self.x = new_x.copy()
        self.y = new_y.copy()
        self.labels = new_labels.copy()
        if new_parameters!=[]:
            self.parameters = copy.deepcopy(new_parameters)
        if new_model!=[]:
            self.model = new_model
        if new_optimizer!=[]:
            self.optimizer = new_optimizer
        if new_px==[]:
            self.px = [1.0 / len(self.x) for i in range(len(self.x))] 
        else:
            self.px = new_px.copy()
        self.sx= [np.concatenate((self.x[i],self.y[i]))*self.px[i] for i in range(len(self.x))]    
        self.K = len(self.x)
    
    ###########################################################################
    ### Tree-Structure Functions
    ###########################################################################
    
    
    # Split Cell: Create Children ODA nodes
    ###########################################################################
    def split(self):
        
        for i in range(len(self.myX[-1])):

            
            self.children.append(ODA(
                # Data
                # No data given. Child will be initialized with the first sample.
                #  train_data_x=None, 
                #  train_data_y=None, 
                #  train_labels=None,
                # Probability domain
                observe_xy = self.observe_xy_arxiv, 
                # Layers
                layers = self.layers,
                # Bregman divergence
                Bregman_phi=self.Bregman_phi_arxiv, 
                # Termination
                Kmax=self.Kmax_arxiv,
                timeline_limit = self.timeline_limit,
                error_type=self.error_type_arxiv,
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
                px_cut=self.py_cut_arxiv,
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
                verbose=self.verbose,
                jit = self.jit)
                )
            
            self.children[-1].parameters=[copy.deepcopy(self.parameters[i])]
            self.children[-1].model=[copy.deepcopy(self.model[i])]
            self.children[-1].optimizer=[self.copy_opt(self.optimizer[i],self.children[-1].model[-1])]
    
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
            # if self.parent:
            #     if (not self.regression) and len(np.unique(self.parent.myLabels[-1]))>1:
            #         if len(np.unique(self.labels))==1:
            #             self.overwrite_codevectors([self.x[0]],[self.labels[0]],[copy.deepcopy(self.parameters[0])],
            #                                        [self.model[0]],[self.optimizer[0]])
            #             self.trained = True
            #             if self.keepscore>2:
            #                 print('*** Same-Class Codevectors Pruned ***')
    
            # Find idle siblings and prune them
            if self.parent:
                for sibling in self.parent.children:
                    if sibling.current_sample == 0:
                        sibling.overwrite_codevectors([sibling.x[0]],[sibling.labels[0]],[copy.deepcopy(sibling.parameters[0])],
                                                      [sibling.model[0]],[sibling.optimizer[0]])
                        sibling.trained = True
                        if self.verbose==2:
                            print('*** Idle Sibling Pruned ***')
    
    # Check Untrained Siblings
    ###########################################################################
    def check_untrained_siblings(self):
        
        # Find idle siblings and prune them
        if self.parent:
            for sibling in self.parent.children:
                if sibling.current_sample == 0:
                    sibling.overwrite_codevectors([sibling.x[0]],[sibling.labels[0]],[copy.deepcopy(sibling.parameters[0])],
                                                  [sibling.model[0]],[sibling.optimizer[0]])
                    sibling.trained = True
                    if self.verbose==2:
                        print('*** Idle Sibling Pruned (check_untrained_siblings) ***')
        
        # Idle siblings have len(myX)=1
                    
                        
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
    
    # Load ODA Model
    # TODO: Extend to go back in timeline 
    ###########################################################################
    def load(self, T=None):
        
        self.perturbed = False
        self.converged = False
        self.trained = False
        self.bb = self.bb_init
        
        if T:
            self.T = T
        
        self.tik = time.perf_counter()
        self.tok = time.perf_counter()            
        
        if len(self.children)>0:
            for child in self.children:
                child.load()
                
    
    ###########################################################################
    ### Score Functions
    ###########################################################################
    
    # Compute Score
    ###########################################################################
    def score(self, data_x, data_y=None, labels=None, recursive=1e3):
        
        if data_x is None: 
            return 1.0, 1.0, 1.0

        if self.parent:
            return self.parent.score(data_x, data_y, labels, recursive)
        
        d_clustering = 0.0
        d_regression = 0.0
        d_classification = 0.0

        for i in range(len(data_x)):
            d_x, d_y, d_c = self.datum_score(data_x[i],data_y[i],labels[i],recursive)
            d_clustering += d_x
            d_regression += d_y
            d_classification += d_c

        return d_clustering/len(data_x), d_regression/len(data_x), d_classification/len(data_x) 
    
    # Compute All Scores between Codebook and Input Vector
    ###########################################################################
    def datum_score(self, datum_x, datum_y=None, label=None, recursive=1e3):
    
        d_clustering = 1.0
        d_regression = 1.0
        d_classification = 1.0

        if datum_x is None: 
            return d_clustering, d_regression, d_classification
        
        x = self.myX[-1]
        j,d = self.winner(x, datum_x[self.layers[self.mylayer]])
        
        if recursive>0 and len(self.children) > 0 and len(self.children[j].myX)>1:
            return self.children[j].datum_score(datum_x, datum_y, label, recursive-1)
        
        d_clustering = d 
        if datum_y is not None:
            y = self.myY[-1][j]
            d_regression = self.BregmanD(datum_y, y)
            # TODO: When I am using model and parameters?
        if label is not None: 
            decision_label = self.myLabels[-1][j]
            d_classification = 0.0 if decision_label == label else 1.0
        
        return d_clustering, d_regression, d_classification 
        
    # Compute Dissimilarity between Codebook and Input Vector
    ###########################################################################
    def datum_dissimilarity(self, datum_x, recursive=1e3):
    
        x = self.myX[-1]
        j,d = self.winner(x, datum_x[self.layers[self.mylayer]])
        
        if recursive>0 and len(self.children) > 0 and len(self.children[j].myX)>1:
            return self.children[j].datum_dissimilarity(datum_x,recursive-1)
        
        return d   
    
    # Compute Regression Error between Codebook and Input Vector
    ###########################################################################
    def datum_regression_error(self, datum_x, datum_y, recursive=1e3):
    
        x = self.myX[-1]
        j,_ = self.winner(x, datum_x[self.layers[self.mylayer]])
        
        if recursive>0 and len(self.children) > 0 and len(self.children[j].myX)>1:
            return self.children[j].datum_regression_error(datum_x,datum_y,recursive-1)
        
        y = self.myY[-1][j]
        d = self.BregmanD(datum_y, y) 
        return d
        
    # Compute Classification Error between Codebook and Input Vector 
    ###########################################################################
    def datum_classification_error(self, datum_x, label, recursive=1e3):
        
        x = self.myX[-1]
        j,_ = self.winner(x, datum_x[self.layers[self.mylayer]])
        
        ## if I have children and the winner child has converged at least once
        if recursive>0 and len(self.children)>0 and len(self.children[j].myX)>1: 
            return self.children[j].datum_classification_error(datum_x,label,recursive-1)
        
        decision_label = self.myLabels[-1][j]
        d = 0 if np.all(decision_label == label) else 1
        return d    
        
    # Find best representative, predict y, predict label
    ###########################################################################
    def predict_x(self, datum_x, recursive=1e3):
    
        x = self.myX[-1]
        j,_ = self.winner(x, datum_x[self.layers[self.mylayer]])
        
        if recursive>0 and len(self.children) > 0 and len(self.children[j].myX)>1:
            return self.children[j].predict(datum_x,recursive-1)
        
        return self.myX[-1][j], self.myY[-1][j], self.myLabels[-1][j], j
        # if len(self.myX)>0 else None

    def predict(self, data_x, recursive=1e3):
    
        hatX = []
        hatY = []
        hatLabel = []
        hatMode = []

        for datum_x in data_x: 
            hX, hY, hL, w = self.predict_x(datum_x, recursive=recursive)
            hatX.append(hX)
            hatY.append(hY)
            hatLabel.append(hL)
            hatMode.append(w)
        
        return hatX, hatY, hatLabel, hatMode

    # Return Codebook
    ###########################################################################
    def codebook(self, recursive=1e3):
    
        if recursive>0 and len(self.children) > 0:
            cb = []
            cby = []
            cbl = []
            for child in self.children:
                # if len(child.myX)>1: # check why. it has to do with prune siblings
                if len(child.myX)>0: 
                    tcb,tcby,tcbl = child.codebook(recursive-1)
                    cb = cb + tcb
                    cby = cby + tcby
                    cbl = cbl + tcbl
            return cb, cby, cbl
        
        return self.myX[-1], self.myY[-1], self.myLabels[-1]   
        # TODO: When using model and parameters?
    
    # Find Winner Codevector
    ###########################################################################
    def winner(self, mu, datum):

        mu = np.array(mu, dtype=DTYPE)  # Ensure mu is a NumPy array
        datum = datum.astype(DTYPE)

        if self.jit:
            j, d = _winner(mu, datum, phi=self.Bregman_phi)
        else:
            dists = [self.BregmanD(datum, yj) for yj in mu]
            j = np.argmin(dists)
            d = dists[j]
            
        return j, d
    
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
    

    #%% ###########################################################################    
    ### Plotting Functions
    ###############################################################################

    # Performance Curve
    ###############################################################################
    def plot_curve(self, error_type=0, 
                   figname='', show = False, save = False,
                    fig_size=(9,5),
                    font_size = 20,
                    label_size = 16,
                    legend_size = 18,
                    line_width = 6,
                    marker_size = 6,
                    fill_size=10,
                    line_alpha = 0.8,
                    txt_size = 32,
                    txt_x = 1.0,
                    txt_y = 0.03,
                    font_weight = 'bold', 
                    ylim = 0.5,
                    dpi=300
                    ):
        
        # Variables        
        
        idx = []
        myK = []
        tK=[]
        tL=[]
        myT = []
        myX = []
        myY = []
        myLabels = []
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
            myX.append(node.myX[node.plt_counter])
            myY.append(node.myY[node.plt_counter])
            myLabels.append(node.myLabels[node.plt_counter])
            myLoops.append(node.myLoops[node.plt_counter])   
            myTime.append(node.myTime[node.plt_counter])
            
            try:
                myTrainError.append(node.myTrainError[node.plt_counter])
            except:
                myTrainError.append([])

            try:
                myTestError.append(node.myTestError[node.plt_counter])
            except:
                myTestError.append([])

            node.plt_counter += 1
        
        import matplotlib.colors as mcolors
        colors = mcolors.TABLEAU_COLORS
        colors = list(colors.keys())
        markers = ['s','D','o','X','P']    
        
        #######################################################################
        # Performance VS Time
        #######################################################################
        
        fig,ax = plt.subplots(figsize=fig_size,tight_layout = {'pad': 1},dpi=dpi)
        
        # Label axes
        ax.set_ylim(-0.01,ylim+0.01)
        ax.set_xlabel('time (s)', fontsize = font_size)
        ylabel = '% error'
        ax.set_ylabel(ylabel, fontsize = font_size)
        ax.tick_params(axis='both', which='major', labelsize=label_size)
        ax.tick_params(axis='both', which='minor', labelsize=8)

        x= [t - myTime[0] for t in myTime]

        try:
            y=[e[error_type] for e in myTrainError]
            clr=colors[0]
            mrkr = markers[0]
            ax.plot(x, y, label='Train',
                color=clr, marker=mrkr,linestyle='solid', 
                linewidth=line_width, markersize=marker_size,alpha=line_alpha)
        except:
            pass
        
        try:     
            y=[e[error_type] for e in myTestError]
            clr=colors[1]
            mrkr = markers[1]
            ax.plot(x, y, label='Test', 
                color=clr, marker=mrkr,linestyle='solid', 
                linewidth=line_width, markersize=marker_size,alpha=line_alpha)
        except:
            pass

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
        
        if False:
            fig,ax = plt.subplots(figsize=fig_size,tight_layout = {'pad': 1},dpi=dpi)
            
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
            
            try:
                y=[e[error_type] for e in myTrainError]
                clr=colors[0]
                mrkr = markers[0]
                ax2 = ax.twinx()
                ax2.plot(x, y, label='Train', 
                    color=clr, marker=mrkr,linestyle='solid', 
                    linewidth=line_width, markersize=marker_size,alpha=line_alpha)
            except: 
                pass

            try:
                y=[e[error_type] for e in myTestError]
                clr=colors[1]
                mrkr = markers[1]
                ax2.plot(x, y, label='Test', 
                    color=clr, marker=mrkr,linestyle='solid', 
                    linewidth=line_width, markersize=marker_size,alpha=line_alpha)
                ylabel = '% error' 
                ax2.set_ylabel(ylabel, fontsize = font_size)
                ax2.tick_params(axis='both', which='major', labelsize=label_size)
                ax2.tick_params(axis='both', which='minor', labelsize=8)
            except:
                pass
            
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

# Stochastic Approximation Update
###############################################################################

@njit(fastmath=True, cache=True)
def _sa_update(mu, labels, px, sx,  
               datum, datum_label, lenx, observe_xy, 
               sep, T, bb, phi='phi_Eucl'):
    
    pzero = np.float32(1e-9)
    T_inv = np.float32( (1-T) / T )
    K = mu.shape[0]
    bb_step = np.float32( 1 / (bb+1) )

    for k in range(K): 

        px_classification = px.astype(px.dtype)
        if sep:
            for i in range(K):
                if datum_label != labels[i]:
                    px_classification[i]=np.float32(0.0)

        # if px_classification[k] == 0.0:
        #     continue 

        pxk = px[k]
        muk = mu[k]
        sxk = sx[k]
        labelsk = labels[k]
        
        # Calculate distances and Gibbs probabilities 
        dists = np.zeros(K, dtype=mu.dtype)
        for i in range(K):
            # if observe_xy:
            #     dists[i] = _BregmanD(datum, mu[i], phi)
            # else:
            weight_x = 1-observe_xy
            weight_y = observe_xy
            dists[i] = weight_x*_BregmanD(datum[:lenx], mu[i][:lenx], phi) + \
                        weight_y*_BregmanD(datum[lenx:], mu[i][lenx:], phi)
        
        gibbs = np.exp(-dists * T_inv)

        # Calculate hat_pmu[k]
        pxkhat = np.float32(0.0)
        pxk_sum = np.sum(px_classification * gibbs) # dot product
        if pxk_sum > 0: 
            pxkhat = px_classification[k] * gibbs[k] / pxk_sum
        else: # e.g. if no codevectors of the same class as observation (not expected behavior)
            pxkhat = np.float32(0.0) # set it zero (or break)

        # SA update 
        pxk = pxk + bb_step * (pxkhat - pxk)
        sxk = sxk + bb_step * (pxkhat*datum - sxk)
        if pxk>0: # (avoiding division by zero)
            muk = sxk/pxk
        # else:
        #     muk = sigmamuk/pmuk

        if phi == 'phi_KL':
            for i in range(muk.shape[0]):
                if muk[i]<pzero:
                    muk[i] = pzero

        px[k] = pxk
        sx[k] = sxk
        mu[k] = muk
        labels[k] = labelsk

    return px, sx, mu


# LVQ Gradient Descent Update
###############################################################################
@njit(fastmath=True, cache=True)
def _lvq_update(mu, labels, datum, datum_label, lenx, bb, phi='phi_Eucl'):
    
    j,_ = _winner(mu[:lenx], datum[:lenx], phi)
    s = 1 if (labels[j]==datum_label) else -1
    
    # LVQ Update
    mu[j] = mu[j] - 1/(bb+1) * s * _dBregmanD(datum,mu[j],phi)
        
    return mu

# Find Winner Codevector
###############################################################################
@njit(cache=True,nogil=True)
def _winner(y, datum, phi='phi_Eucl'):
    dists = np.zeros(len(y))
    for i in range(len(y)):
        dists[i]=_BregmanD(datum,y[i],phi)
    j = np.argmin(dists)
    return j, dists[j]


# Compute Bregman Divergence
###############################################################################
@njit(fastmath=True, cache=True, nogil=True)
def _BregmanD(x, y, phi='phi_Eucl'):
    if phi == 'phi_Eucl':
        d = np.sum((x-y)*(x-y))
    elif phi == 'phi_KL':
        pzero = np.float32(1e-9)
        logx = np.zeros_like(x)
        logy = np.zeros_like(y)
        sxy=np.float32(0)
        for i in range(x.shape[0]):
            if x[i]<pzero:
                x[i]=pzero
            if y[i]<pzero:
                y[i]=pzero    
            logx[i] = np.log(x[i])
            logy[i] = np.log(y[i])
            sxy += x[i]-y[i]
        d = np.sum(x * (logx-logy)) - sxy
    return d


# Compute Bregman Divergence Derivative
###############################################################################
@njit(fastmath=True, cache=True, nogil=True)
def _dBregmanD(x, y, phi='phi_Eucl'):
    if phi == 'phi_Eucl':
        dd = -2*(x-y)
    elif phi == 'phi_KL':
        pzero = 1e-9
        dd = np.zeros([x.shape[0]])
        for i in range(x.shape[0]):
            if x[i]<pzero:
                x[i]=pzero
            if y[i]<pzero:
                y[i]=pzero    
            dd[i] = - 1/y[i] * (x[i]-y[i])
    return dd














"""

Tree-Structured (Multi-Resolution) Online Deterministic Annealing for 
Clustering, Regression, Classification, and Hybrid System Identification

Christos Mavridis 

School of Electrical and Computer Engineering, University of Maryland, College Park and
Department of Electrical Engineering and Computer Science, KTH Royal Institute of Technology

< mavridis (at) kth.se >
< c.n.mavridis (at) gmail.com >

https://mavridischristos.github.io/
https://github.com/MavridisChristos

"""