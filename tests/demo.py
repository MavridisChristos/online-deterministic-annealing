#%%
#!/usr/bin/env python

"""
2D Binary Classification Demo with Gaussian Mixtures
(Multi-Resolution) Online Deterministic Annealing (ODA) for Classification and Clustering

Christos N. Mavridis
Division of Decision and Control Systems,
School of Electrical Engineering and Computer Science, 
KTH Royal Institute of Technology
<mavridis@kth.se>
"""

# Import Modules

import pickle
import numpy as np
import os
# import sys
# sys.path.append(os.path.abspath('.'))
# sys.path.append('/Users/cm/Documents/Github/online-deterministic-annealing/online_deterministic_annealing')

from online_deterministic_annealing.oda import ODA
from tests.utils import domain_plots as dplot

#%% Problem Parameters

data_file = './tests/utils/data'
results_file = './tests/'
clustering = False

plot_curve = True
plot_domain = True

# ODA Parameters 

layers = [1]
observe_xy = [0.5]

Tmax = [0.9,1e-1,1e-2]
Tmin = [1e-2,1e-2,5*1e-4]
gamma_schedule = [[0.1,0.5],[],[]] 
gamma_steady = [0.8]

lvq=[0]
perturb_param = [1e-1]
effective_neighborhood = [1e-0]
px_cut = [1e-6]

Kmax = [16,5,5] 
timeline_limit = 1e3
error_type = [0]
error_threshold = [0.002]
error_threshold_count = [2]

em_convergence = [1e-1]
convergence_counter_threshold = [5]
stop_separation = [1e6-1]
convergence_loops = [0]
bb_init = [0.9]
bb_step = [0.9] 

Bregman_phi = ['phi_Eucl']

plot_curves = True
show_domain = True
keepscore = 2
jit = True

# Load data

# Data: list of (n) lists of (r) m-vectors, where r=dim is the number of resolutions 
# Labels: list of (n) labels
with open(data_file+'.pkl', mode='rb') as file:
    train_data,train_labels,test_data,test_labels = pickle.load(file)

# Resolutions
train_data_x = [[td[r] for r in range(2)] for td in train_data]
train_data_y = [np.array(0).reshape(1) for td in train_data_x]

if clustering:
    train_labels = [0 for i in range(len(train_labels))] 

# Init ODA

clf = ODA(
        # Data
        # No data. Will be initialized with first sample.
        observe_xy = observe_xy, 
        # Layers 
        layers = layers,
        # Bregman divergence
        Bregman_phi=Bregman_phi,
        # Termination
        Kmax=Kmax,
        timeline_limit = timeline_limit,
        error_type = error_type,
        error_threshold=error_threshold,
        error_threshold_count=error_threshold_count,
        # Temperature
        Tmax=Tmax,
        Tmin=Tmin,
        gamma_schedule=gamma_schedule,
        gamma_steady=gamma_steady,
        # Regularization
        lvq=lvq, # 0:ODA, 1:soft clustering with no perturbation/merging, 2: LVQ update with no perturbation/merging
        px_cut=px_cut,
        perturb_param=perturb_param, 
        effective_neighborhood=effective_neighborhood, 
        # Convergence
        em_convergence=em_convergence, 
        convergence_counter_threshold=convergence_counter_threshold,
        convergence_loops=convergence_loops,
        stop_separation=stop_separation,
        bb_init=bb_init,
        bb_step=bb_step,
        # Verbose
        verbose=keepscore,
        jit=jit
        )

print('*** ODA Initialized ***')

# Run ODA

print('*** ODA Start ***')

# Fit Model
clf.fit(train_data_x=train_data_x,
        train_data_y=train_data_y,
        train_labels=train_labels
        )

print('*** ODA Finish ***')

# Plot Curve

if plot_curve:

    clf.plot_curve(error_type=0, 
                   figname='./'+results_file+'/'+'demo',
                   show=True, save = True,
                   ylim = 0.5
                   )

# Plot Domain

if show_domain: 

    if len(layers)<2:
        dplot.show(clf=clf, train_data=train_data, train_labels=train_labels, res=layers, 
                            plot_folder='./'+results_file+'/domain/', 
                            plot_fig = True, save_fig = False)
    else:
        dplot.show_resolutions(clf=clf, train_data=train_data, train_labels=train_labels, res=layers, 
                                        plot_folder='./'+results_file+'/domain/',
                                        plot_fig = True, save_fig = False)











#%%



