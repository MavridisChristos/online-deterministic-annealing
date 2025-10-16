# Online Deterministic Annealing (ODA)

> A general-purpose learning model designed to meet the needs of Cyber-Physical Systems applications in which data and computational/communication resources are limited, and robustness and interpretability are prioritized.

> An **online** prototype-based learning algorithm based on annealing optimization that is formulated as an recursive **gradient-free** stochastic approximation algorithm.

> An interpretable and progressively growing competitive-learning neural network model.

> A hierarchical, multi-resolution learning model. 

> **Applications:** Real-time clustering, classification, regression, (state-aggregation) reinforcement learning, hybrid system identification, graph partitioning, leader detection.

## pip install

	pip install online-deterministic-annealing

Current Version: 

    version = "1.0.1"

Dependencies: 

    dependencies = [
    "numpy>=2.2.6",
    "numba>=0.61.2",
    "matplotlib>=3.10.3",
    "scipy>=1.16.2",
    "shapely>=2.1.2",
    ]
    requires-python = ">=3.8"

License: 

    license = "MIT"

## Demo

https://github.com/MavridisChristos/online-deterministic-annealing/tests/demo.py

## Usage 

The ODA architecture is coded in the ODA class inside ```online_deterministic_annealing/oda.py```:
	
	from online_deterministic_annealing.oda import ODA

Regarding the data format, they need to be a list of *(n)* lists of *(m=1)* *d*-vectors (np.arrays):

	train_data_x = [[np.array], [np.array], [np.array], ...]
	train_data_y = [np.array, np.array, np.array, ...] | make sure it is np.atleast1d()

The simplest way to train ODA on a dataset is:

    clf = ODA()

    clf.fit(train_data_x, train_data_y, train_labels,
            test_data_x, test_data_y, test_labels)

    hatX, hatY, hatLabel, hatMode = clf.predict(test_data_x)

Notice that a dataset is not required, and one can train ODA using observations one at a time as follows:

    while stopping_criterion:
        
        # Stop in the next converged configuration
        tl = len(clf.timeline)
        while len(clf.timeline)==tl and not clf.trained:
            train_datum_x, train_datum_y, train_label = system.observe()
            clf.train(train_datum_x, train_datum_y, train_label)

    hatX, hatY, hatLabel, hatMode = clf.predict(test_data_x)

## Clustering

For clustering set:

    train_data_x = [[np.array], [np.array], [np.array], ...]
    train_data_y = [np.atleast1d(0) for td in train_data_x]
    train_labels = [0 for td in train_data_x] 

## Classification

For classification, the labels need to be a list of *(n)* labels, preferably integer numbers (for numba.jit)

    train_data_x = [[np.array], [np.array], [np.array], ...]
    train_data_y = [np.atleast1d(0) for td in train_data_x]
	train_labels = [ int, int , int, ...]

## Regression

For regression (piece-wise constant function approximation) replace:

    observe_xy = 0.9 | set a value between [0,1] to give more weight to x or y error
    train_data_x = [[np.array], [np.array], [np.array], ...]
    train_data_y = [np.array, np.array, np.array, ...]
    train_labels = [0 for td in train_data_x] 

## Simultaneous Classification and Regression (Hybrid Learning)

    train_data_x = [[np.array], [np.array], [np.array], ...]
    train_data_y = [np.array, np.array, np.array, ...]
    train_labels = [ int, int , int, ...] 

## Prediction

    hatX, hatY, hatLabel, hatMode = clf.predict(test_data_x)
    error_clustering, error_regression, error_classification = clf.score(data_x, data_y, labels)

## All Parameters

Data

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
    
Bregman divergence

    - Bregman_phi = ['phi_Eucl']
        # Defines Bregman divergence d_phi. 
        # Values in {'phi_Eucl', 'phi_KL'} (Squared Euclidean distance, KL divergence)


Termination Criteria

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


Temperature Schedule

    - Tmax = [0.9] 
    - Tmin = [1e-2]
        # lambda max min values in [0,1]. T = (1-lambda)/lambda
    - gamma_steady = [0.95] 
        # T' = gamma * T
    - gamma_schedule = [[0.8,0.8]] 
        # Initial updates can be set to reduce faster


Tree Structure

    - node_id = [0] 
        # Tree/branch parent node
    - parent = None 
        # Pointer used to create tree-structured linked list


Regularization: Perturbation and Merging

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


Convergence 

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

Verbose

    - verbose = 2 
        # Values in {0,1,2}    
        # 0: don't show score
        # 1: show score only on tree node splits 
        # 2: show score after every SA convergence 

Numba Jit

    - jit = True
        # Using jit/python for Bregman divergences

## Model Parameters 

Tree Structure

    - self.id
    - self.parent
    - self.children

Status

    - self.K
    - self.T

    - self.perturbed = False
    - self.converged = False
    - self.trained = False

Variables

    - self.x 
    - self.y 
    - self.labels 
    - self.classes
    - self.parameters 
    - self.model 
    - self.optimizer 

    - self.px 
    - self.sx 

History

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

Convergence Parameters (development mode)

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

## Tree Structure and Multiple Resolutions

For multiple resolutions every parameter becomes a list of *m* parameters.
Example for *m=2*:

	Tmax = [0.9, 0.09]
	Tmin = [0.01, 0.0001]

The training data should look like this:

	train_data = [[np.array, np.array, ...], [np.array, np.array, ...], [np.array, np.array, ...], ...]


## Description of the Optimization Algorithm

The **observed data** are represented by a random variable 
$$X: \Omega \rightarrow S\subseteq \mathbb{R}^d$$
defined in a probability space $(\Omega, \mathcal{F}, \mathbb{P})$.

Given a **similarity measure** (which can be any Bregman divergence, e.g., squared Euclidean distance, Kullback-Leibler divergence, etc.) 
$$d:S\rightarrow \mathrm{ri}(S)$$ 
the goal is to **find a set $\mu$ of $M$ codevectors** 
in the input space **such that** the following average distortion measure is minimized: 

$$ \min_\mu  J(\mu) := E[\min_i d(X,\mu_i)] $$
    
For supervised learning, e.g., classification and regression, each codevector $\mu_i$ is associated with a label $c_i$ as well.
This process is equivalent to finding the most suitable set of $M$
local constant models, and results in a 

> **Piecewise-constant approximation (partition) of the input space $S$**.

To construct a learning algorithm that progressively increases the number 
of codevectors $M$ as needed, 
we define a probability space over an infinite number of local models, 
and constraint their distribution using the maximum-entropy principle 
at different levels.

First we need to adopt a probabilistic approach, and a discrete random variable
$$Q:S \rightarrow \mu$$ 
with countably infinite domain $\mu$.

Then we constraint its distribution by formulating the multi-objective optimization:

$$\min_\mu F(\mu) := (1-\lambda) D(\mu) - \lambda H(\mu)$$
where 
$$D(\mu) := E[d(X,Q)] =\int p(x) \sum_i p(\mu_i|x) d_\phi(x,\mu_i) ~\textrm{d}x$$
and
$$H(\mu) := E[-\log P(X,Q)] =H(X) - \int p(x) \sum_i p(\mu_i|x) \log p(\mu_i|x) ~\textrm{d}x $$
is the Shannon entropy.

This is now a problem of finding the locations $\{\mu_i\}$ and the 
corresponding probabilities
$\{p(\mu_i|x)\}:=\{p(Q=\mu_i|X=x)\}$.

> The **Lagrange multiplier $\lambda\in[0,1]$** is called the **temperature parameter** 

and controls the trade-off between $D$ and $H$.
As $\lambda$ is varied, we essentially transition from one solution of the multi-objective optimization 
(a Pareto point when the objectives are convex) to another, and:

> **Reducing the values of $\lambda$ results in a bifurcation phenomenon that increases $M$ and describes an annealing process**.

The above **sequence of optimization problems** is solved for decreasing values of $\lambda$ using a

> Recursive **gradient-free stochastic approximation** algorithm.

The annealing nature of the algorithm contributes to
avoiding poor local minima, 
offers robustness with respect to the initial conditions,
and provides a means 
to progressively increase the complexity of the learning model
through an intuitive bifurcation phenomenon.

## Cite
If you use this work in an academic context, please cite the following:

    @article{mavridis2023annealing,
        author = {Mavridis, Christos and Baras, John S.},
        journal = {IEEE Transactions on Automatic Control},
        title = {Annealing Optimization for Progressive Learning With Stochastic Approximation},
        year = {2023},
        volume = {68},
        number = {5},
        pages = {2862-2874},
        publisher = {IEEE},
    }

    @article{mavridis2023online,
        title = {Online deterministic annealing for classification and clustering},
        author = {Mavridis, Christos and Baras, John S},
        journal = {IEEE Transactions on Neural Networks and Learning Systems},
        year = {2023},
        volume = {34},
        number = {10},
        pages = {7125-7134},
        publisher = {IEEE},
    }
	  
    @article{mavridis2022multi,
        title = {Multi-Resolution Online Deterministic Annealing: A Hierarchical and Progressive Learning Architecture},
        author = {Mavridis, Christos and Baras, John},
        journal = {arXiv preprint arXiv:2212.08189},
        year = {2024},
    }

## Author 

Christos N. Mavridis, Ph.D. \
Division of Decision and Control Systems \
School of Electrical Engineering and Computer Science, \
KTH Royal Institute of Technology \
https://mavridischristos.github.io/ \
```mavridis (at) kth.se``` 
```c.n.mavridis (at) gmail.com``` 