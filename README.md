# Online Deterministic Annealing (ODA)

> A general-purpose learning model designed to meet the needs of applications in which computational resources are limited, and robustness and interpretability are prioritized.

> Constitutes an **online** prototype-based learning algorithm based on annealing optimization that is formulated as an recursive **gradient-free** stochastic approximation algorithm.

> Can be viewed as an interpretable and progressively growing competitive-learning neural network model.

>Applications include online unsupervised and supervised learning [1], regression,
>reinforcement learning [2], 
>adaptive graph partitioning [3], and swarm leader detection.

![](tests/tutorials/img/2Dapproximation/app0.png)
![](tests/tutorials/img/2Dapproximation/app1.png)
![](tests/tutorials/img/2Dapproximation/app2.png)
![](tests/tutorials/img/2Dapproximation/original.png)

## Contact 

Christos N. Mavridis, Ph.D. \
Division of Decision and Control Systems \
School of Electrical Engineering and Computer Science, \
KTH Royal Institute of Technology \
https://mavridischristos.github.io/ \
```mavridis (at) kth.se``` 

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

$$\min_\mu F(\mu) := (1-T) D(\mu) - T H(\mu)$$
where 
$$D(\mu) := E[d(X,Q)] =\int p(x) \sum_i p(\mu_i|x) d_\phi(x,\mu_i) ~\textrm{d}x$$
and
$$H(\mu) := E[-\log P(X,Q)] =H(X) - \int p(x) \sum_i p(\mu_i|x) \log p(\mu_i|x) ~\textrm{d}x $$
is the Shannon entropy.

This is now a problem of finding the locations $\{\mu_i\}$ and the 
corresponding probabilities
$\{p(\mu_i|x)\}:=\{p(Q=\mu_i|X=x)\}$.

> The **Lagrange multiplier $T\in[0,1]$** is called the **temperature parameter** 

and controls the trade-off between $D$ and $H$.
As $T$ is varied, we essentially transition from one solution of the multi-objective optimization 
(a Pareto point when the objectives are convex) to another, and:

> **Reducing the values of $T$ results in a bifurcation phenomenon that increases $M$ and describes an annealing process** [1, 2].

The above **sequence of optimization problems** is solved for decreasing values of T using a

> Recursive **gradient-free stochastic approximation** algorithm.

The annealing nature of the algorithm contributes to
avoiding poor local minima, 
offers robustness with respect to the initial conditions,
and provides a means 
to progressively increase the complexity of the learning model
through an intuitive bifurcation phenomenon.
	
## Usage (Outdated)

The ODA architecture is coded in the ODA class inside ```OnlineDeterministicAnnealing/oda.py```:
	
	from oda import ODA

Regarding the data format, they need to be a list of *(n)* lists of *(m=1)* *d*-vectors (np.arrays):

	train_data = [[np.array], [np.array], [np.array], ...]

The simplest way to train ODA on a dataset is:

    clf = ODA(train_data=train_data,train_labels=train_labels)
    clf.fit(test_data=test_data,test_labels=test_labels)

Notice that a dataset is not required, and one can train ODA using observations one at a time as follows:

    tl = len(clf.timeline)
    # Stop in the next converged configuration
    while len(clf.timeline)==tl and not clf.trained:
        train_datum, train_label = system.observe()
        clf.train(train_datum,train_label,test_data=test_data,test_labels=test_labels)
	
## Classification

For classification, the labels need to be a list of *(n)* labels, preferably integer numbers (for numba.jit)

	train_labels = [ int, int , int, ...]
            
## Clustering

For clustering replace:

    train_labels = [0 for i in range(len(train_labels))] 

## Regression

For regression (piece-wise constant function approximation) replace:

    train_labels = [ np.array, np.array , np.array, ...]
    clf = ODA(train_data=train_data,train_labels=train_labels,regression=True)

## Prediction

    prediction = clf.predict(test_datum)
    error = clf.score(test_data, test_labels)

## Useful Parameters (Outdated)

### Cost Function

> Bregman Divergence: 
    
    # Values in {'phi_Eucl', 'phi_KL'} (Squared Euclidean distance, KL divergence)
    Bregman_phi = ['phi_Eucl'] 

### Termination Criteria

> Minimum Termperature 

    Tmin = [1e-4]

> Limit in node's children. After that stop growing

    Kmax = [50]

> Desired training error

    error_threshold = [0.0]
    # Stop when reached 'error_threshold_count' times
    error_threshold_count = [2]
    # Make sure keepscore > 2

> ODA vs Soft-Clustering vs LVQ

    # Values in {0,1,2,3}
    # 0:ODA update
    # 1:ODA until Kmax. Then switch to 2:soft clustering with no perturbation/merging 
    # 2:soft clustering with no perturbation/merging 
    # 3: LVQ update (hard-clustering) with no perturbation/merging
    lvq=[0]

> Verbose

    # Values in {0,1,2,3}    
    # 0: don't compute or show score
    # 1: compute and show score only on tree node splits 
    # 2: compute score after every SA convergence and use it as a stopping criterion
    # 3: compute and show score after every SA convergence and use it as a stopping criterion
    keepscore = 3

## Model Progression

The history of all the intermediate models trained is stored in:

    clf.myY, clf.myYlabels, clf.myK, clf.myTreeK, clf.myT, clf.myLoops, clf.myTime, clf.myTrainError, clf.myTestError

## Tree Structure and Multiple Resolutions

For multiple resolutions every parameter becomes a list of *m* parameters.
Example for *m=2*:

	Tmax = [0.9, 0.09]
	Tmin = [0.01, 0.0001]

The training data should look like this:

	train_data = [[np.array, np.array, ...], [np.array, np.array, ...], [np.array, np.array, ...], ...]

## Tutorials (Outdated)

> [Clustering](https://colab.research.google.com/github/MavridisChristos/OnlineDeterministicAnnealing/blob/main/tutorials/tutorial-clustering.ipynb)

![](tests/tutorials/img/clustering/app0.png)
![](tests/tutorials/img/clustering/app1.png)
![](tests/tutorials/img/clustering/app2.png)

> [Classification](https://colab.research.google.com/github/MavridisChristos/OnlineDeterministicAnnealing/blob/main/tutorials/tutorial-classification.ipynb)

![](tests/tutorials/img/classification/app0.png)
![](tests/tutorials/img/classification/app1.png)
![](tests/tutorials/img/classification/app2.png)

> [Regression](https://colab.research.google.com/github/MavridisChristos/OnlineDeterministicAnnealing/blob/main/tutorials/tutorial-regression.ipynb)

![](tests/tutorials/img/2Dapproximation/app0.png)
![](tests/tutorials/img/2Dapproximation/app1.png)
![](tests/tutorials/img/2Dapproximation/app2.png)
![](tests/tutorials/img/2Dapproximation/original.png)

## Citing
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

or

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
	  