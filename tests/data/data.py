#!/usr/bin/env python
"""
Multi-Resolution (2) Binary Classification with underline Gaussian Distributions
Christos Mavridis & John Baras,
Electrical and Computer Engineering Dept.,
University of Maryland
"""

#%% Import Modules

import pickle
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# from scipy.stats import multivariate_normal

# plt.ioff() # turn off interactive mode: only show figures with plt.show()
plt.close('all')

#%%

save_file = 'data.pkl'

# Gaussian Sampling?
gauss_sampling = True
# gauss_sampling = False

# Number of samples per Gaussian
ns = 120

test_ratio = 20.0/100

# Gaussian Density 2D, Symmetric
def gauss_dens(c,s,X,Y):
    # pos = np.empty(X.shape + (2,))
    # pos[:, :, 0] = X
    # pos[:, :, 1] = Y
    # dens = multivariate_normal.pdf(pos, mean=c, cov=np.diag((s,s)))
    dens = 1/(2*np.pi*s) * np.exp(-((X-c[0])**2 + (Y-c[1])**2)/(2.0*s**2))
    return dens

b = 3
c = 0

centers = [[b-1.4,b+0.4],[b+2.2,b+1.0],[b-0.4+c,b-1.4],[b+1.4+c,b-0.4],[b+0.4+c,b+2.0]]
sigmas = [0.75, 0.5, 0.35, 0.75, 0.5]
sigmas = [0.35, 0.25, 0.15, 0.35, 0.25]
sigmas = [0.20, 0.15, 0.10, 0.20, 0.15]

# Contour alpha
aa = 0.05
# samples alpha
aaa = 0.4

np.random.seed(0)
# np.random.seed(13)

#%% Sample Data Set

data = []
labels = []

if gauss_sampling:
    
    # class o
    cx = np.array(centers[0])
    sx = sigmas[0]
    for i in range(ns):
        data.append( np.random.multivariate_normal(cx, [[sx,0],[0,sx]]) )
        labels.append(0)
                    
    # class o
    cx = np.array(centers[1])
    sx = sigmas[1]
    for i in range(ns):
        data.append( np.random.multivariate_normal(cx, [[sx,0],[0,sx]]) )
        labels.append(0)
              
    # class o
    cx = np.array(centers[2])
    sx = sigmas[2]
    for i in range(ns):
        data.append( np.random.multivariate_normal(cx, [[sx,0],[0,sx]]) )
        labels.append(0)                      
    
    # class x
    cx = np.array(centers[3])
    sx = sigmas[3]
    for i in range(ns):
        data.append( np.random.multivariate_normal(cx, [[sx,0],[0,sx]]) )
        labels.append(1)
              
    # class x
    cx = np.array(centers[4])
    sx = sigmas[4]
    for i in range(ns):
        data.append( np.random.multivariate_normal(cx, [[sx,0],[0,sx]]) )
        labels.append(1)
        
else:
    
    def arc_point(c, r, theta):
        c = np.array(c)
        d = np.array([r*np.cos(theta), r*np.sin(theta)])
        return c + d
    
    # class o
    cx = np.array(centers[0])
    for r in [0.5,1]:
        for theta in np.arange(0,2*np.pi,np.pi/5):
            data.append(arc_point(cx,r,theta))
            labels.append(0)
            
    # class o
    cx = np.array(centers[1])
    for r in [0.3,0.5]:
        for theta in np.arange(0,2*np.pi,np.pi/5):
            data.append(arc_point(cx,r,theta))
            labels.append(0)
            
    # class o
    cx = np.array(centers[2])
    for r in [0.1,0.3]:
        for theta in np.arange(0,2*np.pi,np.pi/5):
            data.append(arc_point(cx,r,theta))
            labels.append(0)
            
    # class x
    cx = np.array(centers[3])
    for r in [0.5,1]:
        for theta in np.arange(0,2*np.pi,np.pi/5):
            data.append(arc_point(cx,r,theta))
            labels.append(1)
    
    # class x
    cx = np.array(centers[4])
    for r in [0.3,0.5]:
        for theta in np.arange(0,2*np.pi,np.pi/5):
            data.append(arc_point(cx,r,theta))
            labels.append(1)

#% Convert Data

# data = x_data + y_data    
# labels = [0]*len(x_data) + [1]*len(y_data)

#%% Map to [0,1] and shuffle

# map to [0,1]
train_min = np.min(np.min(data,0)) #-0.1
train_max = np.max(np.max(data,0)) #+0.1
train_domain = train_max-train_min 
# add small margins so that we avoid 0.0 values (for KL divergence)
train_min = train_min - 0.05*train_domain
train_max = train_max + 0.05*train_domain
train_domain = train_max-train_min 
# transform
data = (data-train_min)/train_domain
centers = (centers-train_min)/train_domain
sigmas = sigmas/train_domain


# shuffle data
shi = np.arange(len(data))
np.random.shuffle(shi)

data = [data[i] for i in shi]
labels = [labels[i] for i in shi]

# train_data = [data[i] for i in shi]     
# train_labels = [labels[i] for i in shi]       
# train_samples = len(train_data) 

# test_data = []
# test_labels = []
# test_samples = 1#len(test_data) 


# Split into training and testing sets
train_samples = int(np.floor(len(data)*(1-test_ratio)))
test_samples = int(len(data)-train_samples)

train_data = data[:][:train_samples]
train_labels = labels[:][:train_samples]

test_data = data[:][train_samples:]
test_labels = labels[:][train_samples:]


#%% Algorithm Plot

# Create new Figure
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     # xlim=(b-3, b+3), ylim=(b-3, b+3))
                     xlim=(0, 1), ylim=(0, 1))
# ax.grid(True)
plt.xticks([0,1],'')
plt.yticks([0,1],'')

# data in 2D space
x_plot = [data[i] for i in range(len(data)) if labels[i] == 0]
y_plot = [data[i] for i in range(len(data)) if labels[i] == 1]

## plot data 
ax.plot([x_plot[i][0] for i in range(len(x_plot))],
        [x_plot[i][1] for i in range(len(x_plot))],'k.',alpha=aaa)
ax.plot([y_plot[i][0] for i in range(len(y_plot))],
        [y_plot[i][1] for i in range(len(y_plot))],'r.',alpha=aaa)

## Contours
delta = 0.005
# xm = np.arange(b-3.0, b+3.0, delta)
# ym = np.arange(b-3.0, b+3.0, delta)
xm = np.arange(0.0, 1.0, delta)
ym = np.arange(0.0, 1.0, delta)
Xm, Ym = np.meshgrid(xm, ym)

test = gauss_dens((0.5,0.5),0.05,Xm,Ym)
# class o
cx = np.array(centers[0])
sx = sigmas[0]
Zm =  gauss_dens(cx,sx,Xm,Ym)
ax.contour(Xm, Ym, Zm, alpha=aa, colors='k', levels=[1e-9, 1e-3, 1e-1, 1])

# class o
cx = np.array(centers[1])
sx = sigmas[1]
Zm =  gauss_dens(cx,sx,Xm,Ym)
ax.contour(Xm, Ym, Zm, alpha=aa, colors='k', levels=[1e-9, 1e-3, 1e-1, 1])

# class o
cx = np.array(centers[2])
sx = sigmas[2]
Zm =  gauss_dens(cx,sx,Xm,Ym)
ax.contour(Xm, Ym, Zm, alpha=aa, colors='k', levels=[1e-9, 1e-3, 1e-1, 1])
        
# class x
cx = np.array(centers[3])
sx = sigmas[3]
Zm =  gauss_dens(cx,sx,Xm,Ym)
ax.contour(Xm, Ym, Zm, alpha=aa, colors='r', levels=[1e-9, 1e-3, 1e-1, 1])

# class x
cx = np.array(centers[4])
sx = sigmas[4]
Zm =  gauss_dens(cx,sx,Xm,Ym)
ax.contour(Xm, Ym, Zm, alpha=aa, colors='r', levels=[1e-9, 1e-3, 1e-1, 1])
    

plt.show()
    
#%% Save Data

def proj(x): 
    u = np.array([1,1])
    u = u/np.linalg.norm(u)
    x = np.array(x)
    return np.dot(u,x)

# def proj_back(x):
#     u = np.array([1,1])
#     u = u/np.linalg.norm(u)
#     x = np.array(x)
#     return u*x

train_data = [[np.array([proj(td)]),td] for td in train_data]
test_data = [[np.array([proj(td)]),td] for td in test_data]
    

# For high Resolution only:
# train_data = [[np.array(td)] for td in train_data]
# test_data = [[np.array(td)] for td in test_data]

# Save results to file 
mydata = [train_data,train_labels,test_data,test_labels]
            
with open(save_file, mode='wb') as file:
    pickle.dump(mydata, file) 
   
    