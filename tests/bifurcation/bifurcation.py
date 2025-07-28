#%% Import Modules

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

import os
import sys
sys.path.append(os.path.abspath('.'))

plot_folder = './tests/demo/bifurcation'

#%% Parameters

# Ts_high = np.linspace(0.99,0.09,10)
# Ts_low = np.linspace(0.09,0.005,40)
# Ts = np.concatenate((Ts_high,Ts_low))
Ts = np.linspace(0.99,0.01,50)
ms = np.linspace(0,1,51)
xs= np.array((0.2,0.45,0.55,0.8))

def gibbs(x,ms,T):
    dists = np.abs(ms - x)
    exps = np.exp(-dists * (1-T)/T)
    return exps/np.sum(exps)

def pmu(ms,T):
    gs = [gibbs(x,ms,T) for x in xs]
    return np.sum(gs,axis=0)

def mu_star(ms,T):
    mus = [x*gibbs(x,ms,T) for x in xs]
    return np.sum(mus,axis=0)/pmu(ms,T)

def F(ms,T):
    d = [np.abs(ms - x)**2*gibbs(x,ms,T) for x in xs]
    dsum = np.sum(d,axis=0)
    h = [np.log(gibbs(x,ms,T))*gibbs(x,ms,T) for x in xs]
    hsum = np.sum(h,axis=0)
    return dsum, hsum, (1-T)*dsum+T*hsum

def dF(ms,T):
    dfs = [(ms-x)*gibbs(x,ms,T) for x in xs]
    return np.sum(dfs,axis=0)

def intdF(ms,T):
    return np.cumsum(dF(ms,T)) / len(ms)

def ddF(ms,T):
    dfs = dF(ms,T)
    ddfs = np.diff(dfs)*len(ms)
    return np.append(ddfs,ddfs[-1])

#%% Plot F

for k in range(1):
    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111, autoscale_on=True)
    ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)

    for i,T in enumerate(Ts):
        
        ax.plot(ms,F(ms,T)[0])

    # ax.plot(ms,ms,'k--')
    plt.grid(True)
plt.show()

# %% Plot pmu 

fig = plt.figure(figsize=(5,3))
ax = fig.add_subplot(111, autoscale_on=True)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

for i,T in enumerate(Ts):
    
    ax.plot(ms,pmu(ms,T))
    
plt.show()

# %% Plot mu_star 

fig = plt.figure(figsize=(5,3))
ax = fig.add_subplot(111, autoscale_on=True, aspect='equal')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

for i,T in enumerate(Ts):
    
    ax.plot(ms,mu_star(ms,T))

# ax.plot(ms,ms,'k--')
plt.grid(True)
plt.show()




# %% dF

for i,T in enumerate(Ts):
    
    df = dF(ms,T)
    eqa = np.where(np.diff(np.sign(df))>0)[0]
    meqa = [ms[eq] for eq in eqa]

    fig, (ax, bax) = plt.subplots(1, 2, dpi=300, figsize=(6,3),
                                  gridspec_kw={'width_ratios':[15,1]}
                                  )
    ax.set_xlim(0, 1)
    ax.set_xlabel(r'$X$', 
                  fontsize=14, rotation=0)
    ax.set_ylim(-0.05, 0.05)
    ax.set_ylabel(r'$\frac{{\partial F}}{{\partial \mu}}$', 
                  fontsize=20, rotation=0)
    ax.yaxis.set_label_coords(-0.05, 0.8)
    ax.set_xticks([0,0.25,0.5,0.75,1.0])
    ax.set_xticklabels(['','','','',''])
    ax.tick_params(axis='y', labelsize=14)
    ax.set_yticks([0,-0.02,0.02])
    ax.set_yticklabels(['0','',''])
    bax.tick_params(axis='x', labelsize=14)
    bax.tick_params(axis='y', labelright=True, labelleft=False)
    # pos = bax.get_position()
    # bax.set_position([pos.x0, pos.y0+0.1, pos.width, pos.height - 0.2])
    
    for j,nT in enumerate(Ts[:i]):

        ax.plot(ms,dF(ms,nT),alpha=0.5,linewidth=1, zorder=1)
        bax.bar([f"$\lambda$"],[nT],width=0.1,alpha=0.1, zorder=1)

    ax.plot(ms,dF(ms,T),color='slateblue', alpha=1,linewidth=5, zorder=2)
    bax.bar([f"$\lambda$"],[T],color='slateblue',width=0.1,alpha=1, zorder=2)

    for eq in meqa: 
        ax.scatter(eq,0,color='maroon',marker="o", s=80, zorder=3)
    
    ax.plot(ms,-0.05*np.ones_like(ms),alpha=1,linewidth=5, 
            color='firebrick', zorder=2)
    for x in xs:
        # ax.scatter((x-0.5)*np.exp(-T/(1-T))+0.5,-0.05,color='red',marker="o", s=80, zorder=3)
        ax.scatter(x,-0.05,color='firebrick',marker="o", s=200, zorder=3)
    
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    fig.savefig(plot_folder+f"/{i+1}.png", dpi=300, bbox_inches='tight')

# %% pmu

for i,T in enumerate(Ts):
    
    fig, (ax, bax) = plt.subplots(1, 2, dpi=300, figsize=(6,3),
                                  gridspec_kw={'width_ratios':[15,1]}
                                  )
    ax.set_xlim(0, 1)
    ax.set_xlabel(r'$X$', 
                  fontsize=14, rotation=0)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel(r'$p(\mu)$', 
                  fontsize=14, rotation=0)
    ax.yaxis.set_label_coords(-0.07, 0.9)
    ax.set_xticks([0,0.25,0.5,0.75,1.0])
    ax.set_xticklabels(['','','','',''])
    ax.tick_params(axis='y', labelsize=14)
    ax.set_yticks([0,0.1,0.5])
    ax.set_yticklabels(['0','0.1','0.5'])
    bax.tick_params(axis='x', labelsize=14)
    bax.tick_params(axis='y', labelright=True, labelleft=False)
    # pos = bax.get_position()
    # bax.set_position([pos.x0, pos.y0+0.1, pos.width, pos.height - 0.2])
    
    for j,nT in enumerate(Ts[:i]):

        ax.plot(ms,pmu(ms,nT),alpha=0.5,linewidth=1, zorder=1)
        bax.bar([f"$\lambda$"],[nT],width=0.1,alpha=0.1, zorder=1)

    ax.plot(ms,pmu(ms,T),color='royalblue', alpha=1,linewidth=3, zorder=2)
    bax.bar([f"$\lambda$"],[T],color='royalblue',width=0.1,alpha=1, zorder=2)

    ax.plot(ms,0*np.ones_like(ms),alpha=1,linewidth=5, 
            color='firebrick', zorder=2)
    for x in xs:
        # ax.scatter((x-0.5)*np.exp(-T/(1-T))+0.5,-0.05,color='red',marker="o", s=80, zorder=3)
        ax.scatter(x,0,color='firebrick',marker="o", s=100, zorder=3)
    
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    fig.savefig(plot_folder+f"/{i+1}.png", dpi=300, bbox_inches='tight')

# %% D

for i,T in enumerate(Ts):
    
    fig, (ax, bax) = plt.subplots(1, 2, dpi=300, figsize=(6,3),
                                  gridspec_kw={'width_ratios':[15,1]}
                                  )
    ax.set_xlim(0, 1)
    ax.set_xlabel(r'$X$', 
                  fontsize=14, rotation=0)
    ax.set_ylim(-0.001, 0.025)
    ax.set_ylabel(r'$D(\mu)$', 
                  fontsize=14, rotation=0)
    ax.yaxis.set_label_coords(-0.07, 0.9)
    ax.set_xticks([0,0.25,0.5,0.75,1.0])
    ax.set_xticklabels(['','','','',''])
    ax.tick_params(axis='y', labelsize=14)
    ax.set_yticks([0,0.01,0.02])
    ax.set_yticklabels(['0','0.01','0.02'])
    bax.tick_params(axis='x', labelsize=14)
    bax.tick_params(axis='y', labelright=True, labelleft=False)
    # pos = bax.get_position()
    # bax.set_position([pos.x0, pos.y0+0.1, pos.width, pos.height - 0.2])
    
    for j,nT in enumerate(Ts[:i]):

        ax.plot(ms,F(ms,nT)[0],alpha=0.5,linewidth=1, zorder=1)
        bax.bar([f"$\lambda$"],[nT],width=0.1,alpha=0.1, zorder=1)

    ax.plot(ms,F(ms,T)[0],color='royalblue', alpha=1,linewidth=3, zorder=2)
    bax.bar([f"$\lambda$"],[T],color='royalblue',width=0.1,alpha=1, zorder=2)

    ax.plot(ms,-0.001*np.ones_like(ms),alpha=1,linewidth=5, 
            color='firebrick', zorder=2)
    for x in xs:
        # ax.scatter((x-0.5)*np.exp(-T/(1-T))+0.5,-0.05,color='red',marker="o", s=80, zorder=3)
        ax.scatter(x,-0.001,color='firebrick',marker="o", s=100, zorder=3)
    
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    fig.savefig(plot_folder+f"/{i+1}.png", dpi=300, bbox_inches='tight')

# %% mustar

for i,T in enumerate(Ts):
    
    # eqa = np.where(np.abs(mu_star(ms,T)-ms)<1e-2)[0]
    eqa = np.where(np.diff(np.sign(mu_star(ms,T)-ms))<0)[0]
    meqa = [ms[eq] for eq in eqa]
    meqa = [x for x in meqa if (np.abs(x-0.5)>0.04 or np.abs(x-0.5)<0.01)]

    fig, (ax, bax) = plt.subplots(1, 2, dpi=300, figsize=(4,3),
                                  gridspec_kw={'width_ratios':[15,1]}
                                  )
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_xlabel(r'$\mu_i$', 
                  fontsize=14, rotation=0)
    ax.xaxis.set_label_coords(0.95,-0.01)
    ax.set_ylim(0, 1)
    ax.set_ylabel(r'$\mu_i^\star$', 
                  fontsize=14, rotation=0)
    ax.yaxis.set_label_coords(-0.07, 0.9)
    ax.set_xticks([0,0.25,0.5,0.75,1.0])
    ax.set_xticklabels(['','','','',''])
    ax.tick_params(axis='y', labelsize=14)
    ax.set_yticks([0,0.25,0.5,0.75,1.0])
    ax.set_yticklabels(['','','','',''])
    bax.tick_params(axis='x', labelsize=14)
    bax.tick_params(axis='y', labelright=True, labelleft=False)
    # pos = bax.get_position()
    # bax.set_position([pos.x0, pos.y0+0.1, pos.width, pos.height - 0.2])
    
    for j,nT in enumerate(Ts[:i]):

        ax.plot(ms,mu_star(ms,nT),alpha=0.5,linewidth=1, zorder=1)
        bax.bar([f"$\lambda$"],[nT],width=0.1,alpha=0.1, zorder=1)

    ax.plot(ms,mu_star(ms,T),color='royalblue', alpha=1,linewidth=3, zorder=2)
    bax.bar([f"$\lambda$"],[T],color='royalblue',width=0.1,alpha=1, zorder=2)

    for eq in meqa: 
        ax.scatter(eq,eq,color='black',marker="o", s=40, zorder=3)
    
    # ax.plot(ms,0*np.ones_like(ms),alpha=1,linewidth=5, 
    #         color='firebrick', zorder=2)
    for x in xs:
        # ax.scatter((x-0.5)*np.exp(-T/(1-T))+0.5,-0.05,color='red',marker="o", s=80, zorder=3)
        ax.scatter(x,0,color='black',marker="o", s=100, zorder=3)
        ax.scatter(0,x,color='black',marker="o", s=100, zorder=3)
    
    ax.plot(ms,ms,alpha=1,linewidth=1, 
            color='black', zorder=2)
    
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    fig.savefig(plot_folder+f"/{i+1}.png", dpi=300, bbox_inches='tight')

#%%




























#%% Plot dF colors

if False:

    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111, autoscale_on=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 0.05)

    for T in Ts[:5]:
        x = ms
        y = dF(ms,T)
        magnitudes = dF(ms,T)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection([], cmap='seismic', linewidth=1)
        lc.set_segments(segments)
        lc.set_array(magnitudes)

        ax.add_collection(lc)

    plt.grid(True)
    plt.show()   

# %% Quiver dF

if False:

    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111, autoscale_on=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for T in Ts:
            
        ax.quiver(ms, T*np.ones_like(ms), dF(ms,T), np.zeros_like(ms),dF(ms,T), cmap='viridis')

    plt.grid(True)
    plt.show()







#%% Animations

# %% mu_star gif

if False:

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Initialize the line object to be updated
    line, = ax.plot([], [], lw=2)

    # Animate function
    def animate(i):
        T = Ts[i]
        line.set_data(ms, mu_star(ms, T))
        return line,

    # Create the animation
    ani = FuncAnimation(fig, animate, frames=len(Ts), interval=100, blit=True)

    plt.show()
    ani.save(plot_folder+'/animation.gif', writer='imagemagick')

# %% dF gif

if False:

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    for x in xs:
        ax.scatter(x,0)
    plt.grid(True)

    # Initialize the line object to be updated
    line, = ax.plot([], [], lw=2)
    scat, = ax.scatter([],[],lw=2)

    # Animate function
    def animate(i):
        T = Ts[i]
        fun = dF(ms, T)
        eqa = fun[np.abs(fun)<1e-9]
        line.set_data(ms, dF(ms, T))
        for eq in eqa: 
            scat.set_data(eq)
        return line,

    # Create the animation
    ani = FuncAnimation(fig, animate, frames=len(Ts), interval=100, blit=True)

    plt.show()
    ani.save(plot_folder+'/animation.gif', writer='imagemagick')


#%% dF gif color

if False:
    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111, autoscale_on=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 0.05)
    lc = LineCollection([], cmap='vanimo', linewidth=1)
    ax.add_collection(lc)

    def animate(i):
        T = Ts[i]
        x = ms
        y = dF(ms,T)
        magnitudes = 10*dF(ms,T)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc.set_segments(segments)
        lc.set_array(magnitudes)

        # ax.add_collection(lc)

        return lc,

    ani = FuncAnimation(fig, animate, frames=len(Ts), interval=50, blit=True)

    plt.grid(True)
    plt.show()   

    ani.save(plot_folder+'/animation.gif')


# %%
