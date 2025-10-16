#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Show Domain
Tree-Structured (Multi-Resolution) Online Deterministic Annealing for Classification and Clustering
Christos Mavridis & John Baras,
Department of Electrical and Computer Engineering, University of Maryland
<mavridis@umd.edu>
"""

#%% Import Modules

import numpy as np
import matplotlib.pyplot as plt
from tests.utils import bounded_voronoi as bvoronoi

import os
# from oda import ODA 

# plt.ion()
plt.ioff() # turn off interactive mode: only show figures with plt.show()
plt.close('all')

#%% Read results

def show(clf, train_data, train_labels, 
        res=[0], # resolutions 
        # Plotting Options
        plot_folder = './domain',
        plot_fig = False,
        save_fig = True
        ):


    if save_fig:
        os.makedirs(plot_folder, exist_ok=True)

    # i=0
    # for r in reversed(list(set(res))):
    #     plt_data(clf=clf, plot_counter=i, res=r, 
    #              plot_fig=plot_fig, save_fig=save_fig, plot_folder=plot_folder)
    #     i+=1
        
    ## Initialize plot counters
    for i in range(len(clf.timeline[1:])):
        nid = clf.timeline[i+1]
        node = clf
        for child_id in nid[1:]:
            node=node.children[child_id]   
        node.plt_counter = 0 # init to 1: read after first convergence
        
    for i in range(len(clf.timeline)):
        nid = clf.timeline[i]
        node = clf
        for child_id in nid[1:]:
            node=node.children[child_id]    
        plt_oda(clf=node, train_data=train_data, train_labels=train_labels, res=res, plot_counter=i, bias=len(list(set(res))), 
                plot_fig=plot_fig, save_fig=save_fig, plot_folder=plot_folder)
            
def show_instance(clf, train_data, train_labels, instance=0,
        res=[0], # resolutions 
        # Plotting Options
        plot_folder = './domain',
        plot_fig = False,
        save_fig = True
        ):

    if save_fig:
        os.makedirs(plot_folder, exist_ok=True)

    # i=0
    # for r in reversed(list(set(res))):
    #     plt_data(clf=clf, plot_counter=i, res=r, 
    #              plot_fig=plot_fig, save_fig=save_fig, plot_folder=plot_folder)
    #     i+=1
        
    ## Initialize plot counters
    for i in range(len(clf.timeline[1:])):
        nid = clf.timeline[i+1]
        node = clf
        for child_id in nid[1:]:
            node=node.children[child_id]   
        node.plt_counter = 0 # init to 1: read after first convergence
    
    for i in range(len(clf.timeline)):
        nid = clf.timeline[i]
        node = clf
        for child_id in nid[1:]:
            node=node.children[child_id]    
        if i != instance:
            node.plt_counter += 1
        else:
            ax = plt_oda(clf=node, train_data=train_data, train_labels=train_labels, res=res, plot_counter=i, bias=len(list(set(res))), 
                    plot_fig=plot_fig, save_fig=save_fig, plot_folder=plot_folder)
            break
        
    return ax

def show_resolutions(clf, train_data, train_labels, 
        res=[0], # resolutions 
        # Plotting Options
        plot_folder = './domain',
        plot_fig = False,
        save_fig = True
        ):

    if save_fig:
        os.makedirs(plot_folder, exist_ok=True)

    # Mesh data    
    delta = 0.01
    xm = np.arange(0.0, 1.0, delta)
    ym = np.arange(0.0, 1.0, delta)
    mesh_data = [np.array([xmi,ymi]) for xmi in xm for ymi in ym]
    mesh_data = [[np.array(proj(td)),td] for td in mesh_data]
    # mesh_data = [[td[r] for r in res] for td in mesh_data]
    
        
    ## Plot Data Space for every resolution (0:low, 1:high)
    # i=0
    # for r in reversed(list(set(res))):
    #     plt_data(clf=clf, plot_counter=i, res=r, 
    #              plot_fig=plot_fig, save_fig=save_fig, plot_folder=plot_folder)
    #     i+=1
    axs = []
    for i in range(len(res)):
        ax = plt_oda_resolutions(clf=clf, train_data=train_data, train_labels=train_labels, recursive = i, mesh_data=mesh_data, 
                plot_fig=plot_fig, save_fig=save_fig, plot_folder=plot_folder)
        axs.append(ax)
    
    return axs

def data(clf, train_data, train_labels, 
        res=[0], # resolutions 
        # Plotting Options
        plot_folder = './domain',
        plot_fig = False,
        save_fig = True
        ):

    if save_fig:
        os.makedirs(plot_folder, exist_ok=True)

    i=0
    for r in reversed(list(set(res))):
        plt_data(clf=clf, train_data=train_data, train_labels=train_labels, plot_counter=i, res=r, 
                  plot_fig=plot_fig, save_fig=save_fig, plot_folder=plot_folder)
        i+=1
            
def show_data(data,labels, 
            # Plotting Options
            plot_folder = './domain',
            plot_fig = False,
            save_fig = True,
            title = 'Data Space'
            ):
    
    if save_fig:
        os.makedirs(plot_folder, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=True, aspect='equal')
    
    data = [d[-1] for d in data]
    
    x_plot = [data[i] for i in range(len(data)) if labels[i] == 0]
    y_plot = [data[i] for i in range(len(data)) if labels[i] == 1]
    
    ax.plot([x_plot[i][0] for i in range(len(x_plot))],
            [x_plot[i][1] for i in range(len(x_plot))],'ks',alpha=aa)
    ax.plot([y_plot[i][0] for i in range(len(y_plot))],
            [y_plot[i][1] for i in range(len(y_plot))],'r^',alpha=aa)
   
    plt.title(title)
    
    # save figure
    if save_fig:
        fig.savefig(f'./{plot_folder}/{title}.png', format = 'png')
    
    # show or hide figure
    if plot_fig:
        plt.show()
    else:
        plt.close('all')
        
    return ax
            
#%% Algorithm Plot

# Colors 
a = 1 # centroids
aa = 0.06 # data
aaa = 0.003 # regions

def proj(x): 
    u = np.array([1,1])
    u = u/np.linalg.norm(u)
    x = np.array(x)
    return np.array([np.dot(u,x)])

def proj_back(x):
    u = np.array([1,1])
    u = u/np.linalg.norm(u)
    x = np.array(x)
    return np.array(u*x)

def plt_oda(clf, train_data, train_labels, res, plot_counter=1, bias=0, 
            plot_fig=False, save_fig=True, plot_folder='.'):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=True, aspect='equal')#,
                     #xlim=(-2, 3), ylim=(-5, 5)) #aspect='equal',
    # ax.grid(True)
    plt.xticks([],'')
    plt.yticks([],'')
    
    root = clf
    while root.parent:
        root = root.parent
    
    data = [td[-1] for td in train_data]
    labels = train_labels
    
    # Plot Data   
    
    x_plot = [data[i] for i in range(len(data)) if labels[i] == 0]
    y_plot = [data[i] for i in range(len(data)) if labels[i] == 1]
    
    ax.plot([x_plot[i][0] for i in range(len(x_plot))],
            [x_plot[i][1] for i in range(len(x_plot))],'ks',alpha=aa)
    ax.plot([y_plot[i][0] for i in range(len(y_plot))],
            [y_plot[i][1] for i in range(len(y_plot))],'r^',alpha=aa)
    
    if len(res)>1 and res[0]<res[1] and len(clf.id)<2:
        data = [proj_back(dt[0]) for dt in train_data]
    
        labels = train_labels
        
        x_plot = [data[i] for i in range(len(data)) if labels[i] == 0]
        y_plot = [data[i] for i in range(len(data)) if labels[i] == 1]
        
        ax.plot([x_plot[i][0] for i in range(len(x_plot))],
                [x_plot[i][1] for i in range(len(x_plot))],'ks',alpha=aa)
        ax.plot([y_plot[i][0] for i in range(len(y_plot))],
                [y_plot[i][1] for i in range(len(y_plot))],'r^',alpha=aa)
    
    # Mesh
    delta = 0.01
    xm = np.arange(0.0, 1.0, delta)
    ym = np.arange(0.0, 1.0, delta)
    
    # Call lower Resolution
    if clf.parent:
        
        clf.plt_counter += 1
        
        # Plot partion at R1 
        centroids = clf.parent.myX[clf.parent.plt_counter-1]
        clabels = clf.parent.myLabels[clf.parent.plt_counter-1]
        black_data = []
        red_data = []
        tree_data = [[] for i in clf.parent.children]
        for xi in xm:
            for yi in ym:
                di = proj([xi,yi]) if len(centroids[0])<2 else np.array([xi,yi])
                dists = [clf.BregmanD(di,yj) for yj in centroids]
                j = np.argmin(dists)
                tree_data[j].append(np.array([xi,yi]))
                if clabels[j] == 0:
                    black_data.append(np.array([xi,yi]))
                else:
                    red_data.append(np.array([xi,yi]))
                   
        ax.plot([b[0] for b in black_data],
                [b[1] for b in black_data],'ks',alpha=aaa)
        ax.plot([r[0] for r in red_data],
                [r[1] for r in red_data],'rs',alpha=aaa)
        
        # Plot partion at R2 
        for c in range(len(clf.parent.children)):
            if clf.parent.children[c].plt_counter>0:
                centroids = clf.parent.children[c].myX[clf.parent.children[c].plt_counter]
                clabels = clf.parent.children[c].myLabels[clf.parent.children[c].plt_counter]
                black_data = []
                red_data = []
                for datum in tree_data[c]:
                    di = proj(datum) if len(centroids[0])<2 else datum
                    dists = [clf.BregmanD(di,yj) for yj in centroids]
                    j = np.argmin(dists)
                    if clabels[j] == 0:
                        black_data.append(datum)
                    else:
                        red_data.append(datum)
                           
                ax.plot([b[0] for b in black_data],
                        [b[1] for b in black_data],'ks',alpha=0.5*aaa)
                ax.plot([r[0] for r in red_data],
                        [r[1] for r in red_data],'rs',alpha=1.5*aaa)
    
                # Plot Centroids
                if len(centroids[0])<2:
                    centroids = [proj_back(cd) for cd in centroids]
                for i in range(len(centroids)):
                    # centroids
                    if clabels[i]==0:
                        class_color = 'k'
                    else:
                        class_color = 'r'
                    ax.plot(centroids[i][0],centroids[i][1],color = class_color, marker = 'D',alpha=a)
    else:
        
        # Plot partion at R1 
        centroids = clf.myX[clf.plt_counter]
        clabels = clf.myLabels[clf.plt_counter]
        black_data = []
        red_data = []
        for xi in xm:
            for yi in ym:
                di = proj([xi,yi]) if len(centroids[0])<2 else np.array([xi,yi])
                dists = [clf.BregmanD(di,np.array(yj)) for yj in centroids]
                j = np.argmin(dists)
                if clabels[j] == 0:
                    black_data.append(np.array([xi,yi]))
                else:
                    red_data.append(np.array([xi,yi]))
                   
        ax.plot([b[0] for b in black_data],
                [b[1] for b in black_data],'ks',alpha=aaa)
        ax.plot([r[0] for r in red_data],
                [r[1] for r in red_data],'rs',alpha=aaa)
    
        # Plot Centroids
        if len(centroids[0])<2:
            centroids = [proj_back(cd) for cd in centroids]

        # Bounded Voronoi
        if len(centroids)>1:
            if len(centroids)<4:
                vcentroids = centroids.copy() + [np.array([-10,-10]), np.array([10,10])]
            else:
                vcentroids = centroids.copy()
            
            lower = 0.0
            upper = 1.0
            bounding_polygon = np.array([[lower, lower], [lower, upper], [upper, upper], [upper, lower]])
            regions = bvoronoi.bounded_voronoi(vcentroids,bounding_polygon,radius=20)
            regions = regions[:len(centroids)]

            # plot
            for r,region in enumerate(regions):
                bvoronoi.voronoi_plot_2d(ax,region,color='gray', width=1) # vcentroids[r]

        for i in range(len(centroids)):
            # centroids
            if clabels[i]==0:
                class_color = 'k'
            else:
                class_color = 'r'
            ax.plot(centroids[i][0],centroids[i][1],color = class_color, marker = 'D',alpha=a)
    
    title=f'Obs.: {clf.myLoops[clf.plt_counter]:04d}, T = .{int(clf.myT[clf.plt_counter]*10000):04d}, ' \
              f'K = {root.myTreeK[plot_counter]:03d}'
        
    plt.text(0.5, 0.025, title, horizontalalignment='center',verticalalignment='center', fontsize=12,fontweight='bold')    
    if len(clf.myTrainError)>clf.plt_counter:
        if len(clf.classes)>1:
            acctxt = f'Acc.: {1-clf.myTrainError[clf.plt_counter][2]:.3f}'
        else:
            acctxt = f'Err.: {clf.myTrainError[clf.plt_counter][0]:.3f}'
        plt.text(0.8, 0.95, acctxt, horizontalalignment='center',verticalalignment='center', fontsize=12,fontweight='bold')    
    
    
    if not clf.parent:
        clf.plt_counter += 1
        
    # save figure
    if save_fig:
        fig.savefig(f'./{plot_folder}/{plot_counter+bias:03d}.png', format = 'png')
    
    # show or hide figure
    if plot_fig:
        plt.show()
    
    if not plot_fig:
        plt.close('all')

    return ax

def plt_oda_resolutions(clf, train_data, train_labels, recursive=0, mesh_data = [],
            plot_fig=False, save_fig=True, plot_folder='.'):
    
    ## Figure
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=True, aspect='equal')#,
                     #xlim=(-2, 3), ylim=(-5, 5)) #aspect='equal',
    # ax.grid(True)
    plt.xticks([],'')
    plt.yticks([],'')
    
    
    
    # Plot Data   
    data = [td[-1] for td in train_data]
    labels = train_labels
    
    x_plot = [data[i] for i in range(len(data)) if labels[i] == 0]
    y_plot = [data[i] for i in range(len(data)) if labels[i] == 1]
    
    ax.plot([x_plot[i][0] for i in range(len(x_plot))],
            [x_plot[i][1] for i in range(len(x_plot))],'ks',alpha=aa)
    ax.plot([y_plot[i][0] for i in range(len(y_plot))],
            [y_plot[i][1] for i in range(len(y_plot))],'r^',alpha=aa)
    
    # Plot low D data as well
    if len(data[0])<2 :
        
        ldata = [proj_back(dt[0]) for dt in train_data]
    
        llabels = train_labels
        
        x_plot = [ldata[i] for i in range(len(ldata)) if llabels[i] == 0]
        y_plot = [ldata[i] for i in range(len(ldata)) if llabels[i] == 1]
        
        ax.plot([x_plot[i][0] for i in range(len(x_plot))],
                [x_plot[i][1] for i in range(len(x_plot))],'ks',alpha=aa)
        ax.plot([y_plot[i][0] for i in range(len(y_plot))],
                [y_plot[i][1] for i in range(len(y_plot))],'r^',alpha=aa)
    


    # Plot Voronoi (Mesh)
    for r in range(recursive+1):
        black_data = []
        red_data = []
        for di in mesh_data:
            ci = clf.predict(di,r)[2]
            if ci == 0:
                black_data.append(di[-1])
            else:
                red_data.append(di[-1])
                   
        ax.plot([b[0] for b in black_data],
                [b[1] for b in black_data],'ks',alpha=aaa)
        ax.plot([r[0] for r in red_data],
                [r[1] for r in red_data],'rs',alpha=aaa)

    # Codevectors
    centroids,cy,clabels = clf.codebook(recursive)
    if len(centroids[0])<2:
        centroids = [proj_back(cd) for cd in centroids]
    for i in range(len(centroids)):
        # centroids
        if clabels[i]==0:
            class_color = 'k'
        else:
            class_color = 'r'
        ax.plot(centroids[i][0],centroids[i][1],color = class_color, marker = 'D',alpha=a)
    
    # title=f'Obs.: {clf.myLoops[clf.plt_counter]:04d}, T = .{int(clf.myT[clf.plt_counter]*10000):04d}, ' \
    #           f'K = {root.myTreeK[plot_counter]:03d}'
    title = f'Voronoi Resolution {recursive}'
    plt.title(title)
    
    # plt.text(0.5, 0.015, title, horizontalalignment='center',verticalalignment='center', fontsize=12,fontweight='bold')    
    # if len(clf.myTrainError)>clf.plt_counter:
    #     acctxt = f'Acc.: {1-clf.myTrainError[clf.plt_counter]:.3f}'
    #     plt.text(0.8, 0.95, acctxt, horizontalalignment='center',verticalalignment='center', fontsize=12,fontweight='bold')    
    
    
    # if not clf.parent:
    #     clf.plt_counter += 1
        
    # save figure
    if save_fig:
        fig.savefig(f'./{plot_folder}/{recursive:03d}-Voronoi.png', format = 'png')
    
    # show or hide figure
    if plot_fig:
        plt.show()
    
    if not plot_fig:
        plt.close('all')
        
    return ax

def plt_data(clf, train_data, train_labels, plot_counter=0, res=0, 
             plot_fig=False, save_fig=True, plot_folder='.'):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=True, aspect='equal')
    
    # Plot Data   
    data = [td[-1] for td in train_data]
        
    labels = train_labels
    
    x_plot = [data[i] for i in range(len(data)) if labels[i] == 0]
    y_plot = [data[i] for i in range(len(data)) if labels[i] == 1]
    
    ax.plot([x_plot[i][0] for i in range(len(x_plot))],
            [x_plot[i][1] for i in range(len(x_plot))],'ks',alpha=aa)
    ax.plot([y_plot[i][0] for i in range(len(y_plot))],
            [y_plot[i][1] for i in range(len(y_plot))],'r^',alpha=aa)
    
    if res==0:
        data = [proj_back(dt[0]) for dt in train_data]
    
        x_plot = [data[i] for i in range(len(data)) if labels[i] == 0]
        y_plot = [data[i] for i in range(len(data)) if labels[i] == 1]
        
        ax.plot([x_plot[i][0] for i in range(len(x_plot))],
                [x_plot[i][1] for i in range(len(x_plot))],'ks',alpha=aa)
        ax.plot([y_plot[i][0] for i in range(len(y_plot))],
                [y_plot[i][1] for i in range(len(y_plot))],'r^',alpha=aa)
    
    space = 'Low Resolution' if res==0 else 'High Resolution'
    title=f'#{plot_counter:03d}: Data Space - ' + space 
    plt.title(title)
    
    # save figure
    if save_fig:
        fig.savefig(f'./{plot_folder}/{plot_counter:03d}.png', format = 'png')
    
    # show or hide figure
    if plot_fig:
        plt.show()
    else:
        plt.close('all')
        
    return ax
    
#%%

"""
Show Domain
Tree-Structured (Multi-Resolution) Online Deterministic Annealing for Classification and Clustering
Christos Mavridis & John Baras,
Department of Electrical and Computer Engineering, University of Maryland
<mavridis@umd.edu>
"""
