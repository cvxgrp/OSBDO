import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from osbdo.utils import *
from osbdo.cvx_sol import *



################ plotting functions ##################    
    
def plot_finite_memory(*, memory_bounds, memory, h_true, iters=None, \
                        eps={'eps_rel':10**(-2),'eps_abs':10**(-3)}, title='', fontsize=10, filename=None):
    if iters is None:
        iters = len(memory_bounds[memory[0]]['uk'])
    color = ['b','g','r','c','m','y','k','w']
    fig, ax1 = plt.subplots()
    fig.set_dpi(100)
    lwd = 0.7
    ax1.grid()
    plt.title(title,fontsize=fontsize)
    for i in range(len(memory)):
        m = memory[i]
        Uk = np.array(memory_bounds[m]['uk'][:iters])
        Lk = np.array(memory_bounds[m]['lk'][:iters])
        start_point = 0
        for j in range(iters):
            if Uk[j]==np.inf or Lk[j]==np.inf or np.abs(Uk[j])<1e-5 or np.abs(Lk[j])<1e-5:
                start_point = j+1
        Uk = Uk[start_point:]
        Lk = Lk[start_point:]
        gap1 = (Uk-h_true)/np.abs(h_true)
        gap2 = (Uk-Lk)/np.minimum(np.abs(Uk), np.abs(Lk))
        for j in range(len(Uk)):
            if Uk[j]*Lk[j]<=0:
                gap2[j] = np.inf
        stop = np.inf
        for k in range(len(Uk)):
            if Uk[k]-Lk[k] < eps['eps_abs'] or \
                            (Uk[k]-Lk[k]<eps['eps_rel']*min(np.abs(Uk[k]),np.abs(Lk[k])) and Uk[k]*Lk[k]>0):
                stop = k
                break
        ax1.axvline(stop, color=color[i], linestyle=':')
        ax1.plot(gap1,linewidth=lwd,linestyle='--',color=color[i])
        if m >= iters-1:
            ax1.plot(gap2,linewidth=lwd, linestyle='-',label=r'm = $\infty$', color=color[i])
        else:
            ax1.plot(gap2,linewidth=lwd, linestyle='-',label='m = %d'%m, color=color[i])
    ax1.set_xlabel(r'$k$',fontsize=fontsize)
    ax1.set_yscale('log')
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels , loc=0, fontsize=fontsize)
    plt.show() 
    if filename:
        fig.savefig(filename, bbox_inches='tight')
        
def plot_finite_memory_abs(*, memory_bounds, memory, h_true, iters=None, \
                eps={'eps_rel':10**(-2),'eps_abs':10**(-3)}, title='', fontsize=10, filename=None):
    if iters is None:
        iters = len(memory_bounds[memory[0]]['uk'])
    color = ['b','g','r','c','m','y','k','w']
    fig, ax1 = plt.subplots()
    fig.set_dpi(100)
    lwd = 0.7
    ax1.grid()
    plt.title(title,fontsize=fontsize)
    for i in range(len(memory)):
        m = memory[i]
        Uk = np.array(memory_bounds[m]['uk'][:iters])
        Lk = np.array(memory_bounds[m]['lk'][:iters])
        start_point = 0
        for j in range(iters):
            if Uk[j]==np.inf or Lk[j]==np.inf:
                start_point = j+1
        Uk = Uk[start_point:]
        Lk = Lk[start_point:]
        gap1 = Uk-h_true
        gap2 = Uk-Lk
        stop = np.inf
        for k in range(len(Uk)):
            if Uk[k]-Lk[k] < eps['eps_abs'] or \
                            (Uk[k]-Lk[k]<eps['eps_rel']*min(np.abs(Uk[k]),np.abs(Lk[k])) and Uk[k]*Lk[k]>0):
                stop = k
                break
        ax1.axvline(stop, color=color[i], linestyle=':')
        ax1.plot(gap1,linewidth=lwd,linestyle='--',color=color[i])
        if m >= iters-1:
            ax1.plot(gap2,linewidth=lwd, linestyle='-',label=r'm = $\infty$', color=color[i])
        else:
            ax1.plot(gap2,linewidth=lwd, linestyle='-',label='m = %s'%str(m), color=color[i])
    ax1.set_xlabel(r'$k$',fontsize=fontsize)
    ax1.set_yscale('log')
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels , loc=0, fontsize=fontsize)
    plt.show() 
    if filename:
        fig.savefig(filename, bbox_inches='tight')
        
def plot_agent_failure(*, fail_bounds, fail_probability, h_true, iters=None, \
                    eps={'eps_rel':10**(-2),'eps_abs':10**(-3)}, title='', fontsize=10, filename=None):
    if iters is None:
        iters = len(fail_bounds[fail_probability[0]]['uk'])
    color = ['b','g','r','c','m','y','k','w']
    fig, ax1 = plt.subplots()
    fig.set_dpi(100)
    lwd = 0.7
    ax1.grid()
    plt.title(title,fontsize=fontsize)
    for i in range(len(fail_probability)):
        prob = fail_probability[i]
        Uk = np.array(fail_bounds[prob]['uk'][:iters])
        Lk = np.array(fail_bounds[prob]['lk'][:iters])
        start_point = 0
        for j in range(iters):
            if Uk[j]==np.inf or Lk[j]==np.inf or np.abs(Uk[j])<1e-5 or np.abs(Lk[j])<1e-5:
                start_point = j+1
        Uk = np.array(fail_bounds[prob]['uk'][start_point:])
        Lk = np.array(fail_bounds[prob]['lk'][start_point:])
        gap1 = (Uk-h_true)/np.abs(h_true)
        gap2 = (Uk-Lk)/np.minimum(np.abs(Uk), np.abs(Lk))
        for j in range(len(Uk)):
            if Uk[j]*Lk[j]<0:
                gap2[j] = np.inf
        stop = np.inf
        for k in range(len(Uk)):
            if Uk[k]-Lk[k] < eps['eps_abs'] or \
                                (Uk[k]-Lk[k]<eps['eps_rel']*min(np.abs(Uk[k]),np.abs(Lk[k])) and Uk[k]*Lk[k]>0):
                stop = k
                break
        ax1.axvline(stop, color=color[i], linestyle=':')
        ax1.plot(gap1,linewidth=lwd,linestyle='--',color=color[i])
        ax1.plot(gap2,linewidth=lwd, linestyle='-',label=r'$p$ = %s'%str(prob), color=color[i])
    ax1.set_xlabel(r'$k$',fontsize=fontsize)
    ax1.set_yscale('log')
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels , loc=0, fontsize=fontsize)
    plt.show() 
    if filename:
        fig.savefig(filename, bbox_inches='tight')
        
def plot_agent_failure_abs(*, fail_bounds, fail_probability, h_true, iters=None, \
                    eps={'eps_rel':10**(-2),'eps_abs':10**(-3)}, title='', fontsize=10, filename=None):
    if iters is None:
        iters = len(fail_bounds[fail_probability[0]]['uk'])
    color = ['b','g','r','c','m','y','k','w']
    fig, ax1 = plt.subplots()
    fig.set_dpi(100)
    lwd = 0.7
    ax1.grid()
    plt.title(title,fontsize=fontsize)
    for i in range(len(fail_probability)):
        prob = fail_probability[i]
        Uk = np.array(fail_bounds[prob]['uk'][:iters])
        Lk = np.array(fail_bounds[prob]['lk'][:iters])
        gap1 = Uk-h_true
        gap2 = Uk-Lk
        stop = np.inf
        for k in range(iters):
            if Uk[k]-Lk[k] < eps['eps_abs'] or \
                            (Uk[k]-Lk[k]<eps['eps_rel']*min(np.abs(Uk[k]),np.abs(Lk[k])) and Uk[k]*Lk[k]>0):
                stop = k
                break
        ax1.axvline(stop, color=color[i], linestyle=':')
        ax1.plot(gap1,linewidth=lwd,linestyle='--',color=color[i])
        ax1.plot(gap2,linewidth=lwd, linestyle='-',label=r'$p$ = %s'%str(prob), color=color[i])
    ax1.set_xlabel(r'$k$',fontsize=fontsize)
    ax1.set_yscale('log')
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels , loc=0, fontsize=fontsize)
    plt.show() 
    if filename:
        fig.savefig(filename, bbox_inches='tight')
   
def plot_true_abs_gap_uk(*, lk, uk, h_true, title='', \
                        eps={'eps_rel': 10**(-2), 'eps_abs': 10**(-3)}, fontsize=10,  file_name=None):
    Uk = np.array(uk)
    Lk = np.array(lk)
    fig, ax1 = plt.subplots()
    fig.set_dpi(100)
    lwd = 0.7
    ax1.grid()
    plt.title(title,fontsize=fontsize)
    gap1 = Uk-h_true
    gap2 = Uk-Lk
    stop = np.inf
    for i in range(len(Uk)):
        if Uk[i]-Lk[i] < eps['eps_abs'] or \
                        (Uk[i]-Lk[i] < eps['eps_rel']*min(np.abs(Uk[i]),np.abs(Lk[i])) and Uk[i]*Lk[i]>0):
            stop = i
            break
    ax1.axvline(stop, color='r', linestyle=':')
    ax1.plot(gap1,linewidth=lwd,label=r'$\omega_{true}^{k}$',color='b')
    ax1.set_xlabel(r'$k$',fontsize=fontsize)
    ax1.set_yscale('log')
    ax1.plot(gap2,linewidth=lwd, color='black',label=r'$\omega^{k}$')
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels , loc=0, fontsize=fontsize)
    plt.show() 
    if file_name:
        fig.savefig(file_name, bbox_inches='tight')
        
def plot_true_rel_gap_uk(*, lk, uk, h_true, title='', \
                            eps={'eps_rel': 10**(-2), 'eps_abs': 10**(-3)}, fontsize=10,  file_name=None):
    Uk = np.array(uk)
    Lk = np.array(lk)
    start_point = 0
    for j in range(len(Uk)):
        if Uk[j]==np.inf or Lk[j]==np.inf or np.abs(Uk[j])<1e-5 or np.abs(Lk[j])<1e-5:
            start_point = j+1
    Uk = Uk[start_point:]
    Lk = Lk[start_point:]
    fig, ax1 = plt.subplots()
    fig.set_dpi(100)
    lwd = 0.7
    ax1.grid()
    plt.title(title,fontsize=fontsize)
    gap1 = (Uk-h_true)/np.abs(h_true)
    gap2 = (Uk-Lk)/np.minimum(np.abs(Uk), np.abs(Lk))
    stop = np.inf
    for i in range(len(Uk)):
        if Uk[i]*Lk[i]<0:
            gap2[i] = np.inf
    stop=np.inf
    for i in range(len(Uk)):
        if Uk[i]-Lk[i] < eps['eps_abs'] \
                            or (Uk[i]-Lk[i]<eps['eps_rel']*min(np.abs(Uk[i]),np.abs(Lk[i])) and Uk[i]*Lk[i]>0):
            stop = i
            break
    ax1.axvline(stop, color='r', linestyle=':')
    ax1.plot(gap1,linewidth=lwd,label=r'$\omega_{true}^{k}$',color='b')
    ax1.set_xlabel(r'$k$',fontsize=fontsize)
    ax1.set_yscale('log')
    ax1.plot(gap2,linewidth=lwd, color='black',label=r'$\omega^{k}$')
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels , loc=0, fontsize=fontsize)
    plt.show() 
    if file_name:
        fig.savefig(file_name, bbox_inches='tight')
    
    