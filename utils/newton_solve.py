# newton_solve.py
# Author: Alejandro Diaz

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

def newton_solve(F, Jac, y0, tol=1e-10, maxit=100, print_hist=False, sparse=True):
    '''
    Solve F(y) = 0 using Newton's method.
    inputs:
        F: function F(y)
        Jac: jacobian of F
        y0: initial guess for Newton's method
        tol: [optional] solver tolerance. Default is 1e-10
        maxit: [optional] maximum number of iterations. Default is 100
        print_hist: [optional] Set to True to print iteration history. Default is False
        sparse: [optional] Set to True if Jac returns a sparse matrix. Default is True

    outputs:
        y: solution of F(y) = 0
        res_vecs: array of residual vectors F(y_i) at each iteration, i.e. 
                    res_vecs[i] = F(y_i) = residual at ith iteration
        res_hist: residual iteration history
        step_hist: stepsize history
        iter: number of iterations
    '''
    
    res_hist  = np.zeros(maxit+1)
    step_hist = np.zeros(maxit+1)
    iter = 0
    
    ny  = y0.size
    y   = y0
    
    res = F(y)                # current residual
    
    res_vecs = [res]
    res_hist[iter] = np.linalg.norm(res)
    
    if print_hist:
        print('iter       Stepsize       Residual')
        print('{0:4d}      {1:10.3e}     {2:10.3e}'.format(iter, 0.0, res_hist[iter]))
    
    # chooses a sparse or dense linear solver depending on problem
    if sparse: 
        linsolve = spsolve
    else: 
        linsolve = np.linalg.solve
        
    while (res_hist[iter] >= tol) & (iter < maxit):
        
        jac = Jac(y)
            
        step = -linsolve(jac,res)    # computes line search direction
        
        # Armijo line search
        stepsize = 1.0
        ytmp = y+stepsize*step
        
        rtmp = F(ytmp)      # computes residual at temporary point ytmp
 
        while (np.linalg.norm(rtmp) >= (1 - 2e-4*stepsize)*res_hist[iter]) & (stepsize >= 1e-10):
            stepsize *= 0.5
            ytmp = y + stepsize*step
            rtmp = F(ytmp)

        y   = ytmp
        res = rtmp
        iter += 1
        
        res_vecs.append(res)
        res_hist[iter] = np.linalg.norm(rtmp)
        step_hist[iter] = stepsize
        if print_hist:
            print('{0:4d}      {1:10.3e}     {2:10.3e}'.format(iter, stepsize, res_hist[iter]))
            
        if stepsize < 1e-10:
            print(f'No stepsize found at iteration {iter}.')
            break
            
    res_vecs  = np.vstack(res_vecs)
    res_hist  = res_hist[0:iter+1]
    step_hist = step_hist[0:iter+1]
    if iter == maxit:
        print('Newton failed to converge in ' + str(maxit) + ' iterations.')

    return y, res_vecs, res_hist, step_hist, iter

def newton_solve_combinedFJac(FJac, y0, tol=1e-10, maxit=100, print_hist=False, sparse=True):
    '''
    Solve F(y) = 0 using Newton's method.
    inputs:
        FJac: function that returns F(y) and the jacobian of F at y, i.e.
                F, Jac = FJac(y)
        y0: initial guess for Newton's method
        tol: [optional] solver tolerance. Default is 1e-10
        maxit: [optional] maximum number of iterations. Default is 100
        print_hist: [optional] Set to True to print iteration history. Default is False
        sparse: [optional] Set to True if Jac returns a sparse matrix. Default is True

    outputs:
        y: solution of F(y) = 0
        res_vecs: array of residual vectors F(y_i) at each iteration, i.e. 
                    res_vecs[i] = F(y_i) = residual at ith iteration
        res_hist: residual iteration history
        step_hist: stepsize history
        iter: number of iterations
    '''
    
    res_hist  = np.zeros(maxit+1)
    step_hist = np.zeros(maxit+1)
    iter = 0
    
    ny  = y0.size
    y   = y0
    
    res, jac = FJac(y)                # current residual
    
    res_vecs = [res]
    res_hist[iter] = np.linalg.norm(res)
    
    if print_hist:
        print('iter       Stepsize       Residual')
        print('{0:4d}      {1:10.3e}     {2:10.3e}'.format(iter, 0.0, res_hist[iter]))
    
    # chooses a sparse or dense linear solver depending on problem
    if sparse: 
        linsolve = spsolve
    else: 
        linsolve = np.linalg.solve
        
    while (res_hist[iter] >= tol) & (iter < maxit):
        
        step = -linsolve(jac,res)    # computes line search direction
        
        # Armijo line search
        stepsize = 1.0
        ytmp = y+stepsize*step
        
        rtmp, jtmp = FJac(ytmp)      # computes residual at temporary point ytmp
 
        while (np.linalg.norm(rtmp) >= (1 - 2e-4*stepsize)*res_hist[iter]) & (stepsize >= 1e-10):
            stepsize *= 0.5
            ytmp = y + stepsize*step
            rtmp, jtmp = FJac(ytmp)

        y   = ytmp
        res = rtmp
        jac = jtmp
        iter += 1
        
        res_vecs.append(res)
        res_hist[iter] = np.linalg.norm(rtmp)
        step_hist[iter] = stepsize
        if print_hist:
            print('{0:4d}      {1:10.3e}     {2:10.3e}'.format(iter, stepsize, res_hist[iter]))
            
        if stepsize < 1e-10:
            print(f'No stepsize found at iteration {iter}.')
            break
            
    res_vecs  = np.vstack(res_vecs)
    res_hist  = res_hist[0:iter+1]
    step_hist = step_hist[0:iter+1]
    if iter == maxit:
        print('Newton failed to converge in ' + str(maxit) + ' iterations.')

    return y, res_vecs, res_hist, step_hist, iter