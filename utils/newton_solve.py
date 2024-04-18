# newton_solve.py

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from time import time

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

def newton_solve_combinedFJac(FJac, y0, tol=1e-10, maxit=100, print_hist=False):
    '''
    Solve F(y) = 0 using Newton's method.
    inputs:
        FJac: function that returns F(y) and the jacobian of F at y, i.e.
                F, Jac = FJac(y)
        y0: initial guess for Newton's method
        tol: [optional] solver tolerance. Default is 1e-10
        maxit: [optional] maximum number of iterations. Default is 100
        print_hist: [optional] Set to True to print iteration history. Default is False

    outputs:
        y: solution of F(y) = 0
        res_vecs: array of residual vectors F(y_i) at each iteration, i.e. 
                    res_vecs[i] = F(y_i) = residual at ith iteration
        res_hist: residual iteration history
        step_hist: stepsize history
        iter: number of iterations
    '''
    
    start = time()
    
    res_hist  = np.zeros(maxit+1)
    step_hist = np.zeros(maxit+1)
    iter = 0
    
    ny  = y0.size
    y   = y0
    runtime = time()-start
    
    res, jac, rjtime = FJac(y)                # current residual
    runtime += rjtime
    
    start = time()
    res_vecs = [res]
    res_hist[iter] = np.linalg.norm(res)
    
    if print_hist:
        print('iter       Stepsize       Residual')
        print('{0:4d}      {1:10.3e}     {2:10.3e}'.format(iter, 0.0, res_hist[iter]))
    runtime += time()-start
    
    # chooses a sparse or dense linear solver depending on problem
    if sp.issparse(jac): 
        linsolve = spsolve
    else: 
        linsolve = np.linalg.solve
        
    while (res_hist[iter] >= tol) & (iter < maxit):
        start = time()
        step = -linsolve(jac,res)    # computes line search direction
        
        # Armijo line search
        stepsize = 1.0
        ytmp = y+stepsize*step
        runtime += time()-start
        
        rtmp, jtmp, rjtime = FJac(ytmp)      # computes residual at temporary point ytmp
        runtime += rjtime
        
        while (np.linalg.norm(rtmp) >= (1 - 2e-4*stepsize)*res_hist[iter]) & (stepsize >= 1e-10):
            start = time()
            stepsize *= 0.5
            ytmp = y + stepsize*step
            runtime += time()-start
            rtmp, jtmp, rjtime = FJac(ytmp)
            runtime += rjtime
        
        start = time()
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
        runtime += time()-start
    
    start = time()
    res_vecs  = np.vstack(res_vecs)
    res_hist  = res_hist[0:iter+1]
    step_hist = step_hist[0:iter+1]
    if iter == maxit:
        print('Newton failed to converge in ' + str(maxit) + ' iterations.')
    runtime += time()-start
    
    return y, res_vecs, res_hist, step_hist, iter, runtime

def gauss_newton(res_jac, w0, tol=1e-3, maxit=20, print_hist=False):
    '''
    Solve min 0.5*||r(x)||^2 using Gauss-Newton method. 

    inputs:
        w0: initial guess for Newton's method
        tol: [optional] solver tolerance. Default is 1e-10
        maxit: [optional] maximum number of iterations. Default is 20
        print_hist: [optional] Set to True to print iteration history. Default is False

    outputs:
        w: solution of min ||r(x)||
        conv_hist: convergence iteration history: conv_hist[i] = ||R'r(x_i)||
        step_hist: stepsize history
        iter: number of iterations
    '''

    conv_hist = np.zeros(maxit+1)
    step_hist = np.zeros(maxit+1)
    iter = 0

    w   = w0
    res, jac = res_jac(w)
    norm_res = np.dot(res, res)
    conv_hist[iter] = np.linalg.norm(jac.T@res)

    if print_hist:
        print('iter       Stepsize       ||RTr||')
        print('{0:4d}      {1:10.3e}     {2:10.3e}'.format(iter, 0.0, conv_hist[iter]))

    while (conv_hist[iter] >= tol) & (iter < maxit):
        step, minval = la.lstsq(jac, -res)[0:2]

        # Armijo line search
        stepsize = 1.0
        wtmp = w+stepsize*step

        rtmp, jtmp = res_jac(wtmp)
        norm_rtmp = np.dot(rtmp, rtmp)
        while (norm_rtmp >= norm_res+2e-4*stepsize*(minval-norm_res)) & (stepsize >= 1e-10):
            stepsize *= 0.5
            wtmp = w + stepsize*step
            rtmp, jtmp = res_jac(wtmp)
            norm_rtmp = np.dot(rtmp, rtmp)

        w   = wtmp
        res = rtmp
        jac = jtmp
        norm_res = norm_rtmp
        iter += 1

        conv_hist[iter] = np.linalg.norm(jac.T@res)
        step_hist[iter] = stepsize
        if print_hist:
            print('{0:4d}      {1:10.3e}     {2:10.3e}'.format(iter, stepsize, conv_hist[iter]))

        if stepsize < 1e-10:
            print(f'No stepsize found at iteration {iter}.')
            break

    conv_hist = conv_hist[0:iter+1]
    step_hist = step_hist[0:iter+1]
    if iter == maxit:
        print('Newton failed to converge in ' + str(maxit) + ' iterations.')

    return w, conv_hist, step_hist, iter