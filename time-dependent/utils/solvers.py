import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve 
from time import time

def newton_solve(FJac, y0, tol=1e-10, maxit=20, print_hist=False):
    '''
    Solve F(y) = 0 using Newton's method.
    inputs: 
        FJac:       function that returns F(y) and the jacobian of F at y, i.e.
                       F, Jac = FJac(y)
        y0:         initial guess for Newton's method
        tol:        [optional] solver relative tolerance. Default is 1e-10
        maxit:      [optional] maximum number of iterations. Default is 100
        print_hist: [optional] Set to True to print iteration history. Default is False

    outputs:
        y:         solution of F(y) = 0
        res_vecs:  array of residual vectors F(y_i) at each iteration, i.e. 
                   res_vecs[i] = F(y_i) = residual at ith iteration
        res_hist:  residual iteration history
        step_hist: stepsize history
        iter:      number of iterations
        flag:      termination flag
                    flag=0:    solver converged
                    flag=1:    solver failed to converge in maxit iterations
                    flag=2:    stepsize reduced to below 1e-8
    '''
    
    start = time()
    flag  = 0
    res_hist  = [] #np.zeros(maxit+1)
    step_hist = []# np.zeros(maxit+1)
    iter = 0
    
    ny  = y0.size
    y   = y0
    runtime = time()-start
    
    res, jac, rjtime = FJac(y)                # current residual
    runtime += rjtime
    
    start    = time()
    res_vecs = [res]
    norm_res  = np.linalg.norm(res)
#     norm_res0 = norm_res
    res_hist.append(norm_res)#/norm_res0)
    
    if print_hist:
        print('iter       Stepsize       ||r||')#/||r0||')
        print('{0:4d}      {1:10.3e}     {2:10.3e}'.format(iter, 0.0, res_hist[iter]))
    runtime += time()-start
    
    # chooses a sparse or dense linear solver depending on problem
    if sp.issparse(jac): 
        linsolve = spsolve
    else: 
        linsolve = np.linalg.solve

    while (norm_res >= tol) & (iter < maxit):
#     while (norm_res >= tol*norm_res0) & (iter < maxit):
        start = time()
        step = -linsolve(jac,res)    # computes line search direction
        
        # Armijo line search
        stepsize = 1.0
        ytmp = y+stepsize*step
        runtime += time()-start
        
        rtmp, jtmp, rjtime = FJac(ytmp)      # computes residual at temporary point ytmp
        norm_rtmp = np.linalg.norm(rtmp)
        runtime += rjtime
        
        while (norm_rtmp >= (1 - 2e-4*stepsize)*norm_res) & (stepsize >= 1e-8):
            start = time()
            stepsize *= 0.5
            ytmp = y + stepsize*step
            runtime += time()-start
            rtmp, jtmp, rjtime = FJac(ytmp)
            start = time()
            norm_rtmp = np.linalg.norm(rtmp)
            runtime += rjtime + time()-start
        
        start = time()
        y   = ytmp
        res = rtmp
        jac = jtmp
        norm_res = norm_rtmp
        iter += 1
        
        res_vecs.append(res)
        res_hist.append(norm_res)
#         res_hist.append(norm_res/norm_res0)
        step_hist.append(stepsize)
        if print_hist:
            print('{0:4d}      {1:10.3e}     {2:10.3e}'.format(iter, stepsize, norm_res))#/norm_res0))
            
        if stepsize < 1e-8:
            flag=2
            break
        runtime += time()-start
    
    start = time()
    res_vecs  = np.vstack(res_vecs)
#     res_hist  = res_hist[0:iter+1]
#     step_hist = step_hist[0:iter+1]
    if (iter == maxit) & (res_hist[-1]>=tol):
        flag=1
    runtime += time()-start
    
    return y, res_vecs, res_hist, step_hist, iter, runtime, flag