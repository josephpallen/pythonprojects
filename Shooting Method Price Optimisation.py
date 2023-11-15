# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 13:46:40 2023

@author: joeal
"""
import numpy as np

from matplotlib import pyplot as plt

from scipy.optimize import brentq

from scipy.integrate import odeint

"""
We are faced with a BVP which we need to convert into an IVP and a root finding problem.

"""

def rhs(q,t,L,h,options):
    """
    The right hand side of the expanded Euler-Lagrange equation defining the problem.
    
    Parameters
    ----------
    
    q: array
       Solution vector q = [y,dy]
    t: scalar
       Time
    L: function
       The Lagrangian
    h: scalar
       Step length for finite differencing the Lagrangian
    options: dictionary
             Optional arguments for L
        
    Returns
    -------
    
    dq/dt: array
           Derivative of q
    """    
    assert((not np.any(np.isnan(q))) and np.all(np.isfinite(q))\
    and np.all(np.isreal(q))), \
    "q must be real, finite"
    
    assert((not np.any(np.isnan(t))) and np.all(np.isfinite(t)) and\
    np.all(np.isreal(t)) and np.isscalar(t) and t >= 0), \
    "t must be real, finite and scalar"
    
    assert(hasattr(L, '__call__')), \
    "L must be a callable function"
    
    assert((not np.any(np.isnan(h))) and np.all(np.isfinite(h)) and\
    np.all(np.isreal(h)) and np.isscalar(h) and h > 0), \
    "h must be real, finite and scalar"
    
    assert(isinstance(options, dict)), \
    "options must be a dictionary"    

# Setting out the rhs of the chain rule expanded lagrangian
# In order to convert from a BVP to an IVP, we call dy/dt z. Derivatives of L with respect to y and z are calculated using a step length
# h and using finite differencing to compute derivatives either side of +,- h 
    dqdt=np.zeros_like(q) # Initialising the derivative vector
    
    y=q[0]
    
    dy=q[1]
    
    d2Lddy2=(L(t,y,dy+h,options)+L(t,y,dy-h,options)-L(t,y,dy,options)-L(t,y,dy,options))/h**2
    
    if np.allclose(d2Lddy2,0):
        #In order to avoid division by zero, returns true if two arrays are element-wise equal within a tolerance.
       d2Lddy2=h**2
        
    d2Ldtddy=(L(t+h,y,dy+h,options)-L(t-h,y,dy+h,options)-L(t+h,y,dy-h,options)+L(t-h,y,dy-h,options))/(4*h**2)
       
        
    d2Ldyddy=(L(t,y+h,dy+h,options)-L(t,y-h,dy+h,options)-L(t,y+h,dy-h,options)+L(t,y-h,dy-h,options))/(4*h**2)
        
            
    dLdy=(L(t,y+h,dy,options)-L(t,y-h,dy,options))/(2*h)        
                    
    
        
    dqdt[0]=dy
    
    dqdt[1]=(dLdy-(d2Ldtddy+dy*d2Ldyddy))/d2Lddy2
    
    return dqdt




def lagrangian(t,y,dy,options):
    """
    The Lagrangian defining the problem
    
    Parameters
    ----------
    
    t: scalar
       Time
    y: scalar
       Solution at time t
    dy: scalar
        Derivative of solution at time t
        
    Returns
    -------
    
    L: scalar
       The Lagrangian at this point
    """
    assert((not np.any(np.isnan(t))) and np.all(np.isfinite(t)) and\
    np.all(np.isreal(t)) and np.isscalar(t)), \
    "t must be real, finite and scalar"
    
    assert((not np.any(np.isnan(y))) and np.all(np.isfinite(y)) and\
    np.all(np.isreal(y)) and np.isscalar(y)), \
    "y must be real, finite and scalar"
    
    assert((not np.any(np.isnan(dy))) and np.all(np.isfinite(dy)) and\
    np.all(np.isreal(dy)) and np.isscalar(dy)), \
    "dy must be real, finite and scalar"
    
    assert(isinstance(options, dict)), \
    "options must be a dictionary"
    
    assert('alpha' in options and 'beta' in options), \
    "For this function, the options must contain 'alpha' and 'beta'"    
    
    alpha=options['alpha']
    
    beta=options['beta']
    
    return alpha*dy**2+beta*(t**2-1)*dy**3-y




def el_shooting_error(z,d,bound_cond,L,h,options,tol):
    """
    Error in the given boundary conditions for the shooting method
    
    Parameters
    ----------
    
    z: scalar
       Guess for dy(0) (The aiming of the shot)
    d: array
       domain of t where the solution lies
    bound_cond: array
                Dirichlet boundary conditions for y
    L: function
       The Lagrangian
    h: scalar
       Step length for finite differencing the Lagrangian
    options: dictionary
             Optional arguments for the Lagrangian
    tol: scalar
         Tolerance for odeint
        
    Returns
    -------
    
    el_shooting_error: scalar
                       Error at the right boundary given the guess z
    """
    assert((not np.any(np.isnan(z))) and np.all(np.isfinite(z)) and\
    np.all(np.isreal(z)) and np.isscalar(z)), \
    "z must be real, finite and scalar"
    
    assert((not np.any(np.isnan(d))) and np.all(np.isfinite(d))\
    and np.all(np.isreal(d)) and len(d)==2), \
    "d must be real, finite, length 2"
    
    assert((not np.any(np.isnan(bound_cond))) and np.all(np.isfinite(bound_cond))\
    and np.all(np.isreal(bound_cond)) and len(bound_cond)==2), \
    "Boundary conditions must be real, finite, length 2"
    
    assert(hasattr(L, '__call__')), \
    "L must be a callable function"
    
    assert((not np.any(np.isnan(h))) and np.all(np.isfinite(h)) and\
    np.all(np.isreal(h)) and np.isscalar(h) and h > 0), \
    "h must be real, finite and scalar"
    
    assert(isinstance(options, dict)), \
    "options must be a dictionary"
    
    assert((not np.any(np.isnan(tol))) and np.all(np.isfinite(tol)) and\
    np.all(np.isreal(tol)) and np.isscalar(tol) and tol > 0), \
    "tol must be real, finite and scalar"
    
    q_start=np.array([bound_cond[0],z])
    
    sol=odeint(rhs,q_start,[d[0],d[1]],args=(L,h,options),rtol=tol,atol=tol)
 # I've chosen odeint as the adjustable step feature helps with the convergence later in this coursework                
    y_finish=sol[-1, 0]
    
    return y_finish-bound_cond[1]



def el_shooting(L,d,bound_cond,N,options,tol=1e-4):
    """
    Shooting method for solving Euler-Lagrange equations
    
    Parameters
    ----------
    
    L: function
       The Lagrangian
    d: array
       domain of t where the solution lies
    bound_cond: array
                Boundary conditions for y(t)
    N: scalar
       Number of intervals for finite differencing
    options: dictionary
             Optional arguments for L
    tol: scalar
         Tolerance for odeint and root finding
        
    Returns
    -------
    
    t,y
    """
    assert(hasattr(L, '__call__')), \
    "L must be a callable function"
    
    assert((not np.any(np.isnan(d))) and np.all(np.isfinite(d))\
    and np.all(np.isreal(d)) and len(d)==2), \
    "d must be real, finite, length 2"
    
    assert((not np.any(np.isnan(bound_cond))) and np.all(np.isfinite(bound_cond))\
    and np.all(np.isreal(bound_cond)) and len(bound_cond)==2), \
    "Boundary conditions  must be real, finite, length 2"
    
    assert((not np.any(np.isnan(N))) and np.all(np.isfinite(N)) and\
    np.all(np.isreal(N)) and np.isscalar(N) and np.allclose(N, int(N))), \
    "N must be integer, finite and scalar"
    
    assert(isinstance(options, dict)), \
    "options must be a dictionary"
    
    assert((not np.any(np.isnan(tol))) and np.all(np.isfinite(tol)) and\
    np.all(np.isreal(tol)) and np.isscalar(tol) and tol > 0), \
    "tol must be real, finite and scalar"
    
    t,dt=np.linspace(d[0],d[1],N,retstep=True)
    
    h=tol
    
    z=brentq(el_shooting_error,-0.5,0.5,args=(d,bound_cond,L,h,options,tol),xtol=1e-4*tol,rtol=1e-7*tol) 
# For the non linear root find I've chosen this black-box method as to minimise the the lines of code needed, plus it is easy to use               
    q_start=np.array([1,z])
    
    sol=odeint(rhs,q_start,t,args=(L,h,options),rtol=tol,atol=tol)
    
    return t,sol[:,0] #Splicing the the solution in one vector 



# Maximising profit for the company over two models

# Initialising the finite differencing grid
d=[0,1]

bound_cond=[1,0.9]

N=200 #Number of points for which a solution is formed, N=1/h




options1={'alpha':5,'beta': 5} # for the case alpha=beta
options2={'alpha':1.75,'beta':5}# for the case alpha≠beta

t,y1=el_shooting(lagrangian,d,bound_cond,N,options1)
t,y2=el_shooting(lagrangian,d,bound_cond,N,options2)

plt.figure()
plt.plot(t, y1,color='red')
plt.xlabel(r"$t$")
plt.ylabel(r"$y(t)$")
plt.title('For the case of α=β=5')
plt.figure()
plt.plot(t, y2,color='green')
plt.xlabel(r"$t$")
plt.ylabel(r"$y(t)$")
plt.title('For the case of α=1.75 and β=5')
plt.show()




# Convergence for the second case
# In order to test for convergence we need to factor in all the possible step sizes and tolerances. I do not have the exact solution so
# I check that the algorithm converges to something by looking at the residuals between a high resolution reference solution and the results of the
# algorithm. The residuals (in some norm) must converge to zero to prove convergence.


tolerances=1e-2/2**np.arange(1,10) # Adjusting the step size h
reference_tol=tolerances[-1]/2
differences=np.zeros_like(tolerances)
t,yref=el_shooting(lagrangian,d,bound_cond,N,options2,reference_tol)
plt.figure()
for i,tol in enumerate(tolerances):
    t,ytol=el_shooting(lagrangian,d,bound_cond,N,options2,tol)
    plt.plot(t,ytol,label="Tol={}".format(tol))
    differences[i]=np.linalg.norm(ytol-yref,1) #taking the residuals of the respective norms
plt.xlabel(r"$t$")
plt.ylabel(r"$y(t)$")
plt.title('Testing convergence for a given tolerance')
plt.legend()
plt.show()
plt.figure()
plt.loglog(tolerances, differences,'x', mew=2, color='blue')
plt.xlabel('Tolerances')
plt.ylabel('Difference to reference solution')
plt.title('Correlation between the given tolerance and the reference data')
plt.show()

"""
The problem has not been posed correctly as in the second case the price intially goes above 1.0 before dropping to 0.9.
This is not a real world solution as in reality if prices were to increase first, the company would risk losing customers 
that won't return if prices were to decrease down to 90% of the original rrp.
"""

 



