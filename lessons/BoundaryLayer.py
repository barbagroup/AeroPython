""" Solve boundary layer problems using Pohlhausen's method

This module holds routines to determine the evolution of a boundary layer
in a varying external flow assuming a Pohlhausen velocity profile. 

Example:
    Determine the boundary layer thickness delta, shape parameter
    lam(bda), and index of separation iSep given a sinusoid external
    flow:

        x = numpy.linspace(0,1,16)
        delta,lam,iSep = march(x,u_e=numpy.sin(x),nu=1e-5)

Imports: numpy, bisect from scipy.optimize
"""

import numpy
from scipy.optimize import bisect

# displacement thickness ratios
def disp_ratio(lam): return 3./10.-lam/120.

# momentum thickness ratios
def mom_ratio(lam): return 37./315.-lam/945.-lam**2/9072.

# wall derivative
def df_0(lam): return 2+lam/6.

# momentum function
def g_1(lam): return df_0(lam)-lam*(disp_ratio(lam)+2*mom_ratio(lam))

# use bisect method to find lambda0
lam0 = bisect(g_1,-12,12)

# boundary layer thickness derivative
def ddx_delta(Re_d,lam):
    if Re_d==0: return 0                # stagnation point condition
    return g_1(lam)/mom_ratio(lam)/Re_d

# Heun's method for ODE integration
def heun(g,psi_i,i,dx,*args):
    g_i = g(psi_i,i,*args)              # integrand at i
    tilde_psi = psi_i+g_i*dx            # predicted estimate at i+1
    g_i_1 = g(tilde_psi,i+1,*args)      # integrand at i+1
    return psi_i+0.5*(g_i+g_i_1)*dx     # corrected estimate

# boundary layer ODE integrand function
def g_pohl(delta_i,i,u_e,du_e,nu):
    Re_d = delta_i*u_e[i]/nu            # definition of Re_delta
    lam = delta_i**2*du_e[i]/nu         # definition of lambda
    lam = min(12,max(-12,lam))          # enforce bounds on lambda
    return ddx_delta(Re_d,lam)          # compute derivative

# march along boundary layer from stagnation to separation:
# return the boundary layer thickness, shape function,
# and separation index
def march(x,u_e,nu):
    dx = numpy.diff(x)
    du_e = numpy.gradient(u_e,numpy.gradient(x))
    delta = numpy.full_like(x,0.)
    lam = numpy.full_like(x,lam0)
    
    # Initial conditions must be a stagnation point. If u_e[0] is too
    # fast, assume stagnation is at x=0 and integrate from x=0..x[0].
    if u_e[0]<0.01:                     # stagnation point
        delta[0] = numpy.sqrt(lam0*nu/du_e[0])
    elif x[0]>0:                        # just downstream
        delta[0] = numpy.sqrt(lam0*nu*x[0]/u_e[0])
        delta[0] += 0.5*x[0]*g_pohl(delta[0],0,u_e,du_e,nu)
        lam[0] = delta[0]**2*du_e[0]/nu
    else:
        raise ValueError('x=0 must be stagnation point')
    
    # march!
    for i in range(len(x)-1):
        delta[i+1] = heun(g_pohl,delta[i],i,dx[i],
                          u_e,du_e,nu)  # ...additional arguments
        lam[i+1] = delta[i+1]**2*du_e[i+1]/nu
        
        if lam[i+1] < -12: i-=1; break  # separation condition
    
    return delta,lam,i+1                # return with separation index

# interpolate value of `y` at the separation point
def sep(y,lam,iSep):
    return numpy.interp( 12,-lam[iSep:iSep+2],y[iSep:iSep+2])


### Boundary layers on vortex panels

# split panels into two boundary layer sections
def split(panels):
    top = [p for p in panels if p.gamma<=0]
    bottom = [p for p in panels if p.gamma>=0]
    bottom = bottom[::-1]               # reverse array
    return top,bottom

# get distance array
def distance(panels):
    s = numpy.empty(len(panels))
    s[0] = panels[0].S
    for i in range(len(s)-1):
        s[i+1] = s[i]+panels[i].S+panels[i+1].S
    return s

# march along panels
def panel_march(panels,nu=1e-5):
    s = distance(panels)                    # distance
    u_e = [abs(p.gamma) for p in panels]    # velocity
    return march(s,u_e,nu)                  # march

# predict x,y separation point 
def predict_separation_point(panels):
    delta,lam,iSep = panel_march(panels,nu=1e-5)
    xSep = sep([p.xc for p in panels],lam,iSep)
    ySep = sep([p.yc for p in panels],lam,iSep)
    return xSep,ySep
