""" Solve boundary layer problems using Pohlhausen's method

This module holds routines to determine the evolution of a boundary layer
in a varying external flow assuming a Pohlhausen velocity profile.

Methods:
    disp_ratio, mom_ratio, df_0
    march
    split
    panel_march
    predict_separation_point

Imports: numpy, bisect from scipy.optimize, VortexPanel
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
def _g_1(lam): return df_0(lam)-lam*(disp_ratio(lam)+2*mom_ratio(lam))

# use bisect method to find lambda0
lam0 = bisect(_g_1,-12,12)

# boundary layer thickness derivative
def _ddx_delta(Re_d,lam):
    if Re_d==0: return 0                # stagnation point condition
    return _g_1(lam)/mom_ratio(lam)/Re_d

# Heun's method for ODE integration
def _heun(g,psi_i,i,dx,*args):
    g_i = g(psi_i,i,*args)              # integrand at i
    tilde_psi = psi_i+g_i*dx            # predicted estimate at i+1
    g_i_1 = g(tilde_psi,i+1,*args)      # integrand at i+1
    return psi_i+0.5*(g_i+g_i_1)*dx     # corrected estimate

# boundary layer ODE integrand function
def _g_pohl(delta_i,i,u_e,du_e,nu):
    Re_d = delta_i*u_e[i]/nu            # definition of Re_delta
    lam = delta_i**2*du_e[i]/nu         # definition of lambda
    lam = min(12,max(-12,lam))          # enforce bounds on lambda
    return _ddx_delta(Re_d,lam)          # compute derivative

def march(x,u_e,nu):
    """ March along the boundary layer from stagnation to separation

    Notes:
    Integration stops when lambda<-12. Output array values after that point
    are meaningless.

    Inputs:
    x   -- array of distances along the boundary layer; must be positive and increasing
    u_e -- array of external velocities at locations `x`
    nu  -- kinematic viscosity; must be scalar

    Outputs:
    delta -- array boundary layer thicknesses at locations `x`
    lam   -- array of shape function values at locations `x`
    iSep  -- array index of separation point

    Examples:
    s = numpy.linspace(0,numpy.pi,16)        # define distance array
    u_e = numpy.sin(s)                       # define external velocity (circle example)
    delta,lam,iSep = march(s,u_e,nu=1e-5)    # march along to the point of separation
    """
    dx = numpy.diff(x)
    du_e = numpy.gradient(u_e,numpy.gradient(x))
    du_e[0] = (3.*u_e[0]-4.*u_e[1]+u_e[2])/(2.*(x[0]-x[1]))    # correct initial value
    delta = numpy.full_like(x,0.)
    lam = numpy.full_like(x,lam0)

    # set initial condition on delta
    if du_e[0]>0:
        delta[0] = numpy.sqrt(lam0*nu/du_e[0])
    elif x[0]>0 and u_e[0]>0 and du_e[0]==0: # use flat plate soln.
        lam[0] = 0
        delta[0] = 5.836*numpy.sqrt(nu*x[0]/u_e[0])
    else:
        raise ValueError('bad u_e near x=0')

    # march!
    for i in range(len(x)-1):
        delta[i+1] = _heun(_g_pohl,delta[i],i,dx[i],
                          u_e,du_e,nu)  # ...additional arguments
        lam[i+1] = delta[i+1]**2*du_e[i+1]/nu

        if lam[i+1] < -12: break        # separation condition

    # find separation index
    if(lam[i+1]>-12):
        iSep = i+1
    else:
        iSep = numpy.interp(12,-lam[i:i+2],[i,i+1])

    return delta,lam,iSep               # return with separation index

def sep(y,iSep):
    """ Interpolate value from array at the separation point

    Notes:
    Ignores array values after iSep. See help(march)

    Inputs:
    y     -- array of values to be inpterpolated
    iSep  -- array index of separation point

    Outputs:
    ySep  -- interpolated value at the point lambda=-12

    Examples:
    s = numpy.linspace(0,numpy.pi,16)        # define distance array
    u_e = numpy.sin(s)                       # define external velocity (circle example)
    delta,lam,iSep = bl.march(s,u_e,nu=1e-5) # march along to the point of separation
    sSep = bl.sep(s,iSep)                    # find separation distance
    """
    i = numpy.ceil(iSep)          # round up to nearest integer
    di = i-iSep                   # interpolation `distance`
    return y[i-1]*di+y[i]*(1-di)  # linearly interpolate

### Boundary layers on vortex panels
from VortexPanel import PanelArray

def split(body):
    """Split PanelArray into two boundary layer sections

    Inputs:
    body  -- PanelArray which has be 'solved'

    Outputs:
    top     -- PanelArray defining the top BL
    bottom  -- PanelArray defining the bottom BL

    Examples:
    foil = vp.make_jukowski(N=64)        #1. Define the geometry
    foil.solve_gamma_kutta(alpha=0.1)    #2. Solve for the potential flow
    foil_top,foil_bot = split(foil)      #3. Split the boundary layers
    """
    u_s = -body.get_array('gamma')          # tangential velocity
    top = body.panels[u_s>=0]               # split based on flow direction
    bot = body.panels[u_s<=0]
    bot = bot[::-1]                         # flip to run front to back
    return PanelArray(panels=top),PanelArray(panels=bot)

def panel_march(panels,nu):
    """ March along a set of BL panels

    Inputs:
    panels  -- array of Panels which have been solved and split

    Outputs:
    delta -- array boundary layer thicknesses at locations `x`
    lam   -- array of shape function values at locations `x`
    iSep  -- array index of separation point

    Examples:
    nu = 1e-5
    circle = vp.make_circle(N=64)           #1. make the geometry
    circle.solve_gamma()                    #2. solve the pflow
    top,bottom = bl.split(circle)           #3. split the panels
    delta,lam,iSep = bl.panel_march(top,nu) #4. march along the BL
    """
    s = panels.distance()                   # distance
    u_e = abs(panels.get_array('gamma'))    # velocity
    return march(s,u_e,nu)                  # march

def panel_sep_point(panels):
    """ Predict separation point on a set of BL panels

    Inputs:
    panels    -- array of Panels which have been solved and split

    Outputs:
    xSep,ySep -- x,y location of the boundary layer separation point

    Examples:
    foil = vp.make_jukowski(N=64)         #1. make the geometry
    foil.solve_gamma_kutta(alpha=0.1)     #2. solve the pflow
    top,bottom = bl.split(foil)           #3. split the panels
    x_top,y_top = bl.panel_sep_point(top) #4. find separation point
    """
    delta,lam,iSep = panel_march(panels,nu=1e-5)
    x,y = panels.get_array('xc','yc')
    return sep(x,iSep),sep(y,iSep)
