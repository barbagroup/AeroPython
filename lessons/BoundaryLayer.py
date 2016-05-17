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
    delta = numpy.full_like(x,0.)
    lam = numpy.full_like(x,lam0)

    # set initial condition on delta
    if du_e[0]>0:
        delta[0] = numpy.sqrt(lam0*nu/du_e[0])
    elif x[0]>0 and u_e[0]>0: # use flat plate soln.
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
    delta,lam,iSep = march(s,u_e,nu=1e-5)    # march along to the point of separation
    sSep = sep(s,iSep)                       # find separation distance
    """
    i = numpy.ceil(iSep)-1
    return numpy.interp(iSep,[i,i+1],y[i:i+2])

### Boundary layers on vortex panels
from VortexPanel import distance, get_array

def split(panels):
    """ Split panels into two boundary layer sections

    Inputs:
    panels  -- array of Panels which have be 'solved'

    Outputs:
    top     -- array of Panels defining the top BL
    bottom  -- array of Panels defining the bottom BL

    Examples:
    foil = make_jukowski(N=64)       #1. Define the geometry
    solve_gamma_kutta(foil,alpha=01) #2. Solve for the potential flow
    foilTop,foilBot = split(foil)    #3. Split the boundary layers
    """
    gamma = get_array(panels,'gamma')
    back = numpy.where(numpy.logical_and(
            gamma>=0,numpy.roll(gamma,1)<=0))  # find rear stagnation point
    rolled = numpy.roll(panels,-back[0])       # make this index 0
    top = [p for p in rolled if p.gamma<=0]    # panels where flow aligns with s
    bottom = [p for p in rolled if p.gamma>=0] # panels where flow is against s
    bottom = bottom[::-1]                      # reverse indices to align bottom
    return top,bottom

def panel_march(panels,nu):
    """ March along a set of BL panels

    Inputs:
    panels  -- array of Panels which have been solved and split

    Outputs:
    delta -- array boundary layer thicknesses at locations `x`
    lam   -- array of shape function values at locations `x`
    iSep  -- array index of separation point

    Examples:
    foil = make_jukowski(N=64)         #1. define the geometry
    solve_gamma_kutta(foil,alpha=0.1)  #2. solve the pflow
    top,bottom = split(foil)           #3. split the panels in BL segments
    delta,lam,iSep = panel_march(top)  #4. march along top BL panels
    """
    s = distance(panels)                    # distance
    u_e = abs(get_array(panels,"gamma"))    # velocity
    return march(s,u_e,nu)                  # march

def predict_separation_point(panels):
    """ Predict separation on a set of BL panels

    Inputs:
    panels    -- array of Panels which have been solved and split

    Outputs:
    xSep,ySep -- x,y location of the boundary layer separation point

    Examples:
    foil = make_jukowski(N=64)         #1. define the geometry
    solve_gamma_kutta(foil,alpha=0.1)  #2. solve the pflow
    top,bottom = split(foil)           #3. split the panels in BL segments
    xSep,ySep = panel_march(top)       #4. find separation on top BL panels
    """
    delta,lam,iSep = panel_march(panels,nu=1e-5)
    xSep = sep(get_array(panels,'xc'),iSep)
    ySep = sep(get_array(panels,'yc'),iSep)
    return xSep,ySep
