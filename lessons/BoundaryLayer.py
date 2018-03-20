""" Solve boundary layer problems using Pohlhausen's method

This module holds routines to determine the evolution of a boundary layer
in a varying external flow assuming a Pohlhausen velocity profile.

Methods:
    disp_ratio, mom_ratio, df_0
    march, sep

Imports: numpy, cumtrapz & odeint from scipy.integrate
"""

import numpy
from scipy.integrate import cumtrapz,odeint

# displacement thickness ratios
def disp_ratio(lam): return 3./10.-lam/120.

# momentum thickness ratios
def mom_ratio(lam): return 37./315.-lam/945.-lam**2/9072.

# wall derivative
def df_0(lam): return 2+lam/6.

# look-up table for lambda = f(lambda_2)
_lam_range = numpy.linspace(-17.5,12)
_lam2_range = _lam_range*mom_ratio(_lam_range)**2
def _lam2_inv(lam2): return numpy.interp(lam2,_lam2_range,_lam_range)

def march(s,u_s,nu,thwaites=False):
    """ March along the boundary layer from stagnation to separation

    Notes:
    Output array values after lam<-12 are meaningless.

    Inputs:
    s   -- array of distances along the boundary layer; must be positive and increasing
    u_s -- array of external velocities at locations `x`
    nu  -- kinematic viscosity; must be scalar
    thwaites -- use Thwaites approximate method (default=False)

    Outputs:
    delta2 -- momentum thickness array
    lam   -- shape parameter array
    iSep  -- separation index

    Examples:
    s = numpy.linspace(0,numpy.pi,16)        # define distance array
    u_s = 2*numpy.sin(s)                     # define external velocity (circle example)
    delta2,lam,iSep = march(s,u_s,nu=1e-5)   # march along to the point of separation
    """
    # velocity gradient
    du_s = numpy.gradient(u_s)/numpy.gradient(s)

    if thwaites:
        # integrate for delta_2^2
        delta22 = 0.45*nu/u_s**6*cumtrapz(u_s**5,s,initial=0)

        # check IC
        if du_s[0]==0: delta22 += 0.44*nu/u_s[0]*s[0]

    else:
        # define ODE
        def func(y,t): # y=>delta_2^2, t=>s
            u_t,du_t = numpy.interp(t,s,u_s),numpy.interp(t,s,du_s)
            lam = _lam2_inv(y*du_t/nu)
            F = mom_ratio(lam)
            return 2.*nu/u_t*F*(df_0(lam)-lam*(disp_ratio(lam)+2.*F))

        # set IC
        if du_s[0]>0: y0 = 0.077/du_s[0]*nu # stagnation point
        else: y0 = func(0,0)*s[0]           # flat plate

        # integrate
        delta22 = odeint(func,y0,s)[:,0]

    # find separation point
    lam2 = delta22*du_s/nu; lim = -12*mom_ratio(-12)**2
    i = numpy.count_nonzero(lam2>lim)
    if(i==len(s)):
        iSep = i
    else:
        iSep = numpy.interp(-lim,-lam2[i-1:i+1],[i-1,i])

    return numpy.sqrt(delta22),_lam2_inv(lam2),iSep

def sep(y,iSep):
    """ Interpolate value from array at the separation point

    Notes:
    Ignores array values after iSep. See help(march)

    Inputs:
    y     -- array of values to be interpolated
    iSep  -- array index of separation point

    Outputs:
    ySep  -- interpolated value at the point lambda=-12

    Examples:
    s = numpy.linspace(0,numpy.pi,16)        # define distance array
    u_e = 2*numpy.sin(s)                     # define external velocity (circle example)
    delta,lam,iSep = bl.march(s,u_e,nu=1e-5) # march along to the point of separation
    sSep = bl.sep(s,iSep)                    # find separation distance
    """
    from math import ceil         # numpy doesn't do ceil correctly
    i = ceil(iSep)                # round up to nearest integer
    di = i-iSep                   # interpolation `distance`
    return y[i-1]*di+y[i]*(1-di)  # linearly interpolate
