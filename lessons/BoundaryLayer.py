""" Solve boundary layer problems 

Methods:
    disp_ratio, mom_ratio, df_0
    thwaites, sep

Imports: numpy, cumtrapz from scipy.integrate, ceil from math
"""

import numpy as np
from scipy.integrate import cumtrapz

# displacement thickness ratios
def disp_ratio(lam): return 3./10.-lam/120.

# momentum thickness ratios
def mom_ratio(lam): return 37./315.-lam/945.-lam**2/9072.

# wall derivative
def df_0(lam): return 2+lam/6.

# look-up table for lambda = f(lambda_2)
_lam_range = np.linspace(-17.5,12)
_lam2_range = _lam_range*mom_ratio(_lam_range)**2

def thwaites(s,u_s):
    """ Integrate Thwaites over the boundary layer and find shape factor

    Notes:
    Output array values after separation(lam<-12) are meaningless.

    Inputs:
    s   -- array of distances along the boundary layer; must be positive and increasing
    u_s -- array of external velocities; must be positive

    Outputs:
    delta2 -- momentum thickness array
    lam    -- shape factor array
    iSep   -- distance along array to separation point (iSep=len(s) means no separation)

    Examples:
    s = np.linspace(0,np.pi,16)      # define distance array
    u_s = 2*np.sin(s)                # define external velocity (circle example)
    delta2,lam,iSep = march(s,u_s)   # march along to the point of separation
    """
    # Thwaites approximation for delta_2^2
    delta22 = 0.45/u_s**6*cumtrapz(u_s**5,s,initial=0)

    # adjust for (Blasius) flat plate IC
    if du_s[0]==0: delta22 += 0.441*nu/u_s[0]*s[0]

    # Velocity gradient and shape factor lambda
    du_s = np.gradient(u_s)/np.gradient(s)
    lam = np.interp(delta22*du_s,_lam2_range,_lam_range)
    
    # find separation point
    i = np.count_nonzero(lam>-12)
    if(i==len(s)):
        iSep = i
    else:
        iSep = np.interp(12,-lam[i-1:i+1],[i-1,i])

    # Return delta_2 and lambda and iSep
    return np.sqrt(delta22),lam,iSep

def sep(y,index=False,lam=False):
    """ Interpolate value from array at the separation point

    Notes:
    Ignores array values after iSep. See help(march)

    Inputs:
    y     -- array of values to be interpolated
    iSep  -- array index of separation point

    Outputs:
    ySep  -- interpolated value at the point lambda=-12

    Examples:
    s = np.linspace(0,np.pi,16)        # define distance array
    u_e = 2*np.sin(s)                     # define external velocity (circle example)
    delta,lam,iSep = bl.march(s,u_e,nu=1e-5) # march along to the point of separation
    sSep = bl.sep(s,iSep)                    # find separation distance
    """
    from math import ceil         # numpy doesn't do ceil correctly
    i = ceil(iSep)                # round up to nearest integer
    di = i-iSep                   # interpolation `distance`
    return y[i-1]*di+y[i]*(1-di)  # linearly interpolate
