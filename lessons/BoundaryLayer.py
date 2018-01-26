""" Solve boundary layer problems using Pohlhausen's method

This module holds routines to determine the evolution of a boundary layer
in a varying external flow assuming a Pohlhausen velocity profile.

Methods:
    disp_ratio, mom_ratio, df_0
    march
    split
    panel_march
    panel_sep_point

Imports: numpy, bisect from scipy.optimize, VortexPanel
"""

import numpy
from scipy.optimize import fsolve

# displacement thickness ratios
def disp_ratio(lam): return 3./10.-lam/120.

# momentum thickness ratios
def mom_ratio(lam): return 37./315.-lam/945.-lam**2/9072.

# wall derivative
def df_0(lam): return 2+lam/6.

def march(s,u_s,nu):
    """ March along the boundary layer from stagnation to separation

    Notes:
    Integration stops when lambda<-12. Output array values after that point
    are meaningless.

    Inputs:
    s   -- array of distances along the boundary layer; must be positive and increasing
    u_s -- array of external velocities at locations `x`
    nu  -- kinematic viscosity; must be scalar

    Outputs:
    delta2 -- momentum thickness array
    lam   -- shape parameter array
    iSep  -- separation point in terms of the array indices

    Examples:
    s = numpy.linspace(0,numpy.pi,16)        # define distance array
    u_s = numpy.sin(s)                       # define external velocity (circle example)
    delta2,lam,iSep = march(s,u_s,nu=1e-5)    # march along to the point of separation
    """
    # define functions for delta_2^2 and Delta delta_2^2
    def d22(lam,i):
        return lam*mom_ratio(lam)**2/du_s[i]*nu
    def dd22(lam_i_1,i):
        return nu*(g(lam[i])/u_s[i]+g(lam_i_1)/u_s[i+1])*(s[i+1]-s[i])
    def g(lam):
        F = mom_ratio(lam)
        return F*(df_0(lam)-lam*(disp_ratio(lam)+2.*F))

    # set up arrays with ICs
    du_s = numpy.gradient(u_s)/numpy.gradient(s)
    lam,delta22 = numpy.zeros_like(s),numpy.zeros_like(s)
    if du_s[0]>0: lam[0] = fsolve(g,7)[0]; delta22[0] = d22(lam[0],0)
    else: delta22[0] = 2.*nu*g(0)/u_s[0]*s[0]

    # integrate
    for i in range(len(s)-1):
        if du_s[i+1]==0: lam[i+1] = 0
        else:
            def f(lam): return d22(lam,i+1)-delta22[i]-dd22(lam,i)
            lam[i+1] = fsolve(f,lam[i])[0]
        delta22[i+1] = delta22[i]+dd22(lam[i+1],i)
        if lam[i+1]<-12 : break

    # find separation index
    if(lam[i+1]>-12):
        iSep = i+1
    else:
        iSep = numpy.interp(12,-lam[i:i+2],[i,i+1])

    return numpy.sqrt(delta22),lam,iSep

def thwaites(s,u_s,nu):
    """ Boundary layer integrator using Thwaites approximate method
    Same inputs and outputs as bl.march
    """
    # integrate for delta_2^2
    from scipy.integrate import cumtrapz
    delta22 = 0.45*nu/u_s**6*cumtrapz(u_s**5,s,initial=0)

    # check IC
    du_s = numpy.gradient(u_s)/numpy.gradient(s)
    if du_s[0]==0: delta22 += 0.44*nu/u_s[0]*s[0]

    # find separation point
    lam_2 = delta22*du_s/nu; lim = -12*mom_ratio(-12)**2
    i = numpy.count_nonzero(lam_2>lim)
    if i==len(s):
        iSep = i-1
    else:
        iSep = numpy.interp(-lim,-lam_2[i-1:i+1],[i-1,i])

    # invert for lambda
    def invert(lam_2):
        if lam_2<lim: return 0
        def f(lam): return lam_2-mom_ratio(lam)**2*lam
        return fsolve(f,0)[0]
    lam = numpy.array([invert(lam2) for lam2 in lam_2])

    return numpy.sqrt(delta22),lam,iSep

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
    from math import ceil         # numpy doesn't do ceil correctly
    i = ceil(iSep)                # round up to nearest integer
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
