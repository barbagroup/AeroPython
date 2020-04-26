""" Solve potential flow problems using vortex panels

This module holds routines to determine the potential flow and
separation point around bodies of any shape or number using
vortex panels and laminar boundary layer theory.

Classes:
    Panel, PanelArray

Methods:
    panelize, concatenate
    make_ellipse, make_circle, make_jfoil, make_spline

Imports:
    numpy, pyplot from matplotlib, march & sep from BoundaryLayer
"""
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
from matplotlib import pyplot as plt
from vortexpanel import BoundaryLayer as bl

### Fundamentals

class Panel(object):
    """Vortex panel class

    Attributes:
    xc,yc -- the x and y location of the panel center
    S     -- the half-width of the panel
    sx,sy -- the x and y component of the tangent unit vector
    gamma -- the panel vortex strength
    """

    def __init__(self, x0, y0, x1, y1, gamma=0):
        """Initialize a panel between two points

        Inputs:
        x0,y0 -- the x and y location of the starting point
        x1,y1 -- the x and y location of the ending point
        gamma -- the panel vortex strength; defaults to zero

        Outputs:
        A Panel object.

        Example:
        p_1 = vp.Panel(-1,0,1,0)    # make panel on x-axis with gamma=0
        p_2 = vp.Panel(0,-1,0,1,4)  # make panel on y-axis with gamma=4
        """
        self.x = (x0,x1); self.y = (y0,y1)              # copy end-points
        self.gamma = gamma; self._gamma = (gamma,gamma) # copy gamma
        self.xc = 0.5*(x0+x1); self.yc = 0.5*(y0+y1)    # panel center
        dx = x1-self.xc; dy = y1-self.yc
        self.S = np.sqrt(dx**2+dy**2)                   # half-width
        self.sx = dx/self.S; self.sy = dy/self.S        # tangent
        if np.isclose(self.S,0):                        # check panel size
            raise ValueError("Panels must have non-zero length.\n"+
                             "Your endpoints are on top of each other.")


    def velocity(self, x, y):
        """Compute the velocity induced by the panel

        Inputs:
        x,y   -- the x and y location of the desired velocity

        Outputs:
        u,v   -- the x and y components of the velocity

        Example:
        p_2 = vp.Panel(0,-1,0,1,4)        # make panel on y-axis with gamma=4
        u,v = p_2.velocity(-1,0)          # get induced velocity on x-axis
        """
        if self._gamma[1]-self._gamma[0]: # non-constant gamma
            u0,v0,u1,v1 = self._linear(x,y)
            return (self._gamma[0]*u0+self._gamma[1]*u1,
                    self._gamma[0]*v0+self._gamma[1]*v1)
        else:
            u,v = self._constant(x,y)     # constant gamma
            return self.gamma*u,self.gamma*v

    def plot(self, style='k'):
        """Plot the vortex panel as a line segment

        Inputs:
        style -- a string defining the matplotlib style

        Example:
        my_panel = vp.Panel(0,-1,0,1)  # creates panel on y-axis
        my_panel.plot()                # plot the panel
        """
        return plt.plot(self.x, self.y, style, lw=2)

    def _constant(self, x, y):
        "Constant panel induced velocity"
        lr, dt, _, _ = self._transform_xy(x, y)
        return self._rotate_uv(-dt*0.5/np.pi, -lr*0.5/np.pi)

    def _linear(self, x, y):
        "Linear panel induced velocity"
        lr, dt, xp, yp = self._transform_xy(x, y)
        g, h, c = (yp*lr+xp*dt)/self.S, (xp*lr-yp*dt)/self.S+2, 0.25/np.pi
        return (self._rotate_uv(c*( g-dt), c*( h-lr))
               +self._rotate_uv(c*(-g-dt), c*(-h-lr)))

    def _transform_xy(self, x, y):
        "transform from global to panel coordinates"
        xt = x-self.xc; yt = y-self.yc # shift x,y
        xp = xt*self.sx+yt*self.sy     # rotate x
        yp = yt*self.sx-xt*self.sy     # rotate y
        lr = 0.5*np.log(((xp-self.S)**2+yp**2)/((xp+self.S)**2+yp**2))
        dt = np.arctan2(yp,xp-self.S)-np.arctan2(yp,xp+self.S)
        return lr, dt, xp, yp

    def _rotate_uv(self, up, vp):
        "rotate velocity back to global coordinates"
        u = up*self.sx-vp*self.sy    # reverse rotate u prime
        v = vp*self.sx+up*self.sy    # reverse rotate v prime
        return u, v


class PanelArray(object):
    """Array of vortex panels

    Attributes:
    panels -- the np array of panels
    alpha  -- the flow angle of attack
    """

    def __init__(self, panels, closed=True):
        """Initialize a PanelArray

        Inputs:
        panels -- a np array of panels

        Outputs:
        A PanelArray object.
        """
        self.panels = panels # copy the panels
        self.alpha = 0       # default alpha
        n = len(panels)
        self.bodies = [(0,n)]        # range for a body
        self.left = [n-1]+list(range(n-1)) # index to the left

        # compute area for closed PanelArrays
        if closed:
            yc, S, sx = self.get_array('yc','S','sx')
            self.area = sum(sx*yc*2.*S)
            if self.area<1e-5: # check body area
                raise ValueError("A closed PanelArray must have positive area.\n"+
                                 "Check that your panels wrap clockwise.")

    ### Flow solver

    def solve_gamma(self,alpha=0,kutta=[]):
        """ Set the vortex strength on a PanelArray to enforce the no slip and
        kutta conditions.

        Notes:
        Solves for the normalized gamma by using a unit magnitude background
        flow, |U|=1.

        Inputs:
        alpha   -- angle of attack relative to x-axis; must be a scalar; default 0
        kutta   -- panel indices (a list of tuples) on which to enforce the kutta
                    condition; defaults to empty list

        Outputs:
        gamma of the PanelArray is updated.

        Example:
        foil = vp.make_jfoil(N=32)                    # make a Panel array
        foil.solve_gamma(alpha=0.1, kutta=[(0,-1)])      # solve for gamma
        foil.plot_flow()                                 # plot the flow
        """

        self._set_alpha(alpha)                # set alpha
        A,b = self._construct_A_b()           # construct linear system
        for i in kutta:                       # loop through index pairs
            A[i[0]:i[1],i] += 1                  # apply kutta condition
        gamma = np.linalg.solve(A, b)         # solve for gamma
        for i,p_i in enumerate(self.panels):  # loop through panels
            p_i.gamma = gamma[i]                 # update center gamma
            p_i._gamma = (gamma[i],gamma[i])     # update end-point gammas

    def solve_gamma_O2(self,alpha=0,kutta=[]):
        "special case of solve_gamma for linearly varying panels"
        self._set_alpha(alpha)                   # set alpha
        A,b = self._construct_A_b_O2()           # construct linear system
        if kutta:
            for j,i in kutta:
                A[i,:] = 0; A[i,i] = 1; b[i] = 0
        else:
            S = self.get_array('S')
            for s,e in self.bodies:
                A[s,:] = 0; b[s] = 0
                A[s,s:e] += S[s:e]
                A[s,self.left[s:e]] += S[s:e]
        gamma = np.linalg.solve(A, b)            # solve for gamma!
        for i,p_i in enumerate(self.panels):     # loop through panels
            p_i._gamma = (gamma[self.left[i]],gamma[i])      # update end-point gammas
            p_i.gamma = 0.5*sum(p_i._gamma)         # update center gamma

    def _set_alpha(self,alpha):
        "Set angle of attack, but it must be a scalar"
        if(isinstance(alpha, (set, list, tuple, np.ndarray))):
            raise TypeError('Only accepts scalar alpha')
        self.alpha = alpha

    def _construct_A_b(self):
        "construct the linear system to enforce no-slip on every panel"

        # get arrays
        xc,yc,sx,sy = self.get_array('xc','yc','sx','sy')

        # construct the matrix
        A = np.empty((len(xc), len(xc)))      # empty matrix
        for j, p_j in enumerate(self.panels): # loop over panels
            fx,fy = p_j._constant(xc,yc)           # f_j at all panel centers
            A[:,j] = fx*sx+fy*sy                  # tangential component
        np.fill_diagonal(A,0.5)               # fill diagonal with 1/2

        # construct the RHS
        b = -np.cos(self.alpha)*sx-np.sin(self.alpha)*sy
        return A, b

    def _construct_A_b_O2(self):
        "construct the linear system to enforce no-pen on every panel"

        # get arrays
        xc,yc,sx,sy = self.get_array('xc','yc','sx','sy')

        # construct the matrix
        A = np.zeros((len(xc), len(xc)))         # empty matrix
        for j, p_j in enumerate(self.panels):    # loop over panels
            fx0,fy0,fx1,fy1 = p_j._linear(xc,yc)     # f_j at all panel ends
            A[:,self.left[j]] += -fx0*sy+fy0*sx     # -S end influence
            A[:,j] += -fx1*sy+fy1*sx                # +S end influence

        # construct the RHS
        b = np.cos(self.alpha)*sy-np.sin(self.alpha)*sx
        return A, b


    ### Visualize

    def plot_flow(self,size=2,vmax=None):
        """ Plot the flow induced by the PanelArray and background flow

        Notes:
        Uses unit magnitude background flow, |U|=1.

        Inputs:
        size   -- size of the domain; corners are at (-size,-size) and (size,size)
        vmax   -- maximum contour level; defaults to the field max

        Outputs:
        plot of the flow vectors, velocity magnitude contours, and the panels.

        Example:
        circle = vp.make_circle(N=32)      # make a Panel array
        circle.solve_gamma(alpha=0.1)      # solve for Panel strengths
        circle.plot_flow()                 # plot the flow
        """
        # define the grid
        line = np.linspace(-size, size, 100)  # computes a 1D-array
        x, y = np.meshgrid(line, line)        # generates a mesh grid

        # get the velocity from the free stream and panels
        u, v = self.velocity(x, y)

        # plot it
        plt.figure(figsize=(9,7))                # set size
        plt.xlabel('x', fontsize=14)             # label x
        plt.ylabel('y', fontsize=14, rotation=0) # label y

        # plot contours
        m = np.sqrt(u**2+v**2)
        velocity = plt.contourf(x, y, m, vmin=0, vmax=vmax)
        cbar = plt.colorbar(velocity)
        cbar.set_label('Velocity magnitude', fontsize=14);

        # plot vector field
        plt.quiver(x[::4,::4], y[::4,::4],
                      u[::4,::4], v[::4,::4])
        # plot panels
        self.plot();

    def plot(self, style='k', nlabel=0):
        """Plot the PanelArray panels

        Inputs:
        style -- a string defining the matplotlib style
        nlabel -- add every 'nlabel' panel index labels

        Example:
        circle = vp.make_circle(N=32) # make a circle PanelArray
        circle.plot(style='o-')       # plot the geometry
        """
        for i,p_i in enumerate(self.panels):
            p_i.plot(style)
            if nlabel>0 and i%nlabel == 0:
                x,y = p_i.xc-0.1*p_i.sy,p_i.yc+0.1*p_i.sx
                plt.text(x,y,i,fontsize=12,horizontalalignment='center',verticalalignment='center')

    def velocity(self,x,y):
        "Velocity at (x,y) induced by the free stream and PanelArray"
        u = np.cos(self.alpha)*np.ones_like(x)
        v = np.sin(self.alpha)*np.ones_like(x)

        for p_j in self.panels:
            u_j, v_j = p_j.velocity(x, y)
            u, v = u+u_j, v+v_j

        return u, v


    ## Panel array operations

    def get_array(self,key,*args):
        """ Generate np arrays of panel attributes

        Notes:
        Use help(Panel) to see available attributes

        Inputs:
        key (,*args) -- one or more names of the desired attributes

        Outputs:
        key_vals     -- np arrays filled with the named attributes

        Example:
        circle = vp.make_circle(N=32)           # make a PanelArray
        xc,yc = circle.get_array('xc','yc')     # get arrays of the panel centers
        """
        if not args:
            return np.array([getattr(p,key) for p in self.panels])
        else:
            return [self.get_array(k) for k in (key,)+args]

    def distance(self):
        """ Find the cumulative distance of along the PanelArray

        Notes:
        s[0] = S[0], s[1] = 2*S[0]+S[1], s[2] = 2*S[0]+2*S[1]+S[2], ...

        Example:
        foil = vp.make_jfoil(N=64)   # define the geometry
        s = foil.distance()          # get the panel path distance
        """
        S = self.get_array('S')
        return np.cumsum(2*S)-S


    ### Boundary layers
    def split(self):
        """Split PanelArray into two boundary layer sections

        Outputs:
        top     -- PanelArray defining the top BL
        bottom  -- PanelArray defining the bottom BL

        Example:
        foil = vp.make_jfoil(N=64)        #1. Define the geometry
        foil.solve_gamma_kutta(alpha=0.1) #2. Solve for the potential flow
        foil_top,foil_bot = foil.split()  #3. Split the boundary layers
        """
        # roll the panel index to a stagnation point
        gamma = self.get_array('gamma')
        i = max(np.argmax(gamma<0),np.argmax(gamma>0))
        rolled = np.roll(self.panels,-i)
        # split based on flow direction
        top = [p for p in rolled if p.gamma<=0]
        bot = [p for p in rolled if p.gamma>=0]
        closed = False # split arrays don't make closed loops
        return PanelArray(top,closed),PanelArray(bot[::-1],closed)

    def thwaites(self):
        """ Wrapper for BoundaryLayer.thwaites

        Example:
        circle = vp.make_circle(N=32)     #1. make the geometry
        circle.solve_gamma_O2()           #2. solve the pflow
        top,bottom = circle.split()       #3. split the panels
        delta2,lam,iSep = top.thwaites()  #4. get BL props
        """
        s = self.distance()                # distance
        u_s = abs(self.get_array('gamma')) # velocity
        return bl.thwaites(s,u_s)          # thwaites

    def sep_point(self):
        """ Predict separation point on a set of BL panels

        Outputs:
        x_s,y_s -- location of the boundary layer separation point

        Example:
        circle = vp.make_circle(N=32)  #1. make the geometry
        circle.solve_gamma_O2()        #2. solve the pflow
        top,bottom = circle.split()    #3. split the panels
        x_s,y_s = top.sep_point()      #4. get sep point
        """
        _,_,iSep = self.thwaites()            # only need iSep
        x,y = self.get_array('xc','yc')       # panel centers
        return bl.sep(x,iSep),bl.sep(y,iSep)  # interpolate

### Geometries

def panelize(x,y):
    """Create a PanelArray from a set of points

    Inputs:
    x,y    -- the x and y location of the panel end points

    Outputs:
    A PanelArray object

    Note:
    The first and last point should match for a
    closed shape.
    """
    if len(x)<2:         # check input lengths
        raise ValueError("point arrays must have len>1")
    if len(x)!=len(y):   # check input lengths
        raise ValueError("x and y must be same length")
    return PanelArray([Panel(x[i], y[i], x[i+1], y[i+1])
                for i in range(len(x)-1)])

def make_ellipse(N, t_c, xcen=0, ycen=0):
    """ Make an elliptical PanelArray; defaults to circle

    Inputs:
    N         -- number of panels to use
    t_c       -- thickness/chord of the ellipse
    xcen,ycen -- location of the ellipse center; defaults to origin

    Outputs:
    A PanelArray object; see help(PanelArray)

    Example:
    ellipse = vp.make_ellipse(N=32,t_c=0.5) # make a 1:2 elliptical Panel array
    ellipse.plot()                          # plot the geometry
    """
    theta = np.linspace(0, -2*np.pi, N+1)
    x,y = np.cos(theta)+xcen, np.sin(theta)*t_c+ycen
    return panelize(x,y)

def make_circle(N, xcen=0, ycen=0):
    """ Make an circular PanelArray

    Inputs:
    N         -- number of panels to use
    xcen,ycen -- circle center; defaults to origin

    Outputs:
    A PanelArray object; see help(PanelArray)

    Example:
    circle = vp.make_circle(N=32) # make
    circle.plot()                 # plot
    """
    theta = np.linspace(0, -2*np.pi, N+1)
    x,y = np.cos(theta)+xcen, np.sin(theta)+ycen
    return panelize(x,y)

def make_jfoil(N, xcen=-0.1, ycen=0):
    """Make a foil-shaped PanelArray using the Jukowski mapping

    Note:
    A circle passing through point (1,0) is mapped to create the sharp foil

    Inputs:
    N         -- number of panels to use
    xcen,ycen -- center of the circle before mapping

    Outputs:
    A PanelArray object; see help(PanelArray)

    Example:
    foil = vp.make_jfoil(N=34)  # make
    foil.plot()                 # plot
    """
    # define the circle
    theta = np.linspace(0, -2*np.pi, N+1)
    r = np.sqrt(ycen**2+(1-xcen)**2)
    t0 = np.arctan2(ycen,1-xcen)
    x,y = xcen+r*np.cos(theta-t0), ycen+r*np.sin(theta-t0)

    # apply jukowski mapping and panelize
    r2 = x**2+y**2
    return panelize(x*(1+1/r2)/2,y*(1-1/r2))

def _spline(N,x,y,per=True):
    from scipy import interpolate
    tck, u = interpolate.splprep([x, y], s=0, per=per)
    unew = np.linspace(0, 1, N)
    return interpolate.splev(unew, tck)

def make_spline(N,x,y,sharp=False):
    """Make PanelArray using splines

    Note:
    A closed spline is fit through the coordinates.
    The spline may oscillate between the coords,
    so the output needs to be checked carefully.

    Inputs:
    N     -- number of panels to use
    x,y   -- body coordinate lists
    sharp -- sharp corner between panel 0 and N-1?

    Outputs:
    A PanelArray object; see help(PanelArray)

    Example:
    x = [1,0,-0.5,-1,-0.5,0,1]      # x coords
    y = [0,-.2,-.2,0,.2,.2,0]       # y coords
    plt.plot(x,y,'x',markersize=20) # plot coords
    geom = vp.make_spline(20,x,y)   # fit 20 panels
    geom.plot('-o')                 # plot panels
    """
    xi,yi = _spline(N+1,x,y,per=not sharp)
    return panelize(xi,yi)


def concatenate(a,b,*args):
    """Concatenate PanelArray bodies

    Inputs:
    a,b (,*args)    -- two or more PanelArray bodies

    Outputs:
    A PanelArray object
    """
    if not args:
        na = len(a.panels)
        c = PanelArray(a.panels+b.panels)
        c.bodies = a.bodies+[(s+na,e+na) for s,e in b.bodies]
        c.left = a.left+[l+na for l in b.left]
    else:
        c = concatenate(concatenate(a,b),*args)
    return c
