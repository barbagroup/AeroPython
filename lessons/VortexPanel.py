""" Solve potential flow problems using vortex panels

This module holds routines to determine the potential flow around
bodies of any shape or number using constant strength vortex panels.

Class:
    Panel

Methods:
    panelize
    make_polygon, make_ellipse, make_circle, make_jukowski
    solve_gamma, solve_gamma_kutta
    plot_flow
    get_array
    distance

Imports: numpy, pyplot from matplotlib
"""

import numpy
from matplotlib import pyplot
numpy.seterr(divide='ignore')
### Fundamentals

def _get_u( x, y, S, gamma ):
    "x-component of induced velocity"
    return gamma/(2*numpy.pi)*(numpy.arctan((x-S)/y)-numpy.arctan((x+S)/y))

def _get_v( x, y, S, gamma ):
    "y-component of induced velocity"
    return gamma/(4*numpy.pi)*(numpy.log(((x+S)**2+y**2)/((x-S)**2+y**2)))

class Panel(object):
    """Constant strength vortex panel class

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

        Examples:
        p_1 = vp.Panel(-1,0,1,0)    # make panel on x-axis with gamma=0
        p_2 = vp.Panel(0,-1,0,1,4)  # make panel on y-axis with gamma=4
        """
        self.x, self.y = [x0,x1], [y0,y1]            # copy end-points
        self.gamma = gamma                           # copy gamma
        self.xc, self.yc = 0.5*(x0+x1), 0.5*(y0+y1)  # panel center
        dx, dy = x1-self.xc, y1-self.yc
        self.S = numpy.sqrt(dx**2+dy**2)             # half-width
        self.sx, self.sy = dx/self.S, dy/self.S      # tangent

    def velocity(self, x, y, gamma=None):
        """Compute the velocity induced by the panel

        Inputs:
        x,y   -- the x and y location of the desired velocity
        gamma -- the panel vortex strength; defaults to self.gamma.

        Outputs:
        u,v   -- the x and y components of the velocity

        Examples:
        p_2 = vp.Panel(0,-1,0,1,4)        # make panel on y-axis with gamma=4
        u,v = p_2.velocity(-1,0)          # get induced velocity on x-axis
        u,v = p_2.velocity(-1,0,gamma=1)  # get velocity using gamma=1
        """
        if gamma is None: gamma = self.gamma  # default gamma
        xp,yp = self.__transform_xy(x, y)     # transform
        up = _get_u(xp, yp, self.S, gamma)    # get u prime
        vp = _get_v(xp, yp, self.S, gamma)    # get v prime
        return self.__rotate_uv(up, vp)       # rotate back

    def plot(self, style='k'):
        """Plot the vortex panel as a line segment

        Inputs:
        style -- a string defining the matplotlib style

        Examples:
        my_panel = vp.Panel(0,-1,0,1)  # creates panel on y-axis
        my_panel.plot()                # plot the panel
        """
        return pyplot.plot(self.x, self.y, style, lw=2)

    def __transform_xy(self, x, y):
        "transform from global to panel coordinates"
        xt = x-self.xc               # shift x
        yt = y-self.yc               # shift y
        xp = xt*self.sx+yt*self.sy   # rotate x
        yp = yt*self.sx-xt*self.sy   # rotate y
        return xp, yp

    def __rotate_uv(self, up, vp):
        "rotate velocity back to global coordinates"
        u = up*self.sx-vp*self.sy    # reverse rotate u prime
        v = vp*self.sx+up*self.sy    # reverse rotate v prime
        return u, v


### Geometries

def panelize(N,x,y):
    "Turn x,y arrays (length N+1) into an array of Panels (length N)"
    panels = numpy.empty(N, dtype=object)  # empty array of panels
    for i in range(N):                     # fill the array
        panels[i] = Panel(x[i], y[i], x[i+1], y[i+1])
    return panels

def make_polygon(N,sides):
    """ Make a polygonal array of Panels

    Inputs:
    N     -- number of panels to use
    sides -- number of sides in the polygon

    Outputs:
    panels  -- an array of Panels; see help(Panel)

    Examples:
    triangle = vp.make_polygon(N=33,sides=3)  # make a triangular Panel array
    for panel in triangle: panel.plot()       # plot the geometry
    """
    # define the end-points
    theta = numpy.linspace(0, -2*numpy.pi, N+1)          # equally spaced theta
    r = numpy.cos(numpy.pi/sides)/numpy.cos(             # r(theta)
            theta % (2.*numpy.pi/sides)-numpy.pi/sides)
    x,y = r*numpy.cos(theta), r*numpy.sin(theta)         # get the coordinates
    return panelize(N,x,y)

def make_ellipse(N, t_c, xcen=0, ycen=0):
    """ Make an elliptical array of Panels; defaults to circle

    Inputs:
    N         -- number of panels to use
    t_c       -- thickness/chord of the ellipse
    xcen,ycen -- location of the ellipse center; defaults to origin

    Outputs:
    panels  -- an array of Panels; see help(Panel)

    Examples:
    ellipse = vp.make_circle(N=32,t_c=0.5) # make a 1:2 elliptical Panel array
    for panel in ellipse: panel.plot()     # plot the geometry
    """
    theta = numpy.linspace(0, -2*numpy.pi, N+1)
    x,y = numpy.cos(theta)+xcen, numpy.sin(theta)*t_c+ycen
    return panelize(N,x,y)

def make_circle(N, xcen=0, ycen=0):
    "Make circle as special case of make_ellipse"
    return make_ellipse(N, t_c=1, xcen=xcen, ycen=ycen)

def make_jukowski(N, dx=0.18, dtheta=0, dr=0):
    """ Make a foil-shaped array of Panels using the Jukowski mapping

    Note:
    Foil-shapes are obtained adjusting a circle and then mapping

    Inputs:
    N         -- number of panels to use
    dx        -- negative extent beyond x = -1
    dtheta    -- angle of rotation around (1,0)
    dr        -- radius extent beyond r = 1

    Outputs:
    panels  -- an array of Panels; see help(Panel)

    Examples:
    foil = vp.make_jukowski(N=64)    # make a symmetric foil Panel array
    for panel in foil: panel.plot()  # plot the geometry
    """
    # define the circle
    theta = numpy.linspace(0, -2*numpy.pi, N+1)
    r = (1+dx)/numpy.cos(dtheta)+dr
    x,y = r*numpy.cos(theta)-(r-1-dr), r*numpy.sin(theta)

    #rotate around (1,0)
    ds,dc = numpy.sin(dtheta),numpy.cos(dtheta)
    x2,y2 =  dc*(x-1)+ds*y+1, -ds*(x-1)+dc*y
    r2 = x2**2+y2**2

    # apply jukowski mapping
    x3,y3 = x2*(1+1./r2)/2, y2*(1-1./r2)/2
    return panelize(N,x3,y3)


### Flow solver

def _check_alpha(alpha):
    "catch iterated alpha"
    if(isinstance(alpha, (set, list, tuple, numpy.ndarray))):
        raise TypeError('Only accepts scalar alpha')

def _construct_A_b(panels,alpha=0):
    "construct the linear system to enforce no-slip on every panel"

    # get arrays
    xc,yc,sx,sy = get_array(panels,'xc','yc','sx','sy')

    # construct the matrix
    A = numpy.empty((len(xc), len(xc)))      # empty matrix
    for j, p_j in enumerate(panels):         # loop over panels
        u,v = p_j.velocity(xc,yc,gamma=1)      # f_j at all panel centers
        A[:,j] = u*sx+v*sy                     # tangential component
    numpy.fill_diagonal(A, 0.5)              # fill diagonal with 1/2

    # construct the RHS
    b = -numpy.cos(alpha)*sx-numpy.sin(alpha)*sy
    return A, b

def solve_gamma(panels,alpha=0,kutta=[]):
    """ Determine the vortex strength on an array of Panels needed
    to enforce the no slip and kutta conditions

    Notes:
    Assumes unit magnitude background flow, |U|=1.
    The same alpha must be used in both solve_gamma and plot_flow or
      the wrong flow will be displayed. See example below.

    Inputs:
    panels  -- an array of Panel objects (modified on output)
    alpha   -- angle of attack relative to x-axis; must be a scalar; default 0
    kutta   -- panel indices (a list of tuples) on which to enforce the kutta
                condition; defaults to empty list

    Outputs:
    panels  -- the same array of Panels, but with the gamma attribute updated.

    Examples:
    foil = vp.make_jukowski(N=32)                    # make a Panel array
    vp.solve_gamma(foil, alpha=0.1, kutta=[(0,-1)])  # solve for gamma
    vp.plot_flow(foil, alpha=0.1)                    # plot the flow
    """
    _check_alpha(alpha)                   # check alpha
    A,b = _construct_A_b(panels,alpha)    # construct linear system
    for i in kutta:                       # loop through index pairs
        A[i[0]:i[1],i] += 1               # apply kutta condition
    gamma = numpy.linalg.solve(A, b)      # solve for gamma!
    for i,p_i in enumerate(panels):
        p_i.gamma = gamma[i]              # update panels

def solve_gamma_kutta(panels,alpha=0):
    "special case of solve_gamma with kutta=[(0,-1)]"
    return solve_gamma(panels,alpha,kutta=[(0,-1)])


### Visualize

def _flow_velocity(panels,x,y,alpha=0):
    "get the velocity induced by panels and unit velocity at angle `alpha`"
    # get the uniform velocity ( make it the same size & shape as x )
    u = numpy.cos(alpha)*numpy.ones_like(x)
    v = numpy.sin(alpha)*numpy.ones_like(x)

    # add the velocity contribution from each panel
    for p_j in panels:
        u_j, v_j = p_j.velocity(x, y)
        u, v = u+u_j, v+v_j

    return u, v

def plot_flow(panels,alpha=0,size=2):
    """ Plot the flow induced by a Panel array and the background flow

    Notes:
    Assumes unit magnitude background flow, |U|=1.
    The same alpha must be used in both solve_gamma and plot_flow or
      the wrong flow will be displayed. See example below.

    Inputs:
    panels  -- an array of Panel objects
    alpha   -- angle of attack relative to x-axis; must be a scalar
    size    -- size of the domain; corners are at (-size,-size) and (size,size)

    Outputs:
    pyplot of the flow vectors, velocity magnitude contours, and the panels.

    Examples:
    circle = vp.make_circle(N=32)      # make a Panel array
    vp.solve_gamma(circle, alpha=0.1)  # solve for Panel strengths
    vp.plot_flow(circle, alpha=0.1)    # plot the flow
    """
    # define the grid
    line = numpy.linspace(-size, size, 100)  # computes a 1D-array
    x, y = numpy.meshgrid(line, line)        # generates a mesh grid

    # get the velocity from the free stream and panels
    _check_alpha(alpha)
    u, v = _flow_velocity(panels, x, y, alpha)

    # plot it
    pyplot.figure(figsize=(6,5))        # set size
    pyplot.xlabel('x', fontsize=14)     # label x
    pyplot.ylabel('y', fontsize=14)     # label y

    # plot contours
    m = numpy.sqrt(u**2+v**2)
    velocity = pyplot.contourf(x, y, m, vmin=0)
    cbar = pyplot.colorbar(velocity)
    cbar.set_label('Velocity magnitude', fontsize=14);

    # plot vector field
    pyplot.quiver(x[::4,::4], y[::4,::4],
                  u[::4,::4], v[::4,::4])
    # plot panels
    for p in panels: p.plot()


## Panel array operations

def get_array(panels,key,*args):
    """ Generate numpy arrays of panel attributes

    Notes:
    Use help(panel) to see available attributes

    Inputs:
    panels       -- an array of Panel objects
    key (,*args) -- one or more names of the desired attributes

    Outputs:
    key_vals     -- numpy arrays filled with the named attributes

    Examples:
    circle = vp.make_circle(N=32)           # make a Panel array
    xc,yc = vp.get_array(circle,'xc','yc')  # get arrays of the panel centers
    """
    if not args:
        return numpy.array([getattr(p,key) for p in panels])
    else:
        return [get_array(panels,k) for k in (key,)+args]

def distance(panels):
    """ Find the cumulative distance of the path along a set of panels

    Inputs:
    panels  -- array of Panels

    Outputs:
    s       -- array of distances from the edge of the first
               panel to the center of each panel

    Examples:
    foil = vp.make_jukowski(N=64)       # define the geometry
    s = vp.distance(foil)               # get the panel path distance
    """
    S = get_array(panels,'S')
    return numpy.cumsum(2*S)-S
