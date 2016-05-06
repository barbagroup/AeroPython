""" Solve potential flow problems using vortex panels

This module holds routines to determine the potential flow around
bodies of any shape or number using constant strength vortex panels.

Class:
    Panel

Methods:
    plot_flow
    solve_gamma
    solve_gamma_kutta
    get_array
    make_polygon
    make_ellipse
    make_circle
    make_jukowski

Imports: numpy, pyplot from matplotlib
"""

import numpy
from matplotlib import pyplot

### Fundamentals

# x-component of induced velocity
def _get_u( x, y, S, gamma ):
    return gamma/(2*numpy.pi)*(numpy.arctan((x-S)/y)-numpy.arctan((x+S)/y))

# y-component of induced velocity
def _get_v( x, y, S, gamma ):
    return gamma/(4*numpy.pi)*(numpy.log(((x+S)**2+y**2)/((x-S)**2+y**2)))

# Constant strength vortex panel
class Panel(object):
    """Constant strength vortex panel class.

    Attributes:
    xc,yc -- the x and y location of the panel center
    S     -- the half-width of the panel
    sx,sy -- the x and y component of the tangent unit vector
    gamma -- the panel vortex strength. Defaults to zero.
    """

    def __init__(self, x0, y0, x1, y1, gamma=0):
        """Initialize a panel between two points.

        Inputs:
        x0,y0 -- the x and y location of the starting point
        x1,y1 -- the x and y location of the ending point
        gamma -- the panel vortex strength. Defaults to zero.

        Outputs:
        A Panel object.

        Examples:
        my_panel1 = Panel(-1,0,1,0)    # creates panel on x-axis with gamma=0
        my_panel2 = Panel(0,-1,0,1,4)  # creates panel on y-axis with gamma=4
        """
        self.x,self.y,self.gamma = [x0,x1],[y0,y1],gamma
        self.xc = 0.5*(x0+x1)                # x-center
        self.yc = 0.5*(y0+y1)                # y-center
        self.S = numpy.sqrt((x1-self.xc)**2+
                            (y1-self.yc)**2) # half-width
        self.sx = (x1-self.xc)/self.S        # x-tangent
        self.sy = (y1-self.yc)/self.S        # y-tangent

    def velocity(self, x, y, gamma=None):
        """Compute the velocity induced by the panel

        Inputs:
        x,y -- the x and y location where you want the velocity
        gamma -- the panel vortex strength. Defaults to Panel.gamma.

        Outputs:
        u,v -- the x and y components of the velocity

        Examples:
        my_panel = Panel(0,-1,0,1,4)           # creates panel on y-axis with gamma=4
        u,v = my_panel.velocity(-1,0)          # finds the induced velocity on x-axis
        u,v = my_panel.velocity(-1,0,gamma=1)  # finds the velocity using gamma=1 instead of 4
        """
        if gamma is None: gamma = self.gamma  # default gamma
        xp,yp = self.__transform_xy(x, y)     # transform
        up = _get_u(xp, yp, self.S, gamma)    # get u prime
        vp = _get_v(xp, yp, self.S, gamma)    # get v prime
        return self.__rotate_uv(up, vp)       # rotate back

    def plot(self):
        """Plots the vortex panel as a black line segment

        Examples:
        my_panel = Panel(0,-1,0,1)             # creates panel on y-axis
        my_panel.plot()                        # plot the panel
        """
        return pyplot.plot(self.x,self.y,'k-',lw=2)

    # transform from global to panel coordinates
    def __transform_xy(self, x, y):
        xt = x-self.xc               # shift x
        yt = y-self.yc               # shift y
        xp = xt*self.sx+yt*self.sy   # rotate x
        yp = yt*self.sx-xt*self.sy   # rotate y
        return xp, yp

    # rotate velocity back to global coordinates
    def __rotate_uv(self, up, vp):
        u = up*self.sx-vp*self.sy    # reverse rotate u prime
        v = vp*self.sx+up*self.sy    # reverse rotate v prime
        return u, v


## Panel array operations

def get_array(panels,key):
    """ Generate numpy array of panel attributes

    Inputs:
    panels   -- an array of Panel objects
    key      -- a string naming the desired attribute.
              Use help(panel) to see available attributes

    Outputs:
    key_vals -- a numpy array of length(Panels) filled with the named attribute

    Examples:
    circle = make_circle(N=32)   # make a Panel array
    xc = get_array(circle,'xc')  # get the x-location of each panel center
    S = get_array(circle,'S')    # get the half-width of each panel
    """
    return numpy.array([getattr(p,key) for p in panels])

def distance(panels):
    """ Find the cumulative distance of the path along a set of panels

    Inputs:
    panels  -- array of Panels

    Outputs:
    s       -- array of distances from the edge of the first
               panel to the center of each panel

    Examples:
    foil = make_jukowski(N=64)       # define the geometry
    s = distance(foil)               # get the panel path distance
    """
    S = get_array(panels,'S')
    return numpy.cumsum(2*S)-S


### Visualize

# get the velocity induced by panels and unit velocity at angle `alpha`.
def __flow_velocity(panels,x,y,alpha=0):
    # the flow angle must be a scalar
    if(isinstance(alpha, (list, tuple, numpy.ndarray))):
        raise TypeError('Only accepts scalar alpha')

    # get the uniform velocity ( make it the same size & shape as x )
    u = numpy.cos(alpha)*numpy.ones_like(x)
    v = numpy.sin(alpha)*numpy.ones_like(x)

    # add the velocity contribution from each panel
    for p in panels:
        u0,v0 = p.velocity(x,y)
        u = u+u0
        v = v+v0

    return u, v

def __magnitude(u,v): return numpy.sqrt(u**2+v**2)

# plot the flow on a grid
def plot_flow(panels,alpha=0,size=2,N_grid=100,
              contour_min=0,contour_max=None,
              contour_fun=__magnitude,contour_lab='Velocity magnitude'):
    """ Plot the flow induced by a Panel array and the background flow

    Notes:
    Assumes unit magnitude background flow, |U|=1.
    The same alpha must be used in both solve_gamma and plot_flow or
      the wrong flow will be displayed. See example below.

    Inputs:
    panels  -- an array of Panel objects
    alpha   -- angle of attack relative to x-axis; must be a scalar; default 0
    size    -- size of the domain; corners are at (-size,-size) and (size,size); default 2
    N_grid  -- number of points in the domain grid in each direction; default 100
    contour_min -- the minimum contour value for the velocity magnitude; defaults to 0
    contour_max -- the maximum contour value for the velocity magnitude; defaults to max in domain
    contour_fun -- function of u,v to contour; defaults to |u|
    contour_lab -- contour label; defaults to 'Velocity magnitude'

    Outputs:
    pyplot object of the flow vectors, contours of the velocity magnitude, and the panels.

    Examples:
    circle = make_circle(N=32)      # make a Panel array
    solve_gamma(circle, alpha=0.1)  # solve for Panel strengths
    plot_flow(circle, alpha=0.1)    # plot the flow
    """
    # define the grid
    X = numpy.linspace(-size, size, N_grid) # computes a 1D-array for x
    Y = numpy.linspace(-size, size, N_grid) # computes a 1D-array for y
    x, y = numpy.meshgrid(X, Y)             # generates a mesh grid

    # get the velocity from the free stream and panels
    u,v = __flow_velocity(panels,x,y,alpha)

    # plot it
    pyplot.figure(figsize=(6,5))        # set size
    pyplot.xlabel('x', fontsize=14)     # label x
    pyplot.ylabel('y', fontsize=14)     # label y

    # plot contours
    m = contour_fun(u,v)
    velocity = pyplot.contourf(x, y, m, vmin=contour_min, vmax=contour_max)
    cbar = pyplot.colorbar(velocity)
    cbar.set_label(contour_lab, fontsize=14);

    # plot vector field
    pyplot.quiver(x[::4,::4], y[::4,::4],
                  u[::4,::4], v[::4,::4])
    # plot panels
    for p in panels: p.plot()

### Flow solvers

# define the influence of panel_j on panel_i
def __influence(panel_i,panel_j):
    u,v = panel_j.velocity(panel_i.xc,panel_i.yc,gamma=1)
    return u*panel_i.sx+v*panel_i.sy


# construct the linear system to enforce no-slip on every panel
def __construct_A_b(panels,alpha=0):
    # construct matrix
    N = len(panels)
    A = numpy.empty((N, N))                     # empty matrix
    numpy.fill_diagonal(A, 0.5)                 # fill diagonal with 1/2
    for i, p_i in enumerate(panels):
        for j, p_j in enumerate(panels):
            if i != j:                          # off-diagonals
                A[i,j] = __influence(p_i,p_j)

    # construct the RHS
    if(isinstance(alpha, (list, tuple, numpy.ndarray))):
        raise TypeError('Only accepts scalar alpha')
    b = [-numpy.cos(alpha)*p.sx-numpy.sin(alpha)*p.sy for p in panels]
    return A, b

def solve_gamma(panels,alpha=0):
    """ Determine the vortex strength on an array of Panels

    Notes:
    Assumes unit magnitude background flow, |U|=1.
    The same alpha must be used in both solve_gamma and plot_flow or
      the wrong flow will be displayed. See example below.

    Inputs:
    panels  -- an array of Panel objects (modified on output)
    alpha   -- angle of attack relative to x-axis; must be a scalar; default 0

    Outputs:
    panels  -- the same array of Panels, but with the gamma attribute updated.

    Examples:
    circle = make_circle(N=32)      # make a Panel array
    solve_gamma(circle, alpha=0.1)  # solve for Panel strengths
    plot_flow(circle, alpha=0.1)    # plot the flow
    """
    A,b = __construct_A_b(panels,alpha)# construct linear system
    gamma = numpy.linalg.solve(A, b)   # solve for gamma!
    for i,p_i in enumerate(panels):
        p_i.gamma = gamma[i]           # update panels


def solve_gamma_kutta(panels,alpha=0):
    """ Determine gamma while enforcing the Kutta condition on panels[(0,-1)]

    Notes:
    Same as solve_gamma, but enforces Kutta condition on the first and last panel.
    Assumes unit magnitude background flow, |U|=1.
    The same alpha must be used in both solve_gamma and plot_flow or
      the wrong flow will be displayed. See example below.

    Inputs:
    panels  -- an array of Panel objects (modified on output)
    alpha   -- angle of attack relative to x-axis; must be a scalar; default 0

    Outputs:
    panels  -- the same array of Panels, but with the gamma attribute updated.

    Examples:
    foil = make_jukowski(N=32)          # make a Panel array
    solve_gamma_kutta(foil, alpha=0.1)  # solve for Panel strengths
    plot_flow(foil, alpha=0.1)          # plot the flow
    """
    A,b = __construct_A_b(panels,alpha) # construct linear system
    A[:,(0,-1)] += 1                    # gamma[0] + gamma[N-1] = 0
    gamma = numpy.linalg.solve(A, b)    # solve for gamma!
    for i,p_i in enumerate(panels):
        p_i.gamma = gamma[i]            # update panels


### Geometries

# polygonal shape function
def _polygon(theta,N_sides):
    a = theta % (2.*numpy.pi/N_sides)-numpy.pi/N_sides
    r = numpy.cos(numpy.pi/N_sides)/numpy.cos(a)
    return [r*numpy.cos(theta),r*numpy.sin(theta)]

def make_polygon(N_panels,N_sides):
    """ Make a polygonal array of Panels

    Inputs:
    N_panels -- number of panels to use
    N_sides  -- number of sides in the polygon

    Outputs:
    panels  -- an array of Panels; see help(Panel)

    Examples:
    triangle = make_polygon(N_panels=33,N_sides=3)  # make a triangular Panel array
    for panel in triangle: panel.plot()             # plot the geometry
    """
    # define the end-points
    theta = numpy.linspace(0, -2*numpy.pi, N_panels+1)   # equal radial spacing
    x_ends,y_ends = _polygon(theta, N_sides)             # get the coordinates

    # define the panels
    panels = numpy.empty(N_panels, dtype=object)         # empty array of panels
    for i in range(N_panels):                            # fill the array
        panels[i] = Panel(x_ends[i], y_ends[i], x_ends[i+1], y_ends[i+1])

    return panels

def make_ellipse(N, t_c=1, xcen=0, ycen=0):
    """ Make an elliptical array of Panels; defaults to circle

    Inputs:
    N         -- number of panels to use
    t_c       -- thickness/chord of the ellipse; default 1 (circle)
    xcen,ycen -- x,y location of the ellipse center

    Outputs:
    panels  -- an array of Panels; see help(Panel)

    Examples:
    ellipse = make_circle(N=32,t_c=0.5)   # make a 1:2 elliptical Panel array
    for panel in ellipse: panel.plot()    # plot the geometry
    """
    theta = numpy.linspace(0, -2*numpy.pi, N+1)
    x_ends = numpy.cos(theta)+xcen
    y_ends = numpy.sin(theta)*t_c+ycen

    # define the panels
    ellipse = numpy.empty(N, dtype=object)
    for i in range(N):
        ellipse[i] = Panel(x_ends[i], y_ends[i], x_ends[i+1], y_ends[i+1])

    return ellipse

# make circle and special case of make_ellipse
def make_circle(N, xcen=0, ycen=0):
    return make_ellipse(N, t_c=1, xcen=xcen, ycen=ycen)

def make_jukowski(N, dx=0.18, dtheta=0, dr = 0):
    """ Make a foil-shaped array of Panels using the Jukowski mapping

    Note:
    Different foil-shapes are obtained by scaling and rotating a circle
    before applying the mapping.

    Inputs:
    N         -- number of panels to use
    dx        -- amount to scale circle around (1,0)
    dtheta    -- amount to rotate circle around (1,0)
    dr        -- amount to scale circle around (0,0)

    Outputs:
    panels  -- an array of Panels; see help(Panel)

    Examples:
    foil = make_jukowski(N=64,dtheta=0.1)   # make a cambered foil Panel array
    for panel in foil: panel.plot()         # plot the geometry
    """
    # define the shifted circle
    theta = numpy.linspace(0, -2*numpy.pi, N+1)
    r = (1+dx)/numpy.cos(dtheta)+dr
    x_ends = r*numpy.cos(theta)-(r-1-dr)
    y_ends = r*numpy.sin(theta)

    #rotate around (1,0)
    ds,dc = numpy.sin(dtheta),numpy.cos(dtheta)
    x2_ends =  dc*(x_ends-1)+ds*y_ends+1
    y2_ends = -ds*(x_ends-1)+dc*y_ends
    r2_ends = x2_ends**2+y2_ends**2

    # apply jukowski mapping
    x3_ends = x2_ends*(1+1./r2_ends)/2
    y3_ends = y2_ends*(1-1./r2_ends)/2

    # define the panels
    foil = numpy.empty(N, dtype=object)
    for i in range(N):
        foil[i] = Panel(x3_ends[i], y3_ends[i], x3_ends[i+1], y3_ends[i+1])

    return foil
