""" Solve potential flow problems using vortex panels

This module holds routines to determine the potential flow around
bodies of any shape or number using constant strength vortex panels.

Example:
    Solve and plot the flow around a circle using 64 panels:

        circle = make_circle(N=64)
        solve_gamma(circle)
        plot_flow(circle)

Imports: numpy, pyplot from matplotlib
"""

import numpy
from matplotlib import pyplot

### Fundamentals

# x-component of induced velocity
def get_u( x, y, S, gamma ):
    return gamma/(2*numpy.pi)*(numpy.arctan((x-S)/y)-numpy.arctan((x+S)/y))

# y-component of induced velocity
def get_v( x, y, S, gamma ):
    return gamma/(4*numpy.pi)*(numpy.log(((x+S)**2+y**2)/((x-S)**2+y**2)))

# Constant strength vortex panel
class Panel(object):

    # Initialize a panel between two point with strength gamma
    def __init__(self, x0, y0, x1, y1, gamma=0):
        self.x,self.y,self.gamma = [x0,x1],[y0,y1],gamma
        self.xc = 0.5*(x0+x1)                # x-center
        self.yc = 0.5*(y0+y1)                # y-center
        self.S = numpy.sqrt((x1-self.xc)**2+
                            (y1-self.yc)**2) # half-width
        self.sx = (x1-self.xc)/self.S        # x-tangent
        self.sy = (y1-self.yc)/self.S        # y-tangent

    # get the velocity induced by the panel
    # note: you can adjust gamma as an optional argument
    def velocity(self, x, y, gamma=None):
        if gamma is None: gamma = self.gamma # default gamma
        xp,yp = self.transform_xy(x, y)      # transform
        up = get_u(xp, yp, self.S, gamma)    # get u prime
        vp = get_v(xp, yp, self.S, gamma)    # get v prime
        return self.rotate_uv(up, vp)        # rotate back

    # plot the panel
    def plot(self):
        return pyplot.plot(self.x,self.y,'k-',lw=2)

    # transform from global to panel coordinates
    def transform_xy(self, x, y):
        xt = x-self.xc               # shift x
        yt = y-self.yc               # shift y
        xp = xt*self.sx+yt*self.sy   # rotate x
        yp = yt*self.sx-xt*self.sy   # rotate y
        return xp, yp

    # rotate velocity back to global coordinates
    def rotate_uv(self, up, vp):
        u = up*self.sx-vp*self.sy    # reverse rotate u prime
        v = vp*self.sx+up*self.sy    # reverse rotate v prime
        return u, v


### Visualize

# get the velocity induced by panels and unit velocity at angle `alpha`.
def flow_velocity(panels,x,y,alpha=0):
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


# plot the flow on a grid
def plot_flow(panels,alpha=0,xmax=2,N_grid=100):
    # define the grid
    X = numpy.linspace(-xmax, xmax, N_grid) # computes a 1D-array for x
    Y = numpy.linspace(-xmax, xmax, N_grid) # computes a 1D-array for y
    x, y = numpy.meshgrid(X, Y)             # generates a mesh grid

    # get the velocity from the free stream and panels
    u,v = flow_velocity(panels,x,y,alpha)

    # plot it
    pyplot.figure(figsize=(8,11))       # set size
    pyplot.xlabel('x', fontsize=16)     # label x
    pyplot.ylabel('y', fontsize=16)     # label y
    m = numpy.sqrt(u**2+v**2)           # compute velocity magnitude
    velocity = pyplot.contourf(x, y, m) # plot magnitude contours
    cbar = pyplot.colorbar(velocity, orientation='horizontal')
    cbar.set_label('Velocity magnitude', fontsize=16);
    pyplot.quiver(x[::4,::4], y[::4,::4],
                  u[::4,::4], v[::4,::4]) # plot vector field
    for p in panels: p.plot()


### Flow solvers

# define the influence of panel_j on panel_i
def influence(panel_i,panel_j):
    u,v = panel_j.velocity(panel_i.xc,panel_i.yc,gamma=1)
    return u*panel_i.sx+v*panel_i.sy


# construct the linear system to enforce no-slip on every panel
def construct_A_b(panels,alpha=0):
    # construct matrix
    N = len(panels)
    A = numpy.empty((N, N))                     # empty matrix
    numpy.fill_diagonal(A, 0.5)                 # fill diagonal with 1/2
    for i, p_i in enumerate(panels):
        for j, p_j in enumerate(panels):
            if i != j:                          # off-diagonals
                A[i,j] = influence(p_i,p_j)

    # construct the RHS
    if(isinstance(alpha, (list, tuple, numpy.ndarray))):
        raise TypeError('Only accepts scalar alpha')
    b = [-numpy.cos(alpha)*p.sx-numpy.sin(alpha)*p.sy for p in panels]
    return A, b


# determine the vortex strength on a set of panels
def solve_gamma(panels,alpha=0):
    A,b = construct_A_b(panels,alpha)  # construct linear system
    gamma = numpy.linalg.solve(A, b)   # solve for gamma!
    for i,p_i in enumerate(panels):
        p_i.gamma = gamma[i]           # update panels


# determine gamma while enforcing the Kutta condition on panels[(0,-1)]
def solve_gamma_kutta(panels,alpha=0):
    A,b = construct_A_b(panels,alpha)   # construct linear system
    A[:,(0,-1)] += 1                    # gamma[0] + gamma[N-1] = 0
    gamma = numpy.linalg.solve(A, b)    # solve for gamma!
    for i,p_i in enumerate(panels):
        p_i.gamma = gamma[i]            # update panels


### Geometries

# polynomial shape function
def polynomial(theta,N_sides):
    a = theta % (2.*numpy.pi/N_sides)-numpy.pi/N_sides
    r = numpy.cos(numpy.pi/N_sides)/numpy.cos(a)
    return [r*numpy.cos(theta),r*numpy.sin(theta)]

# make a polynomial array of Panels
def make_poly(N_panels,N_sides):
    # define the end-points
    theta = numpy.linspace(0, -2*numpy.pi, N_panels+1)   # equal radial spacing
    x_ends,y_ends = polynomial( theta, N_sides)          # get the coordinates

    # define the panels
    panels = numpy.empty(N_panels, dtype=object)         # empty array of panels
    for i in range(N_panels):                            # fill the array
        panels[i] = Panel(x_ends[i], y_ends[i], x_ends[i+1], y_ends[i+1])

    return panels

# make an ellipse (defaults to circle)
def make_circle(N, t_c=1, xcen=0, ycen=0):
    # define the end-points of the panels for a unit circle
    theta = numpy.linspace(0, -2*numpy.pi, N+1)
    x_ends = numpy.cos(theta)+xcen
    y_ends = numpy.sin(theta)*t_c+ycen

    # define the panels
    circle = numpy.empty(N, dtype=object)
    for i in range(N):
        circle[i] = Panel(x_ends[i], y_ends[i], x_ends[i+1], y_ends[i+1])

    return circle


# make a jukowski foil
def make_jukowski(N, dx=0.18, dtheta=0, dr = 0):
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
