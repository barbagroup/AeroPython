# Vortex panel method module
import numpy
from matplotlib import pyplot

# velocity component functions
def get_u( x, y, S, gamma ):
    return gamma/(2*numpy.pi)*(numpy.arctan((x-S)/y)-numpy.arctan((x+S)/y))
def get_v( x, y, S, gamma ):
    return gamma/(4*numpy.pi)*(numpy.log(((x+S)**2+y**2)/((x-S)**2+y**2)))

# vortex panel class
class Panel:
    
    # save the inputs and pre-compute factors for the coordinate tranform
    def __init__( self, x0, y0, x1, y1, gamma=0 ):
        self.x,self.y,self.gamma = [x0,x1],[y0,y1],gamma
        self.xc = 0.5*(x0+x1)                # panel x-center
        self.yc = 0.5*(y0+y1)                # panel y-center
        self.S = numpy.sqrt(                 # ...
                            (x1-self.xc)**2+(y1-self.yc)**2) # panel width
        self.sx = (x1-self.xc)/self.S        # unit vector in x
        self.sy = (y1-self.yc)/self.S        # unit vector in y
    
    # get the velocity!
    def velocity( self, x, y, gamma=None ):
        if gamma is None: gamma = self.gamma # default gamma
        xp,yp = self.transform_xy( x, y )    # transform
        up = get_u( xp, yp, self.S, gamma )  # get u prime
        vp = get_v( xp, yp, self.S, gamma )  # get v prime
        return self.rotate_uv( up, vp )      # rotate back
    
    # plot the panel
    def plot(self):
        return pyplot.plot(self.x,self.y,'k-',lw=2)
    
    # transform from global to panel coordinates
    def transform_xy( self, x, y ):
        xt = x-self.xc               # shift x
        yt = y-self.yc               # shift y
        xp = xt*self.sx+yt*self.sy   # rotate x
        yp = yt*self.sx-xt*self.sy   # rotate y
        return [ xp, yp ]
    
    # rotate velocity back to global coordinates
    def rotate_uv( self, up, vp):
        u = up*self.sx-vp*self.sy    # reverse rotate u prime
        v = vp*self.sx+up*self.sy    # reverse rotate v prime
        return [ u, v ]

# plot the flow on a grid
def plot_flow(panels,alpha=0,xmax=2,N_grid=100):
    # define the grid
    X = numpy.linspace(-xmax, xmax, N_grid) # computes a 1D-array for x
    Y = numpy.linspace(-xmax, xmax, N_grid) # computes a 1D-array for y
    x, y = numpy.meshgrid(X, Y)             # generates a mesh grid

    # get the uniform velocity on the grid
    u = numpy.cos(alpha)*numpy.ones((N_grid,N_grid))
    v = numpy.sin(alpha)*numpy.ones((N_grid,N_grid))
    
    # add the velocity contribution from each panel
    for p in panels:
        u0,v0 = p.velocity(x,y)
        u = u+u0
        v = v+v0
    
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
#    pyplot.streamplot(x, y, u, v)       # plots streamlines - this is slow!
    for p in panels: p.plot()

# define the influence of panel_j on panel_i
def influence(panel_i,panel_j):
    u,v = panel_j.velocity(panel_i.xc,panel_i.yc,gamma=1)
    return u*panel_i.sx+v*panel_i.sy

# construct the linear system
def construct_A_b(panels,alpha=0):
    # construct matrix
    N_panels = len(panels)
    A = numpy.empty((N_panels, N_panels), dtype=float) # empty matrix
    numpy.fill_diagonal(A, 0.5)                        # fill diagonal with 1/2
    for i, p_i in enumerate(panels):
        for j, p_j in enumerate(panels):
            if i != j:                                 # off-diagonals
                A[i,j] = influence(p_i,p_j)            # find influence
    
    # computes the RHS
    b = [-numpy.cos(alpha)*p.sx-numpy.sin(alpha)*p.sy for p in panels]
    return [A,b]

# determine the vortex strength on a set of panels
def solve_gamma(panels,alpha=0):
    A,b = construct_A_b(panels,alpha)  # construct linear system
    gamma = numpy.linalg.solve(A, b)   # solve for gamma!
    for i,p_i in enumerate(panels):
        p_i.gamma = gamma[i]           # update panels

# determine the vortex panel strength with Kutta Condition
def solve_gamma_kutta(panels,alpha=0):
    A,b = construct_A_b(panels,alpha)   # construct linear system
    A[:, 0] += 1                        # gamma[0]+ ...
    A[:,-1] += 1                        # gamma[N-1]=0
    gamma = numpy.linalg.solve(A, b)    # solve for gamma!
    for i,p_i in enumerate(panels):
        p_i.gamma = gamma[i]            # update panels


### Geometries

# make a circle
def make_circle(N,t_c=1):
    # define the end-points of the panels
    x_ends = numpy.cos(numpy.linspace(0, -2*numpy.pi, N+1))
    y_ends = numpy.sin(numpy.linspace(0, -2*numpy.pi, N+1))
    y_ends *= t_c
    
    # define the panels
    circle = numpy.empty(N, dtype=object)
    for i in xrange(N):
        circle[i] = Panel(x_ends[i], y_ends[i], x_ends[i+1], y_ends[i+1])
    
    return circle

# make a jukowski foil
def make_jukowski( N, dx = 0.2, dy = 0, dr = 0 ):
    # define the circle
    x_ends = numpy.cos(numpy.linspace(0, -2*numpy.pi, N+1))
    y_ends = numpy.sin(numpy.linspace(0, -2*numpy.pi, N+1))
    
    # shift circle
    r = numpy.sqrt((1+dx)**2+dy**2)+dr
    x2_ends = r*x_ends-dx
    y2_ends = r*y_ends-dy
    r2_ends = x2_ends**2+y2_ends**2
    
    # apply jukowski mapping
    x3_ends = x2_ends*(1+1./r2_ends)/2
    y3_ends = y2_ends*(1-1./r2_ends)/2
    
    # define the panels
    foil = numpy.empty(N, dtype=object)
    for i in xrange(N):
        foil[i] = Panel(x3_ends[i], y3_ends[i], x3_ends[i+1], y3_ends[i+1])
    
    return foil
