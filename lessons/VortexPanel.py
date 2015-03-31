# Vortex panel method module
import numpy
from matplotlib import pyplot

# velocity component functions
def getU( x, y, S, gamma ):
    return gamma/(2*numpy.pi)*(numpy.arctan((x-S)/y)-numpy.arctan((x+S)/y))
def getV( x, y, S, gamma ):
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
        xp,yp = self.transformXY( x, y )     # transform
        up = getU( xp, yp, self.S, gamma )   # get up
        vp = getV( xp, yp, self.S, gamma )   # get vp
        return self.rotateUV( up, vp )       # rotate back
    
    # plot the panel
    def plot(self):
        return pyplot.plot(self.x,self.y,'k-',lw=2)
    
    # transform from global to panel coordinates
    def transformXY( self, x, y ):
        xp = x-self.xc
        yp = y-self.yc
        xpp = xp*self.sx+yp*self.sy
        ypp = yp*self.sx-xp*self.sy
        return [ xpp, ypp ]
    
    # rotate velocity back to global coordinates
    def rotateUV( self, u, v):
        up = u*self.sx-v*self.sy
        vp = v*self.sx+u*self.sy
        return [ up, vp ]

# compute the flow field
def flowVelocity(panels,x,y,alpha=0):
    # get the uniform velocity ( make it the same size & shape as x )
    u = numpy.cos(alpha)*numpy.ones_like(x)
    v = numpy.sin(alpha)*numpy.ones_like(x)
    
    # add the velocity contribution from each panel
    for p in panels:
        u0,v0 = p.velocity(x,y)
        u = u+u0
        v = v+v0
    return [u,v]

# plot the flow on a grid
def plotFlow(panels,alpha=0,xmax=2,N_grid=100):
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
def constructAb(panels,alpha=0):
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
def solveGamma(panels,alpha=0):
    A,b = constructAb(panels,alpha)    # construct linear system
    gamma = numpy.linalg.solve(A, b)   # solve for gamma!
    for i,p_i in enumerate(panels):
        p_i.gamma = gamma[i]           # update panels
