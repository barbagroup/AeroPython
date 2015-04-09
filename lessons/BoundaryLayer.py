import numpy

# thickness ratios
def disp_ratio(lam): return 3./10.-lam/120.
def mom_ratio(lam): return 37./315.-lam/945.-lam**2/9072.

# wall derivative
def df_0(lam): return 2+lam/6.

# momentum function
def g_1(lam): return df_0(lam)-lam*(disp_ratio(lam)+2*mom_ratio(lam))

# use bisect method to find lambda0
from scipy.optimize import bisect
lam0 = bisect(g_1,-12,12)

# boundary layer thickness derivative
def ddx_delta(Re_d,lam):
    if Re_d==0: return 0                     # Stagnation point condition
    return g_1(lam)/mom_ratio(lam)/Re_d      # delta'

# Heun's method for ODE integration
def heun(g,psi_i,i,dx,*args):
    g_i = g(psi_i,i,*args)                      # integrand at i
    tilde_psi = psi_i+g_i*dx                    # predicted estimate at i+1
    g_i_1 = g(tilde_psi,i+1,*args)              # integrand at i+1
    return psi_i+0.5*(g_i+g_i_1)*dx             # corrected estimate

# boundary layer ODE integrand function
def g_pohl(delta_i,i,u_e,du_e,nu):
    Re_d = delta_i*u_e[i]/nu            # compute local Reynolds number
    lam = delta_i**2*du_e[i]/nu         # compute local lambda
    return ddx_delta(Re_d,lam)          # get derivative

# march along boundary layer from stagnation point to separation point
def march(x,u_e,du_e,nu):
    delta0 = numpy.sqrt(lam0*nu/du_e[0])                # set delta0
    delta = numpy.full_like(x,delta0)                   # delta array
    lam = numpy.full_like(x,lam0)                       # lambda array
    for i in range(len(x)-1):                           # march!
        delta[i+1] = heun(g_pohl,delta[i],i,x[i+1]-x[i],    # integrate BL using...
                          u_e,du_e,nu)                          # additional arguments
        lam[i+1] = delta[i+1]**2*du_e[i+1]/nu               # compute lambda
        if abs(lam[i+1])>12: break                          # check stop condition
    return delta,lam,i                                  # return with separation index


# Split panels into two boundary layer sections
def split_panels(panels):
    # positive velocity defines `top` BL
    top = [p for p in panels if p.gamma<=0]
    # negative defines the `bottom`
    bottom = [p for p in panels if p.gamma>=0]
    # reverse array so panel[0] is stagnation
    bottom = bottom[::-1]
    
    return top,bottom

# Pohlhausen Boundary Layer class
class Pohlhausen:
    def __init__(self,panels,nu):
        self.u_e = [abs(p.gamma) for p in panels]   # tangential velocity
        self.s = numpy.empty_like(self.u_e)         # initialize distance array
        self.s[0] = panels[0].S
        for i in range(len(self.s)-1):              # fill distance array
            self.s[i+1] = self.s[i]+panels[i].S+panels[i+1].S
        ds = numpy.gradient(self.s)
        self.du_e = numpy.gradient(self.u_e,ds)     # compute velocity gradient
        
        self.nu = nu                                # kinematic viscosity
        self.xc = [p.xc for p in panels]            # x and ...
        self.yc = [p.yc for p in panels]            # y locations
    
    def march(self):
        # march down the boundary layer until separation
        from BoundaryLayer import march
        self.delta,self.lam,self.iSep = march(self.s,self.u_e,self.du_e,self.nu)
        
        # interpolate values at the separation point
        def sep_interp(y): return numpy.interp(    # interpolate function
            12,-self.lam[self.iSep:self.iSep+2],y[self.iSep:self.iSep+2])
        self.s_sep = sep_interp(self.s)
        self.u_e_sep = sep_interp(self.u_e)
        self.x_sep = sep_interp(self.xc)
        self.y_sep = sep_interp(self.yc)
        self.delta_sep = sep_interp(self.delta)

# solve and plot the boundary layer flow.
def solve_plot_boundary_layers(panels,alpha=0,nu=1e-5):
    from VortexPanel import plot_flow
    from matplotlib import pyplot
    
    # split the panels
    top_panels,bottom_panels = split_panels(panels)
    
    # Set up and solve the top boundary layer
    top = Pohlhausen(top_panels,nu)
    top.march()
    
    # Set up and solve the bottom boundary layer
    bottom = Pohlhausen(bottom_panels,nu)
    bottom.march()
    
    # plot flow with separation points
    plot_flow(panels,alpha)
    pyplot.scatter(top.x_sep, top.y_sep, s=100, c='r')
    pyplot.scatter(bottom.x_sep, bottom.y_sep, s=100, c='g')
    
    return top,bottom