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
