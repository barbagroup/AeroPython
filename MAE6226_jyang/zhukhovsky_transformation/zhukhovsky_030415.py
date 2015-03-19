import numpy
from matplotlib import pyplot
from potential_flow import *

#get grid
c   = 1.0
r1  = 1.15
r2  = 5.0
x_c, y_c = c-r1, 0.
z_c = x_c + 1j * y_c
Nr  = 100
Ntheta = 145
theta  = numpy.linspace(0., 2*numpy.pi, Ntheta)
r      = numpy.linspace(r1, r2, Nr)
thetagrid, rgrid = numpy.meshgrid(theta, r)

z  = rgrid*numpy.exp(1j*thetagrid) + z_c
xi = z + c/z
plot_grid(z, xi,2)

#get streamline
u_inf = 1.0
kappa = r1**2*2*numpy.pi*u_inf
x_doublet, y_doublet = x_c, y_c
z_doublet = x_c + 1j*y_c

u_doublet, v_doublet = get_velocity_doublet(kappa, x_doublet, y_doublet, z.real, z.imag)

psi_doublet = get_stream_function_doublet(kappa, x_doublet, y_doublet, z.real, z.imag)

u_freestream = u_inf * numpy.ones((Nr, Ntheta), dtype=numpy.float64)
v_freestream = numpy.zeros((Nr, Ntheta), dtype=numpy.float64)
psi_freestream = u_inf * z.imag

u = u_freestream + u_doublet
v = v_freestream + v_doublet
psi = psi_freestream + psi_doublet

plot_contourstreamline(z, xi, psi,2)

#get u,v,psi on xi-plane
dxi_dz = 1- (c/z)**2
w_xi = (u-1j*v) / dxi_dz
cp_xi = 1-w_xi*w_xi.conjugate()/u_inf**2

print(w_xi[0][61].real,w_xi[0][61].imag)

pyplot.figure(3,figsize=(14, 7))
pyplot.axis('equal')
contf2 = pyplot.contourf(xi.real, xi.imag, cp_xi.real, levels=numpy.linspace(-1.0, 1.0, 100), extend='both')
cbar = pyplot.colorbar(contf2)
cbar.set_ticks(numpy.linspace(-1.0, 1.0, 9))
pyplot.title(r'$\xi$-plane $C_p$');

#get psi on z plane
aoa = 20./180.*numpy.pi
z1 = (z - (x_c+1j*y_c))*numpy.exp(-1j*aoa)

u_doublet, v_doublet = get_velocity_doublet(kappa, 0.0, 0.0, z1.real, z1.imag)
#u_freestream = u_inf*numpy.zeros_like(z1.real)
#v_freestream = numpy.zeros_like(z1.imag)
u = u_freestream + u_doublet
v = v_freestream + v_doublet

psi_doublet = get_stream_function_doublet(kappa, 0.0, 0.0, z1.real, z1.imag)
psi_freestream = u_inf * z1.imag
psi = psi_freestream + psi_doublet

plot_contourstreamline(z, xi, psi,2)

#AoA = 20
w_z = (u-1j*v)*numpy.exp(-1j*aoa)
w_xi1 = w_z / dxi_dz
cp_z = 1-w_z*w_z.conjugate()/u_inf**2
cp_xi = 1-w_xi1*w_xi1.conjugate()/u_inf**2
print(cp_xi[0][:].real)   #looking for cp=1, which represents stagnation point
print(cp_xi[0][74].real)
print(w_xi1[0][49].real,w_xi1[0][49].imag)

pyplot.figure(5,figsize=(14, 7))
pyplot.axis('equal')
contf2 = pyplot.contourf(xi.real, xi.imag, cp_xi.real, levels=numpy.linspace(-1.0, 1.0, 100), extend='both')
cbar = pyplot.colorbar(contf2)
cbar.set_ticks(numpy.linspace(-1.0, 1.0, 9))
pyplot.title(r'$\xi$-plane $C_p$');

#AoA=20, and adding vortex
gamma = - 4 * numpy.pi * r1 * u_inf * numpy.sin(-aoa)
print(gamma)
rho = 1.0
lift_kutta = rho * u_inf * gamma * numpy.cos(aoa) #lift is in y-dir
print(lift_kutta)

psi_vortex = get_stream_function_vortex(gamma,0.0,0.0,z1.real,z1.imag)
psi = psi + psi_vortex
plot_contourstreamline(z, xi, psi,2)

p = 0.5*rho*cp_z[0,:]* u_inf**2
dA = 2*numpy.pi*r1/(Ntheta-1)
drag = dA * numpy.dot(p, numpy.cos(theta))
print('drag by integration ',drag)

u_vortex,v_vortex = get_velocity_vortex(gamma,0.0,0.0,z1.real,z1.imag)
u = u+u_vortex
v = v+v_vortex
w_z = (u-1j*v)*numpy.exp(-1j*aoa)
w_xi1 = w_z / dxi_dz
cp_z = 1-w_z*w_z.conjugate()/u_inf**2
cp_xi = 1-w_xi1*w_xi1.conjugate()/u_inf**2
#looking for cp_z equal to 1
#for i in range(0,Ntheta):
#    print(cp_z[0][i].real,i)
print('velocity 92th pt from trailing edge: ', w_xi1[0][91].real,w_xi1[0][91].imag)

pyplot.figure(7,figsize=(14, 7))
pyplot.axis('equal')
contf2 = pyplot.contourf(xi.real, xi.imag, cp_xi.real, levels=numpy.linspace(-1.0, 1.0, 100), extend='both')
cbar = pyplot.colorbar(contf2)
cbar.set_ticks(numpy.linspace(-1.0, 1.0, 9))
pyplot.title(r'adding vortex, $\xi$-plane $C_p$');


pyplot.show()
