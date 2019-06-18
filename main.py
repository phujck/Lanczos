import numpy as np
import hub_lats as hub
# from pyscf import fci
import evolve
import hams
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.interpolate import interp1d


def iFT(A):
    """
    Inverse Fourier transform
    :param A:  1D numpy.array
    :return:
    """
    A = np.array(A)
    minus_one = (-1) ** np.arange(A.size)
    result = np.fft.ifft(minus_one * A)
    result *= minus_one
    result *= np.exp(1j * np.pi * A.size / 2)
    return result


def FT(A):
    """
    Fourier transform
    :param A:  1D numpy.array
    :return:
    """
    # test
    A = np.array(A)
    minus_one = (-1) ** np.arange(A.size)
    result = np.fft.fft(minus_one * A)
    result *= minus_one
    result *= np.exp(-1j * np.pi * A.size / 2)
    return result


def progress(total, current):
    if total < 10:
        print("Simulation Progress: " + str(round(100 * current / total)) + "%")
    elif current % int(total / 100) == 0:
        print("Simulation Progress: " + str(round(100 * current / total)) + "%")
    return

number=1
nelec = (number, number)
nx = 4
ny = 0
t = 0.191
# t=1.91
# t=1
U = 0.1*t
delta = 1
cycles = 1
parameternames='-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta' % (nx,cycles,U,t,number,delta)

prop = hams.system(nelec, nx, ny, U, t, delta, cycles, bc='pbc')
# prop = hams.system(nelec, nx, ny, U, t, delta, cycles)

# expectations here
neighbour = []
phi_original = []
J_field = []
phi_reconstruct = [0., 0.]
boundary_1 = []
boundary_2 = []
two_body = []
error=[]
D=[]
branch = 0

# Set Ground State
psi = prop.get_gs()[1].astype(np.complex128)

# Use this to check the error on an individual step
# _,x=evolve.Lanczos(prop,psi,0,delta,3)
# print(x)
# #

# LANCZOS PROP
#
# for j in range(prop.n_time):
#     # print(prop.nsites)
#     progress(prop.n_time, j)
#     neighbour.append(evolve.nearest_neighbour(prop, psi))
#     phi_original.append(prop.phi(j*delta))
#     two_body.append(evolve.two_body2(prop, psi))
#     J_field.append(evolve.current(prop, phi_original[-1], neighbour[-1]))
#     # phi, branch = evolve.phi_reconstruct(prop, J_field[-1], neighbour[-1], phi_reconstruct[-1], phi_reconstruct[-2], branch)
#     # phi_reconstruct.append(phi)
#     # boundary_1.append(evolve.boundary_term_1(prop, psi))
#     # boundary_2.append(evolve.boundary_term_2(prop, psi))
# # del phi_reconstruct[0:2]
#     psi,newerror=evolve.Lanczos(prop,psi,j*delta,delta,3)
#     error.append(newerror)

#RK4- DOESN'T EVEN SLIGHTLY WORK-immediate overflows ensue.
#
# for j in range(prop.n_time):
#     oldpsi = psi
#     psierror=evolve.apply_H(prop,prop.full_1e_ham(t),psi)
#     progress(prop.n_time, j)
#     neighbour.append(evolve.nearest_neighbour(prop, psi))
#     phi_original.append(prop.phi(j*delta))
#     two_body.append(evolve.two_body(prop, psi))
#     J_field.append(evolve.current(prop, phi_original[-1], neighbour[-1]))
#     # phi, branch = evolve.phi_reconstruct(prop, J_field[-1], neighbour[-1], phi_reconstruct[-1], phi_reconstruct[-2], branch)
#     # phi_reconstruct.append(phi)
#     # boundary_1.append(evolve.boundary_term_1(prop, psi))
#     # boundary_2.append(evolve.boundary_term_2(prop, psi))
# # del phi_reconstruct[0:2]
#     psi=evolve.RK4(prop,j*delta,psi)
#     diff=(psi-oldpsi)/delta
#     newerror = np.linalg.norm(diff + 1j * psierror)
#     error.append(newerror)


#ZVODE

r = ode(evolve.integrate_f).set_integrator('zvode', method='bdf')
r.set_initial_value(psi, 0).set_f_params(prop)
branch = 0
while r.successful() and r.t < prop.simT:
    oldpsi=psi
    psierror=evolve.apply_H(prop,prop.full_1e_ham(t),psi)
    r.integrate(r.t + delta)
    psi = r.y
    time = r.t
    # add to expectations

    # double occupancy fails for anything other than half filling.
    # D.append(evolve.DHP(prop,psi))
    progress(prop.n_time, int(time / delta))
    neighbour.append(evolve.nearest_neighbour(prop, psi))
    phi_original.append(prop.phi(time))
    two_body.append(evolve.two_body2(prop, psi))
    J_field.append(evolve.current(prop, phi_original[-1], neighbour[-1]))
    diff=(psi-oldpsi)/delta
    newerror=np.linalg.norm(diff+1j*psierror)
    error.append(newerror)
    # phi, branch = evolve.phi_reconstruct(prop, J_field[-1], neighbour[-1], phi_reconstruct[-1], phi_reconstruct[-2], branch)
    # phi_reconstruct.append(phi)
    # boundary_1.append(evolve.boundary_term_1(prop, psi))
    # boundary_2.append(evolve.boundary_term_2(prop, psi))
del phi_reconstruct[0:2]
#

# #alternative method - can be used for real or scaled time. to use scaled time: uncomment the alternative n_time
# #and phi in hams.py, and the alternative RK4 in evolve.py, and decrease delta
# # time = 0.
# # for i in range(prop.n_time):
# #    psi = evolve.RK4(prop, time, psi)
# #    time += delta
# #     # add to expectations
# #    # D.append(evolve.DHP(prop,psi))
# #    progress(prop.n_time, i)
# #    neighbour.append(evolve.nearest_neighbour(prop, psi))
# #    phi_original.append(prop.phi(time))
# #    two_body.append(evolve.two_body(prop, psi))
# #    J_field.append(evolve.current(prop, phi_original[-1], neighbour[-1]))
# # #    phi, branch = evolve.phi_reconstruct(prop, J_field[-1], neighbour[-1], phi_reconstruct[-1], phi_reconstruct[-2], branch)
# # #    phi_reconstruct.append(phi)
# # #    boundary_1.append(evolve.boundary_term_1(prop, psi))
# # #    boundary_2.append(evolve.boundary_term_2(prop, psi))
# # #del phi_reconstruct[0:2]

# Save for plotting

neighbour = np.array(neighbour)
J_field = np.array(J_field)
phi_original = np.array(phi_original)
phi_reconstruct = np.array(phi_reconstruct)
two_body=np.array(two_body)
error=np.array(error)

np.save('./data/original/Jfield'+parameternames,J_field)
np.save('./data/original/phi'+parameternames,phi_original)
np.save('./data/original/phirecon'+parameternames,phi_reconstruct)
# np.save('./data/original/boundary1'+parameternames,boundary_1)
# np.save('./data/original/boundary2'+parameternames,boundary_2)
np.save('./data/original/neighbour'+parameternames,neighbour)
np.save('./data/original/twobody'+parameternames,two_body)
np.save('./data/original/error'+parameternames,error)





