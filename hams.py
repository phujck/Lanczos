import numpy as np
import hub_lats as hub
from pyscf import fci


class system:
    def __init__(self, nelec, nx, ny, U, t, delta, cycles, lat_type='square', bc=None):
        self.lat = hub.Lattice(nx, ny, lat_type, bc)
        self.nsites = self.lat.nsites
        self.nup = nelec[0]
        self.ndown = nelec[1]
        self.ne = self.nup + self.ndown

        # # converts to a'.u, which are atomic units but with energy normalised to t, so
        # # that 1 hartree=1t. also, hbar=e=m_e=1/4pi*ep_0=1, and c=1/alpha=137
        # self.factor = 1. / (t * 0.036749323)
        # self.U = U / t
        # self.t = 1.
        # # Change this to change perturbation
        # # Set perturbation to HHG perturbation
        # # Set constants required (in future, this should be a kwargs to __init__
        # # input units: THz (field), eV (t, U), MV/cm (peak amplitude), Angstroms (lattice cst)
        # field = 32.9  # Field frequency
        # # field=25
        # # F0 = 10.
        # F0=10
        # # F0 = 12.
        # a = 7.62
        # #field = angular frequency and freq = field/2pi
        # self.field = field * self.factor * 0.0001519828442
        # self.a = (a * 1.889726125) / self.factor
        # self.F0 = F0 * 1.944689151e-4 * (self.factor ** 2.)

        # # trying things in atomic units directly
        field = 0.0049985
        # field=0.01
        a = 7.1739
        F0 = 0.0019
        # F0 = 0.0019/5
        # F0=0.019/a
        # self.factor = 1. / (t * 0.036749323)
        # self.factor=1/t

        # trying without scaling t
        self.factor = 1
        # self.factor=1/(5*field/(2*np.pi))
        # self.factor=1/t
        self.U = U*self.factor
        self.t = t*self.factor
        self.a = a / self.factor
        self.F0 = F0 * (self.factor ** 2.)
        self.field = field * self.factor
        assert self.nup <= self.nsites, 'Too many ups!'
        assert self.ndown <= self.nsites, 'Too many downs!'
        self.h2 = self.two_elec_ham()
        self.h1 = hub.create_1e_ham(self.lat, True)
        self.delta = delta  # timestep

        # cycles and frequency
        self.cycles = cycles
        self.freq = self.field / (2. * np.pi)



        # self.n_time = int(self.cycles / self.delta)  # Number of time points in propagation for scaled time
        self.simT = self.cycles / self.freq
        self.n_time = int(self.simT / self.delta)  # Number of time points in propagation for real time
        print(self.n_time)

        # self.full_1e_ham should return a function which gives the 1e hamiltonian + perturbation with the current time as an argument

    def two_elec_ham(self):
        h2 = np.zeros((self.lat.nsites, self.lat.nsites, self.lat.nsites, self.lat.nsites))
        for i in range(self.lat.nsites):
            h2[i, i, i, i] = self.U
        return h2

    def phi(self, current_time):
        # original phi, scaled time
        # phi = self.a * self.F0 / self.field * (np.sin(np.pi * current_time / self.cycles) ** 2.) * np.sin(2. * np.pi * current_time)

        # #original phi, real time
        # phi = self.a * self.F0 / self.field * (np.sin(self.field * current_time / (2. * self.cycles)) ** 2.) * np.sin(self.field * current_time)

        # testing a constant phi envelope
        # phi = self.a * self.F0 / self.field * (np.sin(self.field * current_time / (2. * 4)) ** 2.) * np.sin(
        #     self.field * current_time)

        # Phi as in thesis. Set periods different to cycles to see a partial pulse.
        # periods = self.cycles
        periods=4
        midpoint = periods / (2*self.freq)
        timediff = current_time - midpoint
        phi = self.a * self.F0 / self.field * ((np.cos(self.field * timediff / (2 * periods))) ** 2) * np.sin(
            self.field * timediff - np.pi / 2) * (
                          np.heaviside(timediff + periods / (2 * self.freq),0) - np.heaviside(
                      timediff - periods / (2 * self.freq),0))

        # alternate envelope, real time
        # phi = self.a * self.F0 / self.field * (np.cos(self.field * current_time / (2. * self.cycles)) ** 2.) * np.sin(self.field * current_time)

        return phi

    def apply_hhg_pert(self, current_time):
        if self.field == 0.:
            phi = 0.
        else:
            phi = self.phi(current_time)
        #note I've swapped tril and triu here, on the basis that h1_ij = c^dag_i H_1 c_j and we have e^(-i*phi) c^dag_j+1 c_j+h.c.
        # ghb bugfix: Deal with boundary conditions correctly.
        h_forwards = np.triu(self.h1)
        h_forwards[0,-1] = 0.0
        h_forwards[-1,0] = self.h1[-1,0]
        h_backwards = np.tril(self.h1)
        h_backwards[-1,0] = 0.0
        h_backwards[0,-1] = self.h1[0,-1]
        #return self.t *np.exp(1j * phi) * np.tril(self.h1) + self.t* np.exp(-1j * phi) * np.triu(self.h1)
        return self.t *np.exp(1j * phi) * h_backwards + self.t* np.exp(-1j * phi) * h_forwards

    full_1e_ham = apply_hhg_pert

    def apply_hhg_pert_simpson(self, current_time, delta):
        if self.field == 0.:
            phi = 0.
        else:
            phi_1 = self.phi(current_time)
            phi_2 = self.phi(current_time + delta / 2)
            phi_3 = self.phi(current_time + delta)
            prefactor = self.t*(np.exp(1j * phi_1) + 4 * np.exp(1j * phi_2) + np.exp(1j * phi_3)) / 6

        # ghb bugfix: Deal with boundary conditions correctly.
        h_forwards = np.triu(self.h1)
        h_forwards[0,-1] = 0.0
        h_forwards[-1,0] = self.h1[-1,0]
        h_backwards = np.tril(self.h1)
        h_backwards[-1,0] = 0.0
        h_backwards[0,-1] = self.h1[0,-1]

        #return prefactor * np.tril(self.h1) + np.conjugate(prefactor) * np.triu(self.h1)
        return prefactor * h_backwards + np.conjugate(prefactor) * h_forwards

    full_1e_ham_simpson = apply_hhg_pert_simpson

    def get_gs(self):
        cisolver = fci.direct_spin1.FCI()
        e, fcivec = cisolver.kernel(self.h1, self.h2, self.lat.nsites, (self.nup, self.ndown))
        return (e, fcivec.flatten())
