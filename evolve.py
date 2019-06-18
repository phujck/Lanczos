import numpy as np
#import hub_lats as hub
from pyscf import fci
import des_cre as dc
from scipy.linalg import eigh_tridiagonal
from scipy.linalg import expm


def Lanczos(sys,psi,t,delta,k):
    """Lanczos propagator. Takes the current state, and uses it to construct a basis for the Krylov subspace Hamiltonian.
     This is then diagonalised to propagate psi, before transforming psi back to the original basis."""
    # initial setup
    psis=np.zeros((k+1,psi.size),dtype=np.complex128)
    psis[0,:]=psi
    alphas=np.zeros(k)
    betas=np.zeros(k)
    # print(psi)


    # Generating the Krylov subspace
    g=apply_H(sys,sys.full_1e_ham_simpson(t,delta),psi)
    testpsi=apply_H(sys,sys.full_1e_ham(t),psi)
    a=np.vdot(psi,g)
    newpsi=g-a*psi
    b=np.linalg.norm(newpsi)
    alphas[0]=a.real
    betas[0]=b.real
    psis[1,:]=newpsi/b

    for j in range (2,k+1):
        oldpsi=psis[j-1,:]
        olderpsi=psis[j-2,:]
        g = apply_H(sys, sys.full_1e_ham_simpson(t,delta), oldpsi)
        # testpsi = apply_H(sys, sys.full_1e_ham(t), oldpsi)
        a = np.vdot(oldpsi, g)
        newpsi = g - a * oldpsi- olderpsi*betas[j-2]
        b=np.linalg.norm(newpsi)
        alphas[j-1] = a.real
        betas[j-1] = b.real
        psis[j,:]=newpsi/b

    # test orthogonality of basis
    #     print(np.dot(psis[j-2,:],psis[j-1,:]))

    # Print Krylov subspace hamiltonian
    # H=np.zeros((k,k))
    # for i in range(0,k-1):
    #     H[i,i]=alphas[i]
    #     H[i,i+1]=betas[i]
    #     H[i+1,i]=betas[i]
    # H[k-1,k-1]=alphas[-1]
    # print(H)


    # Diagonalise the Hamiltonian in Krylov subspace

    w, v = eigh_tridiagonal(alphas, betas[:-1])

    # Htest= np.dot(v,np.transpose(v))
    # v=np.transpose(v)
    # Htest=np.dot(np.dot(v,np.diag(w)),np.transpose(v))
    # Htest=np.dot(v,np.dot(np.diag(w),np.transpose(v)))
    # Htest=v@np.diag(w)@np.transpose(v)
    # print(Htest)
    # print(H)

    # set propagator, with psi_k(t+delta)=v*(e^-iD_k)*v^T * psi_k(t)
    # this should be okay as it's a diagonal matrix
    exponent=np.diag(np.exp(-delta*1j*w))
    # v=np.transpose(v)
    prop=v@exponent@np.transpose(v)

    # First column will be the updated psi_k(t+delta)
    prop=prop[:,0]
    # print(prop)

    # getting psi in the original basis.
    newpsi=np.zeros(psis[0,:].size,dtype=np.complex128)
    for j in range(k):
        newpsi+=prop[j]*psis[j,:]

    #Making sure the thing is normalised, but it should be
    # newpsi=newpsi/np.linalg.norm(newpsi)

    # error estimation
    diff=(newpsi-psi)/delta
    size=np.linalg.norm(diff+1j*testpsi)
    # size=np.linalg.norm(psi)

    return newpsi,size


def apply_H(sys, h1, psi):
    """Apply hamiltonian to arbitrary wavefunction, using given h1 + sys.h2. This is the hubbard hamiltonian + perturbation.
        Real and imaginary parts of wavefunction are dealt with separately"""
    psi_r = psi.real
    psi_i = psi.imag
    h1_r = h1.real
    h1_i = h1.imag
    # H|psi>=(h1+h2)|psi>=(h1_r+ih1_i+h2)|psi>=(h1_r+ih1_i+h2)|psi_r>+i(h1_r+ih1_i+h2)|psi_i>
    pro = one_elec(sys, h1_r, psi_r) + 1j * one_elec(sys, h1_i, psi_r, False) \
              + 1j * one_elec(sys, h1_r, psi_i) - one_elec(sys, h1_i, psi_i, False) + two_elec(sys, psi_r, psi_i)

    # trying neglecting the imaginary part of H
    # pro = one_elec(sys, h1_r, psi_r)+ 1j * one_elec(sys, h1_r, psi_i) + two_elec(sys, psi_r, psi_i)
    return pro

def apply_H_cheat(sys, h1, psi):
    """This includes the if-statement which removes the imaginary part of the 1e-ham.
    I've found that this needs to be included for low U/t (approx <2) for unknown reasons, and
    in particular at U/t=0"""
    psi_r = psi.real
    psi_i = psi.imag 
    h1_r = h1.real
    h1_i = h1.imag
    #H|psi>=(h1+h2)|psi>=(h1_r+ih1_i+h2)|psi>=(h1_r+ih1_i+h2)|psi_r>+i(h1_r+ih1_i+h2)|psi_i>
    if sys.U > 0.0:
        pro = one_elec(sys, h1_r, psi_r) + 1.j*one_elec(sys, h1_i, psi_r, False) \
        + 1.j*one_elec(sys, h1_r, psi_i) - one_elec(sys, h1_i, psi_i, False) + two_elec(sys, psi_r, psi_i)
    else:
        pro = one_elec(sys, h1_r, psi_r) + 1.j*one_elec(sys, h1_r, psi_i) + two_elec(sys, psi_r, psi_i)
    return pro

def one_elec(sys, h1, psi, sym=True):
    """Apply one-electron hamiltonian, h1
         sym tells us whether this is a symmetric (real) hamiltonian or not"""
    if sym:
        return sys.t*fci.direct_spin1.contract_1e(h1, psi, sys.nsites, (sys.nup, sys.ndown))
    else:
        return sys.t*fci.direct_nosym.contract_1e(h1, psi, sys.nsites, (sys.nup, sys.ndown))

def one_elec_debug(sys, h1, psi, sym=True):
    """This can be used for debugging"""
    test = True
    if sym:
        if test:
            pyscf_contract = fci.direct_spin1.contract_1e(h1, psi, sys.nsites, (sys.nup, sys.ndown))
            slow_contract = dc.apply_one_e_ham_slow(h1, psi, sys.nsites, (sys.nup, sys.ndown))
            assert(np.allclose(pyscf_contract,slow_contract))
            return pyscf_contract
        else:
            return fci.direct_spin1.contract_1e(h1, psi, sys.nsites, (sys.nup, sys.ndown))
    else:
        if test:
            pyscf_contract = fci.direct_nosym.contract_1e(h1, psi, sys.nsites, (sys.nup, sys.ndown))
            slow_contract = dc.apply_one_e_ham_slow(h1, psi, sys.nsites, (sys.nup, sys.ndown))
            assert(np.allclose(pyscf_contract,slow_contract))
            return pyscf_contract
        else:
            return fci.direct_nosym.contract_1e(h1, psi, sys.nsites, (sys.nup, sys.ndown))

def two_elec(sys, psi_r, psi_i):
    """ Apply 2-electron hubbard hamiltonian"""
    psi_r=np.transpose(psi_r)
    psi_i=np.conjugate(psi_i)
    pro = 0.5 * fci.direct_uhf.contract_2e_hubbard((0, sys.U, 0), psi_r, sys.nsites, (sys.nup, sys.ndown)) \
          + 0.5 * 1j * fci.direct_uhf.contract_2e_hubbard((0, sys.U, 0), psi_i, sys.nsites, (sys.nup, sys.ndown))
    # pro =  fci.direct_uhf.contract_2e_hubbard((0, sys.U, 0), psi_r, sys.nsites, (sys.nup, sys.ndown)) \
    #       + 1j * fci.direct_uhf.contract_2e_hubbard((0, sys.U, 0), psi_i, sys.nsites, (sys.nup, sys.ndown))

    return pro.flatten()


def RK4(sys, current_time, psi):
    """4th Order Runge-Kutta step for time evolution"""
    
    h1_k1 = sys.full_1e_ham(current_time) 
    k1 = (-1j * sys.delta / sys.freq) * apply_H(sys, h1_k1, psi) #scaled time
    # k1 = -1j * sys.delta * apply_H(sys, h1_k1, psi) #real time
    
    h1_k2 = sys.full_1e_ham(current_time + 0.5 * sys.delta)
    k2 = (-1j * sys.delta / sys.freq) * apply_H(sys, h1_k2, psi + 0.5 * k1) #scaled time
    # k2 = -1j * sys.delta * apply_H(sys, h1_k2, psi + 0.5 * k1) #real time
    
    k3 = (-1j * sys.delta / sys.freq) * apply_H(sys, h1_k2, psi + 0.5 * k2) #scaled time
    # k3 = -1j * sys.delta * apply_H(sys, h1_k2, psi + 0.5 * k2) #real time
    
    h1_k4 = sys.full_1e_ham(current_time + sys.delta)
    k4 = (-1j * sys.delta / sys.freq) * apply_H(sys, h1_k4, psi + k3) #scaled time
    # k4 = -1j * sys.delta * apply_H(sys, h1_k4, psi + k3) #real time
    return psi + (k1 + 2. * k2 + 2. * k3 + k4) / 6.


def RK1(sys, current_time, psi):
    """Euler step for time evolution"""

    h1 = sys.full_1e_ham(current_time)
    #k1 = (-1j * sys.delta / sys.freq) * apply_H(sys, h1, psi)
    k1 = -1j * sys.delta * apply_H(sys, h1, psi)
    return psi + k1


def spin_up(sys, psi):
    """Calculate number of spin up (for testing)"""
    psi = np.reshape(psi,
                     (fci.cistring.num_strings(sys.nsites, sys.nup), fci.cistring.num_strings(sys.nsites, sys.ndown)))
    D = 0.
    for i in range(sys.nsites):
        # D += dc.compute_inner_product(psi,sys.nsites,(sys.nup,sys.ndown),[i,i,i,i],[1,0,1,0],[1,1,0,0])
        D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i, i], [1, 0], [1, 1])
    return D


def spin_down(sys, psi):
    """Calculate number of spin down (testing)"""
    psi = np.reshape(psi,
                     (fci.cistring.num_strings(sys.nsites, sys.nup), fci.cistring.num_strings(sys.nsites, sys.ndown)))
    D = 0.
    for i in range(sys.nsites):
        # D += dc.compute_inner_product(psi,sys.nsites,(sys.nup,sys.ndown),[i,i,i,i],[1,0,1,0],[1,1,0,0])
        D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i, i], [1, 0], [0, 0])
    return D


def nearest_neighbour(sys, psi):
    """Calculate the nearest neighbour expectation (over both spins)"""
    psi = np.reshape(psi,
                     (fci.cistring.num_strings(sys.nsites, sys.nup), fci.cistring.num_strings(sys.nsites, sys.ndown)))
    D = 0.
    for j in [0, 1]:
        for i in range(sys.nsites-1):
            # add expectation for beta electrons
            D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i, i + 1], [1, 0], [j, j])
        # Assuming periodic conditions, add the coupling across the boundary.
        D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [sys.nsites - 1, 0], [1, 0], [j, j])
    return D


def boundary_term_1(sys, psi):
    """"Calculate the first boundary term"""
    D = 0.
    for j in [0, 1]:
        for i in range(sys.nsites - 1):
            D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i + 1, i + 1], [1, 0], [j, j])
            D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i, i], [1, 0], [j, j])

        # Add the periodic transition.
        D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [0, 0], [1, 0], [j, j])
        D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [sys.nsites - 1, sys.nsites - 1], [1, 0],
                                      [j, j])

    return D


def boundary_term_2(sys, psi):
    """"Calculate the second Boundary term"""
    D = 0.
    for j in [0, 1]:
        # Lower boundary term
        D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [sys.nsites - 1, 1], [1, 0], [j, j])
        D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [0, 2], [1, 0], [j, j])
        # Bulk calculation
        for i in range(1, sys.nsites - 2):
            D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i - 1, i + 1], [1, 0], [j, j])
            D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i, i + 2], [1, 0], [j, j])
        # The other terms outside of the bulk sum.
        D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [sys.nsites - 3, sys.nsites - 1], [1, 0],
                                      [j, j])
        D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [sys.nsites - 2, 0], [1, 0], [j, j])
        D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [sys.nsites - 2, 0], [1, 0], [j, j])
        D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [sys.nsites - 1, 1], [1, 0], [j, j])

    return D


def current(sys, phi, neighbour):
    conjugator = np.exp(-1j * phi) * neighbour
    c = -1j * sys.a * sys.t * (conjugator - np.conjugate(conjugator))
    return c


def phi_reconstruct(sys, current, neighbourexpect, phi_previous_1, phi_previous_2, branch_number):
    """This is an ad-hoc method for reconstructing the original field from current calculation"""
    angle = np.angle(neighbourexpect)
    mag = np.abs(neighbourexpect)
    arg = -current / (2. * sys.a * sys.t * mag)
    branch_numbers = [branch_number + k for k in [-1, 0, 1]]
    # arcsin multiple value function, this picks the branch with the minimum second-order gradient.
    _, new_branch_number = min(
        (abs((-1) ** l * np.arcsin(arg + 0j) + l * np.pi + angle - 2 * phi_previous_1 + phi_previous_2), l) for l in
        branch_numbers
    )
    # print([(abs((-1) ** l * np.arcsin(arg + 0j) + l * np.pi + angle-phi_previous), l) for l in branch_numbers])
    new_phi = (-1) ** new_branch_number * np.arcsin(arg + 0j) + new_branch_number * np.pi + angle
    #phi = np.arcsin(arg+0j)
    return new_phi, new_branch_number

def two_body(sys, psi):
    """Contribution from two-body-terms commutator with c*_k c_k+1"""
    psi_r = np.reshape(psi.real,
                     (fci.cistring.num_strings(sys.nsites, sys.nup), fci.cistring.num_strings(sys.nsites, sys.ndown)))
    psi_r = np.reshape(psi.real,
                       (fci.cistring.num_strings(sys.nsites, sys.nup), fci.cistring.num_strings(sys.nsites, sys.ndown)))
    N = sys.nsites - 1
    D = 0.
    # Lower Boundary terms
    D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [0, 1, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0])
    D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [0, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0])
    D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [N, 0, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0])
    D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [0, 0, N, 0], [1, 0, 1, 0], [1, 1, 0, 0])

    # Upper boundary terms
    D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [N, 0, N, N], [1, 0, 1, 0], [1, 1, 0, 0])
    D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [N, N, N, 0], [1, 0, 1, 0], [1, 1, 0, 0])
    D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [N - 1, N, N, N], [1, 0, 1, 0], [1, 1, 0, 0])
    D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [N, N, N - 1, N], [1, 0, 1, 0], [1, 1, 0, 0])
    # Bulk
    for i in range(1, sys.nsites - 1):
        D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i, i + 1, i, i], [1, 0, 1, 0],
                                      [1, 1, 0, 0])
        D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i, i, i, i + 1], [1, 0, 1, 0],
                                      [1, 1, 0, 0])
        D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i - 1, i, i, i], [1, 0, 1, 0],
                                      [1, 1, 0, 0])
        D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i, i, i - 1, i], [1, 0, 1, 0],
                                      [1, 1, 0, 0])
    return D

# Alternative version 
def two_body2(sys, psi):
    """Contribution from two-body-terms commutator with c*_k c_k+1"""   
    D = 0.
    for i in range(sys.nsites):
        w = (i + 1) % sys.nsites
        v = (i - 1) % sys.nsites
        
        D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i, w, i, i], [1, 0, 1, 0], [1, 1, 0, 0])
        
        D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i, i, i, w], [1, 0, 1, 0], [1, 1, 0, 0])
        
        D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [v, i, i, i], [1, 0, 1, 0], [1, 1, 0, 0])
        
        D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i, i, v, i], [1, 0, 1, 0], [1, 1, 0, 0])
        
    return D


# One more check for consistent results
def two_body3(sys, psi):
    """Contribution from two-body-terms commutator with c*_k c_k+1"""
    D = 0.
    for i in range(sys.nsites):
        w = (i + 1) % sys.nsites
        # v = (i - 1) % sys.nsites
        D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i, w, i, i], [1, 0, 1, 0],
                                      [1, 1, 0, 0])

        D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i, i, i, w], [1, 0, 1, 0],
                                      [1, 1, 0, 0])

        D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i, w, w, w], [1, 0, 1, 0],
                                      [1, 1, 0, 0])

        D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [w, w, i, w], [1, 0, 1, 0],
                                            [1, 1, 0, 0])

    return D

def integrate_f(t, psi, sys):
    return -1j * apply_H(sys, sys.full_1e_ham(t), psi)

def DHP(sys,psi):
    """Double occupancy"""
    psi=np.reshape(psi,(fci.cistring.num_strings(sys.nsites,sys.nup),fci.cistring.num_strings(sys.nsites,sys.ndown)))
    D=0.
    for i in range(sys.nsites):
        D+=dc.compute_inner_product(psi,sys.ne,(sys.nup,sys.ndown),[i,i,i,i],[1,0,1,0],[1,1,0,0])
    return D.real/sys.nsites
