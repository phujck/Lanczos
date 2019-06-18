import numpy
from pyscf import lib
from pyscf.fci import cistring


def des_a(ci0, norb, neleca_nelecb, ap_id):
    r'''Construct (N-1)-electron wavefunction by removing an alpha electron from
    the N-electron wavefunction.

    ... math::

        |N-1\rangle = \hat{a}_p |N\rangle

    Args:
        ci0 : 2D array
            CI coefficients, row for alpha strings and column for beta strings.
        norb : int
            Number of orbitals.
        (neleca,nelecb) : (int,int)
            Number of (alpha, beta) electrons of the input CI function
        ap_id : int
            Orbital index (0-based), for the annihilation operator

    Returns:
        2D array, row for alpha strings and column for beta strings.  Note it
        has different number of rows to the input CI coefficients
    '''
    neleca, nelecb = neleca_nelecb
    if neleca <= 0:
        return numpy.zeros_like(ci0)
    if ci0.ndim == 1:
        ci0 = ci0.reshape(cistring.num_strings(norb, neleca),
                          cistring.num_strings(norb, nelecb))
    des_index = cistring.gen_des_str_index(range(norb), neleca)
    na_ci1 = cistring.num_strings(norb, neleca-1)
    ci1 = numpy.zeros((na_ci1, ci0.shape[1]),dtype=ci0[0,0].__class__)

    entry_has_ap = (des_index[:,:,1] == ap_id)
    addr_ci0 = numpy.any(entry_has_ap, axis=1)
    addr_ci1 = des_index[entry_has_ap,2]
    sign = des_index[entry_has_ap,3].astype(complex)
    #print(addr_ci0)
    #print(addr_ci1)
    ci1[addr_ci1] = sign.reshape(-1,1) * ci0[addr_ci0]
    return ci1

def des_b(ci0, norb, neleca_nelecb, ap_id):
    r'''Construct (N-1)-electron wavefunction by removing a beta electron from
    N-electron wavefunction.

    Args:
        ci0 : 2D array
            CI coefficients, row for alpha strings and column for beta strings.
        norb : int
            Number of orbitals.
        (neleca,nelecb) : (int,int)
            Number of (alpha, beta) electrons of the input CI function
        ap_id : int
            Orbital index (0-based), for the annihilation operator

    Returns:
        2D array, row for alpha strings and column for beta strings. Note it
        has different number of columns to the input CI coefficients.
    '''
    neleca, nelecb = neleca_nelecb
    if nelecb <= 0:
        return numpy.zeros_like(ci0)
    if ci0.ndim == 1:
        ci0 = ci0.reshape(cistring.num_strings(norb, neleca),
                          cistring.num_strings(norb, nelecb))
    des_index = cistring.gen_des_str_index(range(norb), nelecb)
    nb_ci1 = cistring.num_strings(norb, nelecb-1)
    ci1 = numpy.zeros((ci0.shape[0], nb_ci1),dtype=ci0[0,0].__class__)

    entry_has_ap = (des_index[:,:,1] == ap_id)
    addr_ci0 = numpy.any(entry_has_ap, axis=1)
    addr_ci1 = des_index[entry_has_ap,2]
    sign = des_index[entry_has_ap,3].astype(complex)
    # This sign prefactor accounts for interchange of operators with alpha and beta spins
    if neleca % 2 == 1:
        sign *= -1
    ci1[:,addr_ci1] = ci0[:,addr_ci0] * sign
    return ci1

def cre_a(ci0, norb, neleca_nelecb, ap_id):
    r'''Construct (N+1)-electron wavefunction by adding an alpha electron in
    the N-electron wavefunction.

    ... math::

        |N+1\rangle = \hat{a}^+_p |N\rangle

    Args:
        ci0 : 2D array
            CI coefficients, row for alpha strings and column for beta strings.
        norb : int
            Number of orbitals.
        (neleca,nelecb) : (int,int)
            Number of (alpha, beta) electrons of the input CI function
        ap_id : int
            Orbital index (0-based), for the creation operator

    Returns:
        2D array, row for alpha strings and column for beta strings. Note it
        has different number of rows to the input CI coefficients.
    '''
    neleca, nelecb = neleca_nelecb
    if neleca >= norb:
        return numpy.zeros_like(ci0)
    if ci0.ndim == 1:
        ci0 = ci0.reshape(cistring.num_strings(norb, neleca),
                          cistring.num_strings(norb, nelecb))
    cre_index = cistring.gen_cre_str_index(range(norb), neleca)
    na_ci1 = cistring.num_strings(norb, neleca+1)
    ci1 = numpy.zeros((na_ci1, ci0.shape[1]),dtype=ci0[0,0].__class__)

    entry_has_ap = (cre_index[:,:,0] == ap_id)
    addr_ci0 = numpy.any(entry_has_ap, axis=1)
    addr_ci1 = cre_index[entry_has_ap,2]
    sign = cre_index[entry_has_ap,3].astype(complex)
    ci1[addr_ci1] = sign.reshape(-1,1) * ci0[addr_ci0]
    return ci1

# construct (N+1)-electron wavefunction by adding a beta electron to
# N-electron wavefunction:
def cre_b(ci0, norb, neleca_nelecb, ap_id):
    r'''Construct (N+1)-electron wavefunction by adding a beta electron in
    the N-electron wavefunction.

    Args:
        ci0 : 2D array
            CI coefficients, row for alpha strings and column for beta strings.
        norb : int
            Number of orbitals.
        (neleca,nelecb) : (int,int)
            Number of (alpha, beta) electrons of the input CI function
        ap_id : int
            Orbital index (0-based), for the creation operator

    Returns:
        2D array, row for alpha strings and column for beta strings. Note it
        has different number of columns to the input CI coefficients.
    '''
    neleca, nelecb = neleca_nelecb
    if nelecb >= norb:
        return numpy.zeros_like(ci0)
    if ci0.ndim == 1:
        ci0 = ci0.reshape(cistring.num_strings(norb, neleca),
                          cistring.num_strings(norb, nelecb))
    cre_index = cistring.gen_cre_str_index(range(norb), nelecb)
    nb_ci1 = cistring.num_strings(norb, nelecb+1)
    ci1 = numpy.zeros((ci0.shape[0], nb_ci1),dtype=ci0[0,0].__class__)

    entry_has_ap = (cre_index[:,:,0] == ap_id)
    addr_ci0 = numpy.any(entry_has_ap, axis=1)
    addr_ci1 = cre_index[entry_has_ap,2]
    sign = cre_index[entry_has_ap,3].astype(complex)
    # This sign prefactor accounts for interchange of operators with alpha and beta spins
    if neleca % 2 == 1:
        sign *= -1
    ci1[:,addr_ci1] = ci0[:,addr_ci0] * sign
    return ci1

def compute_inner_product(civec, norbs, nelecs, ops, cres, alphas):
    neleca, nelecb = nelecs
    ciket = civec.copy()
    assert(len(ops)==len(cres))
    assert(len(ops)==len(alphas))
    for i in reversed(range(len(ops))):
        if alphas[i]:
            if cres[i]:
                ciket = cre_a(ciket, norbs, (neleca, nelecb), ops[i])
                neleca += 1
            else:
                if neleca==0:
                    return 0
                ciket = des_a(ciket, norbs, (neleca, nelecb), ops[i])
                neleca -= 1
        else:
            if cres[i]:
                ciket = cre_b(ciket, norbs, (neleca, nelecb), ops[i])
                nelecb += 1
            else:
                if nelecb==0:
                    return 0
                ciket = des_b(ciket, norbs, (neleca, nelecb), ops[i])
                nelecb -= 1
    return numpy.dot(civec.conj().flatten(), ciket.flatten())

def compute_inner_product_debug(civec, norbs, nelecs, ops, cres, alphas):
    """This produces error messages saying that the imaginary parts are being 
    discarded. This is because of the parts of the above functions that make 
    them usable with complex wavefunctions (e.g. lines 152 and 157 in cre_b)"""
    neleca, nelecb = nelecs
    ciket = civec.copy()
    assert(len(ops)==len(cres))
    assert(len(ops)==len(alphas))
    if civec.dtype == 'complex128':
        civec_r = civec.real.copy()
        civec_i = civec.imag.copy()
        #print civec.dtype,civec_r.dtype,civec_i.dtype
        for i in reversed(range(len(ops))):
            if alphas[i]:
                if cres[i]:
                    civec_r = cre_a(civec_r, norbs, (neleca, nelecb), ops[i])
                    neleca += 1
                else:
                    if neleca==0:
                        return 0
                    civec_r = des_a(civec_r, norbs, (neleca, nelecb), ops[i])
                    neleca -= 1
            else:
                if cres[i]:
                    civec_r = cre_b(civec_r, norbs, (neleca, nelecb), ops[i])
                    nelecb += 1
                else:
                    if nelecb==0:
                        return 0
                    civec_r = des_b(civec_r, norbs, (neleca, nelecb), ops[i])
                    nelecb -= 1
        for i in reversed(range(len(ops))):
            if alphas[i]:
                if cres[i]:
                    civec_i = cre_a(civec_i, norbs, (neleca, nelecb), ops[i])
                    neleca += 1
                else:
                    if neleca==0:
                        return 0
                    civec_i = des_a(civec_i, norbs, (neleca, nelecb), ops[i])
                    neleca -= 1
            else:
                if cres[i]:
                    civec_i = cre_b(civec_i, norbs, (neleca, nelecb), ops[i])
                    nelecb += 1
                else:
                    if nelecb==0:
                        return 0
                    civec_i = des_b(civec_i, norbs, (neleca, nelecb), ops[i])
                    nelecb -= 1
        ciket = civec_r + 1.j*civec_i
    else:
        for i in reversed(range(len(ops))):
            if alphas[i]:
                if cres[i]:
                    ciket = cre_a(ciket, norbs, (neleca, nelecb), ops[i])
                    neleca += 1
                else:
                    if neleca==0:
                        return 0
                    ciket = des_a(ciket, norbs, (neleca, nelecb), ops[i])
                    neleca -= 1
            else:
                if cres[i]:
                    ciket = cre_b(ciket, norbs, (neleca, nelecb), ops[i])
                    nelecb += 1
                else:
                    if nelecb==0:
                        return 0
                    ciket = des_b(ciket, norbs, (neleca, nelecb), ops[i])
                    nelecb -= 1
    return numpy.dot(civec.conj().flatten(), ciket.flatten())

def apply_one_e_ham_slow(h1, psi, n, nelec):
    n_up, n_down = nelec
    psi_new = psi.copy()
    psi_final = numpy.zeros_like(psi)
    psi_tmp = numpy.zeros_like(psi)
    for i in range(n):
        for j in range(n):
            psi_tmp = des_a(psi_new, n, (n_up, n_down), j)
            psi_tmp = cre_a(psi_tmp, n, (n_up-1, n_down), i)
            psi_tmp *= h1[i,j]   # Is this the right way around for the definition?!
            psi_final += psi_tmp.reshape(psi_final.shape)

            psi_tmp = des_b(psi_new, n, (n_up, n_down), j)
            psi_tmp = cre_b(psi_tmp, n, (n_up, n_down-1), i)
            psi_tmp *= h1[i,j]   # Is this the right way around for the definition?! Do we want to take the complex conjugate for correct definition?
            psi_final += psi_tmp.reshape(psi_final.shape)
    return psi_final
