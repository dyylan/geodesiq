import numpy as np
import scipy.linalg as spla
import sympy
import jax.numpy as jnp
import jax
import itertools as it

from . import lie


@jax.jit
def trace_dot_jit(x, y):
    return jnp.trace(x @ y)


def traces(b_1, b_2):
    indices = []
    len_1 = b_1.shape[0]
    len_2 = b_2.shape[0]
    for i in range(len_1):
        for j in range(len_2):
            indices.append([i, j])
    indices = np.stack(indices)
    jself = jnp.array(b_1)
    jother = jnp.array(b_2)
    carry = jnp.empty((len_1, len_2), dtype=complex)

    def scan_body(c, idx):
        idx, jdx = idx
        c = c.at[idx, jdx].set(trace_dot_jit(jself[idx], jother[jdx]))
        return c, None

    carry, _ = jax.lax.scan(scan_body, init=carry, xs=indices)
    return carry


def check_xy_comb(comb):
    if len(np.nonzero(comb)[0]) == 1:
        return True
    elif len(np.nonzero(comb)[0]) > 2:
        return False
    else:
        for i, a in enumerate(comb):
            for j, b in enumerate(comb):
                if (i != j) and (a != b) and (a > 0) and (b > 0):
                    return False
                elif (a == 3) and (b == 3):
                    return False
    return True


def check_Heisenberg_comb(comb):
    if len(np.nonzero(comb)[0]) == 1:
        return True
    elif len(np.nonzero(comb)[0]) > 2:
        return False
    else:
        for i, a in enumerate(comb):
            for j, b in enumerate(comb):
                if (i != j) and (a != b) and (a > 0) and (b > 0):
                    return False
    return True


def construct_restricted_pauli_basis(n: int, restriction):
    I = np.eye(2).astype(complex)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)
    b = []
    l = []
    for comb in list(it.product([0, 1, 2, 3], repeat=n))[1:]:
        p = 1.
        s = ''
        if restriction(comb):
            for c in comb:
                if c == 0:
                    p = np.kron(p, I)
                    s += 'I' 
                elif c == 1:
                    p = np.kron(p, X)
                    s += 'X' 
                elif c == 2:
                    p = np.kron(p, Y)
                    s += 'Y' 
                elif c == 3:
                    p = np.kron(p, Z)
                    s += 'Z' 
            b.append(p)
            l.append(s)

    return lie.Basis(np.stack(b), labels=l)


def construct_Heisenberg_pauli_basis(n: int):
    I = np.eye(2).astype(complex)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)
    b = []
    l = []
    for comb in list(it.product([0, 1, 2, 3], repeat=n))[1:]:
        p = 1.
        s = ''
        if check_Heisenberg_comb(comb):
            for c in comb:
                if c == 0:
                    p = np.kron(p, I)
                    s += 'I' 
                elif c == 1:
                    p = np.kron(p, X)
                    s += 'X' 
                elif c == 2:
                    p = np.kron(p, Y)
                    s += 'Y' 
                elif c == 3:
                    p = np.kron(p, Z)
                    s += 'Z' 
            b.append(p)
            l.append(s)

    return lie.Basis(np.stack(b), labels=l)


def construct_two_body_pauli_basis(n: int):
    I = np.eye(2).astype(complex)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)
    b = []
    for comb in list(it.product([0, 1, 2, 3], repeat=n))[1:]:
        p = 1.
        if len(np.nonzero(comb)[0]) <= 2:
            for c in comb:
                if c == 0:
                    p = np.kron(p, I)
                elif c == 1:
                    p = np.kron(p, X)
                elif c == 2:
                    p = np.kron(p, Y)
                elif c == 3:
                    p = np.kron(p, Z)
            b.append(p)

    return lie.Basis(np.stack(b))


def construct_full_pauli_basis(n: int):
    I = np.eye(2).astype(complex)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)
    labels = []
    b = []
    for comb in list(it.product([0, 1, 2, 3], repeat=n))[1:]:
        p = 1.
        labels.append(comb)
        for c in comb:
            if c == 0:
                p = np.kron(p, I)
            elif c == 1:
                p = np.kron(p, X)
            elif c == 2:
                p = np.kron(p, Y)
            elif c == 3:
                p = np.kron(p, Z)
        b.append(p)

    return lie.Basis(np.stack(b))


def prepare_random_parameters(proj_indices, commuting_matrix, spread=1.0):
    randoms = (2 * np.random.rand(len(proj_indices)) - 1) * spread
    parameters = commuting_matrix @ np.multiply(proj_indices, randoms)
    return parameters


def commuting_ansatz(target_unitary, basis, projected_indices):
    ham = -1.j * spla.logm(target_unitary)
    target_params = lie.Hamiltonian.parameters_from_hamiltonian(ham, basis)
    target_ham = lie.Hamiltonian(basis, target_params)
    h_params = [sympy.Symbol(f'h_{i}') if ind else 0 for i, ind in enumerate(projected_indices)]

    h_mat = None
    for i, b in enumerate(target_ham.basis.basis):
        if h_mat is None:
            h_mat = h_params[i] * sympy.Matrix(b)
        else:
            h_mat += h_params[i] * sympy.Matrix(b)

    sols = sympy.solve(h_mat * target_ham.matrix - target_ham.matrix * h_mat)
    indices = remove_solution_free_parameters(h_params, sols)
    mat = construct_commuting_ansatz_matrix(h_params, sols)
    return indices, mat


def construct_commuting_ansatz_matrix(params, sols):
    mat = np.zeros((len(params), len(params)))
    for j, h in enumerate(params):
        if h:
            h_sub = {m: 0 for m in params if m}
            h_sub[h] = 1
            for i, s in enumerate(params):
                if i == j:
                    mat[i, j] = 1
                if s in sols:
                    mat[i, j] = sols[s].subs(h_sub)
    return mat


def remove_solution_free_parameters(params, sols):
    indices = [0 if h in sols else 1 if h else 0 for h in params]
    return indices


def multikron(matrices):
    product = matrices[0]
    for mat in matrices[1:]:
        product = np.kron(product, mat)
    return product


def golden_section_search(f, a, b, tol=1e-5):
    """Golden-section search.

    Given a function f with a single local minimum in
    the interval [a,b], gss returns a subset interval
    [c,d] that contains the minimum with d-c <= tol.

    Example:
    f = lambda x: (x-2)**2
    a = 1
    b = 5
    tol = 1e-5
    (c,d) = gss(f, a, b, tol)
    print(c, d)
    1.9999959837979107 2.0000050911830893
    source: https://en.wikipedia.org/wiki/Golden-section_search
    """

    invphi = (np.sqrt(5) - 1) / 2  # 1 / phi
    invphi2 = (3 - np.sqrt(5)) / 2  # 1 / phi^2

    (a, b) = (min(a, b), max(a, b))
    h = b - a
    if h <= tol:
        return (a, b)

    # Required steps to achieve tolerance
    n = int(np.ceil(np.log(tol / h) / np.log(invphi)))

    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(c)
    yd = f(d)

    for k in range(n - 1):
        if yc > yd:
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f(c)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = f(d)
    if yc < yd:
        return c, yc
    else:
        return d, yd