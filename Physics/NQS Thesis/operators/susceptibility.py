import netket as nk
from netket.operator.spin import sigmax, sigmay, sigmaz

def m2_neel_2d(hi, g: nk.graph.Lattice):
    sites = g.basis_coords
    id = g.id_from_basis_coords

    m2 = nk.operator.LocalOperator(hi, dtype=complex)
    for r_i in sites:
        for r_j in sites:
            i = int(id(r_i))
            j = int(id(r_j))
            m2 += (sigmax(hi, i)*sigmax(hi, j) + sigmay(hi, i)*sigmay(hi, j) +
                    sigmaz(hi, i)*sigmaz(hi, j)) * (-1)**int(r_i[0] - r_j[0] + r_i[1] - r_j[1])

    return m2

def m2_stripe_2d(hi, g: nk.graph.Lattice):
    sites = g.basis_coords
    id = g.id_from_basis_coords

    m2 = nk.operator.LocalOperator(hi, dtype=complex)
    for r_i in sites:
        for r_j in sites:
            i = int(id(r_i))
            j = int(id(r_j))
            m2 += (sigmax(hi, i)*sigmax(hi, j) + sigmay(hi, i)*sigmay(hi, j) +
                    sigmaz(hi, i)*sigmaz(hi, j)) * (-1)**int(r_i[0] - r_j[0])

    return m2

def m2_1d_neel(hi, g: nk.graph.Lattice):
    m2 = nk.operator.LocalOperator(hi, dtype=complex)
    for i in g.nodes():
        for j in g.nodes():
            m2 += (sigmax(hi, i)*sigmax(hi, j) + sigmay(hi, i)*sigmay(hi, j) +
                    sigmaz(hi, i)*sigmaz(hi, j)) * (-1)**int(i - j)

    return m2
