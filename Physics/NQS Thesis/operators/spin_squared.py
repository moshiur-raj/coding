import netket as nk
from netket.operator.spin import sigmax, sigmay, sigmaz

def s2(hi, g: nk.graph.Lattice):
    sx = nk.operator.LocalOperator(hi, dtype=complex)
    sy = nk.operator.LocalOperator(hi, dtype=complex)
    sz = nk.operator.LocalOperator(hi, dtype=complex)

    for i in g.nodes():
        sx += sigmax(hi, i)
        sy += sigmay(hi, i)
        sz += sigmaz(hi, i)

    s2 = sx * sx + sy * sy + sz * sz

    return s2
