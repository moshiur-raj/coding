import netket as nk

def j1j2(
        j2: float, g: nk.graph.Lattice, size: int, j1: float,
        sign_rule: list[bool], total_sz: int , normalize: bool
        ) -> tuple[nk.operator.GraphOperator, nk.hilbert.Spin, nk.graph.Lattice]:

    hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes, total_sz=total_sz)

    ham = nk.operator.Heisenberg(
            hilbert=hi, graph=g, J=[j1, j2], sign_rule=sign_rule
            ).to_jax_operator()

    if normalize:
        ham /= 4*size

    return ham, hi, g

def j1j2_1d(
        l: int, param: float, j1=1.0, pbc: bool = True,
        sign_rule: list[bool] = [False, False], total_sz: int = 0, normalize: bool = True
        ) -> tuple[nk.operator.GraphOperator, nk.hilbert.Spin, nk.graph.Lattice]:

    g = nk.graph.Chain(length=l, pbc=pbc, max_neighbor_order=2)

    return j1j2(
            j2=param, g=g, size=l, j1=j1,
            sign_rule=sign_rule, total_sz=total_sz, normalize=normalize
            )

def j1j2_2d_rect(
        lx: int, ly: int, param: float, j1: float = 1.0, pbc: bool = True,
        sign_rule: list[bool] = [False, False], total_sz: int = 0, normalize: bool = True
        ) -> tuple[nk.operator.GraphOperator, nk.hilbert.Spin, nk.graph.Lattice]:

    g = nk.graph.Grid((lx, ly), pbc=pbc, max_neighbor_order=2)

    return j1j2(
            j2=param, g=g, size=lx*ly, j1=j1,
            sign_rule=sign_rule, total_sz=total_sz, normalize=normalize
            )

def j1j2_2d_square(
        l: int, param: float, j1: float = 1.0, pbc: bool = True,
        sign_rule: list[bool] = [False, False], total_sz: int = 0, normalize : bool = True
        ) -> tuple[nk.operator.GraphOperator, nk.hilbert.Spin, nk.graph.Lattice]:

    g = nk.graph.Hypercube(length=l, n_dim=2, pbc=pbc, max_neighbor_order=2)

    return j1j2(
            j2=param, g=g, size=l*l, j1=j1,
            sign_rule=sign_rule, total_sz=total_sz, normalize=normalize
            )
