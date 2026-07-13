import numpy as np
from pytest import raises

from discopy import ribbon
from discopy.hopf import (
    HopfAlgebra, Representation, Functor, circle, unlink, hopf_link)


# -- the Hopf algebra layer --------------------------------------------------

def test_group_algebra_is_valid():
    for n in [1, 2, 3, 5]:
        assert HopfAlgebra.cyclic(n).is_valid()
    # a non-cyclic group (Klein four) from its table
    table = [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]]
    assert HopfAlgebra.group_algebra(table).is_valid()


def test_double_is_quasitriangular_hopf_algebra():
    # the general double() applied to k[Z/n], not a hardcoded table
    for n in [2, 3]:
        D = HopfAlgebra.cyclic(n).double()
        assert D.dim == n * n
        assert D.is_valid()
        assert D.is_quasitriangular()


def test_double_of_sweedler():
    # Sweedler's H4 is neither commutative nor cocommutative and has S^2 != id,
    # so its double genuinely exercises the S^-1 in the double's multiplication
    # (a group algebra, being cocommutative with S^2 = id, would not).
    H4 = HopfAlgebra.sweedler()
    assert H4.is_valid() and H4.dim == 4
    assert not np.allclose(H4.antipode @ H4.antipode, np.eye(4))  # S^2 != id
    D = H4.double()
    assert D.dim == 16
    assert D.is_valid() and D.is_quasitriangular()


def test_drinfeld_and_pivotal_element():
    D = HopfAlgebra.cyclic(2).double()
    u = D.drinfeld_element()
    assert u.shape == (4,)
    # no ribbon element on the double -> pivotal element is undefined
    assert D.ribbon_element is None
    with raises(ValueError):
        D.pivotal_element()
    # a group algebra has a (trivial) ribbon element v = 1
    Z2 = HopfAlgebra.cyclic(2)
    assert Z2.ribbon_element is not None
    assert np.allclose(Z2.pivotal_element(), Z2.unit)


# -- representations & structural morphisms ----------------------------------

def _double_and_module():
    D = HopfAlgebra.cyclic(2).double()
    V = Representation.double_sum(D, [(0, -1), (1, 1)])   # e (+) m
    return D, V


def test_representation_is_module():
    D, V = _double_and_module()
    assert V.is_module() and V.dim == 2
    assert Representation.regular(D).is_module()
    for anyon in [(0, 1), (0, -1), (1, 1), (1, -1)]:
        assert Representation.double_sum(D, [anyon]).is_module()


def test_braiding_yang_baxter_and_inverse():
    _, V = _double_and_module()
    d = V.dim
    c = V.braiding()                       # input x output, (d^2, d^2)
    # braiding is invertible and not the swap
    assert not np.isclose(np.linalg.det(c), 0)
    swap = np.zeros((d * d, d * d))
    for a in range(d):
        for b in range(d):
            swap[a * d + b, b * d + a] = 1
    assert not np.allclose(c, swap)
    # Yang-Baxter on the braiding operator R = c^T (output x input)
    R, eye = c.T, np.eye(d)
    R12, R23 = np.kron(R, eye), np.kron(eye, R)
    assert np.allclose(R12 @ R23 @ R12, R23 @ R12 @ R23)


def test_quantum_dimension():
    _, V = _double_and_module()
    assert np.isclose(V.qdim(), 2)         # tr(G) = tr(id) = dim


def test_snake_equations():
    # standard (co)evaluation zig-zag on the tensor level (G = id here)
    _, V = _double_and_module()
    x = ribbon.Ty('x')
    F = Functor(ob={x: V}, ar={})
    left = ribbon.Id(x.l).transpose(left=True)
    right = ribbon.Id(x.r).transpose(left=False)
    assert np.allclose(F(left).array, F(ribbon.Id(x)).array)
    assert np.allclose(F(right).array, F(ribbon.Id(x)).array)


# -- the functor and the topological invariant -------------------------------

def test_reidemeister_moves():
    _, V = _double_and_module()
    x = ribbon.Ty('x')
    F = Functor(ob={x: V}, ar={})
    X = ribbon.Ty('x')
    # R2: a crossing and its inverse cancel
    r2 = ribbon.Braid(X, X) >> ribbon.Braid(X, X).dagger()
    assert np.allclose(F(r2).array, F(ribbon.Id(X @ X)).array)
    # R3 / Yang-Baxter on three strands
    lhs = ribbon.Braid(X, X) @ X >> X @ ribbon.Braid(X, X) \
        >> ribbon.Braid(X, X) @ X
    rhs = X @ ribbon.Braid(X, X) >> ribbon.Braid(X, X) @ X \
        >> X @ ribbon.Braid(X, X)
    assert np.allclose(F(lhs).array, F(rhs).array)


def test_nontrivial_link_invariant():
    _, V = _double_and_module()
    x = ribbon.Ty('x')
    F = Functor(ob={x: V}, ar={})
    # the invariant separates the Hopf link from the unlink
    assert np.isclose(complex(F(circle(x))), 2)       # unknot -> qdim
    assert np.isclose(complex(F(unlink(x))), 4)       # 2 unknots
    assert np.isclose(complex(F(hopf_link(x))), 0)    # Hopf link
    assert not np.isclose(complex(F(hopf_link(x))), complex(F(unlink(x))))


def test_crossing_number_distinguishes_closures():
    _, V = _double_and_module()
    x = ribbon.Ty('x')
    F = Functor(ob={x: V}, ar={})
    X = ribbon.Ty('x')
    values = [
        complex(F(ribbon.Id(X @ X).trace(n=2))),                 # unlink
        complex(F(ribbon.Braid(X, X).trace(n=2))),               # unknot
        complex(F((ribbon.Braid(X, X) >> ribbon.Braid(X, X)).trace(n=2))),
    ]
    assert np.allclose(values, [4, 2, 0])


def test_two_colour_mutual_braiding():
    D = HopfAlgebra.cyclic(2).double()
    e = Representation.double_sum(D, [(0, -1)])
    m = Representation.double_sum(D, [(1, 1)])
    xe, xm = ribbon.Ty('e'), ribbon.Ty('m')
    F = Functor(ob={xe: e, xm: m}, ar={})

    def hopf(a, b):
        return (ribbon.Braid(a, b) >> ribbon.Braid(b, a)).trace(n=2)

    assert np.isclose(complex(F(hopf(xe, xm))), -1)   # mutual statistics -1
    assert np.isclose(complex(F(hopf(xe, xe))), 1)
    assert np.isclose(complex(F(hopf(xm, xm))), 1)


def test_contractors_agree():
    _, V = _double_and_module()
    x = ribbon.Ty('x')
    diagram = hopf_link(x)
    naive = complex(Functor(ob={x: V}, ar={})(diagram))
    einsum = complex(Functor(ob={x: V}, ar={}, contractor='einsum')(diagram))
    assert np.isclose(naive, einsum)


def test_functor_on_generic_box():
    _, V = _double_and_module()
    x = ribbon.Ty('x')
    box = ribbon.Box('f', x, x)
    F = Functor(ob={x: V}, ar={box: np.array([[1, 2], [3, 4]])})
    assert np.allclose(F(box).array, [[1, 2], [3, 4]])


def test_functor_on_type():
    from discopy.tensor import Dim
    _, V = _double_and_module()
    x = ribbon.Ty('x')
    F = Functor(ob={x: V}, ar={})
    assert F(x) == Dim(2)
    assert F(x @ x) == Dim(2, 2)


def test_element_helpers_and_reprs():
    Z2 = HopfAlgebra.cyclic(2)
    assert "HopfAlgebra" in repr(Z2)
    x = Z2.unit
    assert np.allclose(Z2.antipode_of(x), Z2.unit)     # S(1) = 1
    assert np.isclose(Z2.counit_of(Z2.unit), 1)        # eps(1) = 1
    V = Representation.regular(Z2)
    assert "Representation" in repr(V)
    assert V.dual_action().shape == (2, 2, 2)


def test_no_r_matrix_paths():
    Z2 = HopfAlgebra.cyclic(2)
    noR = HopfAlgebra(Z2.unit, Z2.counit, Z2.mult, Z2.comult, Z2.antipode)
    assert noR.R is None
    assert noR.is_quasitriangular() is False
    assert 'quasitriangular' not in noR.validate()
    with raises(ValueError):
        noR.drinfeld_element()


def test_is_module_rejects_non_module():
    Z2 = HopfAlgebra.cyclic(2)
    bogus = np.zeros((2, 2, 2))              # rho(1) != id
    assert not Representation(Z2, 2, bogus).is_module()
    # a homomorphism that fails multiplicativity
    action = np.stack([np.eye(2), np.array([[1., 0], [0, 2]])])
    assert not Representation(Z2, 2, action).is_module()


def test_twist_and_pivotal_with_ribbon_element():
    # a group algebra is a ribbon Hopf algebra with v = 1 (trivial twist)
    Z2 = HopfAlgebra.cyclic(2)
    V = Representation.regular(Z2)
    assert np.allclose(V.pivotal(), np.eye(2))
    assert np.allclose(V.twist(), np.eye(2))
    assert np.isclose(V.qdim(), 2)
    x = ribbon.Ty('x')
    F = Functor(ob={x: V}, ar={})
    twist = ribbon.Twist(x)
    assert np.allclose(F(twist).array, np.eye(2))
    assert np.allclose(F(twist.dagger()).array, np.eye(2))


def test_representation_without_ribbon_element_has_no_twist():
    _, V = _double_and_module()      # double has no ribbon element
    with raises(ValueError):
        V.twist()
    assert np.allclose(V.pivotal(), np.eye(2))   # defaults to identity
