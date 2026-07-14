# -*- coding: utf-8 -*-

"""
The ribbon category of representations of a finite-dimensional Hopf algebra.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    HopfAlgebra
    Representation
    Functor

A finite-dimensional quasitriangular Hopf algebra :math:`H` has a category of
representations :math:`\\mathrm{Rep}(H)` which is braided: the braiding is the
universal R-matrix, cups and caps come from the antipode and the pivotal
element, and (when :math:`H` is *ribbon*) the twist is the ribbon element. A
quantum topological invariant of tangles is then a monoidal functor from the
free :mod:`.ribbon` category into :math:`\\mathrm{Rep}(H)`, evaluated as
concrete tensors (see :mod:`.tensor`).

This module lets you build any finite-dimensional Hopf algebra from its
structure tensors and map any ribbon diagram to the corresponding linear map.

Example
-------
The Drinfeld double of the group algebra of :math:`\\mathbb{Z}/2` gives a
non-trivial link invariant: it separates the Hopf link from the two-component
unlink.

>>> import numpy as np
>>> from discopy import ribbon
>>> H = HopfAlgebra.cyclic(2).double()          # D(Z/2), 4-dimensional
>>> assert H.is_valid() and H.dim == 4
>>> V = Representation.double_sum(H, [(0, -1), (1, 1)])   # V = e (+) m
>>> assert V.is_module()
>>> x = ribbon.Ty('x')
>>> F = Functor(ob={x: V}, ar={})
>>> hopf, split = complex(F(hopf_link(x))), complex(F(unlink(x)))
>>> assert not np.isclose(hopf, split)          # a non-trivial invariant
>>> assert np.isclose(hopf, 0) and np.isclose(split, 4)

Axioms
------
The Hopf-algebra axioms are string diagrams over the signature ``ob``,
``unit``, ``counit``, ``mult``, ``comult``, ``antipode`` (and ``rmatrix``). A
:class:`HopfAlgebra` checks each axiom by evaluating both sides of the equation
with its :meth:`~HopfAlgebra.functor` and comparing the resulting tensors.

Associativity and the bialgebra law (``comult`` is an algebra homomorphism):

>>> from discopy.drawing import Equation
>>> associativity = Equation(mult @ ob >> mult, ob @ mult >> mult)
>>> associativity.draw(path='docs/_static/hopf/associativity.png')

.. image:: /_static/hopf/associativity.png
    :align: center

>>> bialgebra = Equation(
...     mult >> comult,
...     comult @ comult >> ob @ Swap(ob, ob) @ ob >> mult @ mult)
>>> bialgebra.draw(path='docs/_static/hopf/bialgebra.png')

.. image:: /_static/hopf/bialgebra.png
    :align: center

The antipode axiom, with the counit-then-unit on the right-hand side:

>>> antipode_axiom = Equation(
...     comult >> antipode @ ob >> mult, counit >> unit,
...     comult >> ob @ antipode >> mult)
>>> antipode_axiom.draw(path='docs/_static/hopf/antipode.png')

.. image:: /_static/hopf/antipode.png
    :align: center
"""

from __future__ import annotations

import numpy as np

from discopy import ribbon, symmetric, tensor
from discopy.symmetric import Ty, Box, Swap, Id
from discopy.tensor import Dim
from discopy.utils import MappingOrCallable

TOL = 1e-9

# -- the signature of a (quasitriangular) Hopf algebra ----------------------
# One object ``ob`` (the algebra) and one box per generator, so that the axioms
# can be written and drawn as string diagrams and checked by evaluating both
# sides with a :class:`.tensor.Functor` (see :meth:`HopfAlgebra.functor`).
ob = Ty('H')
unit = Box('$\\eta$', Ty(), ob)          #: the unit, ``1 -> H``
counit = Box('$\\epsilon$', ob, Ty())    #: the counit, ``H -> 1``
mult = Box('$\\nabla$', ob @ ob, ob)     #: the multiplication, ``H H -> H``
comult = Box('$\\Delta$', ob, ob @ ob)   #: the comultiplication, ``H -> H H``
antipode = Box('$S$', ob, ob)            #: the antipode, ``H -> H``
rmatrix = Box('$R$', Ty(), ob @ ob)      #: the R-matrix, ``1 -> H H``

# the representation carries a second object ``rep`` with an action of ``ob``
rep = Ty('V')
action = Box('$\\rho$', ob @ rep, rep)        #: the action, ``H V -> V``


class HopfAlgebra:
    """
    A finite-dimensional Hopf algebra given by its structure tensors.

    With respect to a fixed basis :math:`e_0, \\dots, e_{n-1}`:

    Parameters:
        unit : The unit :math:`1_H`, shape ``(n,)``.
        counit : The counit :math:`\\epsilon`, shape ``(n,)``.
        mult : The multiplication, ``mult[i, j, k]`` the coefficient of
            :math:`e_k` in :math:`e_i \\cdot e_j`, shape ``(n, n, n)``.
        comult : The comultiplication, ``comult[i, p, q]`` the coefficient
            of :math:`e_p \\otimes e_q` in :math:`\\Delta(e_i)`, shape
            ``(n, n, n)``.
        antipode : The antipode, ``antipode[i, j]`` the coefficient of
            :math:`e_j` in :math:`S(e_i)``, shape ``(n, n)``.
        R : The universal R-matrix, ``R[i, j]`` the coefficient of
            :math:`e_i \\otimes e_j`, shape ``(n, n)`` (optional).
        ribbon_element : The ribbon element :math:`v \\in H`, shape ``(n,)``,
            making :math:`H` a ribbon Hopf algebra so that ribbon diagrams with
            a :class:`.ribbon.Twist` can be evaluated (optional).
    """
    def __init__(self, unit, counit, mult, comult, antipode,
                 R=None, ribbon_element=None):
        self.unit = np.asarray(unit, dtype=complex)
        self.counit = np.asarray(counit, dtype=complex)
        self.mult = np.asarray(mult, dtype=complex)
        self.comult = np.asarray(comult, dtype=complex)
        self.antipode = np.asarray(antipode, dtype=complex)
        self.dim = len(self.unit)
        n = self.dim
        assert self.mult.shape == (n, n, n)
        assert self.comult.shape == (n, n, n)
        assert self.antipode.shape == (n, n)
        self.R = None if R is None else np.asarray(R, dtype=complex)
        self.ribbon_element = None if ribbon_element is None \
            else np.asarray(ribbon_element, dtype=complex)

    def __repr__(self):
        return f"HopfAlgebra(dim={self.dim})"

    # -- element operations --------------------------------------------------
    def prod(self, x, y):
        """ The product :math:`x \\cdot y` of two elements. """
        return np.einsum('i,j,ijk->k', x, y, self.mult)

    def coprod(self, x):
        """ The coproduct :math:`\\Delta(x)` as an ``(n, n)`` array. """
        return np.einsum('i,ipq->pq', x, self.comult)

    def antipode_of(self, x):
        """ The antipode :math:`S(x)`. """
        return np.einsum('i,ij->j', x, self.antipode)

    def counit_of(self, x):
        """ The counit :math:`\\epsilon(x)`. """
        return complex(self.counit @ x)

    # -- diagrammatic semantics ----------------------------------------------
    def functor(self):
        """
        The :class:`.tensor.Functor` sending each generator of the signature
        (``ob``, ``unit``, ``counit``, ``mult``, ``comult``, ``antipode`` and,
        if present, ``rmatrix``) to its structure tensor. Axioms are checked by
        evaluating both sides of a diagram equation with this functor.
        """
        ar = {unit: self.unit, counit: self.counit, mult: self.mult,
              comult: self.comult, antipode: self.antipode}
        if self.R is not None:
            ar[rmatrix] = self.R
        return tensor.Functor(
            ob={ob: self.dim}, ar=ar, dom=symmetric.Diagram, dtype=complex)

    def check(self, *equations):
        """
        Whether every ``(lhs, rhs)`` diagram equation holds, i.e. the two sides
        evaluate to the same tensor under :meth:`functor`.
        """
        F = self.functor()
        return all(np.allclose(F(lhs).array, F(rhs).array, atol=TOL)
                   for lhs, rhs in equations)

    # -- axioms (as string diagrams) -----------------------------------------
    def is_associative(self):
        """ ``(mult @ ob) >> mult == (ob @ mult) >> mult``. """
        return self.check((mult @ ob >> mult, ob @ mult >> mult))

    def is_unital(self):
        """ The unit is a left and right identity for ``mult``. """
        return self.check(
            (unit @ ob >> mult, Id(ob)), (ob @ unit >> mult, Id(ob)))

    def is_coassociative(self):
        """ ``comult >> (comult @ ob) == comult >> (ob @ comult)``. """
        return self.check(
            (comult >> comult @ ob, comult >> ob @ comult))

    def is_counital(self):
        """ The counit is a left and right identity for ``comult``. """
        return self.check(
            (comult >> counit @ ob, Id(ob)), (comult >> ob @ counit, Id(ob)))

    def is_commutative(self):
        """ ``Swap >> mult == mult`` (a *property*, not an axiom). """
        return self.check((Swap(ob, ob) >> mult, mult))

    def is_cocommutative(self):
        """ ``comult >> Swap == comult`` (a *property*, not an axiom). """
        return self.check((comult >> Swap(ob, ob), comult))

    def is_bialgebra(self):
        """ ``comult`` and ``counit`` are algebra homomorphisms. """
        return self.check(
            (mult >> comult,
             comult @ comult >> ob @ Swap(ob, ob) @ ob >> mult @ mult),
            (mult >> counit, counit @ counit),
            (unit >> comult, unit @ unit),
            (unit >> counit, Id(Ty())))

    def has_antipode(self):
        """ ``comult >> (S @ ob) >> mult == counit >> unit ==
        comult >> (ob @ S) >> mult``. """
        return self.check(
            (comult >> antipode @ ob >> mult, counit >> unit),
            (comult >> ob @ antipode >> mult, counit >> unit))

    def is_quasitriangular(self):
        """
        Whether ``R`` is a universal R-matrix: it intertwines ``comult`` with
        its opposite and satisfies the two hexagon equations.

        >>> from discopy.drawing import Equation
        >>> swap = Swap(ob, ob)
        >>> intertwiner = Equation(
        ...     rmatrix @ comult >> ob @ swap @ ob >> mult @ mult,
        ...     (comult >> swap) @ rmatrix >> ob @ swap @ ob >> mult @ mult)
        >>> intertwiner.draw(path='docs/_static/hopf/quasitriangular.png')

        .. image:: /_static/hopf/quasitriangular.png
            :align: center

        >>> hexagon1 = Equation(
        ...     rmatrix >> comult @ ob,
        ...     rmatrix @ rmatrix >> ob @ swap @ ob >> ob @ ob @ mult)
        >>> hexagon1.draw(path='docs/_static/hopf/hexagon1.png')

        .. image:: /_static/hopf/hexagon1.png
            :align: center

        >>> hexagon2 = Equation(
        ...     rmatrix >> ob @ comult,
        ...     rmatrix @ rmatrix >> ob @ swap @ ob
        ...     >> mult @ ob @ ob >> ob @ swap)
        >>> hexagon2.draw(path='docs/_static/hopf/hexagon2.png')

        .. image:: /_static/hopf/hexagon2.png
            :align: center
        """
        if self.R is None:
            return False
        swap = Swap(ob, ob)
        return self.check(
            # R Delta = Delta^op R
            (rmatrix @ comult >> ob @ swap @ ob >> mult @ mult,
             (comult >> swap) @ rmatrix >> ob @ swap @ ob >> mult @ mult),
            # (Delta (x) id) R = R13 R23
            (rmatrix >> comult @ ob,
             rmatrix @ rmatrix >> ob @ swap @ ob >> ob @ ob @ mult),
            # (id (x) Delta) R = R13 R12
            (rmatrix >> ob @ comult,
             rmatrix @ rmatrix >> ob @ swap @ ob
             >> mult @ ob @ ob >> ob @ swap))

    def validate(self):
        """ A dictionary of all the axiom checks. """
        checks = dict(
            associative=self.is_associative(), unital=self.is_unital(),
            coassociative=self.is_coassociative(), counital=self.is_counital(),
            bialgebra=self.is_bialgebra(), antipode=self.has_antipode())
        if self.R is not None:
            checks['quasitriangular'] = self.is_quasitriangular()
        return checks

    def is_valid(self):
        """ Whether all the axiom checks pass. """
        return all(self.validate().values())

    # -- derived elements ----------------------------------------------------
    def drinfeld_element(self):
        """
        The Drinfeld element :math:`u = \\sum S(R^{(2)}) R^{(1)}`.
        """
        if self.R is None:
            raise ValueError(
                "Drinfeld element needs a quasitriangular structure.")
        # u = sum_ij R[i,j] S(e_j) e_i
        S_R2 = np.einsum('ij,jr->ijr', self.R, self.antipode)  # S(e_j)-> e_r
        return np.einsum('ijr,rik->k', S_R2, self.mult, optimize=True)

    def pivotal_element(self):
        """
        The pivotal (spherical) grouplike element :math:`g = u v^{-1}`, where
        :math:`u` is the Drinfeld element and :math:`v` the ribbon element.
        """
        if self.ribbon_element is None:
            raise ValueError("Pivotal element needs a ribbon element.")
        u = self.drinfeld_element()
        v_inv = self._inverse(self.ribbon_element)
        return self.prod(u, v_inv)

    def _inverse(self, x):
        """ The multiplicative inverse of a central element ``x``. """
        # solve L(x) y = 1 where L(x) is left multiplication by x
        left = np.einsum('i,ijk->kj', x, self.mult)  # (k<-j)
        return np.linalg.solve(left, self.unit)

    # -- constructors --------------------------------------------------------
    @classmethod
    def group_algebra(cls, table):
        """
        The group algebra :math:`k[G]` from a group multiplication ``table``,
        with ``table[i][j]`` the index of :math:`g_i g_j` and ``g_0`` the unit.

        >>> Z2 = HopfAlgebra.group_algebra([[0, 1], [1, 0]])
        >>> assert Z2.is_valid()
        """
        table = [list(row) for row in table]
        n = len(table)
        # inverses: g_i^{-1} is the g_j with g_i g_j = g_0
        inverse = [next(j for j in range(n) if table[i][j] == 0)
                   for i in range(n)]
        unit = np.zeros(n)
        unit[0] = 1
        counit = np.ones(n)
        mult = np.zeros((n, n, n))
        comult = np.zeros((n, n, n))
        antipode = np.zeros((n, n))
        for i in range(n):
            comult[i, i, i] = 1                 # grouplike
            antipode[i, inverse[i]] = 1
            for j in range(n):
                mult[i, j, table[i][j]] = 1
        R = np.zeros((n, n))
        R[0, 0] = 1                             # cocommutative: trivial R
        return cls(unit, counit, mult, comult, antipode, R,
                   ribbon_element=unit.copy())

    @classmethod
    def cyclic(cls, n):
        """
        The group algebra of the cyclic group :math:`\\mathbb{Z}/n`.

        >>> assert HopfAlgebra.cyclic(3).is_valid()
        """
        table = [[(i + j) % n for j in range(n)] for i in range(n)]
        return cls.group_algebra(table)

    @classmethod
    def sweedler(cls):
        """
        Sweedler's four-dimensional Hopf algebra, the smallest one that is
        neither commutative nor cocommutative, with basis :math:`1, g, x, gx`
        (:math:`g^2 = 1`, :math:`x^2 = 0`, :math:`xg = -gx`).

        Its antipode has :math:`S^2 \\neq \\mathrm{id}`, so its
        :meth:`double` genuinely exercises the :math:`S^{-1}` in the double's
        multiplication -- unlike any (cocommutative) group algebra.

        >>> H = HopfAlgebra.sweedler()
        >>> assert H.is_valid() and H.dim == 4
        >>> import numpy as np
        >>> assert not np.allclose(H.antipode @ H.antipode, np.eye(4))
        """
        # basis 0: 1, 1: g, 2: x, 3: gx
        unit = np.array([1, 0, 0, 0])
        counit = np.array([1, 1, 0, 0])
        mult = np.zeros((4, 4, 4))
        for j in range(4):
            mult[0, j, j] = 1                      # 1 . e_j = e_j
        mult[1, 0, 1] = mult[1, 1, 0] = 1          # g.1=g, g.g=1
        mult[1, 2, 3] = mult[1, 3, 2] = 1          # g.x=gx, g.gx=x
        mult[2, 0, 2] = 1                          # x.1=x
        mult[2, 1, 3] = -1                         # x.g = -gx
        mult[3, 0, 3] = 1                          # gx.1=gx
        mult[3, 1, 2] = -1                         # gx.g = -x
        comult = np.zeros((4, 4, 4))
        comult[0, 0, 0] = 1                        # D(1) = 1 (x) 1
        comult[1, 1, 1] = 1                        # D(g) = g (x) g
        comult[2, 2, 0] = comult[2, 1, 2] = 1      # D(x) = x(x)1 + g(x)x
        comult[3, 3, 1] = comult[3, 0, 3] = 1      # D(gx) = gx(x)g + 1(x)gx
        antipode = np.zeros((4, 4))
        antipode[0, 0] = antipode[1, 1] = 1        # S(1)=1, S(g)=g
        antipode[2, 3] = -1                        # S(x) = -gx
        antipode[3, 2] = 1                         # S(gx) = x
        return cls(unit, counit, mult, comult, antipode)

    def double(self):
        """
        The Drinfeld quantum double
        :math:`D(H) = H^{*\\mathrm{cop}} \\otimes H`, a quasitriangular Hopf
        algebra of dimension ``self.dim ** 2``.

        This is the general construction on *any* finite-dimensional Hopf
        algebra with invertible antipode (the group algebra is only one
        example). The
        basis is :math:`f^b \\otimes e_a` at flat index ``b * dim + a``, with
        :math:`\\{f^b\\}` the dual basis of :math:`H^*`.

        >>> D = HopfAlgebra.cyclic(2).double()
        >>> assert D.dim == 4 and D.is_valid() and D.is_quasitriangular()
        """
        n = self.dim
        M, C, U, E, Sa = self.mult, self.comult, self.unit, self.counit, \
            self.antipode
        Sinv = np.linalg.inv(Sa)
        # H* structure constants from the pairing <f^i, e_j> = delta_ij
        Mstar = np.transpose(C, (1, 2, 0))   # f^i f^j = sum_l C[l,i,j] f^l
        Cstar = np.transpose(M, (2, 0, 1))  # Delta* f^k = sum M[i,j,k] f^i f^j
        C2 = np.einsum('ipq,prs->irsq', C, C, optimize=True)
        Cstar2 = np.einsum('bmz,mxy->bxyz', Cstar, Cstar, optimize=True)
        N = n * n

        def flat(b, a):
            return b * n + a

        mult = np.zeros((N, N, N), dtype=complex)
        for b1 in range(n):
            for a1 in range(n):
                for b2 in range(n):
                    for a2 in range(n):
                        coeff = np.einsum(
                            'rsq,xyr,qx,yl,sk->lk', C2[a1], Cstar2[b2], Sinv,
                            Mstar[b1], M[:, a2, :], optimize=True)
                        mult[flat(b1, a1), flat(b2, a2)] = coeff.reshape(N)
        comult = np.zeros((N, N, N), dtype=complex)
        for b in range(n):
            for a in range(n):
                block = np.einsum('ij,pq->jpiq', M[:, :, b], C[a])
                comult[flat(b, a)] = block.reshape(N, N)
        unit = np.outer(E, U).reshape(N)       # eps_H (x) 1_H
        counit = np.outer(U, E).reshape(N)      # <f^b, 1> eps(e_a)
        # R = sum_i (eps_H (x) e_i) (x) (f^i (x) 1_H)
        R = np.zeros((N, N), dtype=complex)
        for b1 in range(n):
            for a1 in range(n):
                for a2 in range(n):
                    R[flat(b1, a1), flat(a1, a2)] += E[b1] * U[a2]
        double = HopfAlgebra(unit, counit, mult, comult, np.eye(N), R)
        # antipode S_D(phi (x) h) = (eps_H (x) S(h)) . (S*^{-1}(phi) (x) 1_H)
        Sstar_inv = Sinv.T
        antipode = np.zeros((N, N), dtype=complex)
        for b in range(n):
            for a in range(n):
                factor1 = np.outer(E, Sa[a]).reshape(N)
                factor2 = np.outer(Sstar_inv[b], U).reshape(N)
                antipode[flat(b, a)] = double.prod(factor1, factor2)
        double.antipode = antipode
        return double


class Representation:
    """
    A finite-dimensional (left) module over a :class:`HopfAlgebra`, i.e. an
    object of :math:`\\mathrm{Rep}(H)`.

    Parameters:
        algebra : The :class:`HopfAlgebra` :math:`H`.
        dim : The dimension :math:`d` of the underlying space :math:`V`.
        action : The action tensor, ``action[i]`` the :math:`d \\times d`
            matrix of :math:`\\rho(e_i)`, shape ``(n, d, d)``.

    The structural morphisms of :math:`\\mathrm{Rep}(H)` are the braiding
    :math:`c = \\tau \\circ (\\rho \\otimes \\rho)(R)`, the (co)evaluations
    built from the antipode and the pivotal element :math:`g`, and the twist
    :math:`\\theta = \\rho(v)`.
    """
    def __init__(self, algebra, dim, action):
        self.algebra = algebra
        self.dim = dim
        self.action = np.asarray(action, dtype=complex)
        assert self.action.shape == (algebra.dim, dim, dim)

    def __repr__(self):
        return f"Representation(dim={self.dim})"

    def act(self, element):
        """ The :math:`d \\times d` matrix :math:`\\rho(a)` of an element. """
        return np.einsum('i,ijk->jk', np.asarray(element, dtype=complex),
                         self.action)

    def functor(self):
        """
        The :class:`.tensor.Functor` extending :meth:`HopfAlgebra.functor` with
        the object ``rep`` (the module :math:`V`) and the generator ``action``.
        """
        H = self.algebra
        ar = {unit: H.unit, counit: H.counit, mult: H.mult, comult: H.comult,
              antipode: H.antipode,
              action: np.transpose(self.action, (0, 2, 1))}
        if H.R is not None:
            ar[rmatrix] = H.R
        return tensor.Functor(
            ob={ob: H.dim, rep: self.dim}, ar=ar,
            dom=symmetric.Diagram, dtype=complex)

    def is_module(self):
        """
        Whether ``action`` is a representation, i.e. the two module axioms hold
        as string diagrams: ``action`` is associative over ``mult`` and unital
        over ``unit``.

        >>> from discopy.drawing import Equation
        >>> associativity = Equation(
        ...     mult @ rep >> action, ob @ action >> action)
        >>> associativity.draw(
        ...     path='docs/_static/hopf/module_associativity.png')

        .. image:: /_static/hopf/module_associativity.png
            :align: center

        >>> unitality = Equation(unit @ rep >> action, Id(rep))
        >>> unitality.draw(path='docs/_static/hopf/module_unitality.png')

        .. image:: /_static/hopf/module_unitality.png
            :align: center
        """
        F = self.functor()

        def eq(lhs, rhs):
            return np.allclose(F(lhs).array, F(rhs).array, atol=TOL)

        return eq(mult @ rep >> action, ob @ action >> action) \
            and eq(unit @ rep >> action, Id(rep))

    def dual_action(self):
        """
        The action on the dual :math:`V^*`:
        :math:`\\rho^*(a) = \\rho(S(a))^T`.
        """
        S = self.algebra.antipode
        return np.einsum('ij,jkl->ilk', S, self.action)  # rho(S(e_i))^T

    def pivotal(self):
        """
        The pivotal operator :math:`G = \\rho(g)` with :math:`g = u v^{-1}`.
        Defaults to the identity when no ribbon element is given (i.e. the
        spherical structure is assumed trivial).
        """
        if self.algebra.ribbon_element is None:
            return np.eye(self.dim, dtype=complex)
        return self.act(self.algebra.pivotal_element())

    def qdim(self):
        """ The quantum dimension :math:`\\mathrm{tr}(G)`. """
        return complex(np.trace(self.pivotal()))

    def braiding(self, other=None):
        """
        The braiding matrix :math:`c_{V,W}: V \\otimes W \\to W \\otimes V`,
        as an ``(dV*dW, dW*dV)`` array indexed ``[input, output]``.
        """
        other = self if other is None else other
        H = self.algebra
        dV, dW = self.dim, other.dim
        R_action = np.zeros((dV * dW, dV * dW), dtype=complex)
        for i in range(H.dim):
            for j in range(H.dim):
                if H.R[i, j] != 0:
                    basis = np.eye(H.dim)
                    R_action += H.R[i, j] * np.kron(
                        self.act(basis[i]), other.act(basis[j]))
        # tau: V (x) W -> W (x) V,  input (a, b) -> output (b, a)
        swap = np.zeros((dW * dV, dV * dW), dtype=complex)
        for a in range(dV):
            for b in range(dW):
                swap[b * dV + a, a * dW + b] = 1
        matrix = swap @ R_action  # output x input
        return matrix.T           # input x output

    def twist(self):
        """ The twist :math:`\\theta = \\rho(v)` as a ``(d, d)`` array. """
        if self.algebra.ribbon_element is None:
            raise ValueError("Twist needs a ribbon element on the algebra.")
        return self.act(self.algebra.ribbon_element)

    # -- constructors --------------------------------------------------------
    @classmethod
    def regular(cls, algebra):
        """ The regular representation (left multiplication). """
        n = algebra.dim
        action = np.einsum('ijk->ikj', algebra.mult)  # rho(e_i)e_j = e_i e_j
        return cls(algebra, n, action)

    @classmethod
    def double_sum(cls, double, anyons):
        """
        The direct sum of anyon modules of the quantum double of a cyclic
        group algebra. Each anyon is a pair ``(flux, charge)`` where ``flux``
        is a group index and the group element ``e_a`` acts by ``charge ** a``.

        >>> D = HopfAlgebra.cyclic(2).double()
        >>> V = Representation.double_sum(D, [(0, -1), (1, 1)])  # e (+) m
        >>> assert V.is_module() and V.dim == 2
        """
        n = int(round(double.dim ** 0.5))
        assert n * n == double.dim, "not the double of an n-dim algebra"
        d = len(anyons)
        action = np.zeros((double.dim, d, d), dtype=complex)
        for k, (flux, charge) in enumerate(anyons):
            for b in range(n):
                for a in range(n):
                    action[b * n + a, k, k] = (charge ** a) if b == flux else 0
        return cls(double, d, action)


def _is_adjoint(ob):
    """ Whether a pivotal object is an adjoint (odd winding). """
    return bool(ob.z % 2)


class Functor(ribbon.Functor):
    """
    A ribbon functor from :mod:`.ribbon` diagrams into
    :math:`\\mathrm{Rep}(H)`, evaluated as concrete :class:`.tensor.Tensor`.

    Parameters:
        ob : A mapping from atomic :class:`.ribbon.Ty` to
            :class:`Representation`.
        ar : A mapping from generating :class:`.ribbon.Box` to arrays.
        contractor : The tensor-network contractor, see
            :class:`.tensor.Functor` (``None`` for the naive functor, or
            ``'einsum'``, ``'tn'``, ...).
        backend : The array backend to evaluate in.

    The braiding is sent to the R-matrix, cups and caps to the (co)evaluations,
    and the twist to the ribbon element. The diagram is first translated into a
    :class:`.tensor.Diagram` of concrete boxes, then contracted through
    :class:`.tensor.Functor` / :class:`.tensor.CMap`.
    """
    dom, cod = ribbon.Diagram, tensor.Diagram

    def __init__(self, ob, ar=None, contractor=None, backend=None):
        self.ob = MappingOrCallable(ob)
        self.ar = MappingOrCallable(ar or {})
        self.contractor, self.backend = contractor, backend
        self._reps = {}
        items = ob.items() if hasattr(ob, "items") else []
        for typ, rep in items:
            name = typ.inside[0].name if hasattr(typ, "inside") else str(typ)
            self._reps[name] = rep

    def rep(self, ob):
        """ The :class:`Representation` assigned to an atomic object. """
        name = ob.name if hasattr(ob, "name") else ob.inside[0].name
        return self._reps[name]

    def dim(self, typ):
        """ The :class:`.tensor.Dim` assigned to a type. """
        return Dim(*[self.rep(ob).dim for ob in typ.inside])

    def _tensor_box(self, box):
        Box = tensor.Box
        if isinstance(box, ribbon.Braid):
            left, right = box.dom.inside
            matrix = self.rep(left).braiding(self.rep(right))
            if box.is_dagger:
                matrix = np.linalg.inv(matrix)
            return Box(str(box), self.dim(box.dom), self.dim(box.cod),
                       matrix.reshape(-1))
        if isinstance(box, ribbon.Twist):
            rep = self.rep(box.dom.inside[0])
            matrix = rep.twist()
            if box.is_dagger:
                matrix = np.linalg.inv(matrix)
            return Box(str(box), self.dim(box.dom), self.dim(box.cod),
                       matrix.reshape(-1))
        if isinstance(box, ribbon.Cap):
            left, right = box.cod.inside
            rep = self.rep(left)
            d = rep.dim
            G = rep.pivotal()
            array = G.T if _is_adjoint(left) else np.eye(d)
            return Box(str(box), Dim(), Dim(d) @ Dim(d), array.reshape(-1))
        if isinstance(box, ribbon.Cup):
            left, right = box.dom.inside
            rep = self.rep(left)
            d = rep.dim
            G = rep.pivotal()
            array = np.linalg.inv(G).T if not _is_adjoint(left) else np.eye(d)
            return Box(str(box), Dim(d) @ Dim(d), Dim(), array.reshape(-1))
        # generic box
        return Box(
            box.name, self.dim(box.dom), self.dim(box.cod), self.ar[box])

    def _to_tensor_diagram(self, diagram):
        result = tensor.Diagram.id(self.dim(diagram.dom))
        for left, box, right in diagram.inside:
            result = result >> (
                tensor.Diagram.id(self.dim(left))
                @ self._tensor_box(box)
                @ tensor.Diagram.id(self.dim(right)))
        return result

    def __call__(self, other):
        if isinstance(other, ribbon.Ty):
            return self.dim(other)
        diagram = self._to_tensor_diagram(other)
        if self.contractor is None and self.backend is None:
            return diagram.eval(dtype=complex)
        return tensor.Functor(
            ob=lambda d: d, ar=lambda b: b.array, dtype=complex,
            contractor=self.contractor, backend=self.backend,
            dom=tensor.Diagram)(diagram)


def circle(typ):
    """ A single loop (unknot) coloured by ``typ``: evaluates to the quantum
    dimension. """
    x, = typ.inside
    x = ribbon.Ty(x.name)
    return ribbon.Cap(x, x.r) >> ribbon.Cup(x, x.r)


def unlink(typ):
    """ The two-component unlink coloured by ``typ``. """
    return circle(typ) @ circle(typ)


def hopf_link(typ):
    """
    The (closed) Hopf link coloured by ``typ``, as the trace closure of the
    square of the braid on two strands.
    """
    x, = typ.inside
    x = ribbon.Ty(x.name)
    braid = ribbon.Braid(x, x) >> ribbon.Braid(x, x)   # sigma^2
    return braid.trace(n=2)
