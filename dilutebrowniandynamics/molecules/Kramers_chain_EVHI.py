import numpy as np
from scipy.linalg.lapack import dposv, dptsv
from numpy.linalg import LinAlgError


LENGTH_TOL = 1e-6
MAXITER = 100
FILTER_NOISE = True


class KramersChainEVHI:
    """Kramers chain with excluded volume and hydrodynamic interactions.

    Attributes
    ----------
    Q : ndarry (N, 3)
        Coorinates of the vectors.
    tensions : ndarray (N,)
        Internal tensions.
    rng : Generator
        Random number generator.
    dW : ndarray (N, 3)
        Random forces.
    dQ : ndarray (N, 3)
        Evolution vector.
    _EV : ndarray (N, 3)
        Excluded volume interactions. Cached property.
    _HI : ndarray (N, 3)
        Explicit hydrodynamic interactions. Cached property.
    _M : ndarray(N+1, N+1, 3, 3)
        Mobility matrices of the beads
    """
    def __init__(self, Q, h_star):
        self.Q = Q
        self.h_star = h_star
        self.tensions = None
        self.rng = np.random.default_rng()
        self.dW = self.stochastic_force()
        self.dQ = None
        self._EV = None
        self._M = None

    def __len__(self):
        """Number of links in the chain"""
        return len(self.Q)

    def stochastic_force(self):
        """Draw random force. Noise filtering if applicable. Noise variance
        proportional to bead size."""

        # Uncorrelated forces on beads...
        Prior = self.rng.standard_normal((len(self)+1, 3))

        # Correlation due to HI
        if self.h_star > 0:
            N = len(self)
            Gamma = self.M.transpose((0, 2, 1, 3)).reshape((3*(N+1), 3*(N+1)))
            try:
                B = np.linalg.cholesky(Gamma)
            except LinAlgError:
                vals = np.linalg.eigvalsh(Gamma)
                print(np.min(vals), np.max(vals))
                import matplotlib.pyplot as plt
                plt.matshow(Gamma)
                plt.show()
                exit()
                exit()
            P2 = B @ Prior.reshape((3*(N+1),))
            Prior = P2.reshape((N+1, 3))

        if FILTER_NOISE:
            # -- Noise reduction:
            # For inner beads, keep component that is only normal to both links
            # Note that vector products of aligned beads can be close to zero,
            # therefore removing scalar products is preferred.
            Prior[1:-1] = (Prior[1:-1]
                           - (np.sum(Prior[1:-1]*self.Q[1:], axis=1)[:, None]
                              * self.Q[1:])
                           - (np.sum(Prior[1:-1]*self.Q[:-1], axis=1)[:, None]
                              * self.Q[:-1])
                           )
            # For start and end, just normal to the link
            Prior[0] = Prior[0] - np.sum(Prior[0]*self.Q[0])*self.Q[0]
            Prior[-1] = Prior[-1] - np.sum(Prior[-1]*self.Q[-1])*self.Q[-1]

        # ...gives correlated Brownian force on rods:
        return Prior[1:] - Prior[:-1]

    @classmethod
    def from_normal_distribution(cls, n_links, h_star):
        """Initialise a Kramers chain as a collection of vectors drawn from a
        normal distribution and rescaled to a unit vector.
        Parameters
        ----------
        n_links : int
            Number of links in the chain.

        Returns
        -------
        KramersChain object
            Random Kramers chain.
        """
        Q = np.random.standard_normal((n_links, 3))
        norms = np.sqrt(np.sum(Q**2, axis=1))
        Q = Q/norms[:, None]
        return cls(Q, h_star)

    @property
    def coordinates(self):
        """Compute coordinates of the beads R0, ...R(N+1), usually for
        plotting.
        The molecule is centered at origin."""
        R = np.vstack(([[0, 0, 0]], np.cumsum(self.Q, axis=0)))
        R = R - np.average(R, axis=0)
        return R

    @property
    def REE(self):
        """End-to-end vector"""
        return np.sum(self.Q, axis=0)

    @property
    def EV(self):
        """Computes excluded volume interactions. Cached property."""
        if self._EV is None:
            X = self.coordinates
            # Interactions between beads
            self._EV = np.zeros_like(X)
            # # Loop over pairs of beads:
            for i in range(len(self)+1):
                for j in range(i+1, len(self)+1):
                    R = X[i]-X[j]
                    drift = fromGaussianPotential(R, height=2., sigma=1.)
                    self._EV[i] += drift
                    self._EV[j] -= drift
        return self._EV

    @property
    def M(self):
        """Compute mobility matrices. Cached property."""
        if self._M is None:
            X = self.coordinates
            N = len(self)
            self._M = np.empty((N+1, N+1, 3, 3))
            for i in range(N+1):
                self._M[i, i] = np.eye(3)
                for j in range(i+1, N+1):
                    R = X[i]-X[j]
                    S = RotnePragerYamakawa(R, h_star=self.h_star)
                    self._M[i, j] = S
                    self._M[j, i] = S
        return self._M

    def solve(self, gradU, dt):
        """Solve tension according to current random forces and constraints.

        Parameters
        ----------
        gradU : (3, 3) ndarray
            Velocity gradient, dvj/dxi convention.
        dt : float
            Time step.
        """
        if self.h_star == 0.:
            self.solve_free_draining(gradU, dt)
            return

        N = len(self)

        # Right hand side
        dW = np.sqrt(2*dt)*self.dW
        Q_gradU = self.Q @ gradU
        Q_gradU_Q = np.sum(Q_gradU*self.Q, axis=1)
        dW_dot_Q = np.sum(dW*self.Q, axis=1)
        # Exculded volume
        M2 = self.M[1:] - self.M[:-1]
        rodEV = (M2.transpose((0, 2, 1, 3)).reshape((3*N, 3*(N+1)))
                 @ self.EV.reshape((3*(N+1),)))
        rodEV = rodEV.reshape((N, 3))
        EV_dot_Q = np.sum(rodEV*self.Q, axis=1)
        # Assemble right-hand-side
        RHS0 = Q_gradU_Q + EV_dot_Q + dW_dot_Q/dt
        RHS = RHS0.copy()

        # Dense matrix
        # Note: `M_Q` is not symmetric while `a` is.
        a = np.empty((N, N))
        M_Q = np.empty((N, N, 3))

        for i in range(N):
            for j in range(N):
                M_Q[i, j] = (self.M[i, j]
                             + self.M[i+1, j+1]
                             - self.M[i+1, j]
                             - self.M[i, j+1]
                             ) @ self.Q[j]
                a[i, j] = M_Q[i, j] @ self.Q[i]

        dQ = np.empty_like(self.Q)

        for i in range(MAXITER):
            with np.errstate(over='raise', invalid='raise'):
                try:
                    # Solve the system, see NOTES
                    tensions = dposv(a, RHS)[1]

                    for k in range(N):
                        dQ[k] = dt*(Q_gradU[k] + rodEV[k]
                                    - np.sum(tensions[:, None]*M_Q[k], axis=0)
                                    ) + dW[k]

                    # Compute new chain
                    new_Q = self.Q + dQ

                    # Compute rod square lengths
                    L = np.sum(new_Q**2, axis=1)
                    err = np.max(np.abs(L-1))
                    # update the right-hand-side
                    RHS = RHS0 + 0.5/dt*np.sum(dQ**2, axis=1)
                except FloatingPointError:
                    raise ValueError('dptsv convergence failed.')
            if err < LENGTH_TOL:
                # Re normalise and exit loop
                new_Q = new_Q/np.sqrt(L[:, None])
                dQ = new_Q - self.Q
                break
            elif i == MAXITER - 1:
                raise ValueError(f"Could not converge in {MAXITER} "
                                 "iterations.")

        self.dQ = dQ
        self.tensions = tensions

    def measure(self):
        """Measure quantities from the systems.

        Returns
        -------
        observables : dict
            Dictionary of observables quantities.
        """
        if self.tensions is None:
            raise RuntimeError("Attempt to measure tension but tension not "
                               "solved.")
        # Molecurlar conformation tensor
        A = np.outer(self.REE, self.REE)
        # Molecular stress
        # Row-wise tensor dot:
        moments = self.tensions[:, None, None]*(self.Q[:, :, None]
                                                * self.Q[:, None, :])
        S = np.sum(moments, axis=0)
        observables = {'A': A, 'S': S}
        return observables

    def evolve(self, **kwargs):
        """Evolve chain by a time step dt. Itô calculus convention.
        Reset random forces.

        Parameters
        ----------
        gradU : (3, 3) ndarray
            Velocity gradient, dvj/dxi convention.
        dt : float
            Time step.
        """
        self.Q += self.dQ

        self.tensions = None
        self._M = None
        # stochastic_force will trigger the calculation of updated self.M
        self.dW = self.stochastic_force()
        if kwargs['first_subit']:
            self._EV = None

    def save_vtk(self, file_name):
        """Save the molecule in vtk 3d format. File can then be imported in
        Paraview.

        Parameters
        ----------
        file_name : str
            Name of file.
        """
        out = []
        # Header
        out.append("# vtk DataFile Version 4.0")
        out.append("Polymer chain")
        out.append("ASCII")
        out.append("DATASET POLYDATA")
        out.append("")

        # Points
        out.append(f"POINTS {len(self)+1} double")
        for x, y, z in self.coordinates:
            out.append(f"{x:e}\t{y:e}\t{z:e}")
        out.append("")

        # Lines
        out.append(f"LINES 1 {len(self)+2}")
        lines = [str(len(self)+1)]
        lines += [str(i) for i in range(len(self)+1)]
        out.append(" ".join(lines))
        out.append("")

        # Bead size (we need this to initialise FIELD on points)
        out.append(f"POINT_DATA {len(self)+1}")
        out.append("SCALARS bead_size double")
        out.append("LOOKUP_TABLE default")
        out.append(" ".join(["1.0"]*(len(self)+1)))
        out.append("")

        # Get number of fields
        if self.tensions is None:
            n_fields = 1
        else:
            n_fields = 2
        out.append(f"FIELD observables {n_fields}")

        if self.tensions is not None:
            # Tensions
            out.append(f"tension 1 {len(self)+1} float")
            out.append(f"{self.tensions[0]}")
            for i in range(len(self)-1):
                out.append(f"{(self.tensions[i] + self.tensions[i+1])/2}")
            out.append(f"{self.tensions[-1]}")
            out.append("")

        # Exculed volume
        out.append(f"excluded_volume 3 {len(self)+1} float")
        for ev in self.EV:
            out.append(f"{ev[0]} {ev[1]} {ev[2]}")
        out.append("")

        # HI
        # out.append(f"hydrodynamic 3 {len(self)+1} float")
        # for hi in self.HI:
        #     out.append(f"{hi[0]} {hi[1]} {hi[2]}")
        # out.append("")

        # Make it a file
        content = '\n'.join(out)
        with open(file_name, 'w') as f:
            f.write(content)

    def solve_free_draining(self, gradU, dt):
        """Solve tension without hydrodynamic interaction.

        Parameters
        ----------
        gradU : (3, 3) ndarray
            Velocity gradient, dvj/dxi convention.
        dt : float
            Time step.

        Notes
        -----
        Scipy `linalg.lapack.dptsv` which does partial Gauss pivoting seems
        to yield at least similar speed as compiled tridigonal algorithm
        from 'Numerical Recipes' tridiagonal solver, which doesn't do pivoting
        and requires smaller time step.
        """
        # Right hand side
        dW = np.sqrt(2*dt)*self.dW
        Q_gradU = self.Q @ gradU
        Q_gradU_Q = np.sum(Q_gradU*self.Q, axis=1)
        dW_dot_Q = np.sum(dW*self.Q, axis=1)
        rodEV = self.EV[1:]-self.EV[:-1]
        EV_dot_Q = np.sum(rodEV*self.Q, axis=1)
        RHS0 = Q_gradU_Q + EV_dot_Q + dW_dot_Q/dt

        # Tridiagonal matrix
        # Low diagonal elements & upper diagonal elements
        dlu = - np.sum(self.Q[1:]*self.Q[:-1], axis=1)
        # Diagonal elements
        d = np.ones(len(self.Q))*2.

        RHS = RHS0.copy()

        dQ = np.zeros_like(self.Q)

        for i in range(MAXITER):
            with np.errstate(over='raise', invalid='raise'):
                try:
                    # Solve the system, see NOTES
                    tensions = dptsv(d, dlu, RHS)[2]

                    dQ[0] = dt*(Q_gradU[0] + rodEV[0] + tensions[1]*self.Q[1]
                                - 2*tensions[0]*self.Q[0]
                                ) + dW[0]

                    dQ[1:-1] = dt*(Q_gradU[1:-1] + rodEV[1:-1]
                                   + tensions[2:, None]*self.Q[2:]
                                   - 2*tensions[1:-1, None]*self.Q[1:-1]
                                   + tensions[:-2, None]*self.Q[:-2]
                                   ) + dW[1:-1]

                    dQ[-1] = dt*(Q_gradU[-1] + rodEV[-1]
                                 - 2*tensions[-1]*self.Q[-1]
                                 + tensions[-2]*self.Q[-2]
                                 ) + dW[-1]

                    # Compute new chain
                    new_Q = self.Q + dQ

                    # Compute rod square lengths
                    L = np.sum(new_Q**2, axis=1)
                    err = np.max(np.abs(L-1))
                    # update the right-hand-side
                    RHS = RHS0 + 0.5/dt*np.sum(dQ**2, axis=1)
                except FloatingPointError:
                    raise ValueError('dptsv convergence failed.')
            if err < LENGTH_TOL:
                # Re normalise and exit loop
                new_Q = new_Q/np.sqrt(L[:, None])
                dQ = new_Q - self.Q
                break
            elif i == MAXITER - 1:
                raise ValueError(f"Could not converge in {MAXITER} "
                                 "iterations.")
        self.dQ = dQ
        self.tensions = tensions


def LennardJones(R, depth=1., radius=1.):
    """Drift due to a Lennard Jones potential.

    Parameters
    ----------
    R : ndarray (3,)
        Distance between beads.
    depth : float, default 1.
        Dimensionless depth of the potential.
    radius : float, default 1.
        Dimensionless radius of the potential (distance of zero energy).

    Returns
    -------
    ndarray (3.)
        Dimensionless drift"""
    R2 = np.sum(R**2)
    if R2 > radius**2 and R2 < 4.:
        return 24*depth*(2*radius**12/R2**7 - radius**6/R2**4)*R
    elif R2 < radius**2:
        return 24*depth/radius*R/np.sqrt(R2)
    else:
        return np.zeros(3)


def fromGaussianPotential(R, height=1., sigma=1.):
    """Drift due to a Gaussian potential.

    Parameters
    ----------
    R : ndarray (3,)
        Distance between beads.
    height : float, default 1.
        Dimensionless height of the potential. Positive for repulsive force.
    sigma : float, default 1.
        Dimensionless standard deviation of the potential.

    Returns
    -------
    ndarray (3,)
        Dimensionless drift"""
    R2 = np.sum(R**2)
    return height/sigma**2*np.exp(-0.5*R2/sigma**2)*R


def RotnePragerYamakawa(R, h_star=0.5641895835477563):
    """Compute Rotne-Prager-Yamakawa mobility tensor.

    Parameters
    ----------
    R : ndarray (3,)
        Distance between beads.
    h_star : float, default 1/np.sqrt(np.pi)
        Strength of hydrodynamic interactions

    Returns
    -------
    ndarray (3, 3)
        Mobility tensor"""
    a = 1.7724538509055159*h_star
    R2 = np.sum(R**2)
    normR = np.sqrt(R2)
    if normR < 1e-6:
        return np.eye(3)
    elif normR < 2*a:
        return ((1-9*normR/(32*a))*np.eye(3)
                + 3./(32*a*normR)*np.outer(R, R))
    else:
        return (0.75*a/normR*((1+2*a**2/(3*R2))*np.eye(3)
                              + (1-2*a**2/R2)*np.outer(R, R)/R2))
