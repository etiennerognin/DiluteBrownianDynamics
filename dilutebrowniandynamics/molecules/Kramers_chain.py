import numpy as np
from scipy.linalg.lapack import dptsv


LENGTH_TOL = 1e-6
MAXITER = 1000
FILTER_NOISE = True


class KramersChain:
    """A Kramers chain, also known as bead-rod model, is a chain of N rigid
    vectors linking N+1 Brownian beads. The tension in each link is given by
    the rigidity constraint. The length of each link 1, so that we can expect
    an equilibrium average end-to-end length of sqrt(N) and a contour length
    (full extension) of N.

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
    """
    def __init__(self, Q):
        self.Q = Q
        self.tensions = None
        self.rng = np.random.default_rng()

        # Uncorrelated forces on beads...
        Prior = self.rng.standard_normal((len(Q)+1, 3))

        if FILTER_NOISE:
            # -- Noise reduction:
            # For inner beads, keep component that is only normal to both links
            vN = np.cross(self.Q[1:], self.Q[:-1])
            vN2 = np.sum(vN**2, axis=1)
            Prior[1:-1] = (np.sum(Prior[1:-1]*vN, axis=1)[:, None]*vN
                           / vN2[:, None])
            # For start and end, just normal to the link
            Prior[0] = Prior[0] - (np.sum(Prior[0]*self.Q[0]) * self.Q[0])
            Prior[-1] = Prior[-1] - (np.sum(Prior[-1]*self.Q[-1]) * self.Q[-1])

        # ...gives correlated Brownian force on rods:
        self.dW = Prior[1:] - Prior[:-1]
        self.dQ = None

    def __len__(self):
        """Number of links in the chain"""
        return len(self.Q)

    @classmethod
    def from_normal_distribution(cls, n_links):
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
        return cls(Q)

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

    def solve(self, gradU, dt):
        """Solve tension according to current random forces and constraints.

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
        RHS0 = Q_gradU_Q + dW_dot_Q/dt

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

                    dQ[0] = dt*(Q_gradU[0] + tensions[1]*self.Q[1]
                                - 2*tensions[0]*self.Q[0]
                                ) + dW[0]

                    dQ[1:-1] = dt*(Q_gradU[1:-1]
                                   + tensions[2:, None]*self.Q[2:]
                                   - 2*tensions[1:-1, None]*self.Q[1:-1]
                                   + tensions[:-2, None]*self.Q[:-2]
                                   ) + dW[1:-1]

                    dQ[-1] = dt*(Q_gradU[-1] - 2*tensions[-1]*self.Q[-1]
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

    def evolve(self, gradU, dt):
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
        # draw new random forces
        self.tensions = None
        # Uncorrelated forces on beads...
        Prior = self.rng.standard_normal((len(self.Q)+1, 3))

        if FILTER_NOISE:
            # -- Noise reduction:
            # For inner beads, keep component that is only normal to both links
            vN = np.cross(self.Q[1:], self.Q[:-1])
            vN2 = np.sum(vN**2, axis=1)
            Prior[1:-1] = (np.sum(Prior[1:-1]*vN, axis=1)[:, None]*vN
                           / vN2[:, None])
            # For start and end, just normal to the link
            Prior[0] = Prior[0] - (np.sum(Prior[0]*self.Q[0]) * self.Q[0])
            Prior[-1] = Prior[-1] - (np.sum(Prior[-1]*self.Q[-1]) * self.Q[-1])

        # ...gives correlated Brownian force on rods:
        self.dW = Prior[1:] - Prior[:-1]

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
        if self.tensions is not None:
            # Tensions
            out.append(f"POINT_DATA {len(self)+1}")
            out.append("SCALARS tension double")
            out.append("LOOKUP_TABLE default")
            out.append(f"{self.tensions[0]}")
            for i in range(len(self)-1):
                out.append(f"{(self.tensions[i] + self.tensions[i+1])/2}")
            out.append(f"{self.tensions[-1]}")
        content = '\n'.join(out)
        with open(file_name, 'w') as f:
            f.write(content)
