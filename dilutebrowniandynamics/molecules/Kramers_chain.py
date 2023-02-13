import numpy as np
from scipy.linalg.lapack import dptsv
from ..simulate import ConvergenceError
from numba import jit

LENGTH_TOL = 1e-6
MAXITER = 100
FILTER_NOISE = True
BROWNIAN_FORCES = False
STICKY_TOL = 0
UNIAXIAL = True


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
    def __init__(self, Q, rng):
        self.Q = Q
        self.tensions = None
        self.rng = rng
        self.dW = self.stochastic_force()
        self.dQ = None
        self.EV = excluded_volume(Q)

    def __len__(self):
        """Number of links in the chain"""
        return len(self.Q)

    def stochastic_force(self):
        """Draw random force. Noise filtering if applicable. Noise variance
        proportional to bead size."""

        if not BROWNIAN_FORCES:
            return 0

        # Uncorrelated forces on beads...
        Prior = self.rng.standard_normal((len(self)+1, 3))

        if FILTER_NOISE:
            # -- Noise reduction:
            # For inner beads, keep component that is only normal to both links
            # Note that vector products of aligned beads can be close to zero,
            # therefore removing scalar products is preferred.
            Prior = filter(Prior, self.Q)

        # ...gives correlated Brownian force on rods:
        return Prior[1:] - Prior[:-1]

    @classmethod
    def from_normal_distribution(cls, n_links, seed=np.random.SeedSequence()):
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
        rng = np.random.default_rng(seed)
        Q = rng.standard_normal((n_links, 3))
        norms = np.sqrt(np.sum(Q**2, axis=1))
        Q = Q/norms[:, None]
        return cls(Q, rng)

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

        # dQ without internal tensions
        dQ0 = dt*(self.Q @ gradU + self.EV) + np.sqrt(2*dt)*self.dW

        # Right hand side
        # ---------------
        RHS0 = np.sum(dQ0*self.Q, axis=1)/dt
        RHS = RHS0.copy()

        # Tridiagonal matrix
        # ------------------
        # Low diagonal elements & upper diagonal elements
        dlu = - np.sum(self.Q[1:]*self.Q[:-1], axis=1)
        # Diagonal elements
        d = np.ones(len(self.Q))*2.

        for i in range(MAXITER):
            with np.errstate(over='raise', invalid='raise'):
                try:
                    # Solve the system, see NOTES
                    tensions = dptsv(d, dlu, RHS)[2]

                    dQ = build_dQ(dQ0, dt, self.Q, tensions)

                    # Compute new chain
                    new_Q = self.Q + dQ
                    if STICKY_TOL:
                        T = np.abs(new_Q[:,0]) > STICKY_TOL
                        new_Q[:,0] *= T
                        #for j, (p1, p2) in enumerate(zip(dlu[:-1], dlu[1:])):
                        #    if p1 < 0 and p2 <0:
                        #        new_Q[j+1][np.abs(new_Q[j+1]) < STICKY_TOL] = 0

                    # Compute rod square lengths
                    L = np.sum(new_Q**2, axis=1)
                    err = np.max(np.abs(L-1))

                    # update the right-hand-side
                    RHS = RHS0 + 0.5/dt*np.sum(dQ**2, axis=1)
                except FloatingPointError:
                    raise ConvergenceError('dptsv convergence failed.')
            if err < LENGTH_TOL:
                # Re normalise and exit loop
                new_Q = new_Q/np.sqrt(L[:, None])
                dQ = new_Q - self.Q
                break
            elif i == MAXITER - 1:
                raise ConvergenceError(f"Could not converge in {MAXITER} "
                                       "iterations.")
        self.dQ = dQ
        self.tensions = tensions

    def solve1(self, gradU, dt):
        """Solve tension according to current random forces and constraints.
        Optimisation method

        Parameters
        ----------
        gradU : (3, 3) ndarray
            Velocity gradient, dvj/dxi convention.
        dt : float
            Time step.

        Notes
        -----
        Optimisation method. Use a pseudo-Newton algorithm where only the
        diagonal of the Hessian is computed.
        """

        # dQ without internal tensions
        dQ0 = dt*(self.Q @ gradU) + np.sqrt(2*dt)*self.dW

        # Inital Tensions
        tensions = np.zeros(len(self))

        dQ = build_dQ(dQ0, dt, self.Q, tensions)

        # Update Q
        new_Q = self.Q + dQ

        for i in range(MAXITER):
            # Cost function
            f = np.sum((np.sum(new_Q**2, axis=1)-1.)**2)
            print(i, f)

            if f < LENGTH_TOL:

                # Re normalise and exit loop
                new_Q = new_Q/np.sqrt(np.sum(new_Q**2, axis=1)[:, None])
                dQ = new_Q - self.Q
                break
            elif i == MAXITER - 1:
                raise ConvergenceError(f"Could not converge in {MAXITER} "
                                       "iterations.")

            # Gradient (normalised by dt)
            df = build_df(self.Q, new_Q)

            # Diag Hessian (normalised by dt²)
            d2f = build_d2f(self.Q, new_Q)

            # Update tensions
            tensions = tensions - 0.01*df/d2f/dt

            dQ = build_dQ(dQ0, dt, self.Q, tensions)

            # Update Q
            new_Q = self.Q + dQ

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
        g = self.tensions
        observables = {'A': A, 'S': S, 'g': g}
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

        if UNIAXIAL:
            # Hook to perfectly align stretched molecules
            R = np.sqrt(np.sum(self.REE**2))
            if R > len(self) - 0.1:
                # print('Elongated')
                self.Q = np.zeros_like(self.Q)
                self.Q[:, 0] = 1.

        # draw new random forces
        self.tensions = None
        self.dW = self.stochastic_force()
        self.EV = excluded_volume(self.Q)

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


@jit(nopython=True)
def filter(Prior, Q):
    """Filter forces to keep only orthogonal part"""
    # Numpy implementation
    # --------------------
    # forces[1:-1] = (forces[1:-1]
    #                 - (np.sum(forces[1:-1]*Qs[1:], axis=1)[:, None]*Qs[1:])
    #                 - (np.sum(forces[1:-1]*Qs[:-1], axis=1)[:, None]*Qs[:-1])
    #                 )
    # # For start and end, just normal to the link
    # forces[0] = forces[0] - np.sum(forces[0]*Qs[0])*Qs[0]
    # forces[-1] = forces[-1] - np.sum(forces[-1]*Qs[-1])*Qs[-1]

    # Numba
    # -----
    Prior[0] = Prior[0] - np.sum(Prior[0]*Q[0])*Q[0]/np.sum(Q[0]**2)

    cross = np.cross(Q[:-1], Q[1:])
    for i in range(len(cross)):
        c2 = np.sum(cross[i]**2)
        if c2 > LENGTH_TOL:
            Prior[i+1] = np.sum(Prior[i+1]*cross[i])*cross[i]/c2
        else:
            Prior[i+1] = (Prior[i+1]
                          - np.sum(Prior[i+1]*Q[i])*Q[i]/np.sum(Q[i]**2)
                          )

    Prior[-1] = Prior[-1] - np.sum(Prior[-1]*Q[-1])*Q[-1]/np.sum(Q[-1]**2)

    return Prior


@jit(nopython=True)
def build_dQ(dQ0, dt, Q, tensions):
    """Compiled version to build dQ"""
    # Numpy implementation
    # dQ = np.empty_like(self.Q)
    # dQ[0] = dt*(Q_gradU[0] + tensions[1]*self.Q[1]
    #             - 2*tensions[0]*self.Q[0]
    #             ) + dW[0]
    #
    # dQ[1:-1] = dt*(Q_gradU[1:-1]
    #                + tensions[2:, None]*self.Q[2:]
    #                - 2*tensions[1:-1, None]*self.Q[1:-1]
    #                + tensions[:-2, None]*self.Q[:-2]
    #                ) + dW[1:-1]
    #
    # dQ[-1] = dt*(Q_gradU[-1] - 2*tensions[-1]*self.Q[-1]
    #              + tensions[-2]*self.Q[-2]
    #              ) + dW[-1]
    dQ = np.empty_like(Q)
    dQ[0] = dQ0[0] + dt*(tensions[1]*Q[1] - 2*tensions[0]*Q[0])
    for i in range(1, len(dQ)-1):
        dQ[i] = dQ0[i] + dt*(tensions[i-1]*Q[i-1]
                             - 2*tensions[i]*Q[i]
                             + tensions[i+1]*Q[i+1])
    dQ[-1] = dQ0[-1] + dt*(- 2*tensions[-1]*Q[-1]
                           + tensions[-2]*Q[-2])
    return dQ


@jit(nopython=True)
def build_df(Q, new_Q):
    """Gradient of the cost function normalised by dt. See optim.lyx"""

    df = np.empty(len(Q))
    df[0] = 4*(-2*(np.sum(new_Q[0]**2)-1.)*np.sum(new_Q[0]*Q[0])
               + (np.sum(new_Q[1]**2)-1.)*np.sum(new_Q[1]*Q[0])
               )
    for i in range(1, len(Q)-1):
        df[i] = 4*((np.sum(new_Q[i-1]**2)-1.)*np.sum(new_Q[i-1]*Q[i])
                   - 2*(np.sum(new_Q[i]**2)-1.)*np.sum(new_Q[i]*Q[i])
                   + (np.sum(new_Q[i+1]**2)-1.)*np.sum(new_Q[i+1]*Q[i])
                   )
    df[-1] = 4*((np.sum(new_Q[-2]**2)-1.)*np.sum(new_Q[-2]*Q[-1])
                - 2*(np.sum(new_Q[-1]**2)-1.)*np.sum(new_Q[-1]*Q[-1])
                )
    return df


@jit(nopython=True)
def build_d2f(Q, new_Q):
    """Gradient of the cost function normalised by dt. See optim.lyx"""

    d2f = np.empty(len(Q))
    d2f[0] = 4*(4*(2*np.sum(new_Q[0]*Q[0])**2 + np.sum(new_Q[0]**2) - 1.)
                + 2*np.sum(new_Q[1]*Q[0])**2 + np.sum(new_Q[1]**2) - 1.
                )
    for i in range(1, len(Q)-1):
        d2f[i] = 4*(2*np.sum(new_Q[i-1]*Q[i])**2 + np.sum(new_Q[i-1]**2) - 1.
                    + 4*(2*np.sum(new_Q[i]*Q[i])**2 + np.sum(new_Q[i]**2) - 1.)
                    + 2*np.sum(new_Q[i+1]*Q[i])**2 + np.sum(new_Q[i+1]**2) - 1.
                    )
    d2f[-1] = 4*(2*np.sum(new_Q[-2]*Q[-1])**2 + np.sum(new_Q[-2]**2) - 1.
                 + 4*(2*np.sum(new_Q[-1]*Q[-1])**2 + np.sum(new_Q[-1]**2) - 1.)
                 )
    return d2f

@jit(nopython=True)
def excluded_volume(Q):
    """Excluded volume force only with neighbours. Avoid stuck folds"""
    # Distance vector between bead i and i+2
    R = Q[:-1] + Q[1:]
    R2 = np.sum(R**2, axis=1)
    F = np.zeros_like(R)
    for i in range(len(F)):
        if R2[i] < 1.:
            F[i] = -R[i]/R2[i]

    EV = np.zeros_like(Q)
    EV[0] = F[1] - F[0]
    EV[1] = F[2] - F[0] - F[1]
    for i in range(2, len(Q)-1):
        EV[i] = F[i+1] - F[i-1] - (F[i] - F[i-2])
    EV[-2] = - F[-2] - (F[-1] - F[-3])
    EV[-1] = - F[-1] - (- F[-2])

    return np.zeros_like(Q)
