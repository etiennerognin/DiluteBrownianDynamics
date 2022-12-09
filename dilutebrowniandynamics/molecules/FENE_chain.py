import numpy as np
from numba import jit
from scipy.linalg.lapack import dptsv

from ..simulate import ConvergenceError

LENGTH_TOL = 1e-6
MAXITER = 100


class FENEChain:
    """A FENE chain is a chain of N FENE vectors linking N+1 Brownian beads.
    The tension in each link is given by the FENE force. All the links
    have the same maximum extension.

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
    L_max : float
        Maximum length for a link.
    """
    def __init__(self, Q, rng, L_max):
        self.Q = Q
        self.L_max = L_max
        self.H = None
        self.tensions = None
        self.rng = rng
        self.dW = self.stochastic_force()
        self.dQ = None

    def __len__(self):
        """Number of links in the chain"""
        return len(self.Q)

    def stochastic_force(self):
        """Draw random force. Noise filtering if applicable. Noise variance
        proportional to bead size."""
        # Uncorrelated forces on beads...
        Prior = self.rng.standard_normal((len(self)+1, 3))
        # ...gives correlated Brownian force on rods:
        return Prior[1:] - Prior[:-1]

    @classmethod
    def from_normal_distribution(cls, n_links, L_max,
                                 seed=np.random.SeedSequence()):
        """Initialise a FENE chain as a collection of vectors drawn from a
        normal distribution of variance 1. Vectors longer than L_max are
        discarded. Note that the equilibrium length is of the order of 3
        (not 1).

        Parameters
        ----------
        n_links : int
            Number of links in the chain.

        Returns
        -------
        FENEChain object
            Random FENE chain.
        """
        rng = np.random.default_rng(seed)
        Q = rng.standard_normal((n_links, 3))
        norms = np.sqrt(np.sum(Q**2, axis=1))
        # Cleaning pass
        for i, L in enumerate(norms):
            while L > L_max - LENGTH_TOL:
                Q[i] = rng.standard_normal(3)
                L = np.sqrt(np.sum(Q[i]**2))

        return cls(Q, rng, L_max)

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
        """Solve tension according to current random forces and constraints."""
        Q2 = np.sum(self.Q**2, axis=1)
        self.H = self.L_max**2/(self.L_max**2-Q2)

        # dQ without internal tensions
        dQ0 = dt*(self.Q @ gradU) + np.sqrt(dt/2)*self.dW

        # Complete
        dQ = build_dQ(dQ0, dt, self.Q, self.H)

        new_Q = self.Q + dQ

        n_iter = 0
        while any(np.sum(new_Q**2, axis=1) > self.L_max**2 - LENGTH_TOL):
            # print(f"Switching to rigid constraint, iteration {n_iter}. dt={dt}")
            Q_star = self.Q.copy()

            if n_iter > MAXITER:
                raise ConvergenceError('Molecule length exceeded L_max.')
            n_iter += 1

            with np.errstate(over='raise', invalid='raise'):
                try:

                    # Freeze length and switch to rigid constraints, L2 is the
                    # length
                    # of links after the evolution step, therefore it should be
                    # contrained.
                    L2 = np.minimum(np.sum(new_Q**2, axis=1), self.L_max**2)

                    # Right hand side
                    # ---------------
                    RHS = 4*np.sum(dQ0*self.Q, axis=1)/dt
                    RHS += - 2*(L2-Q2-np.sum(dQ**2, axis=1))/dt

                    # Tridiagonal matrix
                    # ------------------
                    # Low diagonal elements & upper diagonal elements
                    dlu = - np.sum(Q_star[1:]*Q_star[:-1], axis=1)
                    # Diagonal elements
                    d = 2*Q2

                    # Solve the system, see NOTES

                    g_by_Q = dptsv(d, dlu, RHS)[2]

                    # New tension means rescaling Q
                    Q_star = Q_star/np.sqrt(Q2)[:, None]
                    Q2 = self.L_max**2*(1.-1./g_by_Q)
                    if any(Q2 < 0) or any(np.isnan(Q2)):
                        raise ConvergenceError('dptsv convergence failed.')

                    Q_star = np.sqrt(Q2)[:, None]*Q_star

                    # Update dQ
                    self.H = g_by_Q
                    dQ0 = dt*(Q_star @ gradU) + np.sqrt(dt/2)*self.dW
                    dQ = build_dQ(dQ0, dt, Q_star, self.H)

                    # Compute new chain
                    new_Q = Q_star + dQ
                    # print(f"Maximum length: {np.amax(np.sum(new_Q**2, axis=1))}")
                except FloatingPointError:
                    raise ConvergenceError('dptsv convergence failed.')
        if n_iter:
            self.Q = Q_star
        self.dQ = dQ
        self.tensions = self.H*np.sqrt(Q2)

    def solve_rigid(self, gradU, dt, L2):
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
        # Corresponding length
        L = np.sqrt(L2)

        # dQ without internal tensions (different scaling from Kramers chain)
        dQ0 = dt*(self.Q @ gradU) + np.sqrt(dt/2)*self.dW

        # Right hand side
        # ---------------
        RHS0 = np.sum(dQ0*self.Q, axis=1)/dt
        RHS = RHS0.copy()

        # Tridiagonal matrix
        # ------------------
        # Low diagonal elements & upper diagonal elements
        dlu = - np.sum(self.Q[1:]*self.Q[:-1], axis=1)
        # Diagonal elements
        d = L2

        for i in range(MAXITER):
            with np.errstate(over='raise', invalid='raise'):
                try:
                    # Solve the system, see NOTES
                    g_by_Q = dptsv(d, dlu, RHS)[2]

                    # New tension means rescaling Q
                    Q2 = self.L_max**2*(1.-1./g_by_Q)
                    self.Q = np.sqrt(Q2)/L*self.Q

                    dQ = build_dQ(dQ0, dt, self.Q, g_by_L)

                    # Compute new chain
                    new_Q = self.Q + dQ

                    # Compute rod square lengths
                    new_Q2 = np.sum(new_Q**2, axis=1)
                    # And compare to target
                    err = np.max(np.abs(new_Q2-L2))
                    # update the right-hand-side
                    RHS = RHS0 + 0.5/dt*np.sum(dQ**2, axis=1)
                except FloatingPointError:
                    if FLAG:
                        print(f'dptsv convergence failed with dt={dt}.')
                    raise ConvergenceError('dptsv convergence failed.')
            if err < LENGTH_TOL:
                # Re normalise and exit loop
                new_Q = L[:, None]*new_Q/np.sqrt(new_Q2[:, None])
                dQ = new_Q - self.Q
                break
            elif i == MAXITER - 1:
                if FLAG:
                    print(f"Could not converge in {MAXITER} "
                          "iterations.")
                raise ConvergenceError(f"Could not converge in {MAXITER} "
                                       "iterations.")
        self.dQ = dQ
        self.tensions = g_by_L*L
        if self.avg_tensions is None:
            self.avg_tensions = dt*self.tensions
            self.subit = dt
        else:
            self.avg_tensions += dt*self.tensions
            self.subit += dt

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
        moments = self.H[:, None, None]*(self.Q[:, :, None]
                                         * self.Q[:, None, :])
        S = np.sum(moments, axis=0)

        g_max = np.amax(self.tensions)
        i_max = np.argmax(self.tensions)
        g_centre = self.tensions[len(self)//2]

        observables = {'A': A, 'S': S, 'g_max': g_max, 'i_max': i_max,
                       'g_centre': g_centre}
        return observables

    def evolve(self, **kwargs):
        """Evolve dumbbell by a time step dt. Itô calculus convention.

        Parameters
        ----------
        gradU : (3, 3) ndarray
            Velocity gradient, dvj/dxi convention.
        dt : float
            Time step.
        """
        self.Q += self.dQ
        # draw new random forces
        self.tension = None
        self.dW = self.stochastic_force()

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
def build_dQ(dQ0, dt, Q, H):
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
    dQ[0] = dQ0[0] + dt*(0.25*H[1]*Q[1] - 0.5*H[0]*Q[0])
    for i in range(1, len(dQ)-1):
        dQ[i] = dQ0[i] + dt*(0.25*H[i-1]*Q[i-1]
                             - 0.5*H[i]*Q[i]
                             + 0.25*H[i+1]*Q[i+1])
    dQ[-1] = dQ0[-1] + dt*(- 0.5*H[-1]*Q[-1]
                           + 0.25*H[-2]*Q[-2])
    return dQ
