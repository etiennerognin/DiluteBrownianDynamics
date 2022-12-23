import numpy as np
from numba import jit
from scipy.linalg.lapack import dptsv

from ..simulate import ConvergenceError

DELTA_RIGID = 1e-2
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
            while L > L_max - DELTA_RIGID:
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
        # All links are coiled, explicit solution should be safe
        self.H = self.L_max**2/(self.L_max**2-Q2)

        # Test chain lengths, True if link is close to coiled state
        T = Q2 < (self.L_max - DELTA_RIGID)**2

        if all(T):
            # All links are coiled, explicit solution should be safe

            # dQ without internal tensions
            dQ0 = dt*(self.Q @ gradU) + np.sqrt(dt/2)*self.dW

            # Complete
            dQ = build_dQ(dQ0, dt, self.Q, self.H)

            new_Q = self.Q + dQ

            # Trigger adaptive time step if L_max is breached
            if any(np.sum(new_Q**2, axis=1) > (self.L_max - LENGTH_TOL)**2):
                raise ConvergenceError('Molecule length exceeded L_max.')

            self.dQ = dQ
            self.tensions = self.H*np.sqrt(Q2)

        else:
            print(f'Switch to rigid, dt={dt}')
            with np.errstate(over='raise', invalid='raise'):
                # Some links are extended and should be treated as rigid links.
                try:
                    dQ = np.zeros_like(self.Q)
                    Q_star = self.Q.copy()
                    for i in range(MAXITER):
                        # Number of unknowns
                        n_gs = np.sum(~T)

                        # Right hand side
                        # ---------------
                        # For clarity we build the whole vectors and then trim
                        dQ0 = dt*(Q_star @ gradU) + np.sqrt(dt/2)*self.dW
                        RHS = 4*np.sum(dQ0*Q_star, axis=1)/dt
                        RHS += 2*np.sum(dQ**2, axis=1)/dt

                        # Low diagonal elements & upper diagonal elements
                        dlu = - np.sum(Q_star[1:]*Q_star[:-1], axis=1)

                        # Explicit links should be put to RHS
                        RHS[:-1] += - self.H[:-1]*dlu*T[1:]
                        RHS[1:] += - self.H[1:]*dlu*T[:-1]
                        # And remove form Low/Up diagonals
                        dlu = dlu*~T[1:]*~T[:-1]

                        # Diagonal elements
                        d = 2*Q2

                        if n_gs > 1:
                            # Solve the system, see NOTES
                            g_by_Q = dptsv(d[~T], dlu[~T[:-1]][:-1], RHS[~T])[2]
                        else:
                            g_by_Q = RHS[~T]/d[~T]
                        print(self.H[~T])
                        print(g_by_Q)

                        # New tension means rescaling Q for rigid links
                        Q_star[~T] = Q_star[~T]/np.sqrt(Q2[~T])[:, None]
                        Q2[~T] = self.L_max**2*(1.-1./g_by_Q)
                        if any(Q2 < 0) or any(np.isnan(Q2)):
                            print('neg or nan')
                            raise ConvergenceError('dptsv convergence failed.')

                        Q_star[~T] = np.sqrt(Q2[~T])[:, None]*Q_star[~T]

                        # Update dQ
                        self.H = self.L_max**2/(self.L_max**2-Q2)
                        dQ0 = dt*(Q_star @ gradU) + np.sqrt(dt/2)*self.dW
                        dQ = build_dQ(dQ0, dt, Q_star, self.H)

                        # Compute new chain
                        new_Q = Q_star + dQ

                        print(i, np.amax(Q2), np.amax(np.sum(new_Q**2, axis=1)))
                        print(i, np.argmax(Q2), np.argmax(np.sum(new_Q**2, axis=1)))
                        error = np.abs(Q2[~T] - np.sum(new_Q[~T]**2, axis=1))
                        print(error)
                        if error < LENGTH_TOL**2:
                            break

                    if any(np.sum(new_Q**2, axis=1) > (self.L_max - LENGTH_TOL)**2):
                        raise ConvergenceError('Molecule length exceeded L_max.')
                    self.Q = Q_star
                    self.dQ = dQ
                    self.tensions = self.H*np.sqrt(Q2)
                except FloatingPointError:
                    raise ConvergenceError('dptsv convergence failed.')


    def solve_optim(self, gradU, dt):
        """Solve tension according to current random forces. Use gradient
        descent to comply with maximum elongation"""
        Q2 = np.sum(self.Q**2, axis=1)
        self.H = self.L_max**2/(self.L_max**2-Q2)

        # dQ without internal tensions
        dQ0 = dt*(self.Q @ gradU) + np.sqrt(dt/2)*self.dW

        # Complete
        dQ = build_dQ(dQ0, dt, self.Q, self.H)

        new_Q = self.Q + dQ

        n_iter = 0
        # Test constraint vector
        T = np.sum(new_Q**2, axis=1) > self.L_max**2 - LENGTH_TOL

        while any(T):
            print(n_iter, np.sum(T*(np.sum(new_Q**2, axis=1)-self.L_max**2+LENGTH_TOL)))
            if n_iter > MAXITER:
                raise ConvergenceError('Molecule length exceeded L_max.')
            n_iter += 1
            # Do one descent step
            # Gradient (normalised by dt)
            df = build_df(self.Q, new_Q, T)
            #for grad, h in zip(df, self.H):
            #    print(grad, h)

            # Diag Hessian (normalised by dt²)
            d2f = build_d2f(self.Q, T)

            # Update tensions
            self.H = self.H - 0.01*df

            # Update Q
            Q_star = self.Q.copy()
            Q_star = Q_star/np.sqrt(Q2)[:, None]
            Q2 = self.L_max**2*(1.-1./self.H)
            if any(Q2 < 0) or any(np.isnan(Q2)):
                print('Q2 negative')
                raise ConvergenceError('Gradient descent failed.')
            Q_star = np.sqrt(Q2)[:, None]*Q_star

            # Update dQ
            dQ0 = dt*(Q_star @ gradU) + np.sqrt(dt/2)*self.dW
            dQ = build_dQ(dQ0, dt, Q_star, self.H)

            # Compute new chain
            new_Q = Q_star + dQ

            # Update T
            T = np.sum(new_Q**2, axis=1) > self.L_max**2 - LENGTH_TOL

        if n_iter:
            self.Q = Q_star
            print("Optim success")
        self.dQ = dQ
        self.tensions = self.H*np.sqrt(Q2)

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


@jit(nopython=True)
def build_df(Q, new_Q, T):
    """Gradient of the cost function normalised by dt. See optim.lyx"""

    df = np.empty(len(Q))
    N = new_Q.copy()
    L = np.sqrt(np.sum(new_Q**2, axis=1))
    for j in range(3):
        N[:, j] = N[:, j]/L
    df[0] = np.sum((-2*T[0]*N[0]+T[1]*N[1])*Q[0])
    for i in range(1, len(Q)-1):
        df[i] = np.sum((T[i-1]*N[i-1]-2*T[i]*N[i]+T[i+1]*N[i+1])*Q[i])
    df[-1] = np.sum((T[-2]*N[-2]-2*T[-1]*N[-1])*Q[-1])
    return df


@jit(nopython=True)
def build_d2f(Q, new_Q, T):
    """Diagonal of the Hessian of the cost function normalised by dt.
    See optim.lyx"""

    d2f = np.empty(len(Q))
    L = np.sqrt(np.sum(new_Q**2, axis=1))

    d2f[0] = (4*T[0] + T[1])*np.sum(Q[0]**2)
    for i in range(1, len(Q)-1):
        d2f[i] = (T[i-1]*(np.sum(Q[i]**2)/L[i-1]+np.sum(Q[i]*new_Q[i-1])/L**3)
                  + 4*T[i]*(np.sum(Q[i]**2)/L[i]+np.sum(Q[i]*new_Q[i])/L**3)
                  + T[i+1]**(np.sum(Q[i]**2)/L[i+1]+np.sum(Q[i]*new_Q[i+1])/L**3)
                  )
    d2f[-1] = (T[-2] + 4*T[-1])*np.sum(Q[-1]**2)
    return d2f
