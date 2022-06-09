import numpy as np
from scipy.linalg.lapack import dptsv


LENGTH_TOL = 1e-6
MAXITER = 100
FILTER_NOISE = True
MERGE_MAX_LEVEL = 0    # Maximum level of recursion for merging rods
MERGE_THRESHOLD = 1000.  # Dimensionless tension required to merge segments
FLAG = False


class AdaptiveKramersChain:
    """A Kramers chain, also known as bead-rod model, is a chain of N rigid
    vectors linking N+1 Brownian beads. The tension in each link is given by
    the rigidity constraint. This adaptive version will merge aligned segments
    provided that internal tension is greated than a threshold. This can be
    thought of further coarse graining of the polymer.

    Attributes
    ----------
    Q : ndarray (N, 3)
        Coorinates of the vectors.
    _L2 : ndarray(N,)
        Links length squared. Cached property.
    _beads : ndarray(N+1,)
        Weights (radii) of beads. Cached property.
    tensions : ndarray (N,)
        Internal tensions.
    avg_tensions : ndarray (N,)
        Averaged internal tensions.
    subit : float
        Sub-iteration. This is used in adaptive scheme.
    rng : Generator
        Random number generator.
    dW : ndarray (N, 3)
        Random forces.
    dQ : ndarray (N, 3)
        Evolution vector.
    """
    def __init__(self, Q):
        self.Q = Q
        self._L2 = None
        self._beads = None
        self.tensions = None
        self.avg_tensions = None
        self.subit = 0.
        self.rng = np.random.default_rng()
        self.dW = self.stochastic_force()
        self.dQ = None

    def stochastic_force(self):
        """Draw random force. Noise filtering if applicable. Noise variance
        proportional to bead size."""

        # Uncorrelated forces on beads...
        w = self.beads
        Prior = np.sqrt(w[:, None])*self.rng.standard_normal((len(self)+1, 3))

        if FILTER_NOISE:
            # -- Noise reduction:
            # For inner beads, keep component that is only normal to both links
            # Note that vector products of aligned beads can be close to zero,
            # therefore removing scalar products is preferred.
            Prior[1:-1] = (Prior[1:-1]
                           - (np.sum(Prior[1:-1]*self.Q[1:], axis=1)[:, None]
                              * self.Q[1:]/self.L2[1:, None])
                           - (np.sum(Prior[1:-1]*self.Q[:-1], axis=1)[:, None]
                              * self.Q[:-1]/self.L2[:-1, None])
                           )
            # For start and end, just normal to the link
            Prior[0] = Prior[0] - (np.sum(Prior[0]*self.Q[0])
                                   * self.Q[0]/self.L2[0])
            Prior[-1] = Prior[-1] - (np.sum(Prior[-1]*self.Q[-1])
                                     * self.Q[-1]/self.L2[-1])

        # ...gives correlated Brownian force on rods:
        return Prior[1:] - Prior[:-1]

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
        The molecule is centered at center of friction."""
        R = np.vstack(([[0, 0, 0]], np.cumsum(self.Q, axis=0)))
        R = R - np.average(R, axis=0, weights=self.beads)
        return R

    @property
    def REE(self):
        """End-to-end vector"""
        return np.sum(self.Q, axis=0)

    @property
    def L2(self):
        """Length of each link rounded to the next integer. Note: dtype remains
        float."""
        if self._L2 is None:
            self._L2 = np.rint(np.sum(self.Q**2, axis=1))
        return self._L2

    @property
    def beads(self):
        """Beads radius (or normalised friction) according to the splitting
        rule: 0.5*(L_{i-1} + L_{i})"""
        if self._beads is None:
            w = np.empty(len(self)+1)
            L = np.sqrt(self.L2)
            w[0] = 0.5*(1. + L[0])
            w[-1] = 0.5*(L[-1] + 1.)
            w[1:-1] = 0.5*(L[:-1] + L[1:])
            self._beads = w
        return self._beads

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
        # Get ideal length of each link:
        L2 = self.L2
        # Corresponding inverse
        iL = 1./np.sqrt(L2)
        # Get beads radius
        w = self.beads
        # Centres of action
        X = self.coordinates
        diffwX = w[1:, None]*X[1:] - w[:-1, None]*X[:-1]

        # Right hand side
        dW = np.sqrt(2*dt)*self.dW
        diffwX_gradU = diffwX @ gradU
        diffwX_gradU_Q = np.sum(diffwX_gradU*self.Q, axis=1)
        dW_dot_Q = np.sum(dW*self.Q, axis=1)
        RHS0 = (diffwX_gradU_Q + dW_dot_Q/dt)*iL

        # Tridiagonal matrix
        # Lower and upper diagonal elements
        dlu = - np.sum(self.Q[1:]*self.Q[:-1], axis=1)*iL[1:]*iL[:-1]
        # Diagonal elements
        d = 2*np.ones(len(self.Q))

        RHS = RHS0.copy()

        dQ = np.zeros_like(self.Q)

        for i in range(MAXITER):
            with np.errstate(over='raise', invalid='raise'):
                try:
                    # Solve the system, see NOTES
                    tensions = dptsv(d, dlu, RHS)[2]

                    dQ[0] = dt*(diffwX_gradU[0] + tensions[1]*iL[1]*self.Q[1]
                                - 2*tensions[0]*iL[0]*self.Q[0]
                                ) + dW[0]

                    dQ[1:-1] = dt*(diffwX_gradU[1:-1]
                                   + tensions[2:, None]*iL[2:, None]*self.Q[2:]
                                   - 2*tensions[1:-1, None]*iL[1:-1, None]*self.Q[1:-1]
                                   + tensions[:-2, None]*iL[:-2, None]*self.Q[:-2]
                                   ) + dW[1:-1]

                    dQ[-1] = dt*(diffwX_gradU[-1]
                                 - 2*tensions[-1]*iL[-1]*self.Q[-1]
                                 + tensions[-2]*iL[-2]*self.Q[-2]
                                 ) + dW[-1]

                    # Compute new chain
                    new_Q = self.Q + dQ

                    # Compute rod square lengths
                    new_Q2 = np.sum(new_Q**2, axis=1)
                    # And compare to target
                    err = np.max(np.abs(new_Q2-L2))
                    # update the right-hand-side
                    RHS = RHS0 + 0.5/dt*np.sum(dQ**2, axis=1)*iL
                except FloatingPointError:
                    if FLAG:
                        print('dptsv convergence failed.')
                    raise ValueError('dptsv convergence failed.')
            if err < LENGTH_TOL:
                # Re normalise and exit loop
                new_Q = np.sqrt(L2[:, None])*new_Q/np.sqrt(new_Q2[:, None])
                dQ = new_Q - self.Q
                break
            elif i == MAXITER - 1:
                if FLAG:
                    print(f"Could not converge in {MAXITER} "
                          "iterations.")
                raise ValueError(f"Could not converge in {MAXITER} "
                                 "iterations.")
        self.dQ = dQ
        self.tensions = tensions
        if self.avg_tensions is None:
            self.avg_tensions = dt*tensions
            self.subit = dt
        else:
            self.avg_tensions += dt*tensions
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
        moments = self.tensions[:, None, None]*(self.Q[:, :, None]
                                                * self.Q[:, None, :])
        S = np.sum(moments, axis=0)
        observables = {'A': A, 'S': S}
        return observables

    def evolve(self, first_subit):
        """Evolve chain by a time step dt. Itô calculus convention.
        Reset random forces.

        Parameters
        ----------
        gradU : (3, 3) ndarray
            Velocity gradient, dvj/dxi convention.
        dt : float
            Time step.
        """
        global FLAG
        self.Q += self.dQ

        if first_subit:
            self.tensions = self.avg_tensions/self.subit
            self.adapt()
            self.avg_tensions = None
            self.subit = 0.

        # Reset and draw new random forces
        self.tensions = None
        self.dW = self.stochastic_force()

    def adapt(self):
        """Try to merge or split rods according to internal tension.

        Splitting a rod: any long rod where tension < MERGE_THRESHOLD

        Merging 2 rods:
        - both have tension > MERGE_THRESHOLD*L
        - there is a smooth transition zone (one level of recursion increment)
          with small rods.
        """
        # Check whether adapt is needed:
        if all(self.tensions < MERGE_THRESHOLD) and all(self.L2 < 1.5):
            return

        # Loop over resolution in decreasing order
        for resol in range(MERGE_MAX_LEVEL, 0, -1):
            # Do one pass over the chain
            L = np.rint(np.sqrt(self.L2))
            new_Q = []
            new_tensions = []
            i = 0
            while i < len(self):
                if (L[i] == 2**resol
                        and self.tensions[i] < MERGE_THRESHOLD*2**(resol-1)):
                    # Split
                    new_Q += [self.Q[i]/2]*2
                    new_tensions += [self.tensions[i]]*2
                    # Go to next segment
                    i += 1
                elif (i < len(self)-1  # We don't want the last rod
                      and L[i] == 2**(resol-1)
                      and L[i+1] == 2**(resol-1)
                      and self.tensions[i] > MERGE_THRESHOLD*2**(resol-1)
                      and self.tensions[i+1] > MERGE_THRESHOLD*2**(resol-1)):
                    # 1. Both rods at same resolution
                    # 2. Above tension threshold
                    # Merge
                    new_Q.append(2*self.Q[i])
                    new_tensions.append(self.tensions[i])
                    i += 2  # Skip the next rod wich has been absorbed.
                else:
                    # Do nothing and collect all rods in segment
                    subL = 0
                    while i < len(self) and subL < 2**resol:
                        new_Q.append(self.Q[i])
                        new_tensions.append(self.tensions[i])
                        subL += L[i]
                        i += 1
            self.Q = np.array(new_Q)
            self.tensions = np.array(new_tensions)
            self._L2 = None
            self._beads = None

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

        # Bead size
        out.append(f"POINT_DATA {len(self)+1}")
        out.append("SCALARS bead_size double")
        out.append("LOOKUP_TABLE default")
        for bead in self.beads:
            out.append(str(bead))
        out.append("")

        if self.tensions is not None:
            self.tensions = self.avg_tensions/self.subit
            # Tensions
            out.append("FIELD observables 1")
            out.append(f"tension 1 {len(self)+1} float")
            out.append(f"{self.tensions[0]}")
            for i in range(len(self)-1):
                out.append(f"{(self.tensions[i] + self.tensions[i+1])/2}")
            out.append(f"{self.tensions[-1]}")
            out.append("")

        # Make it a file
        content = '\n'.join(out)
        with open(file_name, 'w') as f:
            f.write(content)
