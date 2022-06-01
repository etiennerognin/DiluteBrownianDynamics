import numpy as np


LENGTH_TOL = 1e-6


class FENEChain:
    """A FENE chain is a chain of N FENE vectors linking N+1 Brownian beads.
    The tension in each link is given by the FENE force. All the links
    have the same maximum extension.

    Attributes
    ----------
    Q : ndarry (N, 3)
        Coorinates of the vectors.
    L_max : float
        Maximum length for a link.
    """
    def __init__(self, Q, L_max):
        self.Q = Q
        self.L_max = L_max

    def __len__(self):
        """Number of links in the chain"""
        return len(self.Q)

    @classmethod
    def from_normal_distribution(cls, n_links, L_max):
        """Initialise a FENE chain as a collection of vectors drawn from a
        normal distribution of variance 1/3. Vectors longer than L_max are
        discarded.

        Parameters
        ----------
        n_links : int
            Number of links in the chain.

        Returns
        -------
        FENEChain object
            Random FENE chain.
        """
        Q = np.sqrt(1./3)*np.random.standard_normal((n_links, 3))
        norm2 = np.sum(Q**2, axis=1)
        # Cleaning pass
        for i, L in enumerate(norm2):
            while L > L_max - LENGTH_TOL:
                Q[i] = np.sqrt(1./3)*np.random.standard_normal(3)
                L = np.sum(Q[i]**2)

        return cls(Q, L_max)

    @property
    def coordinates(self):
        """Compute coordinates of the beads R0, ...R(N+1), usually for
        plotting.
        The molecule is centered at origin."""
        R = np.vstack(([[0, 0, 0]], np.cumsum(self.Q, axis=0)))
        R = R - np.average(R, axis=0)
        return R

    def evolve(self, gradU, dt):
        """Evolve chain by a time step dt. Itô calculus convention.

        Parameters
        ----------
        gradU : (3, 3) ndarray
            Velocity gradient, dvj/dxi convention.
        dt : float
            Time step.

        Returns
        -------
        elemA, elemS : (3, 3) ndarray
            Elementary conformation (QQ) and stress (QF)
        """
        # Uncorrelated forces on beads...
        Prior = np.sqrt(2*dt)*np.random.standard_normal((len(self)+1, 3))
        # ...gives correlated Brownian force on rods:
        dW = Prior[1:]-Prior[:-1]

        tensions = self.L_max**2/(self.L_max**2-np.sum(self.Q**2, axis=1))

        Q_gradU = self.Q @ gradU

        dQ = np.zeros_like(self.Q)

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
        if any(L > self.L_max**2 - LENGTH_TOL):
            raise ValueError('Molecule length exceeded L_max.')

        # End-to-end vector
        REE = np.sum(self.Q, axis=0)
        elemA = np.outer(REE, REE)
        # Row-wise tensor dot:
        moments = tensions[:, None, None]*(self.Q[:, :, None]
                                           * self.Q[:, None, :])
        elemS = np.sum(moments, axis=0)
        self.Q = new_Q
        return elemA, elemS
