import numpy as np
from ..simulate import ConvergenceError

LENGTH_TOL = 1e-6


class RigidDumbbell:
    """Rigid dumbbell molecule object. A rigid Dumbbell is essentially a
    vector of fixed length connecting two beads. The tension is given by the
    rigidity constraint. The squared length is 3 to stick to Hookean dumbbell
    convention.

    Attributes
    ----------
    Q : ndarry (3,)
        Coorinates of the vector.
    H : float
        Dimensionless spring coefficient. Here given by constraint
    rng : Generator
        Random number generator.
    dW : ndarray (3,)
        Random forces.
    dQ : ndarray (3,)
        Evolution vector.
    """
    def __init__(self, Q, rng):
        self.Q = Q
        self.tension = None
        self.rng = rng
        self.dW = self.stochastic_force()
        self.dQ = None

    @classmethod
    def from_normal_distribution(cls, seed=np.random.SeedSequence()):
        """Initialise a rigid Dumbbell with a random vector drawn from a
        normal distribution and rescaled to vector of length sqrt(3)."""
        rng = np.random.default_rng(seed)
        Q = rng.standard_normal(3)
        Q = Q/np.sqrt(np.sum(Q**2))*np.sqrt(3)
        return cls(Q, rng)

    def stochastic_force(self):
        """Draw random force removing colinear part (noise)"""
        dW = self.rng.standard_normal(3)
        return dW - np.sum(dW*self.Q)*self.Q/3

    @property
    def coordinates(self):
        """Compute coordinates of the beads R0 and R1, usually for plotting.
        The molecule is centered at origin."""
        R = np.vstack((-self.Q[None, :]/2, self.Q[None, :]/2))
        return R

    def solve(self, gradU, dt):
        """Solve tension according to current random forces and constraints."""
        # Part without tension
        dQ0 = (self.Q @ gradU)*dt + np.sqrt(dt)*self.dW
        # To enforce rigidity, we need 2Q.dQ + dQ²=0, which leeds to solving
        # 3/4*x^2 - x*(3+Q.dQ0) + 2*Q.dQ0 + dQ0^2 = 0
        Q_dot_dQ0 = np.sum(self.Q*dQ0)
        self.H = (3 + Q_dot_dQ0
                  - np.sqrt(9 + Q_dot_dQ0**2 - 3*np.sum(dQ0**2))
                  )/(3*dt/2)

        self.dQ = dQ0 - 0.5*self.H*self.Q*dt

        new_Q = self.Q + self.dQ
        if np.abs(np.sum(new_Q**2) - 3.) > LENGTH_TOL:
            # Rigidity is broken
            raise ConvergenceError('Molecule length exceeded sqrt(3)')

    def measure(self):
        """Measure quantities from the systems.

        Returns
        -------
        observables : dict
            Dictionary of observables quantities.
        """
        # Molecurlar conformation tensor
        A = np.outer(self.Q, self.Q)
        # Molecular stress
        S = np.outer(self.H*self.Q, self.Q)
        observables = {'A': A, 'S': S}
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
        self.dW = self.stochastic_force()
