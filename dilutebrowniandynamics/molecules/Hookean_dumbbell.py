import numpy as np


class HookeanDumbbell:
    """Hookean dumbbell molecule object. A Hookean Dumbbell is essentially a
    vector connecting two beads. The tension is proportional to the length of
    the connector.

    Attributes
    ----------
    Q : ndarray (3,)
        Coorinates of the dumbbell vector.
    """
    def __init__(self, Q):
        self.Q = Q

    @classmethod
    def from_normal_distribution(cls):
        """Initialise a Dumbbell with a random vector drawn from a
        normal distribution of variance 1/3."""
        Q = np.sqrt(1./3)*np.random.standard_normal(3)
        return cls(Q)

    @property
    def coordinates(self):
        """Compute coordinates of the beads R0 and R1, usually for plotting.
        The molecule is centered at origin."""
        R = np.vstack((-self.Q[None, :]/2, self.Q[None, :]/2))
        return R

    def evolve(self, gradUt, dt):
        """Evolve dumbbell by a time step dt. Itô calculus convention.

        Parameters
        ----------
        gradUt : (3, 3) ndarray
            Velocity gradient, dvj/dxi convention.
        dt : float
            Time step.

        Returns
        -------
        elemA, elemS : (3, 3) ndarray
            Elementary conformation (QQ) and stress (QF)
        """
        dW = np.sqrt(dt/3)*np.random.standard_normal(3)
        elemA = np.outer(self.Q, self.Q)
        elemS = np.outer(self.Q, self.Q)
        self.Q += dt*(self.Q @ (gradUt - 0.5*np.eye(3))) + dW
        return elemA, elemS
