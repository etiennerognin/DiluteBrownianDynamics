import numpy as np


class HookeanDumbbell:
    """Hookean dumbbell molecule object. A Hookean Dumbbell is essentially a
    vector connecting two beads. The tension is proportional to the length of
    the connector.

    Attributes
    ----------
    Q : ndarray (3,)
        Coorinates of the dumbbell vector.
    tension : float
        Internal tension.
    rng : Generator
        Random number generator.
    dW : ndarray (3,)
        Random forces.
    dQ : ndarray (3,)
        Evolution vector
    """
    def __init__(self, Q):
        self.Q = Q
        self.tension = None
        self.rng = np.random.default_rng()
        self.dW = self.rng.standard_normal(3)
        self.dQ = None

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

    def solve(self, gradU, dt):
        """Solve tension according to current random forces and constraints."""
        self.tension = 1.
        self.dQ = dt*(self.Q @ (gradU - 0.5*np.eye(3))) + np.sqrt(dt/3)*self.dW

    def measure(self):
        """Measure quantities from the systems.

        Returns
        -------
        observables : dict
            Dictionary of observables quantities.
        """
        if self.tension is None:
            raise RuntimeError("Attempt to measure tension but tension not "
                               "solved.")
        # Molecurlar conformation tensor
        A = np.outer(self.Q, self.Q)
        # Molecular stress
        S = np.outer(self.Q, self.Q)
        observables = {'A': A, 'S': S}
        return observables

    def evolve(self, gradU, dt):
        """Evolve dumbbell by a time step dt. Itô calculus convention.

        Parameters
        ----------
        gradUt : (3, 3) ndarray
            Velocity gradient, dvj/dxi convention.
        dt : float
            Time step.
        """
        self.Q += self.dQ
        # draw new random forces
        self.tension = None
        self.dW = self.rng.standard_normal(3)
