import numpy as np


LENGTH_TOL = 1e-6


class FENEDumbbell:
    """FENE dumbbell molecule object. A FENE Dumbbell is essentially a
    vector connecting two beads. The tension is given by the

    Attributes
    ----------
    Q : ndarray (3,)
        Coorinates of the vector.
    L_max : float
        Maximim dumbbell length
    tension : float
        Internal tension.
    rng : Generator
        Random number generator.
    dW : ndarray (3,)
        Random forces.
    dQ : ndarray (3,)
        Evolution vector.
    """
    def __init__(self, Q, L_max):
        self.Q = Q
        self.L_max = L_max
        self.tension = None
        self.rng = np.random.default_rng()
        self.dW = self.rng.standard_normal(3)
        self.dQ = None

    @classmethod
    def from_normal_distribution(cls, L_max):
        """Initialise a Dumbbell with a random vector drawn from a
        normal distribution of variance 1/3."""
        Q = np.sqrt(1./3)*np.random.standard_normal(3)
        while np.sum(Q**2) > L_max**2 - LENGTH_TOL:
            # Draw another molecule
            Q = np.sqrt(1./3)*np.random.standard_normal(3)
        return cls(Q, L_max)

    @property
    def coordinates(self):
        """Compute coordinates of the beads R0 and R1, usually for plotting.
        The molecule is centered at origin."""
        R = np.vstack((-self.Q[None, :]/2, self.Q[None, :]/2))
        return R

    def solve(self, gradU, dt):
        """Solve tension according to current random forces and constraints."""
        self.tension = self.L_max**2/(self.L_max**2-np.sum(self.Q**2))

        self.dQ = (dt*(self.Q @ (gradU - 0.5*self.tension*np.eye(3)))
                   + np.sqrt(dt/3)*self.dW)

        new_Q = self.Q + self.dQ
        if np.sum(new_Q**2) > self.L_max**2 - LENGTH_TOL:
            # Finite extensibility is broken
            raise ValueError('Molecule length exceeded L_max.')

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
        S = np.outer(self.tension*self.Q, self.Q)
        observables = {'A': A, 'S': S}
        return observables

    def evolve(self, gradU, dt):
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
        self.dW = self.rng.standard_normal(3)
