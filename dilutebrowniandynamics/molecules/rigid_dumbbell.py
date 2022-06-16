import numpy as np
from ..simulate import ConvergenceError

LENGTH_TOL = 1e-6


class RigidDumbbell:
    """Rigid dumbbell molecule object. A rigid Dumbbell is essentially a
    vector of fixed length connecting two beads. The tension is given by the
    rigidity constraint. The length is 1.

    Attributes
    ----------
    Q : ndarry (3,)
        Coorinates of the vector.
    tension : float
        Internal tension.
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
        self.dW = self.rng.standard_normal(3)
        # discard colinear part (note that dumbbell is unit vector)
        Q_dot_dW = np.sum(self.Q*self.dW)
        self.dW = self.dW - Q_dot_dW*self.Q
        self.dQ = None

    @classmethod
    def from_normal_distribution(cls, seed=np.random.SeedSequence()):
        """Initialise a rigid Dumbbell with a random vector drawn from a
        normal distribution and rescaled to a unit vector."""
        rng = np.random.default_rng(seed)
        Q = rng.standard_normal(3)
        Q = Q/np.sqrt(np.sum(Q**2))
        return cls(Q, rng)

    @property
    def coordinates(self):
        """Compute coordinates of the beads R0 and R1, usually for plotting.
        The molecule is centered at origin."""
        R = np.vstack((-self.Q[None, :]/2, self.Q[None, :]/2))
        return R

    def solve(self, gradU, dt):
        """Solve tension according to current random forces and constraints."""
        A = (self.Q @ gradU)*dt + np.sqrt(dt/3)*self.dW
        QdotA = np.sum(self.Q*A)
        A2 = np.sum(A**2)
        self.tension = 2*(QdotA + 1 - np.sqrt(1 + QdotA**2 - A2))/dt

        self.dQ = (dt*(self.Q @ (gradU - 0.5*self.tension*np.eye(3)))
                   + np.sqrt(dt/3)*self.dW)

        new_Q = self.Q + self.dQ
        if np.abs(np.sum(new_Q**2) - 1.) > LENGTH_TOL:
            # Rigidity is broken
            raise ConvergenceError('Molecule length exceeded 1.')

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
        self.dW = self.rng.standard_normal(3)
        # discard colinear part (note that dumbbell is unit vector)
        Q_dot_dW = np.sum(self.Q*self.dW)
        self.dW = self.dW - Q_dot_dW*self.Q
