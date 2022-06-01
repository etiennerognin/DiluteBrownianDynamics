import numpy as np


LENGTH_TOL = 1e-6


class RigidDumbbell:
    """Rigid dumbbell molecule object. A rigid Dumbbell is essentially a
    vector of fixed length connecting two beads. The tension is given by the
    rigidity constraint. The length is 1.

    Attributes
    ----------
    Q : ndarry (3,)
        Coorinates of the vector.
    """
    def __init__(self, Q):
        self.Q = Q

    @classmethod
    def from_normal_distribution(cls):
        """Initialise a rigid Dumbbell with a random vector drawn from a
        normal distribution and rescaled to a unit vector."""
        Q = np.random.standard_normal(3)
        Q = Q/np.sqrt(np.sum(Q**2))
        return cls(Q)

    @property
    def coordinates(self):
        """Compute coordinates of the beads R0 and R1, usually for plotting.
        The molecule is centered at origin."""
        R = np.vstack((-self.Q[None, :]/2, self.Q[None, :]/2))
        return R

    def evolve(self, gradU, dt):
        """Evolve dumbbell by a time step dt. Itô calculus convention.

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
        dW = np.sqrt(dt/3)*np.random.standard_normal(3)
        # discard colinear part (note that dumbbell is unit vector)
        Q_dot_dW = np.sum(self.Q*dW)
        dW = dW - Q_dot_dW*self.Q

        A = (self.Q @ gradU)*dt+dW
        QdotA = np.sum(self.Q*A)
        A2 = np.sum(A**2)
        tension = 2*(QdotA+1-np.sqrt(1+QdotA**2-A2))/dt

        dQ = dt*(self.Q @ (gradU - 0.5*tension*np.eye(3))) + dW

        new_Q = self.Q + dQ

        if np.abs(np.sum(new_Q**2) - 1.) > LENGTH_TOL:
            # Rigidity is broken
            raise ValueError('Molecule length exceeded 1.')

        elemA = np.outer(self.Q, self.Q)
        elemS = np.outer(self.Q, tension*self.Q)
        self.Q = new_Q
        return elemA, elemS
