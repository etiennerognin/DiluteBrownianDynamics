# from molecule import Molecule
# import numpy as np
#
#
# class Dumbbell(Molecule):
#     """Base class for dumbbell objects."""
#
#     def __init__(self, Q):
#         self.Q = Q
#
#     @property
#     def coordinates(self):
#         """Compute coordinates of the beads R0 and R1, usually for plotting.
#         The molecule is centered at origin."""
#         R = np.vstack((-self.Q[None, :]/2, self.Q[None, :]/2))
#         return R
