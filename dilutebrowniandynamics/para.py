from multiprocessing import Pool
from functools import partial
import numpy as np


# def foo(arg1, arg2, arg3):
#     return arg1 + arg2, arg1 - arg2, arg3**arg3
#
#
# args_list = [0, 1, 2, 3]
#
# with Pool(4) as p:
#     results = list(p.imap(partial(foo, arg2=1, arg3=2), args_list))
# print(*results)

# a = [1.]*9
# MAX_LEVEL = 3
#
# def merge(middle, begin=None, end=None, recursion_level=0):
#     if begin is None:
#         begin = []
#     if end is None:
#         end = []
#
#     if not middle:
#         return begin + end
#     if len(middle) == 1 or recursion_level == MAX_LEVEL:
#         return begin + middle + end
#     if len(middle) % 2 == 1:
#         begin.append(middle.pop())
#     # Smooth transition:
#     begin.append(middle.pop())
#     end = [middle.pop(-1)] + end
#     merged = []
#     for i in range(0, len(middle), 2):
#         merged.append(middle[i]+middle[i+1])
#     return merge(merged, begin, end, recursion_level+1)

a = np.arange(3)
b = np.arange(3)
print(a@np.eye(4)@b)
