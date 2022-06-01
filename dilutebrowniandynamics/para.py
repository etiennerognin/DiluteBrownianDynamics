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

a=[10,23]
b=f"{a[:]}"
print(b)
