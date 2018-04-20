import numpy as np
from functools import reduce


def composition(*functions):
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def max_columns(m, amt):
    sum_ls = np.sum(m, axis=0)
    columns_sorted = sorted(enumerate(sum_ls), key=lambda x: x[1])
    res_columns = columns_sorted[-amt:]
    column_iter = sorted(
        map(lambda x: x[0], res_columns)
    )
    return m[:, column_iter]
