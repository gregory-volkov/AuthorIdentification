from functools import reduce


def composition(*functions):
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

