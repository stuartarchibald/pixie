import numpy as np
from numba_analyser import numba_analysis

np.random.seed(0)


y = np.random.random()


def baz():
    print("This is baz!")
    return 11


def foo(a, k='string'):
    print("This is foo!", a, k)
    return a * 2, baz()


def bar(a, k='string'):
    print("This is bar!", a, k)
    object()
    return a * 2, baz()


def work():
    x = 7
    print("Result from main foo(x):          ", foo(x))
    print("Result from main foo(x * 2, k=y): ", foo(x * 2, k=y))
    print("Result from main baz():           ", baz())
    print("Result from main bar(x * 2, k=y): ", bar(x * 2, k=y))
    print("Result from main foo(object()):   ", foo((object(),),))


def analyser_demo():
    with numba_analysis() as analysis:
        work()

    analysis.process()


if __name__ == "__main__":
    analyser_demo()
