import pybamm
import numpy as np
from FoKL import FoKLRoutines

def diffusion_FoKLbuild(sto, T, D):
    diffmodel = FoKLRoutines.FoKL()
    diffmodel.fit([sto,T],D)

    return diffmodel

def diffusion_FoKLEvaluate(diffmodel, sto, T):
    D = diffmodel.evaluate(inputs=[sto,T])
    return D