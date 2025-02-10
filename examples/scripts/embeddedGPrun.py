import timeit

import pybamm
import pybamm as pb
import numpy as np
from FoKL import FoKLRoutines

from embedded_gp import Experimental_Embedded_GPs

from FoKL.JAX_Eval import *
import pandas as pd
import matplotlib.pyplot as plt

import warnings


warnings.filterwarnings("ignore")


pb.set_logging_level("NOTICE")
batmodel = pybamm.lithium_ion.SPM()
param = pb.ParameterValues("Mohtat2020")
 # remove events (not supported in jax)
batgeometry = batmodel.default_geometry

# Define the phis (basis functions) used
phis = np.array(FoKLRoutines.getKernels.sp500())


def normalize_inputs(inputs, min, max):
    normalized = (inputs - min)/(max - min)
    return normalized

C = pd.read_csv('chargecycle.csv', header=None)
D = pd.read_csv('dischargecycle.csv', header=None)

C = C.to_numpy()
D = D.to_numpy()

TC = C[0, :]
TD = D[0, :] + C[0, -1:] + 0.000001

IC = C[1, :]
ID = D[1, :]

VC = C[2, :]
VD = D[2, :]

I = np.concatenate((IC, ID))
t = np.concatenate((TC, TD))
V = np.concatenate((VC, VD))
IJ = np.array(I)
# Define inputs and normalize
inputs = np.array([I])
inputs_norm = normalize_inputs(inputs, np.min(I), np.max(I))

# Create object for each individual GP
GPj0p = Experimental_Embedded_GPs.GP()
GPj0n = Experimental_Embedded_GPs.GP()

# Create of model and define the number of GP's in it
model = Experimental_Embedded_GPs.Embedded_GP_Model(GPj0p, GPj0n)

# Define appropriate parameters to model
model.inputs = np.transpose(inputs_norm)
model.phis = phis
model.data = np.transpose(V)


#pmap gradient, FD gradient estimator


minI = -4.0304
maxI = 1.5145



current_interpolant = pybamm.Interpolant(t, I, pybamm.t)
param["Current function [A]"] = current_interpolant

# T1 = timeit.default_timer()
solver = pybamm.CasadiSolver(mode="fast")
sim = pybamm.Simulation(batmodel, parameter_values=param, solver=solver)

solution = sim.solve(t)



#
# def j0(self, c_e, c_s_surf, T, lithiation=None):
#     """Dimensional exchange-current density [A.m-2]"""
#     tol = pybamm.settings.tolerances["j0__c_e"]
#     c_e = pybamm.maximum(c_e, tol)
#     tol = pybamm.settings.tolerances["j0__c_s"]
#     c_s_surf = pybamm.maximum(
#         pybamm.minimum(c_s_surf, (1 - tol) * self.c_max), tol * self.c_max
#     )
#     domain, Domain = self.domain_Domain
#     if lithiation is None:
#         lithiation = ""
#     else:
#         lithiation = lithiation + " "
#     inputs = {
#         "Current [A]": pybamm.electrical_parameters.current_with_time,
#     }
#     return pybamm.FunctionParameter(
#         f"{self.phase_prefactor}{Domain} electrode {lithiation}"
#         "exchange-current density [A.m-2]",
#         inputs,
#     )
# pybamm.parameters.lithium_ion_parameters.j0 = j0

def equation(betas, mtx):

    param1 = pb.ParameterValues("Mohtat2020")


    def j0p(I_pb):

        x1 = I

        predictions = np.array(GP_results[0])

        # Create interpolant with separate 1D arrays for each dimension and the associated children

        interp = pybamm.Interpolant(x1, predictions, (I_pb-minI/(maxI-minI)), interpolator="linear")
        return interp
    def j0n(I_pb):

        n = 986
        x1 = I

        predictions = np.array(GP_results[1])

        # Create interpolant with separate 1D arrays for each dimension and the associated children

        interp = pybamm.Interpolant(x1, predictions, (I_pb-minI/(maxI-minI)), interpolator="linear")
        return interp

    param1["Positive electrode exchange-current density [A.m-2]"] = j0p
    param1["Negative electrode exchange-current density [A.m-2]"] = j0n
    param1["Current function [A]"] = current_interpolant

    # T1 = timeit.default_timer()
    solver = pybamm.CasadiSolver(mode="fast")
    sim = pybamm.Simulation(batmodel, parameter_values=param1, solver=solver)

    solution = sim.solve(t)
    Vpb = solution["Voltage [V]"].entries
    # T2 = timeit.default_timer() - T1
    # print(T2)
    return Vpb


model.set_equation(equation)

samples, matrix, BIC = model.full_routine(draws = 1000, tolerance = 0)

pos_j0_model = model.evaluate(np.linspace(0,1,986).reshape(-1,1), GP_number=0, draws=1000, burn=500, ReturnBounds=0)
neg_j0_model  = model.evaluate(np.linspace(0,1,986).reshape(-1,1), GP_number=1, draws=1000, burn=500, ReturnBounds=0)

def pos_j0(I):
    n = 986
    x1 = np.linspace(0, 1, n)

    predictions = pos_j0_model

    # Create interpolant with separate 1D arrays for each dimension and the associated children

    interp = pybamm.Interpolant(x1, predictions, (I+4.0304)/(1.5145+4.0304), interpolator="linear")
    return interp

def neg_j0(I):
    n = 986
    x1 = np.linspace(0, 1, n)

    predictions = neg_j0_model

    # Create interpolant with separate 1D arrays for each dimension and the associated children

    interp = pybamm.Interpolant(x1, predictions, (I + 4.0304) / (1.5145 + 4.0304), interpolator="linear")
    return interp


batmodel = pb.lithium_ion.SPM()

C = pd.read_csv('chargecycle.csv', header=None)
D = pd.read_csv('dischargecycle.csv', header=None)

C = C.to_numpy()
D = D.to_numpy()

TC = C[0, :]
TD = D[0, :] + C[0, -1:] + 0.00001

IC = C[1, :]
ID = D[1, :]

I = np.concatenate((IC, ID))
t = np.concatenate((TC, TD))

current_interpolant = pybamm.Interpolant(t, I, pybamm.t)

param1["Positive electrode exchange-current density [A.m-2]"] = pos_j0
param1["Negative electrode exchange-current density [A.m-2]"] = neg_j0
param1["Current function [A]"] = current_interpolant

solver = pybamm.CasadiSolver(mode="fast")
sim = pybamm.Simulation(batmodel, parameter_values=param1, solver=solver)

solution = sim.solve(jnp.array(t))
Vpb = solution["Voltage [V]"].entries

plt.plot(t, Vpb,'r')
plt.plot(t, V,'g')
plt.show()


