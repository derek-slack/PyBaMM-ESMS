import timeit

import pybamm
import pybamm as pb
import numpy as np
from FoKL import FoKLRoutines
import jax.numpy as jnp
from embedded_gp import Experimental_Embedded_GPs
import jax.scipy

from new_eval import evaluate_pybamm
import pandas as pd

import matplotlib.pyplot as plt, mpld3
from FoKL import getKernels
import warnings
phis = getKernels.sp500()
warnings.filterwarnings("ignore")

k = "symmetric Butler-Volmer"
pb.set_logging_level("NOTICE")
batmodel = pybamm.lithium_ion.SPM({"intercalation kinetics": k})


 # remove events (not supported in jax)
batgeometry = batmodel.default_geometry

# Set Parameter values
param1 = pb.ParameterValues("Mohtat2020")

# Define the phis (basis functions) used



def normalize_inputs(inputs, min, max):
    normalized = (inputs - min)/(max - min)
    return normalized

C = pd.read_csv('Chargecycle.csv', header=None)
D = pd.read_csv('Dischargecycle.csv', header=None)

C = C.to_numpy()
D = D.to_numpy()

TC = C[0, 3:]
TD = D[0, 3:] - max(C[0,3:]) + 0.000001

IC = -C[1, 3:]
ID = -D[1, 3:]


VC = C[2, 3:]
VD = D[2, 3:]

ID_M = []
VD_M = []
TD_M = []

for i in range(336):
    if np.mod(i,2) == 1:
        ID_M.append(ID[i])
        VD_M.append(VD[i])
        TD_M.append(TD[i])

I = np.array(ID_M)
t = np.array(TD_M)-40
V = np.array(VD_M)
IJ = jnp.array(I)
# Define inputs and normalize
inputs = jnp.array([I])
inputs_norm = normalize_inputs(inputs, np.min(I), np.max(I))

# Create object for each individual GP
GPj0p = Experimental_Embedded_GPs.GP()
GPj0n = Experimental_Embedded_GPs.GP()
GPUP = Experimental_Embedded_GPs.GP()
GPUN = Experimental_Embedded_GPs.GP()

# Create of model and define the number of GP's in it
model = Experimental_Embedded_GPs.Embedded_GP_Model(GPj0p,GPj0n,GPUP, GPUN)

# Define appropriate parameters to model
model.inputs = np.transpose(inputs_norm)
model.phis = phis
model.data = np.transpose(V)


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

#pmap gradient, FD gradient estimator

# pybamm.parameters.lithium_ion_parameters.j0 = j0

minI = -4.0304
maxI = 1.5145

current_interpolant = pybamm.Interpolant(t, I, pybamm.t, interpolator="JAX")#, _num_derivatives=0)
VV = []
Dps = [1e-10,5e-10]  # Positive particle diffusivity [m^2/s]
Dns = [5e-4, 1e-4]     # Negative particle diffusivity [m^2/s]
j0s = [1e-3]              # Exchange-current density [A/m^2]
Cs  = [2.1]
OCPM = [4.0, 4.2, 4.3]
OCPN = [1.4, 1.6, 1.8]
soc = np.linspace(0.6,0.8,4)# Nominal cell capacity [Ah]
for i in range(len(j0s)):
    for j in range(len(Dps)):
        for k in range(len(Cs)):
            for l in range(len(Dns)):
                for m in range(len(soc)):
                    for o in range(len(OCPM)):
                        for p in range(len(OCPN)):
                            param1["Current function [A]"] = current_interpolant
                            param1['Nominal cell capacity [A.h]'] = Cs[k]
                            param1["Lower voltage cut-off [V]"] = 1.2
                            param1["Upper voltage cut-off [V]"] = 4.4
                            param1["Positive electrode exchange-current density [A.m-2]"] = j0s[i]
                            # # param1["Initial concentration in positive electrode [mol.m-3]"] =47513.0
                            param1["Open-circuit voltage at 0% SOC [V]"] = OCPN[p]
                            param1["Open-circuit voltage at 100% SOC [V]"] = OCPM[o]
                            param1["Positive particle diffusivity [m2.s-1]"] = Dps[j]
                            # param1["Negative particle diffusivity [m2.s-1]"] = Dns[j]


                            # print(f"time to build = {t2-t1}")
                            # print(f"time to solve = {t3-t2}")

                            solver = pybamm.CasadiSolver(mode="fast",dt_max=25)

                            sim = pybamm.Simulation(batmodel, parameter_values=param1, solver=solver)
                            try:
                                solution = sim.solve(t, initial_soc=soc[m])
                            except:
                                print(f'fail at Dp = {Dps[j]}, Dn = {Dns[l]}, j0 = {j0s[i]}, C = {Cs[k]}, soc = {soc[m]}')
                            else:
                                solution = sim.solve(t, initial_soc=soc[m])
                                VV = solution["Voltage [V]"].entries
                                n = len(VV)
                                plt.plot(t[0:n], VV, label=f'Pybamm soc = {soc[m]}, OCPM = {OCPM[o]}, OCPN = {OCPN[p]}, j0 = {j0s[i]}, C = {Cs[k]}')
                                print(f' Completed Pybamm at soc = {soc[m]}, OCPM = {OCPM[o]}, OCPN = {OCPN[p]}, j0 = {j0s[i]}, C = {Cs[k]}')

n = len(VV)
plt.plot(t[0:n],V[0:n], label='Data')
plt.legend()
mpld3.show()

