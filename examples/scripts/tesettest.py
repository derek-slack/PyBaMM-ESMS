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
import matplotlib.pyplot as plt
from FoKL import getKernels
import warnings
import os
phis = getKernels.sp500()
warnings.filterwarnings("ignore")

k = "symmetric Butler-Volmer"
pb.set_logging_level("NOTICE")
batmodel = pybamm.lithium_ion.SPMe({"intercalation kinetics": k})
batmodel.convert_to_format = 'jax'
batmodel.events = []

save_folder = "figures"
os.makedirs(save_folder, exist_ok=True)

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

minI = -4.0304
maxI = 1.5145

current_interpolant = pybamm.Interpolant(t, I, pybamm.t, interpolator="JAX")#, _num_derivatives=0)

solver = pybamm.JaxSolver()


Dp = [1e-15, 1e-13, 1e-11, 1e-9, 1e-7, 1e-5]
Np = [1e-15, 1e-13, 1e-11, 1e-9, 1e-7, 1e-5]
Cp = [31513, 25000]
Cn = [48]
CpM = [1, 1.05]
CnM = [1, 1.05]
for i in range(len(Dp)):
    for j in range(len(Np)):
        # for k in range(len(Cp)):
        #     for l in range(len(Cn)):
                # for m in range(len(CnM)):
        #     for n in range(len(CnM)):
        param1["Current function [A]"] = 4
        # param1['Nominal cell capacity [A.h]'] = 2.8154
        # param1["Lower voltage cut-off [V]"] = 1.4
        # param1["Upper voltage cut-off [V]"] = 4.4
        # param1["Positive particle diffusivity [m2.s-1]"] = Dp[i]
        # param1["Negative particle diffusivity [m2.s-1]"] = Np[j]
                        # param1["Initial concentration in positive electrode [mol.m-3]"] = Cp[k]
                        # param1["Initial concentration in negative electrode [mol.m-3]"] = Cn[l]
                        # param1["Maximum concentration in positive electrode [mol.m-3]"] = Cp[k] * CpM[m] * 1.1
                        # param1["Maximum concentration in negative electrode [mol.m-3]"] = Cn[l] * CnM[n] * 1.1


        # param1["Initial concentration in positive electrode [mol.m-3]"] = 47513.0 * 0.58
        sim = pybamm.Simulation(batmodel, parameter_values=param1, solver=solver)

        geometry = batmodel.default_geometry
        param1.process_geometry(geometry)
        # set mesh
        mesh = pybamm.Mesh(geometry, batmodel.default_submesh_types, batmodel.default_var_pts)

        # discretise model
        disc = pybamm.Discretisation(mesh, batmodel.default_spatial_methods)
        disc.process_model(batmodel)

        # solve model for 1 hour

        solver = pybamm.JaxSolver()

        solution = solver.solve(batmodel, t)
        sol = solver.get_solve(batmodel,t)
        # print(f"Run failed {[i, j]}")

        print(f"Run Succeeded {[i, j]}")
        Vpbi = solution['Voltage [V]'].entries
        plt.plot(Vpbi,label=f'{[i, j]}')
        plt.plot(V)
        plt.show()