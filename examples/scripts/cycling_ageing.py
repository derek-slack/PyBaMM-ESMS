import timeit

import pybamm
import pybamm as pb
import numpy as np
from FoKL import FoKLRoutines
from functionBuilder import gridmaker, grid_to_saved_predictions

pb.set_logging_level("NOTICE")
model = pb.lithium_ion.SPM(
    {
        "SEI": "ec reaction limited",
        "SEI film resistance": "distributed",
        "SEI porosity change": "true",
        "lithium plating": "irreversible",
        "lithium plating porosity change": "true",
    }
)
# Create j0 data
#
# pj0 = np.genfromtxt("pj0f.csv",delimiter=" ")
# nj0 = np.genfromtxt("nj0f.csv",delimiter=" ")
# c_e_f = np.genfromtxt("ce.csv",delimiter=",")
# c_s_n_avg = np.genfromtxt("csfn.csv",delimiter=",")
# c_s_p_avg = np.genfromtxt("csf.csv",delimiter=",")
#
# add noise to j0 data
# noise_percentage = 0.05
# for i in range(len(pj0)):
#     pj0[i] = pj0[i] * np.random.uniform(1-noise_percentage,1+noise_percentage)
#     nj0[i] = nj0[i] * np.random.uniform(1-noise_percentage,1+noise_percentage)

# inputs_pos = np.transpose(c_s_p_avg)
# inputs_neg = np.transpose(c_s_n_avg)
# # initialize FoKL models for both
# positive_j0_model = FoKLRoutines.FoKL()
# negative_j0_model = FoKLRoutines.FoKL()
#
# positive_j0_model.minmax = [[0, 35839]]
# negative_j0_model.minmax = [[0, 35839]]
#
# positive_j0_model.fit(inputs=inputs_pos, data=pj0, clean=True)
# negative_j0_model.fit(inputs=inputs_neg, data=nj0, clean=True)
#
# positive_j0_model.save('positive_j0.fokl')
# negative_j0_model.save('negative_j0.fokl')
# Set Parameter values
param1 = pb.ParameterValues("Mohtat2020")
param2 = pb.ParameterValues("Mohtat2020")


# Load FoKL Model
# modparams_rad = np.loadtxt('modparams_rad.txt')
# data = np.loadtxt('data.txt', delimiter=',')
mtx_D = np.loadtxt('mtx_D.txt', delimiter=',')
betas_D_NCM = np.loadtxt('betas_D_NCM.txt', delimiter=',')

# # set FoKL model parameters
fmodel = FoKLRoutines.FoKL()
fmodel.minmax = [[0,1]]
fmodel.betas = betas_D_NCM
fmodel.avg_betas = betas_D_NCM
fmodel.mtx = np.c_[mtx_D]
#
fmodel.save('negativemodelD.fokl')
# #
n = 50
x = np.linspace(0, 1, n)
minmax = fmodel.minmax
t = timeit.default_timer()
tt = gridmaker(fmodel.minmax,n)
preds = fmodel.evaluate(inputs=tt, betas = betas_D_NCM, avgbetas=True)
fmodel.predictions = preds
fmodel.save('negativemodelD.fokl')
#
positive_j0_model = FoKLRoutines.load('positive_j0_model.fokl')
n = 100
x1 = np.linspace(0,35839,n)
preds = positive_j0_model.evaluate(inputs=x1, clean = True)
positive_j0_model.predictions = preds
positive_j0_model.save('positive_j0_model.fokl')

negative_j0_model = FoKLRoutines.load('negative_j0_model.fokl')
n = 100
x1 = np.linspace(0,35839,n)
tt = gridmaker(negative_j0_model.minmax,n)
preds = negative_j0_model.evaluate(inputs=x1, clean=True)
negative_j0_model.predictions = preds
negative_j0_model.save('negative_j0_model.fokl')
# # Generate grid for 2D interpolation
#
# # Evaluate model
# p = grid_to_saved_predictions(tt, fmodel)
# # Reshape predictions to match 2D structure of xx, yy


def pf(sto,T):
    fmodel = FoKLRoutines.load('negativemodelD.fokl')
    n = 50
    x = np.linspace(0,1,n)

    predictions = fmodel.predictions
    # predictions_reshaped = predictions.reshape(n, 1)
    # Create interpolant with separate 1D arrays for each dimension and the associated children

    interp = pybamm.Interpolant(x, predictions, [sto], interpolator="linear")
    return interp

def pj0(I):
    fmodel = FoKLRoutines.load('positive_j0_model.fokl')
    n = 100
    x1 = np.linspace(fmodel.minmax[0][0], fmodel.minmax[0][1], n)

    predictions = fmodel.predictions

    # Create interpolant with separate 1D arrays for each dimension and the associated children

    interp = pybamm.Interpolant(x1, predictions, c_s_surf, interpolator="linear")
    return interp

def nj0(I):
    fmodel = FoKLRoutines.load('negative_j0_model.fokl')

    n = 100
    x1 = np.linspace(fmodel.minmax[0][0], fmodel.minmax[0][1], n)

    predictions = fmodel.predictions

    # Create interpolant with separate 1D arrays for each dimension and the associated children

    interp = pybamm.Interpolant(x1, predictions, c_s_surf, interpolator="linear")
    return interp
#
param1["Positive particle diffusivity [m2.s-1]"] = pf
param1["Negative particle diffusivity [m2.s-1]"] = pf
param1["Positive electrode exchange-current density [A.m-2]"] = pj0
param1["Negative electrode exchange-current density [A.m-2]"] = nj0

experiment = pb.Experiment(
    [
        (
            "Charge at 1 C until 4.2 V",
            "Hold at 4.2 V until C/10",
            "Rest for 5 minutes",
            "Discharge at 1 C until 2.8 V",
            "Rest for 5 minutes",
        )
    ]
    * 2
    + [
        (
            "Charge at 1 C until 4.2 V",
            "Hold at 4.2 V until C/20",
            "Rest for 30 minutes",
            "Discharge at C/3 until 2.8 V",
            "Rest for 30 minutes",
        ),
        (
            "Charge at 1 C until 4.2 V",
            "Hold at 4.2 V until C/20",
            "Rest for 30 minutes",
            "Discharge at 1 C until 2.8 V",
            "Rest for 30 minutes",
        ),
        (
            "Charge at 1 C until 4.2 V",
            "Hold at 4.2 V until C/20",
            "Rest for 30 minutes",
            "Discharge at 2 C until 2.8 V",
            "Rest for 30 minutes",
        ),
        (
            "Charge at 1 C until 4.2 V",
            "Hold at 4.2 V until C/20",
            "Rest for 35 minutes",
            pb.step.string("Discharge at 3 C until 2.8 V", period=10),
            "Rest for 30 minutes",
        ),
    ]
)
t_og_sim1 = timeit.default_timer()
sim1 = pb.Simulation(model, experiment=experiment, parameter_values=param1)
time_og_sim = timeit.default_timer() - t_og_sim1
t_og_solve1 = timeit.default_timer()
sim1_sol = sim1.solve(solver=pb.CasadiSolver(mode="fast with events"))
time_og_solve = timeit.default_timer() - t_og_solve1
t_new_sim1 = timeit.default_timer()
sim2 = pb.Simulation(model, experiment=experiment, parameter_values=param2)
time_new_sim = timeit.default_timer() - t_new_sim1
t_new_solve1 = timeit.default_timer()
sim2_sol = sim2.solve(solver=pb.CasadiSolver(mode="fast with events"))
time_new_solve = timeit.default_timer() - t_new_solve1
sims = [sim1, sim2]

# print(f'Time for original simulation: {time_og_sim}\n')
# print(f'Time for new simulation: {time_new_sim}\n')
print(f'Time for original solve: {sim1_sol.solve_time}\n')
print(f'Time for new solve: {sim2_sol.solve_time}\n')
print(f'Time for integration: {sim1_sol.integration_time}\n')
print(f'Time for integration: {sim2_sol.integration_time}\n')


solution = sim2.solution
c_s_surf_positive = solution["Positive particle concentration [mol.m-3]"].entries
c_s_surf_negative = solution["Negative particle concentration [mol.m-3]"].entries
c_e = solution["Electrolyte concentration [mol.m-3]"].entries

pb.dynamic_plot(sims, [
        "Current [A]",
        "Total current density [A.m-2]",
        "Voltage [V]",
        "Discharge capacity [A.h]",
        "Positive electrode exchange current density [A.m-2]",
        "Negative electrode exchange current density [A.m-2]",
    ], labels=["default", "diff model"])

h=1

