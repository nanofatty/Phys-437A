import numpy as np
import random
import functions as fn
from joblib import Parallel, delayed

""" INPUT """
path = "Data/nov3_pad3_YFP/monoculture_nov3_pad3_22hr-02_s2t"  # file name until timestamp
fc = "Cy3"  # fluorescent channel name
time_i = 0  # initial time frame
time_f = 23  # final time frame
nump = 50  # number of pixels to combine when resizing
umpp = 0.91  # um per pixel
thresh_ph = 0  # background subtraction value for phase channel images
thresh_fl = 1  # background subtraction value for fluorescence channel images
dthr = 1  # time interval between frames in hr

input = [path, fc, time_i, time_f, nump, umpp, thresh_ph, thresh_fl, dthr]

""" STARTING PARAMETER VALUES, SMALLEST STEP SIZE IN PARAMETER SPACE AND BOUNDS"""

log10k = 0            # log10 of conversion factor from fluorescence channel to phase channel intensity
step_log10k = 0.1
l_log10k, u_log10k = -4, 4

mu = 0.45/3600        # max growth rate of strain 1 (in 1/sec)
step_mu = 0.1 / 3600
l_mu, u_mu = 0, 2.50 / 3600

Rhf = 150              # cell density parameter to limit accumulating growth
step_Rhf = 10
l_Rhf, u_Rhf = 0, 1e10

Qhf = 150              # cell density parameter determining dependence of migration on local density
step_Qhf = 10
l_Qhf, u_Qhf = 0, 1e10

Phf = 150              # cell density parameter determining dependence of migration on neighboring density
step_Phf = 10
l_Phf, u_Phf = 0, 1e10

Y = 160               # biomass yield (in pixel intensity per unit S0)
step_Y = 10
l_Y, u_Y = 0, 1e10

log10Ks = -2          # rate of nutrient consumption by Monod kinetics (in unit log10S0)
step_log10Ks = 1
l_log10Ks, u_log10Ks = -100, 0

DC = 1200             # spreading coefficient of cells (in um^2)
step_DC = 100
l_DC, u_DC = 0, 1e10

DS = 500              # diffusion coefficient of glucose (in um^2/sec)
step_DS = 10
l_DS, u_DS = 0, 720

Parameters = [log10k, mu, Rhf, Qhf, Phf, Y, log10Ks, DC, DS]
step_Parameters = [step_log10k, step_mu, step_Rhf, step_Qhf, step_Phf, step_Y, step_log10Ks, step_DC, step_DS]
l_Parameters = [l_log10k, l_mu, l_Rhf, l_Qhf, l_Phf, l_Y, l_log10Ks, l_DC, l_DS]
u_Parameters = [u_log10k, u_mu, u_Rhf, u_Qhf, u_Phf, u_Y, u_log10Ks, u_DC, u_DS]

""" ANNEALING PARAMETERS """
an_iters = 1000
maxTemp = 1e-6  # 1e6
alpha = 0.9
beta = 0.95
max_step = 10  # 10

sa_input = [an_iters, maxTemp, alpha, beta, max_step]
""" DEFINE ANNEALING """


def simulated_annealing(marker):
    an_iters, maxTemp, alpha, beta, max_step = tuple(sa_input)

    """ READ DATA """
    Phase, FC = fn.read_data_as_array(input)
    print("Data imported {}".format(marker))

    Parameters = np.array([log10k, mu, Rhf, Qhf, Phf, Y, log10Ks, DC, DS])
    errors = []
    Temps = []
    Parameters_dat = []

    print("Beginning annealing {} ...".format(marker))
    current_error = fn.get_error(Phase, FC, Parameters, l_Parameters, u_Parameters, input)

    """ BEGIN ITERATIONS """
    iter = 0
    while iter < an_iters:

        Temp = maxTemp * alpha ** iter

        new_Parameters = Parameters + \
                         np.random.uniform(-1, 1, len(Parameters)) * np.array(step_Parameters) * (
                                 1 + max_step * beta ** iter)
        #print(new_Parameters)

        """ Use below lines to avoid recomputing error for revisited parameter vectors """
        # if new_Parameters.tolist() in Parameters_dat:
        #    new_error = errors[Parameters_dat.index(new_Parameters.tolist())]
        # else:
        #    new_error = fn.get_error(Phase, FC, new_Parameters, l_Parameters, u_Parameters, input)
        new_error = fn.get_error(Phase, FC, new_Parameters, l_Parameters, u_Parameters, input)

        Derror = new_error - current_error

        if Derror <= 0:
            """ Accept all downhill moves """
            Parameters = new_Parameters
            current_error = new_error
            #print('accepted', iter)
        else:
            """ Accept uphill moves with a probability"""
            metropolis = 2.7183 ** (-Derror / Temp)
            rand = random.uniform(0, 1)
            if rand < metropolis:
                Parameters = new_Parameters
                current_error = new_error
                #print('accepted', iter)
            #else:
                #print('declined', iter)

        errors.append(current_error)
        Temps.append(Temp)
        Parameters_dat.append(Parameters.tolist())

        iter += 1

    print(marker, ", Error: ", current_error, ", Parameters: ", Parameters)
    data = np.column_stack((Temps, errors, Parameters_dat))
    np.savetxt("annealing_data_{}.txt".format(marker), data, delimiter=",")

    return errors, Temps, Parameters_dat


""" RUNNING SIMULATION ANNEALING FOR PARAMETER ESTIMATION """
#errors, Temps, Parameters_dat = simulated_annealing(0)

#data = np.column_stack((Temps, errors, Parameters_dat))

#plt.plot(np.array(errors))
#plt.grid()
#plt.ylabel('Error')
#plt.xlabel('Iteration')
#plt.show()

#np.savetxt("annealing_data.txt", data, delimiter=",")


""" REDEFINE ANNEALING FOR PROFILE LIKELIHOOD """

def simulated_annealing_exc_par(idx):
    """ idx is the index of the parameter vector that remains unchanged throughout optimization"""

    an_iters, maxTemp, alpha, beta, max_step = tuple(sa_input)

    """ READ DATA """
    Phase, FC = fn.read_data_as_array(input)
    print("Data imported")

    Parameters = np.array([log10k, mu, Rhf, Qhf, Phf, Y, log10Ks, DC, DS])
    errors = []
    Temps = []
    Parameters_dat = []

    print("Beginning annealing...")
    current_error = fn.get_error(Phase, FC, Parameters, l_Parameters, u_Parameters, input)

    """ BEGIN ITERATIONS """
    iter = 0
    while iter < an_iters:

        Temp = maxTemp * alpha ** iter

        step = np.random.uniform(-1, 1, len(Parameters)) * np.array(step_Parameters) * (1 + max_step * beta ** iter)
        step[idx] = 0
        new_Parameters = Parameters + step

        print(new_Parameters)

        """ Use below lines to avoid recomputing error for revisited parameter vectors """
        # if new_Parameters.tolist() in Parameters_dat:
        #    new_error = errors[Parameters_dat.index(new_Parameters.tolist())]
        # else:
        #    new_error = fn.get_error(Phase, FC, new_Parameters, l_Parameters, u_Parameters, input)
        new_error = fn.get_error(Phase, FC, new_Parameters, l_Parameters, u_Parameters, input)

        Derror = new_error - current_error

        if Derror <= 0:
            """ Accept all downhill moves """
            Parameters = new_Parameters
            current_error = new_error
            print('accepted', iter)
        else:
            """ Accept uphill moves with a probability"""
            metropolis = 2.7183 ** (-Derror / Temp)
            rand = random.uniform(0, 1)
            if rand < metropolis:
                Parameters = new_Parameters
                current_error = new_error
                print('accepted', iter)
            else:
                print('declined', iter)

        errors.append(current_error)
        Temps.append(Temp)
        Parameters_dat.append(Parameters.tolist())

        iter += 1

    data = np.column_stack((Temps, errors, Parameters_dat))
    np.savetxt("profile_par{}_{}.txt".format(idx, marker), data, delimiter=",")

    return errors, Temps, Parameters_dat

""" RUN ANNEALING FOR DIFFERENT MARKERS AS PARALLEL PROCESSES """

markers = np.linspace(0,10,10)

def run_model():
    # set markers to run
    modelParList = markers
    # run model
    nJobs = min(len(modelParList), -1)
    print('starting with %i jobs' % len(modelParList))
    results = Parallel(n_jobs=nJobs, verbose=9, timeout=1.E5)(
        delayed(simulated_annealing)(par) for par in modelParList)

run_model()