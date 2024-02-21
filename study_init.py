import numpy as np
import functions as fn

""" Parameters"""
# ** = estimated from monoculture model calibration

mu1 = 0.18 / 3600  # growth rate of strain 1 (1/dt) **
mu2 = 0.18 / 3600  # growth rate of strain 2 (1/dt) **
Rhf = 180  # accumulation parameter **
Qhf = 40  # pressure parameter **
Phf = 100  # friction parameter **
Y = 40  # glucose yield **
Ks = 0.001  # half-maximal growth concentration of glucose (1/S0) **
DC = 2200  # biomass dispersal coefficient (dx^2) **
DS = 120  # glucose diffusion coefficient (dx^2/dt) **
DA1 = 100  # antibiotic 1 diffusion coefficient (dx^2/dt)
DA2 = 100  # antibiotic diffusion coefficient (dx^2/dt)
IC1 = 0.5  # half-maximal inhibition concentration of antibiotic 1 (A10)
IC2 = 0.5  # half-maximal inhibition concentration of antibiotic 2 (A20)
n1 = 1  # slope parameter of antibiotic 1 penalty
n2 = 4  # slope parameter of antibiotic 2 penalty
Km1 = 0.1  # half-maximal concentration of antibiotic 1 inactivation (A10)
Km2 = 0.1  # half-maximal concentration of antibiotic 2 inactivation (A20)
V1 = 0.001  # speed of antibiotic 1 inactivation (A10/dt)
V2 = 0.001  # speed of antibiotic 2 inactivation (A20/dt)

model_parameters = np.array([mu1, mu2, Rhf, Qhf, Phf, Y, Ks, DC, DS, DA1, DA2, IC1, IC2, n1, n2, Km1, Km2, V1, V2])

r = 5  # radius of initial spot in pixels
I0 = 10  # initial total biomass density
f1 = 0.5  # initial fractional abundance of strain 1
Lx, Ly = 82, 102  # frame dimensions of frame in pixels

init_parameters = np.array([r, I0, f1, Lx, Ly])

dx = 50  # distance between pixels in um
dt = 1  # time interval between simulation steps in seconds
T = 2000  # Total simulation time in seconds

res = dt / dx ** 2

""" Initialization """


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def disk_init(init_parameters):
    """ Initialize a size (Lx, Ly) array with initial total density I0 and fractional abundance f1 for strain 1,
    in a circular region in the centre with radius = r pixels"""
    r, I0, f1, Lx, Ly = init_parameters

    C1 = np.ones((int(Lx) + 1, int(Ly) + 1),dtype=int) * I0 * f1
    C2 = np.ones((int(Lx) + 1, int(Ly) + 1),dtype=int) * I0 * (1 - f1)
    S = np.ones((int(Lx) + 1, int(Ly) + 1),dtype=int)
    A1 = np.ones((int(Lx) + 1, int(Ly) + 1),dtype=int)
    A2 = np.ones((int(Lx) + 1, int(Ly) + 1),dtype=int)

    # mask a circular region in the middle to load cells. Clear cells everywhere else
    h, w = C1.shape[:2]
    mask = create_circular_mask(h, w, radius=r)
    C1[~mask] = 0
    C2[~mask] = 0

    return (C1, C2, S, A1, A2)


""" Simulation """


def run_model(init_parameters, model_parameters, res, dt):
    C1_time = []
    C2_time = []

    C1, C2, S, A1, A2 = disk_init(init_parameters)
    C1_time.append(C1)
    C2_time.append(C2)

    t = 0
    while t < T:
        C1, C2, S, A1, A2 = fn.model_update(C1, C2, S, A1, A2, model_parameters, res, dt)
        C1_time.append(C1)
        C2_time.append(C2)

        t += 1

    return (C1_time, C2_time)
run_model(init_parameters, model_parameters,res,dt)