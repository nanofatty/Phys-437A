import numpy as np
from PIL import Image

def resolve_down(im, factor):
    """averages value over a square region of pixels of size factor x factor"""

    # get pixel size of image
    Lx, Ly = im.size

    # get pixel offset towards all 4 margins and crop them out
    LR_off = Lx % factor
    L_off = int(LR_off / 2)
    R_off = Lx - LR_off + L_off
    TB_off = Ly % factor
    T_off = int(TB_off / 2)
    B_off = Ly - TB_off + T_off

    im1 = np.array(im.crop((L_off, T_off, R_off, B_off)))

    # resolve down image along the rows
    im2 = []
    for a in im1:
        b = a.reshape(-1, factor).mean(axis=1)
        im2.append(b)
    im2 = np.array(im2).T

    # resolve down the image along the columns
    im3 = []
    for a in im2:
        b = a.reshape(-1, factor).mean(axis=1)
        im3.append(b)
    im3 = np.array(im3).T

    return im3

def read_data_as_array(input):
    """ for given file location, time-series data of Phase and fluorescence channels are imported as numpy arrays """

    # Read input parameters for data formatting
    path, fc1, fc2, time_i, time_f, nump, umpp, thresh, dthr, sd_min = tuple(input)

    Phase = []
    FC1 = []
    FC2 = []

    t = time_i+1
    while t < time_f:
        imp = Image.open(path + str(t).zfill(2) + "_Phase.jpg").convert('L')
        imp_resized = resolve_down(imp, nump)
        imf1 = Image.open(path + str(t).zfill(2) + "_{}.jpg".format(fc1)).convert('L')
        imf1_resized = resolve_down(imf1, nump)
        imf2 = Image.open(path + str(t).zfill(2) + "_{}.jpg".format(fc2)).convert('L')
        imf2_resized = resolve_down(imf2, nump)

        Phase.append(imp_resized)
        FC1.append(imf1_resized)
        FC2.append(imf2_resized)
        t += 1

    Phase = np.array(Phase)
    FC1 = np.array(FC1)
    FC2 = np.array(FC2)

    """ Use fluorescence channel as a mask to eliminate intensities in 'cell-free' regions of phase channel images """
    """ Using a threshold to account for dispersion of emitted light """
    for i in range(len(Phase)):
        mask1 = FC1[i] > thresh * np.max(FC1[i])
        mask2 = FC2[i] > thresh * np.max(FC2[i])
        FC1[i][~mask1] = 0
        FC2[i][~mask2] = 0
        Phase[i][mask1 + mask2] = 0

    return(Phase, FC1, FC2)

def nbr_sum(M):
    """sum of neighboring values of elements of matrix M"""
    L = np.roll(M, (0, -1), (0, 1))   # right neighbor
    L += np.roll(M, (0, +1), (0, 1))  # left neighbor
    L += np.roll(M, (-1, 0), (0, 1))  # top neighbor
    L += np.roll(M, (+1, 0), (0, 1))  # bottom neighbor

    return L

def nbr_sum_jit(M):
    L = np.zeros(M.shape)
    for y in range(M.shape[1]-1):
        L[:, y] = M[:, y+1] + M[:, y-1]
    for x in range(M.shape[0]-1):
        L[x, :] = M[x+1, :] + M[x-1, :]
    return L

def discrete_laplacian(M):
    """Get the discrete Laplacian of matrix M"""
    L = -4 * M
    L += nbr_sum(M)

    return L

def divide(M1, M2):
    """To divide two numpy arrays M1/M2 element-wise and avoid division by zero"""
    f = np.divide(M1, M2, out=np.zeros_like(M1), where=M2 != 0)
    return f

def hill(A, Ahf):
    return A/(Ahf+A)

def inv_hill(A, Ahf):
    return Ahf/(Ahf+A)

def penalty(A, IC, n):
    pen = 1/(1+(A/IC)**n)
    return pen

def model_update(C1, C2, S, A1, A2, Parameters, res, dt):
    """Update state variables according to the single-species continuum model"""
    mu1, mu2, Rhf, Qhf, Phf, Y, Ks, DC, DS, DA1, DA2, IC1, IC2, n1, n2, Km1, Km2, V1, V2 = Parameters

    C = C1 + C2
    R, Q, P = inv_hill(C, Rhf), hill(C, Qhf), inv_hill(C, Phf)

    # maximum growth rate
    M1 = mu1 * C1 * penalty(A2, IC2, n2) * hill(S, Ks)
    M2 = mu2 * C2 * penalty(A1, IC1, n1) * hill(S, Ks)
    M = M1 + M2

    f1 = divide(C1, C)
    f2 = divide(C2, C)

    Sdiff = res * DS * discrete_laplacian(S) - dt * M * R / Y
    C1diff = res * DC * (P * nbr_sum(f1 * M * Q) - f1 * M * Q * nbr_sum(P)) + dt * M1 * R
    C2diff = res * DC * (P * nbr_sum(f2 * M * Q) - f2 * M * Q * nbr_sum(P)) + dt * M2 * R
    A1diff = res * DA1 * discrete_laplacian(A1) - dt * C1 * hill(A1, Km1)
    A2diff = res * DA2 * discrete_laplacian(A2) - dt * C2 * hill(A2, Km2)

    C1 += C1diff
    C2 += C2diff
    S += Sdiff
    A1 += A1diff
    A2 += A2diff

    # Remove negative values
    S[S < 0] = 0
    C1[C1 < 0] = 0
    C2[C2 < 0] = 0
    A1[A1 < 0] = 0
    A2[A2 < 0] = 0
    # Reflecting boundaries
    C1[0, :], C1[-1, :], C1[:, 0], C1[:, -1] = C1[1, :], C1[-2, :], C1[:, 1], C1[:, -2]
    C2[0, :], C2[-1, :], C2[:, 0], C2[:, -1] = C2[1, :], C2[-2, :], C2[:, 1], C2[:, -2]
    S[0, :], S[-1, :], S[:, 0], S[:, -1] = S[1, :], S[-2, :], S[:, 1], S[:, -2]
    A1[0, :], A1[-1, :], A1[:, 0], A1[:, -1] = A1[1, :], A1[-2, :], A1[:, 1], A1[:, -2]
    A2[0, :], A2[-1, :], A2[:, 0], A2[:, -1] = A2[1, :], A2[-2, :], A2[:, 1], A2[:, -2]


    return C1, C2, S, A1, A2
