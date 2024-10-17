import numpy as np
import matplotlib.pyplot as plt

# Parameters
Nh = 2**9           # spatial domain resolution
Nt = 2**11          # time steps
Q = 1               # FDTD quality factor
ome0 = 5 * 2 * np.pi    # center frequency pulse, THz
pulseW = 10 / ome0     # envelope of pulse in time, ps
EA = 100             # amplitude of pulse, ab.unit
t0 = 1               # launch time of pulse, ps
d = 500              # thickness of simulated region, mum
eps00 = 13           # background refractive index, GaAs n = 3.6^2
Nmax = 1e5           # density of free electrons, mum^-3
g = 0.5 * 2 * np.pi  # electron scattering rate, ps^-1
rb = 0.01            # bulk recombination rate, ps^-1
w0 = 5 * 2 * np.pi   # resonance frequency, ps^-1
meff = 1             # effective electron mass, meff*me = me*
rs = 1               # surface recombination velocity, mum/ps
D = 50               # diffusion coefficient, mum^2/ps
abp = 100            # absorption depth, mum
Mst = 100            # medium surface
Amax = 0.186         # absorption coefficient of PML
Am = 1.429           # PML m parameter
Ah = 20              # width of PML
c0 = 300             # light speed, mum/ps
e0 = 0.16            # electronic charge, muAps
eps0 = 8.0657        # vacuum permittivity, muA^2 ps^4/me mum^3
h = d / Nh           # spatial step size, mum
dt = Q * h / c0      # time step size, ps

# Initiate material response
A = np.zeros(Nh)
A[:Ah] = Amax * (1 - (np.arange(Ah)) / Ah)**Am  # left part PML E
A[-Ah:] = Amax * (np.arange(Ah) / Ah)**Am       # right part PML E
App = 1 + A            # plus PML E vector
Amm = 1 - A            # minus PML E vector
AH = (A[1:] + A[:-1]) / 2
AH = np.append(AH, AH[-1])
AHpp = 1 + AH          # plus PML H vector
AHmm = 1 - AH          # minus PML H vector

z = np.linspace(0, h * Nh, Nh)  # position of E, mum
E = np.zeros(Nh)                # electric field, ab.unit.
H = np.zeros(Nh)                # magnetic field, 
z0 = z[Mst]                     # medium surface
ep = np.ones(Nh)                # permittivity vector
ep[Mst:] = eps00                # setup permittivity vector from eps00
eee = e0 / (ep * eps0)          # response parameter
Ms = Nh - Mst                   # medium size
D = D * np.ones(Ms)             # diffusion constant, mum^2/ps
g = g * np.ones(Ms)             # electron scattering rate, ps^-1
al = (4 - 2 * w0**2 * dt**2) / (2 + g * dt)  # alpha parameter
be = (g * dt - 2) / (g * dt + 2)             # beta parameter
th = 2 * e0 * dt**2 / meff / (2 + g * dt)    # theta parameter
x0 = np.zeros(Ms)             # temporary displacement vector
x = np.zeros(Ms)              # current displacement vector
zz = (z[(Mst):] - z0)         # reduced position vector
N = Nmax * np.exp(-zz / abp)  # Nmax*ones(Ms,1)

# Time-stepping loop
for n in range(Nt):
    # Source
    S = EA * np.sin(ome0 * (n * dt - t0)) * np.exp(-((n * dt - t0) ** 2) / (2 * pulseW ** 2))
    E[Ah + 5] += S               # adding current source 
    H[Ah + 5] += Q * S           # unidirectional propagation
    
    # Store old density and evolve N
    N0 = N.copy()
    N[1:Ms-1] = N[1:Ms-1] * (1 - dt * rb - 2 * dt / h**2 * D[1:Ms-1]) \
                + dt * D[1:Ms-1] / h**2 * (N[2:Ms] + N[0:Ms-2])
    N[Ms-1] = N0[Ms-1] * (1 - dt * rb - dt / h**2 * D[Ms-1]) \
              + dt * D[Ms-1] / h**2 * N0[Ms-2]
    N[0] = N0[0] * (1 - dt * rs / h - dt * rb - dt / h**2 * D[0]) \
           + dt * D[0] / h**2 * N0[1]

    # Evolve x fields (displacement)
    x1 = x0.copy()
    x0 = x.copy()
    x = al * x0 + be * x1 - th * E[Mst:]
    J = N * (x - x0) * eee[Mst:]

    # Ensure consistent array shapes
    Nh = len(E)

    # Evolve E field (align dimensions carefully)
    E[0] = 1.0 / App[0] * (Amm[0] * E[0] - Q / ep[0] * H[0])
    E[1:Mst] = 1.0 / App[1:Mst] * (Amm[1:Mst] * E[1:Mst] - Q / ep[1:Mst] * (H[1:Mst] - H[0:Mst-1]))
    E[Mst:Nh] = 1.0 / App[Mst:Nh] * (Amm[Mst:Nh] * E[Mst:Nh] - Q / ep[Mst:Nh] * (H[Mst:Nh] - H[Mst-1:Nh-1])) + J

    # Evolve H field
    H[:-1] = 1 / AHpp[:-1] * (AHmm[:-1] * H[:-1] - Q * (E[1:] - E[:-1]))
    H[-1] = 1 / AHpp[-1] * (AHmm[-1] * H[-1] + Q * E[-1])

    # Plot in 'realtime'
    if n % 10 == 0:  # Plot every 10 steps
        plt.cla()
        plt.plot(z, E / EA * 2, label='E field')
        plt.plot(zz + z0, N / Nmax, label='N density')
        plt.ylim([-0.5, 1.1])
        plt.legend()
        plt.pause(0.01)

plt.show()
