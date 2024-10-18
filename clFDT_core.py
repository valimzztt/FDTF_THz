import numpy as np
import matplotlib.pyplot as plt


def clFDTcore(c, m, ps, tp=None, usepump=None):
    """
    Core function for FDTD-TRTS in Python
    """
    if tp is not None:
        c.tp = tp
    if usepump is not None:
        c.usepump = usepump

    o = {}  # output structure

    # Constants
    T = 300  # temperature in K
    c0 = 299792458 * 1e-6  # light speed, µm/ps
    eps0 = 8.065618045839289  # vacuum permittivity, µA^2 ps^4/me µm^3
    e0 = 0.1602176487  # elementary charge, µA ps
    me = 1  # electron mass, 1 me
    Vp = 1.097769292728596  # voltage
    kb = 1.515635503336524e-5  # boltzmann constant
    hcon = 6.62606896e-022  # planck constant in J ps
    h = c.d / (c.Nh - 2 * c.Ah - 6 * c.sph)  # spatial step size, µm
    dt = c.Q * h / c0  # time step size, ps

    if c.xpmtype == 2:  # 1D pump
        dt *= c.tpfac1D  # increase time step by tpfac1D

    tmax = dt * c.Nt  # propagation time, ps
    zmax = h * c.Nh  # domain length, µm
    npoles = len(m)  # number of poles

    # Setup vectors
    A = np.zeros(c.Nh)
    A[:c.Ah] = c.Amax * (1 - (np.arange(1, c.Ah + 1) - 1) / c.Ah) ** c.Am
    A[-c.Ah:] = c.Amax * (np.arange(1, c.Ah + 1) / c.Ah) ** c.Am

    App = 1 + A
    Amm = 1 - A
    AH = (A[1:c.Nh] + A[:c.Nh - 1]) / 2
    AH = np.append(AH, AH[-1])

    AHpp = 1 + AH
    AHmm = 1 - AH

    o['t'] = np.linspace(0, tmax, c.Nsav)  # time vector, ps
    o['z'] = np.linspace(0, zmax, c.Nh)  # position, µm

    E = np.zeros(c.Nh)  # electric field, Vp/µm
    H = np.zeros(c.Nh)  # magnetic field
    o['detR'] = np.zeros(c.Nsav)  # reflection detector
    o['detT'] = np.zeros(c.Nsav)  # transmission detector
    o['detJ'] = np.zeros(c.Nsav)  # polarization detector
    o['detx'] = np.zeros((c.Nsav, npoles))  # displacement detector
    o['detN'] = np.zeros((c.Nsav, npoles))  # density detector

    Mst = c.Ah + 4 * c.sph + 1  # start of medium
    Mend = c.Nh - c.Ah - 2 * c.sph  # end of medium
    Ms = Mend - Mst + 1  # size of medium
    Msource = c.Ah + 2 * c.sph + 1  # position of source

    prpp = np.zeros((npoles, npoles - 1))  # density rate equation vector
    Nskip = c.Nt // c.Nsav  # store every Nskip time-steps

    ep = np.full(c.Nh, c.epsin)
    ep[Mst:Mend] = c.eps00
    ep[Mend:] = c.epsout
    eee = e0 / (ep * eps0)

    tl = np.linspace(0, tmax, c.Nt)  # long time vector, ps

    if c.useEref == 1:  # use 'Eref.dat' as probe input
        tempdat = np.loadtxt('Eref.dat')  # read data
        tEref = tempdat[:, 0]  # time vector
        Eref = tempdat[:, 1]  # E field vector
        SE = np.interp(tl, tEref + c.t0, Eref)  # interpolate
        o['detI'] = np.interp(o['t'], tEref + c.t0, Eref)  # stored times
        c.EA = max(o['detI']) / np.sqrt(c.eps00)
    else:  # use exp * sin as probe input
        SE = c.EA * np.sin(c.ome0 * (tl - c.t0) * (1 + c.chirp * (tl - c.t0))) * np.exp(-(tl - c.t0) ** 2 / (2 * c.pulseW ** 2))
        o['detI'] = c.EA * np.sin(c.ome0 * (o['t'] - c.t0) * (1 + c.chirp * (o['t'] - c.t0))) * np.exp(-(o['t'] - c.t0) ** 2 / (2 * c.pulseW ** 2))

    # Dynamic material and pump parameters
    o['Nmax'] = c.F * 1e-6 * (1e-4) ** 2 * c.lamp * (1 - c.Rpump) / (hcon * c0 * c.abp)  # maximum carrier density
    Iamp = o['Nmax'] / (np.sqrt(np.pi) * c.pumpW * c.abp)
    z0 = o['z'][Mst]  # start position of medium
    o['x'] = o['z'][Mst:Mend] - z0  # reduced position vector

    pumpWz = c0 * c.pumpW / c.ng  # pump width in space
    pumpprop = 10 * max(c.abp, pumpWz)  # length of propagation of pump
    nattem = round(((-10 * pumpWz - z0) * c.ng / c0 - c.tp + c.t0) / dt)  # first pump time step
    natten = round(((pumpprop - z0) * c.ng / c0 - c.tp + c.t0) / dt)  # last pump time step

    Ik1 = o['x'] - c0 / c.ng * (c.tp - c.t0)  # pump parameter 1
    Ik2 = c0 / c.ng * dt  # pump parameter 2
    Ik3 = 1 / pumpWz ** 2  # pump parameter 3
    I0 = c.usepump * Iamp * np.exp(-o['x'] / c.abp)  # pump vector
    I = np.zeros(Ms)

    # Initialize material response
    for p in range(npoles):
        m[p].N = m[p].y * o['Nmax'] * np.ones(Ms) * np.exp(-1)
        prpp[p, :] = np.delete(np.arange(npoles), p)
        m[p].th0 = 2 * e0 * dt ** 2 / m[p].meff
        m[p].D = m[p].D * np.ones(Ms)
        m[p].g = m[p].g * np.ones(Ms)
        m[p].al = (4 - 2 * m[p].w0 ** 2 * dt ** 2) / (2 + m[p].g * dt)
        m[p].be = (m[p].g * dt - 2) / (m[p].g * dt + 2)
        m[p].mue = e0 / (m[p].g[0] * m[p].meff)
        m[p].muh = m[p].mue
        m[p].th = m[p].th0 / (2 + m[p].g * dt)
        m[p].x0 = np.zeros(Ms)
        m[p].x = np.zeros(Ms)
        m[p].N0 = np.zeros(Ms)

    # Check for stability conditions
    if ps.numpar  == 1:
        print(f'Nh: {c.Nh}, h: {h * 1e3}nm, Zmax: {zmax}µm, d: {c.d}µm | Nt: {c.Nt}, dt: {dt * 1e3}fs, Tmax: {tmax}ps, df: {1/tmax}THz')
        for p in range(npoles):
            print(f'Term: {p}, F: {m[p].y * c.F:.2f} µJ/cm^2, '
            f'Nmax: {m[p].y * o["Nmax"] * 1e12:.3g} cm^-3, '
            f'mu_e: {m[p].mue * Vp * 1e4:.4g} cm^2/Vs, '
            f'mu_h: {m[p].muh * Vp * 1e4:.4g} cm^2/Vs, '
             f'D: {m[p].D[0] * 1e4:.3g} cm^2/s, '
            f's: {m[p].rs * 1e8:.3g} cm/s, '
            f'gamma: {m[p].g[0]:.3g} aTHz'
           )

    # Implement real-time plotting setup if needed using Matplotlib
    if ps.real > 0 and c.xpmtype != 3:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        ax1, ax2, ax3, ax4 = axs.flatten()
        ax1.set_xlabel('Position, µm')
        ax1.set_ylabel('Electric field and ε(z)/ε∞')
        ax2.set_xlabel('Time, ps')
        ax2.set_ylabel('Trans. and Reflec.')
        ax3.set_xlabel('Position, µm')
        ax3.set_ylabel('Carrier Density, 10^{17} cm^{-3}')
        ax4.set_xlabel('Time, ps')
        ax4.set_ylabel('Near Surface Carrier Density, 10^{17} cm^{-3}')

    # Rest of the code handles the time evolution of the electric, magnetic fields, carrier densities,
    # and dynamic updates for each time step.

    # Your implementation continues here following the same logic as the MATLAB code.
    # Use Matplotlib to visualize in real-time and NumPy for efficient computation of field evolution.
    
    return o

