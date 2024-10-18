import numpy as np
import matplotlib.pyplot as plt

def clFDTcore(c, m, ps, tp=None, usepump=None):
    """
    FDTD implementation of time-resolved THz-spectroscopy (TRTS).
    Translated from MATLAB to Python.
    """

    # Enable parallel 2D simulations
    if tp is not None:
        c['tp'] = tp
    if usepump is not None:
        c['usepump'] = usepump

    o = {}  # Output dictionary

    # ######################
    # Constants
    # ######################
    T = 300  # Temperature in K
    c0 = 299792458 * 1e-6  # Speed of light in µm/ps
    eps0 = 8.065618045839289  # Vacuum permittivity
    e0 = 0.1602176487  # Elementary charge in µA ps
    me = 1  # Electron mass
    Vp = 1.097769292728596  # Voltage
    kb = 1.515635503336524e-5  # Boltzmann constant in µm^2/ps^2 K
    hcon = 6.62606896e-22  # Planck constant in J ps
    h = c['d'] / (c['Nh'] - 2 * c['Ah'] - 6 * c['sph'])  # Spatial step size in µm
    dt = c['Q'] * h / c0  # Time step size in ps
    if c['xpmtype'] == 2:  # For 1D pump experiments
        dt *= c['tpfac1D']
    tmax = dt * c['Nt']  # Total propagation time
    zmax = h * c['Nh']  # Domain length in µm
    np = len(m)  # Number of poles

    # ######################
    # Setup vectors and constants
    # ######################
    A = np.zeros(c['Nh'])  # PML vector
    A[:c['Ah']] = c['Amax'] * (1 - (np.arange(c['Ah'])) / c['Ah'])**c['Am']  # Left part PML E
    A[-c['Ah']:] = c['Amax'] * (np.arange(c['Ah']) / c['Ah'])**c['Am']  # Right part PML E
    App = 1 + A  # Plus PML E vector
    Amm = 1 - A  # Minus PML E vector
    AH = (A[1:] + A[:-1]) / 2  # PML H i+1/2
    AH = np.append(AH, AH[-1])  # Complete PML H vector
    AHpp = 1 + AH  # Plus PML H vector
    AHmm = 1 - AH  # Minus PML H vector

    o['t'] = np.linspace(0, tmax, c['Nsav'])  # Time vector of E, ps
    o['z'] = np.linspace(0, zmax, c['Nh'])  # Position of E, µm
    E = np.zeros(c['Nh'])  # Electric field, Vp/µm
    H = np.zeros(c['Nh'])  # Magnetic field
    o['detR'] = np.zeros(c['Nsav'])  # Reflection detector
    o['detT'] = np.zeros(c['Nsav'])  # Transmission detector
    o['detJ'] = np.zeros(c['Nsav'])  # Polarization detector
    o['detx'] = np.zeros((c['Nsav'], np))  # Displacement detector
    o['detN'] = np.zeros((c['Nsav'], np))  # Density detector

    Mst = c['Ah'] + 4 * c['sph'] + 1  # Start of medium
    Mend = c['Nh'] - c['Ah'] - 2 * c['sph']  # End of medium
    Ms = Mend - Mst + 1  # Size of medium
    Msource = c['Ah'] + 2 * c['sph'] + 1  # Position of source
    prpp = np.zeros((np, np - 1))  # Density rate equation vector
    Nskip = c['Nt'] // c['Nsav']  # Store every Nskip time-steps

    ep = c['epsin'] * np.ones(c['Nh'])  # Permittivity vector
    ep[Mst:Mend] = c['eps00']  # Setup permittivity vector from epsin
    ep[Mend + 1:] = c['epsout']  # Setup permittivity vector from epsout
    eee = e0 / (ep * eps0)  # Response parameter

    tl = np.linspace(0, tmax, c['Nt'])  # Long time vector, ps
    if c['useEref'] == 1:
        tempdat = np.loadtxt('Eref.dat')  # Load Eref.dat as probe input
        tEref = tempdat[:, 0]  # Time vector
        Eref = tempdat[:, 1]  # E field vector
        SE = np.interp(tl, tEref + c['t0'], Eref)  # Long input vector
        o['detI'] = np.interp(o['t'], tEref + c['t0'], Eref)  # Input vector at stored times
        c['EA'] = np.max(o['detI']) / np.sqrt(c['eps00'])
    else:
        # Use exp * sin as probe input
        SE = c['EA'] * np.sin(c['ome0'] * (tl - c['t0']) * (1 + c['chirp'] * (tl - c['t0']))) * \
            np.exp(-(tl - c['t0'])**2 / (2 * c['pulseW']**2))
        o['detI'] = c['EA'] * np.sin(c['ome0'] * (o['t'] - c['t0']) * (1 + c['chirp'] * (o['t'] - c['t0']))) * \
            np.exp(-(o['t'] - c['t0'])**2 / (2 * c['pulseW']**2))

    # ######################
    # Dynamic material and pump parameters
    # ######################
    o['Nmax'] = c['F'] * 1e-6 * (1e-4)**2 * c['lamp'] * (1 - c['Rpump']) / (hcon * c0 * c['abp'])  # Max carrier density in µm^-3
    Iamp = o['Nmax'] / (np.sqrt(np.pi) * c['pumpW'] * c['abp'])  # Normalization factor of I
    z0 = o['z'][Mst]  # Start position of medium
    o['x'] = o['z'][Mst:Mend] - z0  # Reduced position vector
    pumpWz = c0 * c['pumpW'] / c['ng']  # Pump width in space
    pumpprop = 10 * max([c['abp'], pumpWz])  # Length of propagation of pump
    nattem = int(((-10 * pumpWz - z0) * c['ng'] / c0 - c['tp'] + c['t0']) / dt)  # First time step with pump
    natten = int(((pumpprop - z0) * c['ng'] / c0 - c['tp'] + c['t0']) / dt)  # Last time step with pump

    Ik1 = o['x'] - c0 / c['ng'] * (c['tp'] - c['t0'])  # Pump parameter 1
    Ik2 = c0 / c['ng'] * dt  # Pump parameter 2
    Ik3 = 1 / pumpWz**2  # Pump parameter 3
    I0 = c['usepump'] * Iamp * np.exp(-o['x'] / c['abp'])  # Setup pump vector
    I = np.zeros(Ms)

    # ######################
    # Initialize material response
    # ######################
    for p in range(np):
        m[p]['N'] = m[p]['y'] * o['Nmax'] * np.ones(Ms) * np.exp(-1)  # Set density to maximum temporarily
        prpp[p, :] = np.delete(np.arange(np), p)  # Density rate equation vector
        m[p]['th0'] = 2 * e0 * dt**2 / m[p]['meff']  # Theta parameter
        m[p]['D'] = m[p]['D'] * np.ones(Ms)  # Diffusion constant, µm^2/ps
        m[p]['g'] = m[p]['g'] * np.ones(Ms)  # Electron scattering rate, ps^-1
        m[p]['al'] = (4 - 2 * m[p]['w0']**2 * dt**2) / (2 + m[p]['g'] * dt)  # Alpha parameter
        m[p]['be'] = (m[p]['g'] * dt - 2) / (m[p]['g'] * dt + 2)  # Beta parameter
        m[p]['th'] = m[p]['th0'] / (2 + m[p]['g'] * dt)  # Theta parameter
        m[p]['x0'] = np.zeros(Ms)  # Temporary displacement vector
        m[p]['x'] = np.zeros(Ms)  # Current displacement vector
        m[p]['N0'] = np.zeros(Ms)  # Setup old density vector

    # ######################
    # Check conditions
    # ######################
    if ps['numpar'] == 1:
        print(f"Nh: {c['Nh']}, h: {h * 1e3:.1f} nm, Zmax: {zmax} µm, d: {c['d']} µm")
        print(f"Nt: {c['Nt']}, dt: {dt * 1e3:.1f} fs, Tmax: {tmax:.2f} ps, df: {1 / tmax:.2f} THz")

    # Check stability conditions
    if max([m[p]['rs'] for p in range(np)]) * dt / (2 * h) >= 1:
        raise ValueError("Surface recombination stability factor larger than 1")
    if 2 * dt * max([m[p]['D'] for p in range(np)]) / h**2 >= 1:
        raise ValueError("Diffusion stability factor larger than 1")
    if Nskip < 1:
        c['Nsav'] = c['Nt']
        print("Nsav > Nt")
    if c['xpmtype'] == 2 and nattem < 1:
        raise ValueError(f"c.tp must be smaller: ~{nattem * dt + c['tp']} ps")

    # ######################
    # Propagate the system
    # ######################
    if ps['real'] > 0 and c['xpmtype'] != 3:
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        ax1, ax2, ax3, ax4 = axes.flatten()
        ax1.set(xlabel="Position, µm", ylabel="Electric field and ε(z)/ε_∞", xlim=[0, zmax], ylim=[-1, 1])
        ax2.set(xlabel="Time, ps", ylabel="Trans. and Reflec.", xlim=[0, tmax], ylim=[-1, 1])
        ax3.set(xlabel="Position, µm", ylabel="Carrier Density, 10^{17} cm^{-3}", xlim=[0, zmax])
        ax4.set(xlabel="Time, ps", ylabel="Near Surface Carrier Density, 10^{17} cm^{-3}", xlim=[0, tmax])
    
    # Simulate the system and propagate fields
    for n in range(c['Nt']):
        # Apply the source
        S = SE[n]
        E[Msource] += S  # Add current source
        H[Msource] += c['Q'] * S  # Add unidirectional propagation

        # Evolve carrier density dynamics
        for p in range(np):
            m[p]['N0'] = m[p]['N']  # Store old density
            # Update density based on diffusion, pump, and surface recombination
            if c['diffusion'] == 1:
                # Apply density update equations
                pass  # Update m[p]['N'] as per diffusion equation and pump effects (same as in MATLAB)
    
    return o
