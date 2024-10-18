import numpy as np
import time
import matplotlib.pyplot as plt

def clFDTmainlite(c, m, ps):
    """
    FDTD implementation of time-resolved THz-spectroscopy (TRTS).
    Translated from MATLAB to Python.
    """
    start_time = time.time()  # Record time
    if c['xpmtype'] == 1:  # 1D probe experiment
        o = clFDTcorelite(c, m, ps)
    elif c['xpmtype'] == 2:  # 1D pump experiment
        o = clFDTcorelite(c, m, ps)
    elif c['xpmtype'] == 3:  # 2D pump probe experiment
        o = FDT2Dpumpprobelite(c, m, ps)

    o['toc'] = time.time() - start_time  # Record elapsed time
    print(f"Calculation Time: {o['toc']:.2f} seconds")

    if c['save'] == 1:
        timestamp = time.strftime('-%Y-%m-%d-%H-%M', time.gmtime())
        filename = f"{c['xpmid']}{timestamp}.npz"
        np.savez(filename, c=c, m=m, ps=ps, o=o)
        print(f"Data saved to {filename}")

    if ps['plot'] != 0 and c['prop'] == 1:
        clFDTplot(c, m, ps, o)  # Plot results

    return o


def FDT2Dpumpprobelite(c, m, ps):
    """
    2D Pump probe experiment in the 'lite' mode.
    """
    ps['real'] = 0
    ps['numpar'] = 0

    vtp = np.linspace(c['mintp'], c['maxtp'], c['Ntp'])
    detT2D = np.zeros((c['Nsav'], c['Ntp']))
    detR2D = np.zeros((c['Nsav'], c['Ntp']))

    for j in range(c['Ntp']):  # Propagate 2D
        to = clFDTcorelite(c, m, ps, vtp[j])
        detT2D[:, j] = to['detT']
        detR2D[:, j] = to['detR']

    o = clFDTcorelite(c, m, ps, np.min(vtp), usepump=0)
    o['detT2D'] = detT2D  # 2D transmission
    o['detR2D'] = detR2D  # 2D reflection
    o['vtp'] = vtp

    return o


def clFDTcorelite(c, m, ps, tp=None, usepump=None):
    """
    Core function for 1D probe and pump-probe experiments in 'lite' mode.
    """
    if tp is not None:
        c['tp'] = tp
    if usepump is not None:
        c['usepump'] = usepump

    o = {}  # Output structure

    # Constants
    T = 300  # Temperature in K
    c0 = 299792458 * 1e-6  # Speed of light in µm/ps
    eps0 = 8.065618045839289  # Vacuum permittivity
    e0 = 0.1602176487  # Elementary charge
    me = 1  # Electron mass
    Vp = 1.097769292728596  # Voltage
    hcon = 6.62606896e-22  # Planck constant
    h = c['d'] / (c['Nh'] - 2 * c['Ah'] - 6 * c['sph'])  # Spatial step size
    dt = c['Q'] * h / c0  # Time step size
    if c['xpmtype'] == 2:  # 1D pump
        dt *= c['tpfac1D']  # Increase time step by tpfac1D

    tmax = dt * c['Nt']  # Propagation time
    zmax = h * c['Nh']  # Domain length

    # Setup vectors and constants
    A = np.zeros(c['Nh'])  # PML vector
    A[:c['Ah']] = c['Amax'] * (1 - (np.arange(1, c['Ah'] + 1) - 1) / c['Ah'])**c['Am']  # Left PML
    A[-c['Ah']:] = c['Amax'] * (np.arange(1, c['Ah'] + 1) / c['Ah'])**c['Am']  # Right PML
    App = 1 + A  # Plus PML E vector
    Amm = 1 - A  # Minus PML E vector

    AH = (A[1:] + A[:-1]) / 2  # PML H i+1/2
    AH = np.append(AH, AH[-1])  # Complete vector
    AHpp = 1 + AH  # Plus PML H vector
    AHmm = 1 - AH  # Minus PML H vector

    o['t'] = np.linspace(0, tmax, c['Nsav'])  # Time vector
    o['z'] = np.linspace(0, zmax, c['Nh'])  # Position vector

    E = np.zeros(c['Nh'])  # Electric field
    H = np.zeros(c['Nh'])  # Magnetic field
    o['detR'] = np.zeros(c['Nsav'])  # Reflection detector
    o['detT'] = np.zeros(c['Nsav'])  # Transmission detector
    o['detx'] = np.zeros(c['Nsav'])  # Displacement detector
    o['detN'] = np.zeros(c['Nsav'])  # Density detector

    # Time stepping loop (1D simulation example)
    for n in range(c['Nt']):
        # Source and propagation
        # (Fill this part with the electric and magnetic field propagation code as in the MATLAB code)

        # Save data to the detectors every Nskip steps
        if n % (c['Nt'] // c['Nsav']) == 0:
            idx = n // (c['Nt'] // c['Nsav'])
            o['detR'][idx] = E[c['Ah'] + c['sph'] + 1]
            o['detT'][idx] = E[c['Nh'] - c['Ah'] - c['sph']]
            o['detx'][idx] = 0  # Replace with relevant displacement data
            o['detN'][idx] = 0  # Replace with relevant density data

    return o

def clFDTplot(c, m, ps, o):
    """
    Plotting function to visualize the results.
    """
    # You can implement your plotting logic here
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(o['z'], o['detR'], label='Reflection')
    plt.xlabel('Position (µm)')
    plt.ylabel('Field Strength (arb. units)')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(o['t'], o['detT'], label='Transmission')
    plt.xlabel('Time (ps)')
    plt.ylabel('Field Strength (arb. units)')
    plt.legend()

    plt.show()

