import numpy as np
import matplotlib.pyplot as plt

def clFDTplot(c, m, ps):
    """
    Function to plot the results of FDTD simulation.
    It calls the subplot function for each result.
    """
    no = len(o)
    for j in range(no):
        clsubplots(c, m, ps, o[j])

def clsubplots(c, m, ps, o):
    """
    Function to generate subplots based on the experiment type.
    """
    if c.xpmtype == 1:  # 1D probe
        if ps.plot == 1:  # Analyze simulation
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            
            # Plot time domain signals
            axs[0, 0].plot(o['t'], o['detI'], label='E_in')
            axs[0, 0].plot(o['t'], o['detR'], label='E_R')
            axs[0, 0].plot(o['t'], o['detT'], label='E_T')
            axs[0, 0].plot(o['t'], o['detN'] / o['Nmax'] * c.EA, label='N_surf')
            axs[0, 0].set_xlim([0, max(o['t'])])
            axs[0, 0].set_xlabel('Time (ps)')
            axs[0, 0].set_ylabel('ab.units')
            axs[0, 0].legend()
            axs[0, 0].set_title('Time domain signals')

            # Frequency domain analysis (FFT)
            f, FI, absFI, angFI = clfft(o['t'], o['detI'], ps.Nfft)
            f, FR, absFR, angFR = clfft(o['t'], o['detR'], ps.Nfft)
            f, FT, absFT, angFT = clfft(o['t'], o['detT'], ps.Nfft)
            T = absFT / absFI
            R = absFR / absFI

            print(f"df: {f[1] - f[0]}")

            # Plot FFT
            axs[1, 0].plot(f, 10 * np.log10(absFI), label='E_in')
            axs[1, 0].plot(f, 10 * np.log10(absFR), label='E_R')
            axs[1, 0].plot(f, 10 * np.log10(absFT), label='E_T')
            axs[1, 0].set_xlim(ps.fxlim)
            axs[1, 0].set_xlabel('Frequency (THz)')
            axs[1, 0].set_ylabel('E(f) (dB)')
            axs[1, 0].legend()
            axs[1, 0].set_title('Frequency domain (FFT)')

            # Reflectance and Transmittance
            axs[1, 1].plot(f, R**2, label='Reflectance')
            axs[1, 1].plot(f, T**2 * np.sqrt(c.epsout), label='Transmittance')
            axs[1, 1].set_xlim(ps.fxlim)
            axs[1, 1].set_ylim([0, 1])
            axs[1, 1].set_xlabel('Frequency (THz)')
            axs[1, 1].legend()
            axs[1, 1].set_title('Reflectance and Transmittance')

            plt.tight_layout()
            plt.show()

    # Other cases for pump-probe (xpmtype == 2) and 2D (xpmtype == 3)
    # Add as needed.

def clfft(t, data, Nfft):
    """
    Perform FFT on the time domain data.
    """
    f = np.fft.fftfreq(Nfft, d=(t[1] - t[0]))  # Frequency axis
    fft_data = np.fft.fft(data, Nfft)
    abs_fft_data = np.abs(fft_data)
    ang_fft_data = np.angle(fft_data)
    
    return f, fft_data, abs_fft_data, ang_fft_data
