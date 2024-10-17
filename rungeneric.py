import numpy as np
from fdtd_plotting import clFDTplot

# ######################
# Numerical stepping parameters
# ######################
class Config:
    def __init__(self):
        self.xpmtype = 1         # experiment type: 1 (1Dprobe), 2 (1Dpump), 3 (2Dpump-probe)
        self.prop = 1            # evolve system, 0 or 1
        self.parallel = 0        # Enable parallel processing of 2Dpumpprobe
        self.diffusion = 0       # activation of diffusion and surface recombination of carriers
        self.xpmid = 'test'      # name of experiment (used for saving)
        self.save = 0            # save input and output data
        self.Nh = 2**9           # spatial domain resolution
        self.Nt = 2**16          # time steps
        self.Nsav = 2**10        # number of time steps stored
        self.Q = 1               # FDTD quality factor

# ######################
# Probe pulse
# ######################
class ProbePulse:
    def __init__(self, config):
        self.useEref = 0         # use the eref in file 'Eref.dat'
        self.EA = 100            # amplitude of pulse
        self.t0 = 1              # time of start of pulse, ps
        if self.useEref == 0:
            self.ome0 = 3 * 2 * np.pi     # center frequency, THz
            self.pulseW = 1.5 / self.ome0 # envelope of pulse in time, ps
            self.chirp = 0                # chirp of probe pulse

# ######################
# Pump pulse
# ######################
class PumpPulse:
    def __init__(self):
        self.usepump = 0         # use the pump: 0 or 1
        self.tp = 1              # pump delay [1Dprobe] or first pump delay [1Dpump], ps
        self.ng = 4              # group index of refraction of pump
        self.pumpW = 0.1         # pump time-width, ps
        self.F = 80              # pump fluence, µJ/cm^2/pulse
        self.lamp = 0.8          # pump wavelength, µm
        self.abp = 100           # absorption depth of pump, µm
        self.Rpump = 0.33        # reflectance of pump at pump wavelength
        
        # Specific for 2D pumpprobe
        self.mintp = -4          # first pump delay, ps
        self.maxtp = 2           # last pump delay, ps
        self.Ntp = 10            # number of pump delays

        # Specific for 1D pump
        self.tpfac1D = 200       # increase time step size by this factor

        # When usepump = 0
        self.pumpexp = 0         # exponential density at start (for long times)
        self.densityconst = 1    # homogeneous density at start

# ######################
# Medium
# ######################
class Medium:
    def __init__(self):
        self.d = 10              # thickness of simulated region, µm
        self.eps00 = 13          # background relative permittivity
        self.epsin = 1           # relative permittivity at input side
        self.epsout = self.eps00 # relative permittivity at output side

# First excited medium
class ExcitedMedium:
    def __init__(self):
        self.g = 0.01 * 2 * np.pi     # electron scattering rate, ps^-1
        self.rb = 0                  # bulk recombination rate, ps^-1
        self.w0 = 2 * 2 * np.pi      # resonance frequency, THz
        self.meff = 0.067            # effective electron mass, meff*me = me*
        self.rs = 1e6 * 1e-8         # surface recombination velocity, µm/ps
        self.D = 21 * 1e-4           # diffusion coefficient, µm^2/ps
        self.y = 1                   # yield of pump
        self.rpp = [0, 0, 0]         # rate from other poles [p1 p2 p3], ps^-1

# ######################
# PML (boundaries)
# ######################
class PML:
    def __init__(self):
        self.Amax = 0.18574656389     # absorption coefficient of PML
        self.Am = 1.4293377164138     # PML m parameter
        self.Ah = 20                  # width of PML
        self.sph = 3                  # size of space between elements in the domain

# ######################
# Plot parameters
# ######################
class PlotParams:
    def __init__(self):
        self.numpar = 1               # display numerical and material parameters message
        self.real = 2**9              # "live" plotting every # iterations
        self.plot = 1                 # analysis of results by various plots
        self.fxlim = [0.3, 3]         # plot frequency limits (determined by bandwidth of probe)
        self.Nfft = 2**12             # resolution of FFT when plotting

# ######################
# Setup and function call
# ######################

# Create instances for configuration
config = Config()
probe_pulse = ProbePulse(config)
pump_pulse = PumpPulse()
medium = Medium()
excited_medium1 = ExcitedMedium()

# Initialize PML and plot parameters
pml = PML()
plot_params = PlotParams()

# Medium is a list if there are multiple
mediums = [excited_medium1]
o = [{
    't': np.linspace(0, 10, 1000),
    'detI': np.sin(np.linspace(0, 10, 1000)),
    'detR': np.sin(np.linspace(0, 10, 1000) * 0.8),
    'detT': np.sin(np.linspace(0, 10, 1000) * 0.6),
    'detN': np.sin(np.linspace(0, 10, 1000) * 0.4),
    'Nmax': 1
}]
 
# Call main FDTD function
output = clFDTplot(config, mediums, plot_params)

# Print the output for verification (for example)
print(output)
