import numpy as np
import matplotlib.pyplot as plt
from clfft import clfft

def clFDTplot(c, m, ps, o_list):
    """
    FDTD-TRTS plot function.
    """
    # If batch-simulation, then plot all of them
    no = len(o_list)
    for j in range(no):
        clsubplots(c, m, ps, o_list[j])

def clsubplots(c, m, ps, o):
    """
    Subplots for different experiment types.
    """
    if c.xpmtype == 1:  # 1D probe
        if ps.plot == 1:  # analyze simulation
            plt.figure()
            plt.subplot(2, 2, (1, 2))
            plt.plot(o['t'], o['detI'], label='E_in')
            plt.plot(o['t'], o['detR'], label='E_R')
            plt.plot(o['t'], o['detT'], label='E_T')
            plt.plot(o['t'], o['detN'] / o['Nmax'] * c.EA, label='N_surf')
            plt.xlim([0, max(o['t'])])
            plt.xlabel('Time, ps')
            plt.ylabel('ab.units')
            plt.legend()
            f, FI, absFI, angFI = clfft(o['t'], o['detI'], 'tf', ps.Nfft)
            f, FR, absFR, angFR = clfft(o['t'], o['detR'], 'tf',ps.Nfft)
            f, FT, absFT, angFT = clfft(o['t'], o['detT'], 'tf', ps.Nfft)
            T = absFT / absFI
            R = absFR / absFI

            print(f'df: {f[1] - f[0]}')

            plt.subplot(2, 2, 3)
            plt.plot(f, 10 * np.log10(absFI), label='E_in')
            plt.plot(f, 10 * np.log10(absFR), label='E_R')
            plt.plot(f, 10 * np.log10(absFT), label='E_T')
            plt.xlim(ps.fxlim)
            plt.xlabel('Frequency, THz')
            plt.ylabel('E(f), dB')
            plt.legend()

            plt.subplot(2, 2, 4)
            plt.plot(f, R ** 2, label='Reflectance')
            plt.plot(f, T ** 2 * np.sqrt(c.epsout), label='Transmittance')
            plt.xlim(ps.fxlim)
            plt.ylim([0, 1])
            plt.xlabel('Frequency, THz')
            plt.legend()

            plt.show()

        elif ps.plot == -1:  # evaluate loss
            loss = -10 * np.log10(np.sum(o['detI'] ** 2) / np.sum(o['detR'] ** 2))
            print(f'Loss in dB: {-loss}')

    elif c.xpmtype == 2:  # 1D pump
        if ps.plot == 1:
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(o['t'], o['detN'])
            plt.title('Spatial integral of the density α -ΔT')
            plt.xlabel('Time, ps')
            plt.ylabel('Normalized carriers')
            plt.xlim([min(o['t']), max(o['t'])])

            plt.subplot(2, 1, 2)
            plt.plot(o['t'], o['detx'] * 1e12)
            plt.title('Near surface density')
            plt.xlabel('Time, ps')
            plt.ylabel('Density, cm^-3')
            plt.xlim([min(o['t']), max(o['t'])])

            plt.show()

    elif c.xpmtype == 3:  # 2D pump probe
        if ps.plot == 1:
            figcon = plt.figure()
            nlines = 20
            nskip = 1
            viewposT = [31.5, 34]
            ylabelrot = -45

            o['detT2Dref'] = np.tile(o['detT'], (1, c.Ntp))
            o['DdetT2D'] = o['detT2D'] - o['detT2Dref']
            o['DdetT2Dfix'] = np.zeros_like(o['DdetT2D'])

            for j in range(c.Nsav):
                o['DdetT2Dfix'][j, :] = np.interp(o['vtp'] - 0 + o['t'][j], o['DdetT2D'][j, :], o['vtp'], left=0, right=0)

            o['detT2Dfix'] = o['DdetT2Dfix'] + o['detT2Dref']
            o['f'], o['Ewr'], o['absEwr'], o['phaEwr'] = clfft(o['t'], o['detT2Dref'], 'tf', ps.Nfft)
            o['f'], o['Ewp'], o['absEwp'], o['phaEwp'] = clfft(o['t'], o['detT2Dfix'], 'tf', +ps.Nfft)

            o['T'] = o['absEwp'] / o['absEwr']
            o['Phi'] = np.unwrap(o['phaEwp'] - o['phaEwr'], axis=1)

            o['nnumin'] = np.where(o['f'] >= ps.fxlim[0])[0][0]
            o['nnumax'] = np.where(o['f'] >= ps.fxlim[1])[0][0]
            o['vecnu'] = np.arange(o['nnumin'], o['nnumax'])
            o['fnu'] = o['f'][o['vecnu']]

            plt.subplot(2, 2, 1)
            plt.contour(o['t'], o['vtp'], -o['DdetT2D'].T, nlines)
            plt.xlabel('THz delay t, ps')
            plt.ylabel('Pump-emitter delay t_pe, ps')
            plt.title('-ΔE(t,t_pe)')
            plt.grid(True)

            plt.subplot(2, 2, 2)
            plt.contour(o['t'], o['vtp'], -o['DdetT2Dfix'].T, nlines)
            plt.xlabel('THz delay t, ps')
            plt.ylabel('Pump-sampling delay t_p, ps')
            plt.title('-ΔE(t,t_p)')
            plt.colorbar(location='east')
            plt.grid(True)

            a1 = plt.subplot(2, 2, 3, projection='3d')
            a1.plot_surface(o['vtp'][::nskip], o['fnu'][::nskip], o['T'][o['vecnu'], ::nskip], color=[1, 1, 1])
            a1.view_init(viewposT[0], viewposT[1])
            plt.ylabel('Frequency, THz')
            plt.xlabel('t_p, ps')
            plt.xlim([min(o['vtp']), max(o['vtp'])])
            plt.ylim([min(o['fnu']), max(o['fnu'])])
            plt.grid(True)

            a2 = plt.subplot(2, 2, 4, projection='3d')
            a2.plot_surface(o['vtp'][::nskip], o['fnu'][::nskip], -o['Phi'][o['vecnu'], ::nskip], color=[1, 1, 1])
            a2.view_init(viewposT[0], viewposT[1])
            plt.ylabel('Frequency, THz')
            plt.xlabel('t_p, ps')
            plt.xlim([min(o['vtp']), max(o['vtp'])])
            plt.ylim([min(o['fnu']), max(o['fnu'])])
            plt.grid(True)

            plt.show()
