import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor

def clFDTmain(c, m, ps):
    """
    Main function for FDTD-TRTS.
    """
    uselite = clfinduselite(c, m)
    if uselite == 1:
        print('The lite version is used to increase efficiency.')
        o = clFDTmainlite(c, m, ps)  # Call the lite version (not implemented here)
    else:
        c = clfindvp(c, m, ps)  # Enable batch calculations
        start_time = time.time()
        if c['xpmtype'] == 1:  # 1D probe experiment
            o = FDT1Dprobe(c, m, ps)
        elif c['xpmtype'] == 2:  # 1D pump experiment
            o = FDT1Dpump(c, m, ps)
        elif c['xpmtype'] == 3:  # 2D pump probe experiment
            o = FDT2Dpumpprobe(c, m, ps)
        end_time = time.time()
        print(f"Calculation time: {end_time - start_time} seconds")

        if c['save'] == 1:
            current_time = time.strftime("-%Y-%m-%d-%H-%M")
            np.save(f"{c['xpmid']}{current_time}.npy", {'c': c, 'm': m, 'ps': ps, 'o': o})

        if ps['plot'] != 0 and c['prop'] == 1:
            clFDTplot(c, m, ps, o)  # Plot results (assume it's already implemented)
    
    return o


def clfinduselite(c, m):
    """
    Determines whether to use the lite version of the simulation based on the conditions.
    """
    uselite = 1
    if c['useEref'] == 1 or len(m) > 1:
        uselite = 0
    if len(m) < 2:
        for field in c:
            if isinstance(c[field], (list, np.ndarray)) and len(c[field]) > 1:
                uselite = 0
        for field in m:
            if isinstance(m[field], (list, np.ndarray)) and len(m[field]) > 1:
                uselite = 0
    return uselite


def clfindvp(c, m, ps):
    """
    Finds the variable parameter for batch calculations.
    """
    nvar = 0
    vp = [0, 0, 0]  # Default variable parameter
    
    for field in c:
        if isinstance(c[field], (list, np.ndarray)) and len(c[field]) > 1:
            nvar += 1
            print(f'"{field}" is found as the variable parameter')
            vp = [1, field, 1]
    
    for field in m:
        if isinstance(m[field], (list, np.ndarray)) and len(m[field]) > 1:
            nvar += 1
            print(f'"{field}" is found as the variable parameter')
            vp = [2, field, 1]
    
    if nvar > 1:
        print('Only one variable parameter is allowed.')
        return c
    
    if nvar == 0 and ps['numpar'] == 1:
        print('No variable parameters entered, continuing...')
    
    if vp[0] == 1:
        varvec = c[vp[1]]
    elif vp[0] == 2:
        varvec = m[vp[2]][vp[1]]
    
    c['nvar'] = nvar
    c['vp'] = vp
    c['varvec'] = varvec if nvar > 0 else []
    c['nvp'] = len(c['varvec']) if nvar > 0 else 0
    
    return c


