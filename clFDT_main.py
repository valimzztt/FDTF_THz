import numpy as np
import time
from clFDT_core import clFDTcore
from clFDT_plot import clFDTplot


def clFDTmain(c, m_list, ps):
    """
    Main function for FDTD-TRTS in Python.
    """
    uselite = clfinduselite(c, m_list)
    
    if uselite == 1:
        print('The lite version is used to increase efficiency.')
        o = clFDTmainlite(c, m_list, ps)  # Call the lite version (assumed to be implemented elsewhere)
    else:
        c = clfindvp(c, m_list, ps)  # Enable batch calculations
        
        start_time = time.time()  # Record time
        if c.xpmtype == 1:  # 1D probe experiment
            o = FDT1Dprobe(c, m_list, ps)
        elif c.xpmtype == 2:  # 1D pump experiment
            o = FDT1Dpump(c, m_list, ps)
        elif c.xpmtype == 3:  # 2D pump probe experiment
            o = FDT2Dpumpprobe(c, m_list, ps)
        
        end_time = time.time()  # Display time of calculation
        print(f"Calculation time: {end_time - start_time:.2f} seconds")
        
        if c.save == 1:
            current_time = time.strftime("-%Y-%m-%d-%H-%M")
            np.save(f"{c.xpmid}{current_time}.npy", {'c': c, 'm': m_list, 'ps': ps, 'o': o})
        
        if ps.plot != 0 and c.prop == 1:
            print("Calling the plotting functiin")
            clFDTplot(c, m_list, ps, o)  # Call the plotting function
    
    return o

def clfinduselite(c, m_list):
    """
    Determines whether to use the lite version of the simulation based on conditions.
    """
    uselite = 1
    if c.useEref == 1 or len(m_list) > 1:
        uselite = 0

    if len(m_list) < 2:
        for field_name, value in vars(c).items():
            if isinstance(value, (list, np.ndarray)) and len(value) > 1:
                uselite = 0
        
        for medium in m_list:
            for field_name, value in vars(medium).items():
                if isinstance(value, (list, np.ndarray)) and len(value) > 1:
                    uselite = 0

    return uselite

def clfindvp(c, m_list, ps):
    """
    Finds the variable parameter for batch calculations.
    """
    nvar = 0
    vp = [0, 0, 0]  # Initialize vp
    
    for field_name, value in vars(c).items():
        if isinstance(value, (list, np.ndarray)) and len(value) > 1:
            nvar += 1
            print(f'"{field_name}" is found as the variable parameter')
            vp = [1, field_name, 1]
    
    for medium_index, medium in enumerate(m_list):
        for field_name, value in vars(medium).items():
            if isinstance(value, (list, np.ndarray)) and len(value) > 1:
                nvar += 1
                print(f'"{field_name}" is found as the variable parameter')
                vp = [2, field_name, medium_index + 1]  # `medium_index + 1` to match MATLAB's 1-based indexing

    if nvar > 1:
        print("############ Only one variable parameter is allowed - choose one! (check orientation of non-variable vectors)")
        return c

    if nvar == 0 and ps['numpar'] == 1:
        print("No variable parameters entered, continuing...")
    
    if vp[0] == 1:
        varvec = getattr(c, vp[1])
    elif vp[0] == 2:
        varvec = getattr(m_list[vp[2] - 1], vp[1])  # Convert to 0-based indexing

    c.nvar = nvar
    c.vp = vp
    c.vpfname = vp[1] if vp[0] > 0 else ""
    c.varvec = varvec if nvar > 0 else []
    c.nvp = len(c.varvec) if nvar > 0 else 0
    
    return c

 
def FDT1Dprobe(c, m_list, ps):
    v = {}
    for jj in range(c.nvp):
        if c.nvar > 0:
            if c.vp[0] == 1:
                setattr(c, c.vpfname, c.varvec[jj])
            elif c.vp[0] == 2:
                setattr(m_list[c.vp[2] - 1], c.vpfname, c.varvec[jj])
        
        o = clFDTcore(c, m_list, ps)
        
        if c.nvar > 0:
            # o.varpar = c.varvec[jj]
            o['varpar'] = c.varvec[jj]

            v[jj] = o
    
    if c.nvar > 0:
        result = []
        for jj in range(len(c.varvec)):
            result.append(v[jj])
        return result
    
    return o


def FDT1Dpump(c, m_list, ps):
    v = {}
    for jj in range(c.nvp):
        if c.nvar > 0:
            if c.vp[0] == 1:
                setattr(c, c.vpfname, c.varvec[jj])
            elif c.vp[0] == 2:
                setattr(m_list[c.vp[2] - 1], c.vpfname, c.varvec[jj])
        
        o = clFDTcore(c, m_list, ps)
        
        if c.nvar > 0:
            o.varpar = c.varvec[jj]
            v[jj] = o
        else:
            o.varpar = []
    
    if c.nvar > 0:
        result = []
        for jj in range(len(c.varvec)):
            result.append(v[jj])
        return result
    
    return o


def FDT2Dpumpprobe(c, m_list, ps):
    ps.real = 0
    ps.numpar = 0
    v = {}
    vtp = np.linspace(c.mintp, c.maxtp, c.Ntp)
    
    if c.parallel == 1:
        # Example: using multiprocessing in Python for parallel processing
        import multiprocessing as mp
    
    for jj in range(c.nvp):
        if c.nvar > 0:
            if c.vp[0] == 1:
                setattr(c, c.vpfname, c.varvec[jj])
            elif c.vp[0] == 2:
                setattr(m_list[c.vp[2] - 1], c.vpfname, c.varvec[jj])
        
        detT2D = np.zeros((c.Nsav, c.Ntp))
        detR2D = np.zeros((c.Nsav, c.Ntp))

        # Parallel processing can be done using multiprocessing.Pool or similar libraries
        pool = mp.Pool(mp.cpu_count())
        results = [pool.apply_async(clFDTcore, args=(c, m_list, ps, vtp[j])) for j in range(c.Ntp)]
        pool.close()
        pool.join()

        for j, res in enumerate(results):
            to = res.get()  # Collecting the results from parallel processing
            detT2D[:, j] = to.detT
            detR2D[:, j] = to.detR

        o = clFDTcore(c, m_list, ps, min(vtp), 0)
        o.detT2D = detT2D
        o.detR2D = detR2D
        o.vtp = vtp

        if c.nvar > 0:
            o.varpar = c.varvec[jj]
            v[jj] = o
        else:
            o.varpar = []

    if c.nvar > 0:
        result = []
        for jj in range(c.nvp):
            result.append(v[jj])
        return result
    
    if c.parallel == 1:
        # Close the multiprocessing pool if parallel processing was used
        pool.close()
    
    return o
