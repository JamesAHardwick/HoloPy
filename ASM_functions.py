import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools as it
import math as math
from scipy.special import comb

    
def ASM_fw(f, cell_spacing, target_dist, k):
    
    """
    
    Forward ASM
    
    args:
        f: complex surface pressure
        cell_spacing: seperation between cells on the metasurface
        target_dist: distance to propagation plane
        k: wavenumber
        
    returns:
    
    """
    
    Nfft = f.shape
    kx = (2*np.pi*(np.arange(-Nfft[0]/2, (Nfft[0])/2)/((cell_spacing)*Nfft[0]))) \
    .reshape((1, Nfft[0])) # kx vector
    ky = (2*np.pi*(np.arange(-Nfft[1]/2, (Nfft[1])/2)/((cell_spacing)*Nfft[1]))) \
    .reshape((1, Nfft[1])) # ky vector
    F = np.fft.fft2(f) # 2D FT
    F = np.fft.fftshift(F) # Shift to the centre
    
    ## Propagate forwards    
    H = np.exp(1j*np.lib.scimath.sqrt(k**2 - kx**2 - (ky**2).T)*target_dist) # propagator function
    Gf = F*H.T # propagating the signal forward in Fourier space
    gf = np.fft.ifft2(np.fft.ifftshift(Gf)) # IFT & shift to return to real space
    return gf

def ASM_bw(f, cell_spacing, target_plane, k):
    
    """
    
    Backward ASM
    
    args:
        f: complex surface pressure
        cell_spacing: seperation between cells on the metasurface
        target_dist: distance to propagation plane
        k: wavenumber
    
    returns:
        
        
    """
    
    Nfft = f.shape
    kx = (2*np.pi*(np.arange(-Nfft[0]/2, (Nfft[0])/2)/((cell_spacing)*Nfft[0]))) \
    .reshape((1, Nfft[0])) # kx vector
    ky = (2*np.pi*(np.arange(-Nfft[1]/2, (Nfft[1])/2)/((cell_spacing)*Nfft[1]))) \
    .reshape((1, Nfft[1])) # ky vector
    F = np.fft.fft2(f) # 2D FT
    F = np.fft.fftshift(F) # Shift to the centre
    
    ## Propagate backwards
    H = np.exp(1j * np.lib.scimath.sqrt(k ** 2 - kx ** 2 - (ky ** 2).T) * target_plane)  # propagator function
    Hb = np.conj(H)  # conjugate of propagator will result in backpropagation
    Gb = F * Hb.T  # propagating backwards from target to lens
    gb = np.fft.ifft2(np.fft.ifftshift(Gb))  # IFT & shift to return to real space
    return gb  # return backpropagation
    
def ASM_prop(surface_pressure, cell_spacing, target_dist, resolution, k):
    """
    Propagate complex pressure forward to a parallel plane at a given resolution.
    Returns a complex pressure field.
    
    args:
        surface_pressure: complex surface pressure
        cell_spacing: seperation between cells on the metasurface
        target_dist: distance to propagation plane
        resolution: returns this many sample points for each point in the input phasemap
        k: wavenumber
    
    returns:
        
    """
    f = np.kron(surface_pressure, np.ones((resolution, resolution)))
    Nfft = f.shape
    kx = (2*np.pi*(np.arange(-Nfft[0]/2, (Nfft[0])/2)/((cell_spacing/resolution)*Nfft[0]))).reshape((1, Nfft[0])) # kx vector
    ky = (2*np.pi*(np.arange(-Nfft[1]/2, (Nfft[1])/2)/((cell_spacing/resolution)*Nfft[1]))).reshape((1, Nfft[1])) # ky vector
    F = np.fft.fft2(f) # 2D FT
    F = np.fft.fftshift(F) # Shift to the centre
    ## Propagate forwards    
    H = np.exp(1j*np.lib.scimath.sqrt(k**2 - kx**2 - (ky**2).T)*target_dist) # propagator function
    Gf = F*H.T # propagating the signal forward in Fourier space
    gf = np.fft.ifft2(np.fft.ifftshift(Gf)) # IFT & shift to return to real space
    return gf
    
    
def ASM_prop_perpendicular(surface_pressure, cell_spacing, target_dist, z_height, cut_axis, resolution, k):
    """
    Propagate complex pressure forward to a perpendicular plane at a given resolution.
    Returns a complex pressure field.
    
    args:
        surface_pressure: complex surface pressure.
        cell_spacing: seperation between cells on the metasurface.
        target_dist: distance to propagation plane.
        z_height; how far in the z-axis we want to propagate [m]
        cut_axis; are we slicing in the x or y axes?
        resolution: returns this many sample points for each point in the input phasemap.
        k: wavenumber
        
    returns:
        
    """
    if cut_axis == "x":
        centrepoint = int(surface_pressure.shape[0]*resolution/2)
        
    elif cut_axis == "y":
        centrepoint = int(surface_pressure.shape[1]*resolution/2)
        
    else:
        print(cut_axis, "is not a valid axis, please enter 'x' or 'y'.")
        return 
    
    step = cell_spacing/resolution
    prop_range = np.arange(0, z_height + step, step)
    xy_pressure_list = []
    for target_dist in prop_range:
        propagation = ASM_prop(surface_pressure, cell_spacing, target_dist, resolution, k)
        if cut_axis == "x":
            central_abs_pressure_vector = abs(propagation)[centrepoint]
        elif cut_axis == "y":
            central_abs_pressure_vector = abs(propagation.T)[centrepoint]
        xy_pressure_list.append(central_abs_pressure_vector)

    return np.flipud(np.array(xy_pressure_list))
    
    
def ASM_Iterative_GS(target_image, incident_surface_pressure, iterations, cell_spacing, target_dist, k):
    
    """
    
    implements the iterative GS algorithm to create phasemaps for a chosen target a chosen distance away.
    In this version we assume a non planewave source (board or transducer).
    
    args:
        target_image: the target image as a normalised numpy array (all elems should be 0-1).
        incident_surface_pressure: the complex pressure from the source incident on the metasurface.
        iterations: number of iterations for the algorithm to run (normally asymptotes ~200).
        cell_spacing: seperation between cells on the metasurface [m].
        target_dist: distance to propagation plane [m].
        k: wavenumber.
        
    returns:
        lpp: complex pressure map at the target plane.
        
    """
    
    apsize = target_image.shape
    N = 3*apsize[0]
    pw = ((int((N - apsize[0]) / 2), int((N - apsize[0]) / 2)),
          (int((N - apsize[1]) / 2), int((N - apsize[1]) / 2)))
    surface_amplitude, surface_phase = abs(incident_surface_pressure), np.angle(incident_surface_pressure)
    aperture = ((surface_amplitude != 0).astype(int))
    
    padded_target = np.pad(target_image, pw, 'constant', constant_values = 0) # Pad with zeros 
    padded_aperture = np.pad(aperture, pw, 'constant', constant_values = 0)
    padded_surface_amplitude = np.pad(surface_amplitude, pw, 'constant', constant_values = 0)
    
    tpp = padded_target*np.exp(1j*padded_target*np.pi) # initial pressure at target plane
    
    for it in range(iterations):
        lpp = ASM_bw(tpp, cell_spacing, target_dist, k) # inverse ASM to backpropagate complex pressure to lens-plane
        lpp = padded_surface_amplitude*np.exp(1j*np.angle(lpp)*padded_aperture) # isolate over aperture
        tpp = ASM_fw(lpp, cell_spacing, target_dist, k) # forward ASM to propagate updated phase as complex pressure to the target-plane
        tpp = padded_target*np.exp(1j*np.angle(tpp)*padded_target) # isolate target area
        
    lpp = ASM_bw(tpp, cell_spacing, target_dist, k)
    lpp = padded_aperture*np.exp(1j*np.angle(lpp)*padded_aperture) # isolate
    tpp = ASM_fw(lpp, cell_spacing, target_dist, k)
    
    return lpp[pw[0][0]:pw[0][0] + apsize[0], pw[1][0]:pw[1][0] + apsize[1]]