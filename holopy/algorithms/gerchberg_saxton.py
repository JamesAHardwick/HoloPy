import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools as it
import math as math
from scipy.special import comb


def GF_propagator_function_builder(reflector_points, eval_points, normals, areas, k):
    
    """
    
    http://www.personal.reading.ac.uk/~sms03snc/fe_bem_notes_sncw.pdf
    
    builds a scattering matrix for the sound propagation of a set elements/sources that cover
    a finite area in which they are assumed to have a constant pressure.

    args:
        reflector_points: matrix of x,y,z coords for the reflecting elements.
        eval_points: matrix of evaluation x,y,z coords at the propagation plane.
        normals: for a flat metasurface you get n*m times the vector [0, 0, 1].
        area: vector of the areas covered by each element (1, n*m).
        k: wavenumber.
        

    returns:
        H: Gives distance matrix of all distances between reflector and evaluation points (n*m, p*q).
        
    """
    # assign variables for x, y and z coord vectors for reflectors, evaluation points and normals
    rp_x, rp_y, rp_z = reflector_points.T
    ep_x, ep_y, ep_z = eval_points.T
    nm_x, nm_y, nm_z = normals
    
    # compute distances between eval_points and reflecting elements
    r = np.sqrt((rp_x.reshape(-1, 1) - ep_x.reshape(1, -1))**2 + \
                (rp_y.reshape(-1, 1) - ep_y.reshape(1, -1))**2 + \
                (rp_z.reshape(-1, 1) - ep_z.reshape(1, -1))**2)
    
    # partial of greens w.r.t normals
    g = -(1/(4*np.pi)) * np.exp(1j*k*r) * (1j*k*r-1)/(r**3)
    
    # find infinities and set them to zero.
    g[g == np.inf] = 0 
    
    # equation 2.21 in the pdf
    g = g * ((ep_x.reshape(1, -1) - rp_x.reshape(-1, 1)) * nm_x.T + \
             (ep_y.reshape(1, -1) - rp_y.reshape(-1, 1)) * nm_y.T + \
             (ep_z.reshape(1, -1) - rp_z.reshape(-1, 1)) * nm_z.T)
    
    # include reflector areas to build propagator function H
    H = g * areas.T
    
    return H


def GF_prop(complex_pressure, H, prop_direction):
    
    """
    
    find the product of the propagator, H, and AMM surface pressure to find pressure at evaluation plane.
    
    args:
        complex_pressure: The complex pressure matrix at our source plane. For forward propagation,
        this is the pressure on the reflective surface, for backwards propagation this it the pressure at
        the target.
        H: the propagator function defined using "GF_propagator_function_builder" 
        prop_direction: direction of propagation ("forward" or "backward").
        
    returns:
        complex pressure matrix at the evaluation plane.
    
    """
    
    # forward propagate from AMM plane to evaluation plane
    if prop_direction == "forward":
        return 2 * np.dot(complex_pressure, H) 
    
    # backward propagate from evaluation plane to AMM
    elif prop_direction == "backward":
        return 2 * np.dot(complex_pressure, np.conj(H)) 
    
    else:
        print(prop_direction, "is not a valid propagation direction, please specifiy either 'forward' or 'backward'.")
        
        


def GFGS(H, incident_surface_pressure, abs_target, iterations=50, verbose=False):

    # initialise - target plane pressure
    tpp = abs_target*np.exp(1j*np.pi*abs_target)

    # algorise - run GS
    for it in range(iterations):

        # backpropagate to find surface plane pressure
        spp = np.dot(np.conj(H), tpp)

        # reset amplitude on the surface
        spp = abs(incident_surface_pressure).reshape(-1, 1)*np.exp(1j*np.angle(spp))

        # propagate back to target plane
        tpp = np.dot(H.T, spp)

        # isolate over the target area
        tpp = abs_target*np.exp(1j*np.angle(tpp)*abs_target)

        if verbose:
            print("Iteration:", str(it), "... done!")

    # finalise
    spp = np.dot(np.conj(H), tpp)
    spp = incident_surface_pressure.reshape(-1, 1)*np.exp(1j*np.angle(spp))
    
    return spp