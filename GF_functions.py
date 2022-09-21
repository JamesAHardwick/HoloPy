import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools as it
import math as math
from scipy.special import comb


def points_vector_builder(centrepoint, extents, pixel_spacing):
    
    """
    
    We can define an evalution plane using 3 inputs:

    args:
        centrepoint: (x, y, z) tuple describing the central point of the evaulation plane (meters).
        extents: list of tuples in the form [(+x, -x), (+y, -y), (+z, -z)] describing the distances which the plane
        extends in each +ve and -ve direction from the centrepoint. In order to create a valid 2D plane, one of
        (+x, -x), (+y, -y), or (+z, -z) must be (0, 0).
        pixel_spacing: distance between pixels on the evaluation plane (meters).
    
    returns:
        points_vector_list: list of x, y and z coordinate arrays.
    
    """

    # side vectors for evaluation point matrix
    x = np.arange(centrepoint[0] - (extents[0][0]/2) + (pixel_spacing/2),
                  centrepoint[0] + (extents[0][1]/2),
                  pixel_spacing)
    
    y = np.arange(centrepoint[1] - (extents[1][0]/2) + (pixel_spacing/2),
                  centrepoint[1] + (extents[1][1]/2),
                  pixel_spacing)
    
    z = np.arange(centrepoint[2] - (extents[2][0]/2) + (pixel_spacing/2),
                  centrepoint[2] + (extents[2][1]/2),
                  pixel_spacing)
    
    # if yz plane
    if extents[0] == (0, 0): 
        yy, zz = np.meshgrid(y, z)
        xx = centrepoint[0]*np.ones(len(y)*len(z))
    
    # if xz plane
    elif extents[1] == (0, 0): 
        xx, zz = np.meshgrid(x, z)
        yy = centrepoint[1]*np.ones(len(x)*len(z))
        
    # if xy plane    
    elif extents[2] == (0, 0):
        xx, yy = np.meshgrid(x, y)
        zz = centrepoint[2]*np.ones(len(x)*len(y))
    
    # return a list of x, y and z vectors
    points_vector_list = [xx.reshape((1, -1)),
                          yy.reshape((1, -1)),
                          zz.reshape((1, -1))]
    
    return points_vector_list
    
    
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
    rp_x, rp_y, rp_z = reflector_points
    ep_x, ep_y, ep_z = eval_points
    nm_x, nm_y, nm_z = normals
    
    # compute distances between eval_points and reflecting elements
    r = np.sqrt((rp_x.T - ep_x)**2 + (rp_y.T - ep_y)**2 + (rp_z.T - ep_z)**2)
    
    # partial of greens w.r.t normals
    g = -(1/(4*np.pi)) * np.exp(1j*k*r) * (1j*k*r-1)/(r**3)
    
    # find infinities and set them to zero.
    g[g == np.inf] = 0 
    
    # equation 2.21 in the pdf
    g = g * ((ep_x - rp_x.T) * nm_x.T + (ep_y - rp_y.T) * nm_y.T + (ep_z - rp_z.T) * nm_z.T)
    
    # include reflector areas to build propagator function H
    H = g * areas.T
    
    return H


def GF_prop(AMM_surface_pressure, H, prop_direction):
    
    """
    
    find the product of the propagator, H, and AMM surface pressure to find pressure at evaluation plane.
    
    args:
        AMM_surface_pressure: complex pressure matrix on AMM surface.
        H: the propagator function defined using "GF_propagator_function_builder" 
        prop_direction: direction of propagation ("forward" or "backward").
        
    returns:
        complex pressure matrix at the evaluation plane.
    
    """
    
    # forward propagate from AMM plane to evaluation plane
    if prop_direction == "forward":
        return 2 * np.dot(AMM_surface_pressure, H) 
    
    # backward propagate from evaluation plane to AMM
    elif prop_direction == "backward":
        return 2 * np.dot(AMM_surface_pressure, np.conj(H).T) 
    
    else:
        print(prop_direction, "is not a valid propagation direction, please specifiy either 'forward' or 'backward'.")