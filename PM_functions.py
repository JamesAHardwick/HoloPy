import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools as it
import math as math
from scipy.special import comb


def rotate_2d(x_vals, y_vals, theta):
    
    """
    
    Create a rotation matrix to find new x & z-coords for an angled board (y-coords will not change)
    
    params:
    x_vals = vector or matrix of x coords
    y_vals = vector or matrix of z coords
    theta = the angle by which we wish to rotate the coords. In degrees.
    (be careful, as the angle you actually want may be 90-theta - please check before you use this function).
    
    """
    
    r = np.array(( (np.cos(np.radians(theta)), -np.sin(np.radians(theta))),
                   (np.sin(np.radians(theta)), np.cos(np.radians(theta))) ))
    v = np.array((x_vals, y_vals))
    return r.dot(v)
    
    
def transducer_rotate_and_translate(x, y, z, rotation_axis, tran_dist, tran_angle):
    
    """
    rotates and translates a transducer coord system from being parallel to the AMM to being positioned correctly.
    
    params:
    x,y,z = coords of the tranducer plane before rotation.
    rotation_axis = axis in which the plane will be rotated w.r.t the origin.
    tran_dist = distance between centre of AMM plan and centre of transducer plane (metres).
    tran_angle = angle of rotation (degrees).
    
    """
    
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros(xx.shape)
    zz = zz + tran_dist # bring up to correct height above the AMM
    if rotation_axis == "x":
        # rotate
        rot_y, rot_z = rotate_2d(y, z, tran_angle)
        rot_yy, rot_zz = np.meshgrid(rot_y, rot_z)
        rot_zz = rot_zz + tran_dist # bring up to correct height above he AMM
        # translate
        offset = tran_dist * np.sin(np.radians(tran_angle))/np.sin(np.radians(90 - tran_angle))
        return xx + offset, rot_yy, rot_zz
    elif rotation_axis == "y":
        # rotate
        rot_x, rot_z = rotate_2d(x, z, tran_angle)
        rot_xx, rot_zz = np.meshgrid(rot_x, rot_z)
        rot_zz = rot_zz + tran_dist # bring up to correct height above he AMM
        # translate
        offset = tran_dist * np.sin(np.radians(tran_angle))/np.sin(np.radians(90 - tran_angle))
        return rot_xx, yy + offset, rot_zz
    else:
        print("'"+rotation_axis+"' is not a valid translation axis.")


def pesb(side_length_x, side_length_y, cell_spacing, angle, tx, ty, tz):
    ''' Pressure evaluation space builder '''
    # side length vectors for the plane being propagated to
    ex = np.arange(-side_length_x/2 + cell_spacing/2, side_length_x/2 + cell_spacing/2, cell_spacing) 
    ey = np.arange(-side_length_y/2 + cell_spacing/2, side_length_y/2 + cell_spacing/2, cell_spacing) 
    
    exx, eyy = np.meshgrid(ex, ey)
    # x, y & z vectors for evaluation-plane sample points:
    px, py, pz = exx.flatten(), eyy.flatten(), np.zeros_like(exx.flatten())
    
    # Grids to describe the vector distances between each transducer & evaluation plane sample point.
    txv, pxv = np.meshgrid(tx,px)
    tyv, pyv = np.meshgrid(ty,py)
    tzv, pzv = np.meshgrid(tz,pz)
    
    rxyz = np.sqrt((txv-pxv)**2 + (tyv-pyv)**2 + (tzv-pzv)**2) # Pythagoras for xyz distances
    rxy = np.sqrt((txv-pxv)**2 + (tyv-pyv)**2) # Pythagoras for xy distances
    return rxyz, rxy


# def pesb(side_length_x, side_length_y, AMM_dist, cell_spacing, tx, ty, tz):

    # """pressure_evaluation_space_builder"""
    
    # ex = np.arange(-side_length_x/2 + cell_spacing/2, side_length_x/2 + cell_spacing/2, cell_spacing) # x side length vector 
    # ey = np.arange(-side_length_y/2 + cell_spacing/2, side_length_y/2 + cell_spacing/2, cell_spacing) # y side length vector
    
    # exx, eyy = np.meshgrid(ex, ey)
    # # x, y & z vectors for evaluation-plane sample points:
    # px, py, pz = exx.flatten(), eyy.flatten(), AMM_dist*np.ones_like(exx.flatten())

    
    # # Grids to describe the vector distances between each transducer & evaluation plane sample point.
    # txv, pxv = np.meshgrid(tx,px)
    # tyv, pyv = np.meshgrid(ty,py)
    # tzv, pzv = np.meshgrid(tz,pz)
    
    # rxyz = np.sqrt((txv-pxv)**2 + (tyv-pyv)**2 + (tzv-pzv)**2) # Pythagoras for xyz distances
    # rxy = np.sqrt((txv-pxv)**2 + (tyv-pyv)**2) # Pythagoras for xy distances
    
    # return rxyz, rxy

def pesb_old(n, side_length, cell_spacing, angle, res, x_tcoords, y_tcoords, z_tcoords):
    '''Pressure_evaluation_space_builder'''
    # rotation matrix, assuming that the board is rotated in the x-z plane.
    x_tcoords, z_tcoords = rotate_2d(x_tcoords, z_tcoords, angle) 
    xx, yy = np.meshgrid(x_tcoords, y_tcoords) # Use meshgrid to create grid versions of x and y coords
    zz = np.array([z_tcoords for i in range(n)]) # create a meshgrid for z without making all of them 3D arrays
    # x, y & z vectors for transducer-plane sample points:
    tx, ty, tz = xx.flatten(), yy.flatten(), zz.flatten() 
    
    ev = np.arange(-side_length, side_length, cell_spacing/res) # side length vector for the plane being propagated to
    ex, ey = np.meshgrid(ev,ev)
    # x, y & z vectors for evaluation-plane sample points:
    ez = np.zeros(len(ex)*len(ey))
    px, py, pz = ex.flatten(), ey.flatten(), ez.flatten() 
    
    # Grids to describe the vector distances between each transducer & evaluation plane sample point.
    txv, pxv = np.meshgrid(tx,px)
    tyv, pyv = np.meshgrid(ty,py)
    tzv, pzv = np.meshgrid(tz,pz)
    
    rxyz = np.sqrt((txv-pxv)**2 + (tyv-pyv)**2 + (tzv-pzv)**2) # Pythagoras for xyz distances
    rxy = np.sqrt((txv-pxv)**2 + (tyv-pyv)**2) # Pythagoras for xy distances
    return rxyz, rxy
    

def PM_pesb(tm_shape, ev_res, dx, z_plane_height):
    """
    Piston model pressure evaluation space builder. This function creates a coordinate system
    connecting two planes between which the piston model can propagate pressure.
    
    params:
    tm_shape = shape of the transducer matrix as a 2 element tuple (x, y)
    ev_res = resolution of the evaluation plane.
    dx = cell spacing on transducer plane
    z_plane_height = distance between centrepoints of the two planes.
    
    TO DO: implement a condition system to allow propagtion to xy and yz planes.
    TO DO: add the ability to rotate the transducer plane using the rotate_2d function to model an angled source.
    
    # rotation matrix, assuming that the board is rotated in the x-z plane.
    # tran_x, tran_y = rotate_2d(x_tcoords, z_tcoords, angle) 
    """
    # define transducer plane
    tran_x = np.arange(-(tm_shape[0]/2 - .5)*dx, (tm_shape[0]/2 + .5)*dx, dx)
    tran_y = np.arange(-(tm_shape[1]/2 - .5)*dx, (tm_shape[1]/2 + .5)*dx, dx)
    tran_xx, tran_yy = np.meshgrid(tran_x, tran_y)
    tran_zz = np.zeros(len(tran_y)*len(tran_x))
    tran_x_vec, tran_y_vec, tran_z_vec = tran_xx.flatten(), tran_yy.flatten(), tran_zz.flatten()

    # define evaluation plane (xz)
    ev_x = np.arange(-(tm_shape[0]/2 - .5)*dx, (tm_shape[0]/2 + .5)*dx, dx/ev_res)
    ev_z = np.arange(0, z_plane_height, dx/ev_res)
    ev_xx, ev_zz = np.meshgrid(ev_x, ev_z)
    ev_yy = np.zeros(len(ev_x)*len(ev_z))
    ev_x_vec, ev_y_vec, ev_z_vec = ev_xx.flatten(), ev_yy.flatten(), ev_zz.flatten()
    
    # define evaluation plane (xy)
    # TO DO
    
    # define evaluation plane (yz)
    # TO DO

    # Grids to describe the vector distances between each tran & evaluation plane sample point.
    txv, pxv = np.meshgrid(tran_x_vec, ev_x_vec)
    tyv, pyv = np.meshgrid(tran_y_vec, ev_y_vec)
    tzv, pzv = np.meshgrid(tran_z_vec, ev_z_vec)

    rxyz = np.sqrt((txv-pxv)**2 + (tyv-pyv)**2 + (tzv-pzv)**2) # Pythagoras for xyz distances
    rxy = np.sqrt((txv-pxv)**2 + (tyv-pyv)**2) # Pythagoras for xy distances
    return rxyz, rxy


# Transducer piston model
def piston_model_matrix(rxy, rxyz, k, p0=8.02, d=10.5/1000):
    """
    Piston model calculator. Finds the complex pressure propagated by transducers from
    one plane to another, determined using the PM_pesb function. (see GS-PAT eq.2).
    
    params:
    rxy = matrix describing Pythagorean distances between each transducer and each evaluation point in x and y
    rxyz = matrix describing Pythagorean distances between each transducer and each evaluation point in x, y and z.
    k = wavenumber
    p0 = (8.02 [Pa]) refernce pressure for Murata transducer measured at a distance of 1m
    d = (10.5/1000 [m]) spacing between Murata transducers in a lev board.
    """
    # 1st order Bessel function
    b = k*(d/2)*(rxy/rxyz)
    # taylor expansion of the bessel function
    tay = (1/2)-(b**2/16)+(b**4/384)-(b**6/18432)+(b**8/1474560)-(b**10/176947200)
    return 2*p0*(tay/rxyz)*np.exp(1j*k*rxyz)
    # return 2*(p0/b)*(tay/rxyz)*np.exp(1j*k*rxyz)