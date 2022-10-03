import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools as it
import math as math
from scipy.special import comb


def find_tp_mat(tran_centrepoint, eval_points):
    
    """
    find the matrix of vector distances between transducer (t) and each point in the evaluation plane (p_{i,j,k}).
    
    args:
        tran_centrepoint: 
        eval_points: list of [x, y, z] coords for the evaluation points in form of flat numpy arrays.
        
    returns:
        tp_mat: flat matrix containing all vector distances between t and p_{i,j,k}
    
    """
    
    tp_mat = np.sqrt((tran_centrepoint[0] - eval_points[0])**2 + \
                     (tran_centrepoint[1] - eval_points[1])**2 + \
                     (tran_centrepoint[2] - eval_points[2])**2)
    
    return tp_mat


def find_sin_theta_mat(tran_normal, eval_points):
    
    """
    find the sin of the angle between the transducer normal, tp and each point in the evaulation plane (p_{i,j,k}).
    
    args:
        tran_normal: 3d vector describing the normal of the transducer. We usually point this at the centre of the AMM.
        eval_points: list of [x, y, z] coords for the evaluation points in form of flat numpy arrays.
        
    returns:
        sin_theta_mat: flat matrix containing sin of the angle between the tran normal and each point.
    
    """
    
    def vmag3D(vector):
        
        """
        find the magnitude of a 3D vector.
        
        """
        
        return np.sqrt((vector[0] ** 2) + vector[1] ** 2 + vector[2] ** 2)
        
    eval_array = np.array([[float(eval_points[0].T[i]),
                        float(eval_points[1].T[i]),
                        float(eval_points[2].T[i])] for i in range(len(eval_points[0].T))])

    cross_product = np.cross(tran_normal.T, eval_array)
    dot_product = vmag3D(tran_normal)*vmag3D(eval_array.T)

    sin_theta = vmag3D(cross_product.T) / dot_product

    return sin_theta.reshape((1, -1))


def PM_propagator_function_builder(tp_mat, sin_theta_mat, k, p0=8.02, d=10.5/1000):
    
    """ 
    Piston model calculator. Finds the complex pressure propagated by transducers from
    one plane to another, determined using the PM_pesb function. (see GS-PAT eq.2).
    
    args:
        rxy = matrix describing Pythagorean distances between each transducer and each evaluation point in x and y
        rxyz = matrix describing Pythagorean distances between each transducer and each evaluation point in x, y and z.
        k = wavenumber
        p0 = (8.02 [Pa]) refernce pressure for Murata transducer measured at a distance of 1m
        d = (10.5/1000 [m]) spacing between Murata transducers in a lev board.
    
    returns:
        
        
    """
    # argument of 1st order Bessel function
    J_arg = k*(d/2)*sin_theta_mat
    
    # taylor expansion of first order Bessel function over its agrument (J_1(J_arg)/J_arg)
    tay = (1/2)-(J_arg**2/16)+(J_arg**4/384)-(J_arg**6/18432)+(J_arg**8/1474560)-(J_arg**10/176947200)
    
    # propagator function
    H = 2*p0*(tay/tp_mat)*np.exp(1j*k*tp_mat)
    
    return H


def PM_prop(transducer_surface_pressure, H, prop_direction):
    
    """
    find the product of the propagator, H, and transducer surface pressure to find pressure at evaluation plane.
    
    args:
        transducer_surface_pressure: complex pressure matrix on transdcuer plane surface.
        H: the propagator function defined using "PM_propagator_function_builder" 
        prop_direction: direction of propagation ("forward" or "backward").
        
    returns:
        complex pressure matrix at the evaluation plane.
    
    """
    
    # forward propagate from AMM plane to evaluation plane
    if prop_direction == "forward":       
        return np.dot(transducer_surface_pressure, H) 
    
    # backward propagate from evaluation plane to AMM
    elif prop_direction == "backward":
        return np.dot(transducer_surface_pressure, np.conj(H).T) 
    
    else:
        print(prop_direction, "is not a valid propagation direction, please specifiy either 'forward' or 'backward'.")


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



# def pesb(side_length_x, side_length_y, cell_spacing, angle, tx, ty, tz):
    # ''' Pressure evaluation space builder '''
    # # side length vectors for the plane being propagated to
    # ex = np.arange(-side_length_x/2 + cell_spacing/2, side_length_x/2 + cell_spacing/2, cell_spacing) 
    # ey = np.arange(-side_length_y/2 + cell_spacing/2, side_length_y/2 + cell_spacing/2, cell_spacing) 
    
    # exx, eyy = np.meshgrid(ex, ey)
    # # x, y & z vectors for evaluation-plane sample points:
    # px, py, pz = exx.flatten(), eyy.flatten(), np.zeros_like(exx.flatten())
    
    # # Grids to describe the vector distances between each transducer & evaluation plane sample point.
    # txv, pxv = np.meshgrid(tx,px)
    # tyv, pyv = np.meshgrid(ty,py)
    # tzv, pzv = np.meshgrid(tz,pz)
    
    # rxyz = np.sqrt((txv-pxv)**2 + (tyv-pyv)**2 + (tzv-pzv)**2) # Pythagoras for xyz distances
    # rxy = np.sqrt((txv-pxv)**2 + (tyv-pyv)**2) # Pythagoras for xy distances
    # return rxyz, rxy


# # Transducer piston model
# def piston_model_matrix(rxy, rxyz, k, p0=8.02, d=10.5/1000):
    # """
    # Piston model calculator. Finds the complex pressure propagated by transducers from
    # one plane to another, determined using the PM_pesb function. (see GS-PAT eq.2).
    
    # params:
    # rxy = matrix describing Pythagorean distances between each transducer and each evaluation point in x and y
    # rxyz = matrix describing Pythagorean distances between each transducer and each evaluation point in x, y and z.
    # k = wavenumber
    # p0 = (8.02 [Pa]) refernce pressure for Murata transducer measured at a distance of 1m
    # d = (10.5/1000 [m]) spacing between Murata transducers in a lev board.
    # """
    # # 1st order Bessel function
    # J_arg = k*(d/2)*(rxy/rxyz)
    
    # # taylor expansion of first order Bessel function over its agrument (J_1(J_arg)/J_arg)
    # tay = (1/2)-(J_arg**2/16)+(J_arg**4/384)-(J_arg**6/18432)+(J_arg**8/1474560)-(J_arg**10/176947200)
    
    # return 2*p0*(tay/rxyz)*np.exp(1j*k*rxyz)