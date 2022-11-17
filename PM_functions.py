import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools as it
import math as math
from scipy.special import comb


def vmag3D(vector):

    """ find the magnitude of a 3D vector. """

    return np.sqrt((vector[0] ** 2) + (vector[1] ** 2) + (vector[2] ** 2))
    
    
def find_tp_vec(eval_points, tran_points):
    
    """
    find the matrix of vector distances between transducer (t) and each point in the evaluation plane (p_{i,j,k}).
    
    args:
        eval_points: list of [x, y, z] coords for the evaluation points in form of flat numpy arrays.
        tran_points: list of [x, y, z] coords for the transducer points in form of flat numpy arrays.
        
    returns:
        tp_vec: vector of distances between t and p_{i,j,k}
    
    """
    
    # find the distance vector in the x, y and z planes.
    tp_vec = np.array([np.array([float(eval_points[0].T[i]) - float(tran_points[0]),
                                 float(eval_points[1].T[i]) - float(tran_points[1]),
                                 float(eval_points[2].T[i]) - float(tran_points[2])]) \
                                 for i in range(len(eval_points[0].T))])
    
    return tp_vec


def find_sin_theta(tp_vec, tran_normal):
    
    """
    find the sin of the angle between the transducer normal, tp and each point in the evaulation plane (p_{i,j,k}).
    
    args:
        tp_vec: flat array of vector distances between transducer and evaluation points.
        tran_normal: vector describing the normal of the transducer.
        
    returns:
        sin_theta: flat matrix containing sin of the angle between the tran normal and each point.
    
    """

    # find cross and dot product
    cross_product = np.cross(tp_vec, tran_normal)
    dot_product = vmag3D(tp_vec.T)*vmag3D(tran_normal.T)
    
    # find sin theta
    sin_theta = vmag3D(cross_product.T)/dot_product
    
    return sin_theta
    
    
def PM_propagator_function_builder(tp_vec, sin_theta, k, p0=8.02, d=10.5/1000):
    
    """ 
    Piston model calculator. Finds the complex pressure propagated by transducers from
    one plane to another, determined using the PM_pesb function. (see GS-PAT eq.2).
    
    args:
        tp_vec = vector of Pythagorean distances between each transducer and each evaluation point in x, y and z.
        sin_theta = sin of angle between tran normal and vector drawn between the transducer and each evaluation point.
        k = wavenumber
        p0 = (8.02 [Pa]) reference pressure for Murata transducer measured at a distance of 1m
        d = (10.5/1000 [m]) spacing between Murata transducers in a lev board.
    
    returns:
        
        
    """
    # argument of 1st order Bessel function
    J_arg = k*(d/2)*sin_theta
    
    # taylor expansion of first order Bessel function over its agrument (J_1(J_arg)/J_arg)
    tay = (1/2)-(J_arg**2/16)+(J_arg**4/384)-(J_arg**6/18432)+(J_arg**8/1474560)-(J_arg**10/176947200)
    
    # propagator function
    H = 2*p0*(tay/tp_vec)*np.exp(1j*k*tp_vec)
    
    return H
    
    
def hexagon_diameter_to_coordinates(d, x_spacing=10.5/1000, y_spacing=9/1000) -> list((float, float, float)):
    
    """
    
    Coordinate system for d-transducers diameter hexagon. Centrepoint of central transducer is at origin (0,0,0).
    Array begins with the bottom left transducer.
    
    args: 
        d:          diameter of hexagon (longest row) in transducer units 
        x_spacing:  interspacing between elements in the x axis
        y_spacing:  interspacing between elements in the y axis
        f_tran:     focal length of the PAT [m]
    
    returns:
        coords: nx3 array of coords for this hexagon, with [0, 0, 0] as the centrepoint.
    """
    
    # from the diameter in transducer units (central and longest row) calculate array with transducers count 
    # for bottom row up to central row:
    
    bottom_to_central_row_tran_count = np.arange(np.floor((d+1)/2), np.floor(d+1), 1, dtype=int)

    # calculate array with rows' transducers count:
    
    rows_transducer_count = np.concatenate((bottom_to_central_row_tran_count,
                                            np.flip( bottom_to_central_row_tran_count )[1:]), axis=0)

    coords = []
    
    # for each row, depending on whether it is offset or not (i.e. shifted in relation to central row), 
    # calculate and assign X Y coordinates to each transducer:
    
    for row, row_length in enumerate(rows_transducer_count):

        for elem in range(row_length):
            
            coord_x = x_spacing * ( elem - row_length/2 + .5 )
            coord_y = -sys.maxsize - 1
            coord_z = 0

            if d % 2 != 0:
                coord_y = y_spacing * (row - (d-1)/2)
                
            else:
                coord_y = y_spacing * (row - d/2)
                
            coords.append((coord_x, coord_y, coord_z))  
    
    return np.array(coords)
    

def rotate_and_translate(old_plane_points, old_plane_centre, new_plane_centre):
    
    
    """

    Rotates and translates a plane described by an nx3 array of points "old_plane_points" with centrepoint
    "old_plane_centre" and a normal in the +z direction, such that it's new centrepoint is "new_plane_centre"
    and it's normal is pointing from here towards the old centrepoint at "old_plane_centre".
    
    https://python.plainenglish.io/reference-frame-transformations-in-python-with-numpy-and-matplotlib-6adeb901e0b0

    args:
        old_plane_points: points describing positions of elements on the plane we want to rotate and translate.
        old_plane_centre: centre point of the old plane (convention is for this to be [0, 0, 0]).
        new_plane_centre: centre point of the new plane.


    returns:
        new_plane_points: points describing positions of elements in the new plane.
        new_plane_normal_vector: normal vector of the new plane.

    """

    
    def unit_axis_angle(a, b):
        
        """

        Finds the axis of rotation for a pair of 3d vectors a & b.

        """

        an = np.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
        bn = np.sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2])
        ax, ay, az = a[0]/an, a[1]/an, a[2]/an
        bx, by, bz = b[0]/bn, b[1]/bn, b[2]/bn
        nx, ny, nz = ay*bz-az*by, az*bx-ax*bz, ax*by-ay*bx
        nn = np.sqrt(nx*nx + ny*ny + nz*nz)
        return (nx/nn, ny/nn, nz/nn), np.arccos(ax*bx + ay*by + az*bz)

    def rotation_matrix(axis, angle):
        
        """

        Finds the rotation matrix for a given pair of 3d vectors connected by an axis and a given angle.

        """

        ax, ay, az = axis[0], axis[1], axis[2]
        s = np.sin(angle)
        c = np.cos(angle)
        u = 1 - c
        return np.array([( ax*ax*u + c,    ax*ay*u - az*s, ax*az*u + ay*s ),
                         ( ay*ax*u + az*s, ay*ay*u + c,    ay*az*u - ax*s ),
                         ( az*ax*u - ay*s, az*ay*u + ax*s, az*az*u + c    )])
    
    def rotate_plane(vector, points):

        """

        Rotates an nx3 array of 3D coordinates from the +z normal to an arbitrary new normal vector.
        
        https://stackoverflow.com/questions/63287960/
        python-rotate-plane-set-of-points-to-match
        -new-normal-vector-using-scipy-spat

        """

        import vg
        from scipy.spatial.transform import Rotation as Rot
        from pytransform3d.rotations import matrix_from_axis_angle

        vector = vg.normalize(vector)
        axis = vg.perpendicular(vg.basis.z, vector)
        angle = vg.angle(vg.basis.z, vector, units='rad')

        a = np.hstack((axis, (angle,)))
        R = matrix_from_axis_angle(a)

        r = Rot.from_matrix(R)
        rotmat = r.apply(points)

        return rotmat

    # define the vector pointing upwards (+ve z direction) from the centre of the old plane:
    old_plane_normal_vector_start = old_plane_centre
    old_plane_normal_vector_end = np.array([0, 0, np.linalg.norm(new_plane_centre)])
    old_plane_normal_vector = old_plane_normal_vector_end - old_plane_normal_vector_start

    # define the vector pointing from the centre of the transducer plane towards the centre of the old plane:
    new_plane_normal_vector_start = new_plane_centre
    new_plane_normal_vector_end = old_plane_centre
    new_plane_normal_vector = new_plane_normal_vector_end - new_plane_normal_vector_start
    
    # calculate the rotation matrix for translating the old vector into the tran vector:
    axis, angle = unit_axis_angle(old_plane_normal_vector/np.linalg.norm(old_plane_normal_vector),
                                  new_plane_normal_vector/np.linalg.norm(new_plane_normal_vector))
    
    # find the rotation matrix
    R = rotation_matrix(axis, angle)
    
    # use R to find the rotation vector top transform the old plane into the new plane:
    rotation_vector = np.linalg.norm(new_plane_normal_vector) * \
                      R.dot(old_plane_normal_vector/np.linalg.norm(old_plane_normal_vector))
    
    # rotate:
    r_old_plane_points = rotate_plane(rotation_vector, old_plane_points)

    # translate:
    new_plane_points = r_old_plane_points + new_plane_centre
    
    return new_plane_points, new_plane_normal_vector

# def rotate_2d(x_vals, y_vals, theta):
    
    # """
    
    # Create a rotation matrix to find new x & z-coords for an angled board (y-coords will not change)
    
    # params:
    # x_vals = vector or matrix of x coords
    # y_vals = vector or matrix of z coords
    # theta = the angle by which we wish to rotate the coords. In degrees.
    # (be careful, as the angle you actually want may be 90-theta - please check before you use this function).
    
    # """
    
    # r = np.array(( (np.cos(np.radians(theta)), -np.sin(np.radians(theta))),
                   # (np.sin(np.radians(theta)), np.cos(np.radians(theta))) ))
    # v = np.array((x_vals, y_vals))
    # return r.dot(v)
    
    
# def transducer_rotate_and_translate(x, y, z, rotation_axis, tran_dist, tran_angle):
    
    # """
    # rotates and translates a transducer coord system from being parallel to the AMM to being positioned correctly.
    
    # params:
    # x,y,z = coords of the tranducer plane before rotation.
    # rotation_axis = axis in which the plane will be rotated w.r.t the origin.
    # tran_dist = distance between centre of AMM plan and centre of transducer plane (metres).
    # tran_angle = angle of rotation (degrees).
    
    # """
    
    # xx, yy = np.meshgrid(x, y)
    # zz = np.zeros(xx.shape)
    # zz = zz + tran_dist # bring up to correct height above the AMM
    # if rotation_axis == "x":
        # # rotate
        # rot_y, rot_z = rotate_2d(y, z, tran_angle)
        # rot_yy, rot_zz = np.meshgrid(rot_y, rot_z)
        # rot_zz = rot_zz + tran_dist # bring up to correct height above he AMM
        # # translate
        # offset = tran_dist * np.sin(np.radians(tran_angle))/np.sin(np.radians(90 - tran_angle))
        # return xx + offset, rot_yy, rot_zz
    # elif rotation_axis == "y":
        # # rotate
        # rot_x, rot_z = rotate_2d(x, z, tran_angle)
        # rot_xx, rot_zz = np.meshgrid(rot_x, rot_z)
        # rot_zz = rot_zz + tran_dist # bring up to correct height above he AMM
        # # translate
        # offset = tran_dist * np.sin(np.radians(tran_angle))/np.sin(np.radians(90 - tran_angle))
        # return rot_xx, yy + offset, rot_zz
    # else:
        # print("'"+rotation_axis+"' is not a valid translation axis.")



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