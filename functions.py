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
    
    params:
    f = complex surface pressure
    cell_spacing = seperation between cells on the metasurface
    target_dist = distance to propagation plane
    k = wavenumber
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
    
    params:
    f = complex surface pressure
    cell_spacing = seperation between cells on the metasurface
    target_dist = distance to propagation plane
    k = wavenumber
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
    Propagate a phasemap forward to a parallel plane at a given resolution.
    Returns a complex pressure field. Assumes non-plane wave source.
    
    params:
    surface_pressure = complex surface pressure
    cell_spacing = seperation between cells on the metasurface
    target_dist = distance to propagation plane
    res = returns this many sample points for each point in the input phasemap
    k = wavenumber
    """
    f = np.kron(surface_pressure, np.ones((resolution, resolution)))
    Nfft = f.shape
    kx = (2*np.pi*(np.arange(-Nfft[0]/2, (Nfft[0])/2)/((cell_spacing/resolution)*Nfft[0]))) \
    .reshape((1, Nfft[0])) # kx vector
    ky = (2*np.pi*(np.arange(-Nfft[1]/2, (Nfft[1])/2)/((cell_spacing/resolution)*Nfft[1]))) \
    .reshape((1, Nfft[1])) # ky vector
    F = np.fft.fft2(f) # 2D FT
    F = np.fft.fftshift(F) # Shift to the centre
    ## Propagate forwards    
    H = np.exp(1j*np.lib.scimath.sqrt(k**2 - kx**2 - (ky**2).T)*target_dist) # propagator function
    Gf = F*H.T # propagating the signal forward in Fourier space
    gf = np.fft.ifft2(np.fft.ifftshift(Gf)) # IFT & shift to return to real space
    return gf
    

def points_vector_builder(centrepoint, x_sl, y_sl, pixel_spacing):
    """
    We can define an evalution plane using 4 inputs:

    INPUTS:
    centrepoint = (x, y, z) tuple describing the central point of the evaulation plane.
    x_sl, y_sl = side lengths of the evaluation plane in the x and y axes respectively (meters)
    pixel_spacing = distance between pixels on the evaluation plane (meters)
    
    OUTPUT:
    points_vector_list = list of x, y and z coordinate arrays.
    """

    # side vectors for evaluation point matrix
    x = np.arange(centrepoint[0] - (x_sl/2) + (pixel_spacing/2), centrepoint[0] + (x_sl/2), pixel_spacing)
    y = np.arange(centrepoint[1] - (y_sl/2) + (pixel_spacing/2), centrepoint[1] + (y_sl/2), pixel_spacing)
    xx, yy = np.meshgrid(x, y)
    zz = centrepoint[2]*np.ones(xx.size)
    
    points_vector_list = [xx.reshape((1, len(x)*len(y))),
                          yy.reshape((1, len(x)*len(y))),
                          zz.reshape((1, len(x)*len(y)))]
    
    return points_vector_list

def GF_prop(AMM_surface_pressure, reflector_points, eval_points, normals, areas, k, prop_direction):
    """
    http://www.personal.reading.ac.uk/~sms03snc/fe_bem_notes_sncw.pdf
    
    builds a scattering matrix for the sound propagation of a set elements/sources that cover
    a finite area in which they are assumed to have a constant pressure.

    INPUTS:
    reflector_points = matrix of x,y,z coords for the reflecting elements.
    eval_points = matrix of evaluation x,y,z coords at the propagation plane.
    normals = for a flat metasurface you get n*m times the vector [0, 0, 1].
    area = vector of the areas covered by each element (1, n*m).
    k = wavenumber.
    prop_direction = direction of propagation ("forward" or "backward").

    OUTPUT:
    A: Gives distance matrix of all distances between reflector and evaluation points (n*m, p*q).
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
    
    # find the product of the propagator and AMM surface pressure to find pressure at evaluation plane
    
    # forward propagate from AMM plane to evaluation plane
    if prop_direction == "forward":
        return 2 * np.dot(AMM_surface_pressure, H) 
    
    # backward propagate from evaluation plane to AMM
    elif prop_direction == "backward":
        return 2 * np.dot(AMM_surface_pressure, np.conj(H).T) 
    
    else:
        print(prop_direction, "is not a valid propagation direction, please specifiy either 'forward' or 'backward'.")


def target_builder_image(filename):
    '''Convert an image in .png format into a bitmap target in .npy format'''
    import numpy as np, matplotlib.pyplot as plt
    target_image = plt.imread(filename+'.png')
    if len(target_image.shape) > 2: # remove rgb vestige if it is present
        target_image = target_image[:,:,:1]
        target_image = np.reshape(target_image[:,:],(target_image.shape[0], target_image.shape[1]))
    return abs((target_image/np.amax(target_image)) - 1)

    
def target_builder_chars(char_list, font_file, fontsize, im_h):
    """
    Creates an array of numpy array images to be used as targets with the iterative GS function.
    
    params:
    char_list = list of characters to be made into targets.
    font_file = .tff file containing the character font.
    fontsize = font size (in pts, knowing that 10pts = 13px).
    im_h = height of the numpy array in pixels, the function assumes a square array.
    """
    from PIL import Image, ImageDraw, ImageFont 
    bg_color = (255, 255, 255) # Image background color (white)
    slice_threshold = 0.05 # sum values in row or column and delete them if below this threshold
    target_images = []
    for count, char in enumerate(char_list):
        fnt = ImageFont.truetype(font_file, fontsize) # Create font object
        w, h = fnt.getsize(str(char))
        im = Image.new('RGB', (w, h), color = bg_color)
        draw = ImageDraw.Draw(im)
        draw.text((0, 0), str(char), font=fnt, fill="black")
    #     im = im.rotate(45)

        target_image = np.array(im)[:,:,:1]
        target_image = np.reshape(target_image[:,:],(target_image.shape[0], target_image.shape[1]))
        target_image = 1 - target_image/255 # normalise

        # remove rows < threshold
        x_del = []
        for i, x in enumerate(np.sum(target_image, axis=1)):
            if x < slice_threshold:
                x_del.append(i)
        target_image = np.delete(target_image, x_del, axis=0)

        # remove columns < threshold
        y_del = []
        for j, y in enumerate(np.sum(target_image, axis=0)):
            if y < slice_threshold:
                y_del.append(j)
        target_image = np.delete(target_image, y_del, axis=1)

        # pad zeros around the characters
        target_dummy = target_image
        w, h = target_dummy.shape[0], target_dummy.shape[1]
        target_image = np.zeros((im_h, im_h))    
        target_image[int((im_h-w)/2): int((im_h+w)/2), int((im_h-h)/2):int((im_h+h)/2)] = target_dummy   
        target_images.append(target_image) # save to list
    return target_images
    
    
def Iterative_GS(target_image, iterations, cell_spacing, target_dist, k):
    """
    implements the iterative GS algorithm to create phasemaps for a chosen target a chosen distance away.
    Assumes plane wave source.
    
    params:
    target_image = the target image as a normalised numpy array (all elems should be 0-1).
    iterations = number of iterations for the algorithm to run (normally assymptotes ~200).
    ell_spacing = seperation between cells on the metasurface.
    target_dist = distance to propagation plane.
    k = wavenumber.      
    """
    apsize = target_image.shape
    N = 3*apsize[0]
    pw = ((int((N - apsize[0]) / 2), int((N - apsize[0]) / 2)),
          (int((N - apsize[1]) / 2), int((N - apsize[1]) / 2)))
    padded_target = np.pad(target_image, pw, 'constant', constant_values = 0) # Pad with zeros 
    aperture = np.ones(apsize)
    padded_aperture = np.pad(aperture, pw, 'constant', constant_values = 0)
    tpp = padded_target*np.exp(1j*padded_target*np.pi) # initial pressure at target plane
    for it in range(iterations):
        lpp = ASM_bw(tpp, cell_spacing, target_dist, k) # inverse ASM to backpropagate complex pressure to lens-plane
        lpp = padded_aperture*np.exp(1j*np.angle(lpp)*padded_aperture) # isolate over aperture
        tpp = ASM_fw(lpp, cell_spacing, target_dist, k) # forward ASM to propagate updated phase as complex pressure to the target-plane
        tpp = padded_target*np.exp(1j*np.angle(tpp)*padded_target) # isolate target area
    lpp = ASM_bw(tpp, cell_spacing, target_dist, k)
    lpp = padded_aperture*np.exp(1j*np.angle(lpp)*padded_aperture) # isolate
    return np.angle(lpp)[pw[0][0]:pw[0][0] + apsize[0], pw[1][0]:pw[1][0] + apsize[1]]


def PMP_Iterative_GS(target_image, incident_surface_pressure, iterations, cell_spacing, target_dist, k):
    """
    implements the iterative GS algorithm to create phasemaps for a chosen target a chosen distance away.
    In this version we assume a non planewave source (board or transducer).
    
    params:
    target_image = the target image as a normalised numpy array (all elems should be 0-1).
    incident_surface_pressure = the complex pressure from the source incident on the metasurface.
    iterations = number of iterations for the algorithm to run (normally assymptotes ~200).
    ell_spacing = seperation between cells on the metasurface.
    target_dist = distance to propagation plane.
    res = returns this many sample points for each point in the input phasemap.
    k = wavenumber.
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
    return np.angle(lpp)[pw[0][0]:pw[0][0] + apsize[0], pw[1][0]:pw[1][0] + apsize[1]]
    

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
    '''
    rotates and translates a transducer coord system from being parallel to the AMM to being positioned correctly.
    
    params:
    x,y,z = coords of the tranducer plane before rotation.
    rotation_axis = axis in which the plane will be rotated w.r.t the origin.
    tran_dist = distance between centre of AMM plan and centre of transducer plane (metres).
    tran_angle = angle of rotation (degrees).
    '''
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros(xx.shape)
    zz = zz + tran_dist # bring up to correct height above he AMM
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


# def pesb(n, side_length, cell_spacing, angle, res, x_tcoords, y_tcoords, z_tcoords):
    # '''Pressure_evaluation_space_builder'''
    # # rotation matrix, assuming that the board is rotated in the x-z plane.
    # x_tcoords, z_tcoords = rotate_2d(x_tcoords, z_tcoords, angle) 
    # xx, yy = np.meshgrid(x_tcoords, y_tcoords) # Use meshgrid to create grid versions of x and y coords
    # zz = np.array([z_tcoords for i in range(n)]) # create a meshgrid for z without making all of them 3D arrays
    # # x, y & z vectors for transducer-plane sample points:
    # tx, ty, tz = xx.flatten(), yy.flatten(), zz.flatten() 
    
    # ev = np.arange(-side_length, side_length, cell_spacing/res) # side length vector for the plane being propagated to
    # ex, ey = np.meshgrid(ev,ev)
    # # x, y & z vectors for evaluation-plane sample points:
    # ez = np.zeros(len(ex)*len(ey))
    # px, py, pz = ex.flatten(), ey.flatten(), ez.flatten() 
    
    # # Grids to describe the vector distances between each transducer & evaluation plane sample point.
    # txv, pxv = np.meshgrid(tx,px)
    # tyv, pyv = np.meshgrid(ty,py)
    # tzv, pzv = np.meshgrid(tz,pz)
    
    # rxyz = np.sqrt((txv-pxv)**2 + (tyv-pyv)**2 + (tzv-pzv)**2) # Pythagoras for xyz distances
    # rxy = np.sqrt((txv-pxv)**2 + (tyv-pyv)**2) # Pythagoras for xy distances
    # return rxyz, rxy
    

def pesb(n, side_length, cell_spacing, angle, tx, ty, tz):

    """pressure_evaluation_space_builder"""
    
    ev = np.arange(-side_length, side_length, cell_spacing) # side length vector for the plane being propagated to
    ex, ey = np.meshgrid(ev,ev)
    # x, y & z vectors for evaluation-plane sample points:
    px, py, pz = ex.reshape(len(ev)**2), ey.reshape(len(ev)**2), np.zeros(len(ev)**2) 
    
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


def heightmap_builder(phasemap, wavelength, discretisation_flag, discretisation=16):
    """
    builds a discrete heightmap using an analogue phasemap. Discretising the heightmap
    is usesful for 3d printing or testing with programs like comsol.
    
    params:
    phasemap = the analogue phasemap to be converted to a heightmap.
    wavelength = we kee this as a variable in case we want to do multifrequency modulation.
    discretisation = how many discrete pahse should the heightmap be limited to?
    They will be evenly spread throughout a 2pi range.
    """
    if discretisation_flag:
        db = [np.round(i, 4) for i in np.arange(-np.pi, np.pi, (2 * np.pi) / discretisation)]
        dis_phase_map = []
        n = phasemap.shape[0]
        for phase in np.array(phasemap).reshape(n ** 2):  # reshape into n**2-by-1 vector and cycle through elements
            current = db[0]  # dummy variable set to the first discretised brick phase delay value
            for brickID in db[1:]:  # cycle through analogue phase delays
                if np.abs(phase - current) > np.abs(phase - brickID):  # assign closest discrete brick value
                    current = brickID  # redefine dummy variable as closest discrete phase delay
            dis_phase_map.append(np.round(current, 4))  # discretised phase-delay map for each element
        dis_phase_map = np.array(dis_phase_map).reshape((n, n))  # reshape by into n-by-n matrix
        normalised_phasemap = (dis_phase_map + np.round(np.pi, 4)) / (2 * np.round(np.pi, 4))  # normalise between 0 and 1
        phase_delay_map = 1 - normalised_phasemap  # delays phase values are 1 - phase on the surface
        height_map = phase_delay_map * (wavelength / 2)  # convert to height between 0 and wavelength/2 (meters)
        return height_map
    
    else:
        normalised_phasemap = (phasemap + np.pi) / (2 * np.pi)  # normalise between 0 and 1
        phase_delay_map = 1 - normalised_phasemap  # delays phase values are 1 - phase on the surface
        height_map = (wavelength / 2) * phase_delay_map # convert to height between 0 and wavelength/2 (meters)
        return height_map

def brickmap_builder(phasemap, discretisation):
    """
    Convert a phase delay map to a brickmap.
    
    params:
    phasemap = the phasemap to be converted.
    discretisation = the number of discrete brick IDs (for interact bricks this is 16).
    """
    db = [np.round(i, 4) for i in np.arange(-np.pi, np.pi, (2 * np.pi) / discretisation)]
    brickmap = []
    n = phasemap.shape[0]
    # reshape into n**2-by-1 vector and cycle through elements
    for analogue_phase in np.array(phasemap).reshape(n ** 2):  
        current = db[0]  # dummy variable set to the first discretised brick phase delay value
        for brick_phase in db[1:]:  # cycle through analogue phase delays
            # assign closest discrete brick value
            if np.abs(analogue_phase - current) > np.abs(analogue_phase - brick_phase): 
                current = brick_phase  # redefine dummy variable as closest discrete phase delay
        brickmap.append(db.index(current))  # discretised phase-delay map for each element
    brickmap = np.array(brickmap).reshape((n, n))  # reshape by into n-by-n matrix
    return brickmap
    
    
def circular_distance(angle1, angle2):
    '''Find the circular distance between two angles'''
    return np.pi - abs(np.pi - abs(angle1 - angle2))


def circular_mean(phases):
    """
    find the circular mean of a set of phases
    """
    return np.arctan2(np.sum(np.sin(phases)), np.sum(np.cos(phases)))
    
    
def signed_circular_distance(angle1, angle2):
    '''
    Find the circular difference between two angles.
    The sign indicates whether the angle2 is clockwise or anticlockwise w.r.t angle1.
    This is useful when one intends to convert from phase to heights.
    '''
    return min(angle2-angle1, angle2-angle1+2*np.pi, angle2-angle1-2*np.pi, key=abs)


def seg_phasemap_builder_constant_differences(CS, input_phasemaps):
    
    """
        1. Loop through each segment in the CS.
        2. We find the "signed circular difference" between angles (pixels) across each configuration,
        using the zeroth configuration as our reference.
        3. Once we have a difference vector for each configuration, we can find the average difference vector.
        This represents the constant differences present between pixels in each segment.
        4. With this, we can generate new phase values for each configuration each segment is required to be in.
        5. Finally, we combine these new phases into segmented phasemap numpy arrays representing each configuration of
        the Segmented SSM.
    """
    
    flat_phasemaps = [phasemap.flatten() for phasemap in input_phasemaps]
    segmented_constant_diff_vectors = np.zeros_like(flat_phasemaps)
    
    for seg_ID in range(len(CS)):
        seg_phase_list, differences_list, mean_differences, corrected_phase_list = [], [], [], []

        # Find the "signed circular difference" between angles (pixels) in a given segment.
        for phasemap in input_phasemaps:
            seg_phase = phasemap.flatten()[list(CS[seg_ID])]
            differences = [signed_circular_distance(seg_phase[0], phase) for phase in seg_phase]
            seg_phase_list.append(seg_phase)
            differences_list.append(differences)

        # Calculate the mean differences accross all patterns.
        differences_array = np.array(differences_list).T
        for row in differences_array:
            mean_differences.append(np.mean(row))

        # save the new phases for the segments, now with constant differences between them for each configuration.
        for i in range(len(seg_phase_list)):
            corrected_phase_list.append([seg_phase_list[i][0] + mean_difference for mean_difference in mean_differences])

            for pixel_ID, new_phase in enumerate(corrected_phase_list[i]):
                segmented_constant_diff_vectors[i][CS[seg_ID][pixel_ID]] = new_phase
      
    segmented_constant_diff_phasemaps = np.array([phasemap.reshape(input_phasemaps[0].shape) for phasemap in segmented_constant_diff_vectors])
    return segmented_constant_diff_phasemaps
    

def naive_pattern_generator(pattern, elem_shape):
    '''
    generate homogeneous naive phasemaps for a given pattern by naively
    combining groups of elements and taking the circular mean.
    
    params:
    pattern = the pattern to be segmented.
    elem_shape = the shape of the naive segments.
    '''
    naive_pattern = np.zeros(pattern.shape) 
    for row in range(int(pattern.shape[0]/elem_shape[0])):
        for col in range(int(pattern.shape[0]/elem_shape[1])):
            a1, a2, b1, b2 = row*elem_shape[0], (row+1)*elem_shape[0], col*elem_shape[1], (col+1)*elem_shape[1]
            elem_mean = circular_mean(pattern[a1:a2, b1:b2])
            naive_pattern[a1:a2, b1:b2] = np.tile(elem_mean, (elem_shape[0], elem_shape[1])) 
    return naive_pattern
    
    
def combinations_without_repetition(r, iterable=None, values=None, counts=None):
    """
    iterates over unique combinations between elements in iterable of size r without
    repeating combos of identical elements in different orders.

    params:
    r = size of the combinations.
    iterable = the set of elements to be iterated over.
    values = the unique elements present in the set.
    counts = number of instances of each uniqe value in the set.
    """
    from itertools import chain, repeat, islice, count
    from collections import Counter
    if iterable:
        values, counts = zip(*Counter(iterable).items())
    f = lambda i, c: chain.from_iterable(map(repeat, i, c))
    n = len(counts)
    indices = list(islice(f(count(), counts), r))
    if len(indices) < r:
        return
    while True:
        yield tuple(values[i] for i in indices)
        for i, j in zip(reversed(range(r)), f(reversed(range(n)), reversed(counts))):
            if indices[i] != j:
                break
        else:
            return
        j = indices[i] + 1
        for i, j in zip(range(i, r), f(count(j), islice(counts, j, None))):
            indices[i] = j


def full_combinations_without_repetition(r, iterable):
    """
    This function allows the above function to work as intended
    even when some number of pixels have identical phase values
    
    params:
    r = size of the combinations.
    iterable = the set of elements to be iterated over.
    """
    keys = np.arange(1, len(iterable) + 1)
    it_dict = dict(zip(np.arange(len(keys)), iterable))
    temp_combos = list(combinations_without_repetition(r, iterable=it_dict.keys()))
    true_combos = []
    for combo in list(temp_combos):
        true_combos.append([list(it_dict.values())[i] for i in combo])
    return true_combos
 

def ssim_metric(imageA, imageB, registration_flag=False):
    """
    The SSIM image quality metric.
    Compares luminosity constrast and structure of two images to
    return a score between 1 (perfectly similar) and 0 (perfectly dissimilar).
    
    params:
    imageA, imageB = The images to be compared. order does not matter.
    registration_flag = If True, will perform a registration to align the features of the two images before
    SSIM comparison is made.
    
    """
    from skimage.metrics import _structural_similarity as ssim
    if registration_flag:
        from skimage.registration import phase_cross_correlation
        from scipy.ndimage import fourier_shift
        shift, error, diffphase = phase_cross_correlation(imageA, imageB)
        aligned_imageB = fourier_shift(np.fft.fftn(imageB), shift)
        imageB = np.fft.ifftn(aligned_imageB).real
    s = ssim.structural_similarity(imageA, imageB)
    return s
 
    
def mse_metric(imageA, imageB):
    '''Calculate the mean squared error between two images or arrays of equal size'''
    import numpy as np
    err = np.sum((imageA - imageB) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err
    
    
def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np

    if type not in ('bright', 'soft'):
        print('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')
    return random_colormap
  
  
def contour_rect(im):
    ''' Creates a set of vectors which can be plotted to draw lines around the edges of coalitions'''
    lines = []
    pad = np.pad(im, [(2, 2), (2, 2)])  # zero padding
    
    im0 = np.abs(np.diff(pad, n=1, axis=0))[:, 1:]
    im1 = np.abs(np.diff(pad, n=1, axis=1))[1:, :]

    im0 = np.diff(im0, n=1, axis=1)
    starts = np.argwhere(im0 == 1) - 1
    ends = np.argwhere(im0 == -1) - 1
    lines += [([s[0]-.5, s[0]-.5], [s[1]+.5, e[1]+.5]) for s, e in zip(starts, ends)]

    im1 = np.diff(im1, n=1, axis=0).T
    starts = np.argwhere(im1 == 1) - 1
    ends = np.argwhere(im1 == -1) - 1
    lines += [([s[1]+.5, e[1]+.5], [s[0]-.5, s[0]-.5]) for s, e in zip(starts, ends)]
    return lines
 