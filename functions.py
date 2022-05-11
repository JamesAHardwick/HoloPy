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
    Gb = F * Hb  # propagating backwards from target to lens
    gb = np.fft.ifft2(np.fft.ifftshift(Gb))  # IFT & shift to return to real space
    return gb  # return backpropagation


def prop(phasemap, cell_spacing, target_dist, res, k):
    """
    Propagate a phasemap forward to a parallel plane at a given resolution.
    Returns a complex pressure field. Assumes plane wave source.
    
    params:
    phasemap = the phasemap.
    cell_spacing = seperation between cells on the metasurface
    target_dist = distance to propagation plane
    res = returns this many sample points for each point in the input phasemap
    k = wavenumber
    """
    aperture = ((phasemap != 0).astype(int))
    lpp = aperture * np.exp(1j * phasemap)
    f = np.kron(lpp, np.ones((res, res)))
    Nfft = f.shape
    
    ## Calculate and shape k vectors
    kx = 2*np.pi*(np.arange(-Nfft[0]/2,(Nfft[0])/2)/((cell_spacing/res)*Nfft[0])).reshape((1, Nfft[0]))
    ky = 2*np.pi*(np.arange(-Nfft[1]/2,(Nfft[1])/2)/((cell_spacing/res)*Nfft[1])).reshape((1, Nfft[1]))
    
    ## Fourier transform
    F = np.fft.fft2(f) # 2D FT
    F = np.fft.fftshift(F) # Shift to the centre
    
    ## Propagate forwards    
    H = np.exp(1j*np.lib.scimath.sqrt(k**2 - kx**2 - (ky**2).T)*target_dist) # propagator function
    Gf = F*H.T # propagating the signal forward in Fourier space
    gf = np.fft.ifft2(np.fft.ifftshift(Gf)) # IFT & shift to return to real space
    return gf
    
    
def target_builder(char_list, font_file, fontsize, im_h):
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


def padwidth(N, array):
    """
    Dumb function which needs to be deleted. currently GS relies on this, but i need to remove that reliance.
    
    TO DO: eliminate the need for this function. 
    """
    return ((int((N - array.shape[0]) / 2), int((N - array.shape[0]) / 2)),
            (int((N - array.shape[1]) / 2), int((N - array.shape[1]) / 2))) 
    
    
def Iterative_GS(target_image, iterations, cell_spacing, target_dist, res, k):
    """
    implements the iterative GS algorithm to create phasemaps for a chosen target a chosen distance away.
    Assumes plane wave source.
    
    params:
    target_image = the target image as a normalised numpy array (all elems should be 0-1).
    iterations = number of iterations for the algorithm to run (normally assymptotes ~200).
    ell_spacing = seperation between cells on the metasurface.
    target_dist = distance to propagation plane.
    res = returns this many sample points for each point in the input phasemap.
    k = wavenumber.
    """
    apsize = target_image.shape
    n = apsize[0]
    N = 3*n
    pw = padwidth(N, np.zeros(apsize))
    padded_target = np.pad(target_image, pw, 'constant', constant_values = 0) # Pad with zeros 
    aperture = np.ones(apsize)
    padded_aperture = np.pad(aperture, pw, 'constant', constant_values = 0)
    tpp = padded_target*np.exp(1j*padded_target*np.pi) # initial pressure at target plane
    for it in range(iterations):
        lpp = ASM_bw(tpp, cell_spacing, target_dist, k) # inverse ASM to backpropagate complex pressure to lens-plane
        lpp = padded_aperture*np.exp(1j*np.angle(lpp)*padded_aperture) # isolate over aperture
        tpp = ASM_fw(lpp, cell_spacing, target_dist, res, k) # forward ASM to propagate updated phase as complex pressure to the target-plane
        tpp = padded_target*np.exp(1j*np.angle(tpp)*padded_target) # isolate target area
    lpp = ASM_bw(tpp, cell_spacing, target_dist, k)
    lpp = padded_aperture*np.exp(1j*np.angle(lpp)*padded_aperture) # isolate
    tpp = ASM_fw(lpp, cell_spacing, target_dist, res, k)
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
    n = apsize[0]
    N = 4*n
    pw = padwidth(N, np.zeros(apsize))
    surface_amplitude, surface_phase = abs(incident_surface_pressure), np.angle(incident_surface_pressure)
    aperture = ((surface_amplitude != 0).astype(int))
    
    padded_target = np.pad(target_image, pw, 'constant', constant_values = 0) # Pad with zeros 
    padded_aperture = np.pad(aperture, pw, 'constant', constant_values = 0)
    padded_surface_amplitude = np.pad(surface_amplitude, pw, 'constant', constant_values = 0)
    
    tpp = padded_target*np.exp(1j*padded_target*np.pi) # initial pressure at target plane
    
    for it in range(iterations):
        lpp = ASM_bw(tpp, cell_spacing, target_dist, k) # inverse ASM to backpropagate complex pressure to lens-plane
        lpp = padded_surface_amplitude*np.exp(1j*np.angle(lpp)*padded_aperture) # isolate over aperture
        tpp = ASM_fw(lpp, cell_spacing, target_dist, 1, k) # forward ASM to propagate updated phase as complex pressure to the target-plane
        tpp = padded_target*np.exp(1j*np.angle(tpp)*padded_target) # isolate target area
        
    lpp = ASM_bw(tpp, cell_spacing, target_dist, k)
    lpp = padded_aperture*np.exp(1j*np.angle(lpp)*padded_aperture) # isolate
    tpp = ASM_fw(lpp, cell_spacing, target_dist, 1, k)
    return np.angle(lpp)[pw[0][0]:pw[0][0] + apsize[0], pw[1][0]:pw[1][0] + apsize[1]]
    

def rotate_2d(cccc):
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


def heightmap_builder(phasemap, wavelength, discretisation):
    """
    builds a discrete heightmap using an analogue phasemap. Discretising the heightmap
    is usesful for 3d printing or testing with programs like comsol.
    
    params:
    phasemap = the analogue phasemap to be converted to a heightmap.
    wavelength = we kee this as a variable in case we want to do multifrequency modulation.
    discretisation = how many discrete pahse should the heightmap be limited to?
    They will be evenly spread throughout a 2pi range.
    """
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