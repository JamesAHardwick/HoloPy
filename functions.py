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
    x = np.arange(centrepoint[0] - (extents[0][0]) + (pixel_spacing/2),
                  centrepoint[0] + (extents[0][1]),
                  pixel_spacing)
    
    y = np.arange(centrepoint[1] - (extents[1][0]) + (pixel_spacing/2),
                  centrepoint[1] + (extents[1][1]),
                  pixel_spacing)
    
    z = np.arange(centrepoint[2] - (extents[2][0]) + (pixel_spacing/2),
                  centrepoint[2] + (extents[2][1]),
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

    
    
def circular_distance(angle1, angle2):
    '''Find the circular distance between two angles'''
    return np.pi - abs(np.pi - abs(angle1 - angle2))
    
    
# def signed_circular_distance(angle1, angle2):
    # '''
    # Find the circular difference between two angles.
    # The sign indicates whether the angle2 is clockwise or anticlockwise w.r.t angle1.
    # This is useful when one intends to convert from phase to heights.
    # '''
    # return min(angle2-angle1, angle2-angle1+2*np.pi, angle2-angle1-2*np.pi, key=abs)
    
    
def signed_circular_distance(angle1, angle2):
    
    """
    
    Find the circular difference between two angles. The sign indicates whether the angle2 is clockwise or anticlockwise w.r.t angle1.
    This is useful when one intends to convert from phase to heights.
    
    args:
        angle1:
        angle2:
        
    returns:
        circular_difference_matrix:
        
    """
    
    vector1, vector2 = np.array(angle1).flatten(), np.array(angle2).flatten()
    circular_difference_vector = np.zeros_like(vector1)
    
    for i in range(len(circular_difference_vector)):
        circular_difference_vector[i] = min(vector2[i]-vector1[i],
                                  vector2[i]-vector1[i]+2*np.pi,
                                  vector2[i]-vector1[i]-2*np.pi,
                                  key=abs)
    
    circular_difference_matrix = circular_difference_vector.reshape(angle1.shape)    
    
    return circular_difference_matrix


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


def target_builder_image(filename):
    
    """
    
    Convert an image in .png format into a grayscale target in .npy format
    
    """
    
    import numpy as np, matplotlib.pyplot as plt
    target_image = plt.imread(filename+'.png')
    if len(target_image.shape) > 2: # remove rgb vestige if it is present
        target_image = target_image[:,:,:1]
        target_image = np.reshape(target_image[:,:],(target_image.shape[0], target_image.shape[1]))
    return abs((target_image/np.amax(target_image)) - 1)

    
def target_builder_chars(char_list, font_file, fontsize, im_h):
    
    """
    
    Creates an array of numpy array images to be used as targets with the iterative GS function.
    
    args:
        char_list: list of characters to be made into targets.
        font_file: .tff file containing the character font.
        fontsize: font size (in pts, knowing that 10pts = 13px).
        im_h: height of the numpy array in pixels, the function assumes a square array.
        
    returns:
        target_images: array of target images.
        
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

 
def prop_thresholder(prop, threshold):
    
    """
    
    x
    
    args:
    
    returns:
    
    """
    
    prop[prop < threshold] = 0
    return prop
     
    
def focus_phasemap_builder(AMM_points, focal_point_position, k):

    """
    Builds a phasemap for an AMM or PAT creating a focus at some point in 3D space.

    args:
        AMM_points: matrix describing the postitions of elements on the AMM or PAT surface.
        focal_point_position: (x, y, z) coords of the focus.
        k: wavenumber
        
    returns:
        norm_phase_array: matrix of phase delays from -pi to pi, with the same size as AMM_points, describing the
        phase delays to build a focus at focal_point_position.

    """

    # matrix of distances from centre of each elem to focus
    travel_distance_array = np.sqrt((AMM_points[0] + focal_point_position[0])**2 + \
                                    (AMM_points[1] + focal_point_position[1])**2 + \
                                    (AMM_points[2] + focal_point_position[2])**2)

    # total change in phase of waves as they travel this distance.
    total_phase_array = -travel_distance_array * k 

    # normalise between 0 and 2π [rads].
    norm_phase_array = np.remainder(total_phase_array, 2*np.pi) - np.pi

    return norm_phase_array


def heightmap_builder_simple(phasemap, wavelength):
    normalised_phasemap = phasemap / (2 * np.pi)  # normalise between 1 and -1
    phase_delay_map = -normalised_phasemap  # invert the sign to create phase delays
    height_map = (wavelength/2) * phase_delay_map # multiply by wavelength/2 to convert to height (meters)
    return height_map


def phasemap_normaliser(phasemap):
    phasemap[phasemap>np.pi] = phasemap[phasemap>np.pi] - 2*np.pi
    phasemap[phasemap<-np.pi] = phasemap[phasemap<-np.pi] + 2*np.pi
    return phasemap


def heightmap_builder(phasemap, wavelength, discretisation_flag, discretisation=16):
    """
    builds a discrete heightmap using an analogue phasemap. Discretising the heightmap
    is usesful for 3d printing or testing with programs like comsol.
    
    params:
    phasemap = the analogue phasemap to be converted to a heightmap.
    wavelength = we keep this as a variable in case we want to do multifrequency modulation.
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
    m, n = phasemap.shape
    # reshape into n*m-by-1 vector and cycle through elements
    for analogue_phase in np.array(phasemap).reshape(m*n):  
        current = db[0]  # dummy variable set to the first discretised brick phase delay value
        for brick_phase in db[1:]:  # cycle through analogue phase delays
            # assign closest discrete brick value
            if np.abs(analogue_phase - current) > np.abs(analogue_phase - brick_phase): 
                current = brick_phase  # redefine dummy variable as closest discrete phase delay
        brickmap.append(db.index(current))  # discretised phase-delay map for each element
    brickmap = np.array(brickmap).reshape((m, n))  # reshape by into m-by-n matrix
    return brickmap