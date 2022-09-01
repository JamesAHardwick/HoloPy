import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools as it
import math as math
from scipy.special import comb


def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    
    """
    
    https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
    
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    
    args:
        nlabels: Number of labels (size of colormap)
        type: 'bright' for strong colors, 'soft' for pastel colors
        first_color_black: Option to use first color as black, True or False
        last_color_black: Option to use last color as black, True or False
        verbose: Prints the number of labels and shows the colormap. True or False
    
    returns:
        colormap for matplotlib
        
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
    
    """
    
    Creates a set of vectors which can be plotted to draw lines around the edges of coalitions
    
    args:
        im: image
    
    returns:
        lines: list of line vectors to be drawn onto a plot
    
    """
    
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
    
    
def coalition_line_drawer(ax, segmented_CS, m, n, num_pixels, edge_line_width):
    
    """
    
    Creates a set of vectors which can be plotted to draw lines around the edges of coalitions
    
    args:
        ax:
        segmented_CS:
        m, n:
        num_pixels:
        edge_line_width:
    
    returns:
        None
        
    """
    
    CS_map_array = np.zeros(num_pixels) # segment array = positions of the various coalitions   
    for i, coalition in enumerate(segmented_CS):
        for pixel in coalition:
            CS_map_array[pixel] = i
    CS_map_array = np.reshape(CS_map_array, (m, n))
    
    contour_lines = []
    for ID in range(len(segmented_CS)):
        coalition = np.zeros(CS_map_array.size)
        coalition[list(segmented_CS[ID])] = 1
        coalition = coalition.reshape(CS_map_array.shape)
        contour_lines.append(contour_rect(coalition))
    for lines in contour_lines:
        for line in lines:
            ax.plot(line[1], line[0], color='k', lw=edge_line_width)
    return None
        
        
def plot_edger():

    """
    
    Creates a set of vectors which can be plotted to draw lines around the edges of coalitions
    
    args:
        ax:
        edge_line_width:
    
    returns:
        None
    
    """

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(edge_line_width)    
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False,
                   right=False, labelbottom=False, labelleft=False)
    return None    
    
    
def CS_structure_plotter(ax, segmented_CS, font_size, edge_line_width, CS_labels=True, coalition_lines=True):

    """
    
    Creates a set of vectors which can be plotted to draw lines around the edges of coalitions
    
    args:
        ax:
        segmented_CS:
        font_size:
        edge_line_width:
        CS_labels:
        coalition_lines:
    
    returns:
        None
    
    """

    CS_map_array = np.zeros(input_phasemaps[0].size) # segment array = positions of the various coalitions  
    
    for i, coalition in enumerate(segmented_CS):
        for pixel in coalition:
            CS_map_array[pixel] = i
    CS_map_array = np.reshape(CS_map_array, (m, n))

    cmap_list = [[1., 1., 1., 1.] for i in range(target_CS_length)]
    CS_cmap = ListedColormap(cmap_list)
    
    ax.imshow(CS_map_array, cmap=plt.get_cmap(CS_cmap))

    plot_edger(ax, edge_line_width)

    if CS_labels:
        for i, value in enumerate(np.reshape(CS_map_array, CS_map_array.size)):
            coord = PixIDToPos((m, n), i)
            ax.text(coord[1], coord[0], str(int(value+1)), ha="center", va="center", fontsize=font_size)
    
    if coalition_lines:
        coalition_line_drawer(ax, segmented_CS, m, n, num_pixels, edge_line_width)
    
    return None