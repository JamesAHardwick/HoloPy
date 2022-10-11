import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools as it
import math as math
from scipy.special import comb


def pressure_calculator(voltage_list, V_Pa=3.16/1000):
    
    """
    
    Calculate absolute pressure from a list of voltage measurements.
    
    args:
        voltage_list: List of voltage measurements for this data point.
        V_Pa: Amplifier reference.
        
    returns:
        Measured absolute pressure at this datapoint.
        
    """
    
    V_peak = (np.amax(voltage_list) - np.min(voltage_list))/2
    
    return V_peak / V_Pa


def phase_calculator(voltage_list, sample_period=250):
        
    """
    
    Calculate phase from a list of voltage measurements.
    
    args:
        voltage_list: List of voltage measurements for this data point.
        sample_period: Time in microseconds between voltage samples.
        
    returns:
        Measured phase at this datapoint.
        
    """
    
    V_min = min(voltage_list)
    V_min_time_index = list(voltage_list).index(V_min)
    
    return  (2*np.pi/sample_period) * np.mod(V_min_time_index-1, sample_period) - np.pi


def free_field_mic_correction(complex_pressure_matrix, microphone_correction_dB):
    
    """
    
    https://kiptm.ru/images/Production/bruel/table_pdf/microphones_preamps/2017.02/4138.pdf
    
    Microphone correction to account for the angle of beast measurement.
    
    args:
        complex_pressure_matrix: Matrix of complex pressures measured by the Beast.
        microphone_correction_dB: Correction detrmined by the graph of page 2 of the linked PDF file.
        
    returns:
        complex_pressure_matrix_Pa: Actual complex pressure measurement having accounted for the angle of the microphone.
        
    """
    
    complex_pressure_matrix_SPL = 20*np.log10(complex_pressure_matrix/20e-6) # convert to SPL
    complex_pressure_matrix_SPL = complex_pressure_matrix_SPL - microphone_correction_dB # subtract
    corrected_complex_pressure_matrix = (20e-6) * 10**(complex_pressure_matrix_SPL/20)*np.sqrt(2) # convert back Pascals
        
    return corrected_complex_pressure_matrix


def plot_shape_finder(scan_size, step):
    
    """
    
    find the shape of the plot in units of 'step' rather than mm.
    
    args:
        scan_size: tuple with (x,y,z) dimensions of the scan [mm]
        step: size of steps between datapoints [mm]
        
    returns:
        shape of the plot in units of 'step'.
        
    """
    
    if scan_size[0] == 0:  # yz
        return (int(scan_size[1]/step)+1, int(scan_size[2]/step)+1)
    
    elif scan_size[1] == 0: # xz
        return (int(scan_size[0]/step)+1, int(scan_size[2]/step)+1)

    elif scan_size[2] == 0: # xy
        return (int(scan_size[0]/step)+1, int(scan_size[1]/step)+1)
        
    else:
        print(scan_size, "is not a valid scan size, exactly 1 dimension must equal zero.")
        
        
def read_csv_data(folder_path, filename):
    
    """
    
    x
    
    args:
        folder_path:
        filename:
        
    returns:
        coords:
        voltages:
        
    """
    
    import csv
    
    coords, voltages = [], []
        
    # load in the csv file
    with open(folder_path+"/csv/"+filename+".csv", newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        for _ in range(8): # skip the headers
            next(reader)

        # extract coords and voltages
        for i, row in enumerate(reader):
            if i % 2 == 0: 
                coords.append(tuple([float(elem) for elem in row[:3]]))
            else:
                voltages.append([float(elem) for elem in row])
                
    return coords, voltages
    
def find_complex_pressure_matrix(coords, voltages, step, microphone_correction_dB):
        
    """
    
    x
    
    args:
        coords:
        voltages:
        step:
        microphone_correction_dB:
        
    returns:
        complex_pressure_matrix:
        
    """ 
        
    # calculate pressure and phase and save in dicts with datapoint coord tuple as the key  
    abs_pressure_dict, phase_dict = {}, {}

    for i, coord in enumerate(coords):
        abs_pressure_dict[tuple(coord)] = pressure_calculator(voltages[i], V_Pa=3.16/1000)
        phase_dict[tuple(coord)] = phase_calculator(voltages[i], sample_period=250)

    # reformat into lists without inverted rows
    abs_pressure_list, phase_list = [], []

    starting_coord = coords[0] # first coord measured by the beast
    ending_coord = coords[-1]  # final coord measured by the beast
    scan_size = tuple(map(lambda i, j: round(abs(i - j), 2), starting_coord, ending_coord)) # size of the 2d scan
    plot_shape = plot_shape_finder(scan_size, step) # shape of the sacn plane

    for dx in np.arange(0, scan_size[0]+step, step).round(2):
        for dy in np.arange(0, scan_size[1]+step, step).round(2):
            for dz in np.arange(0, scan_size[2]+step, step).round(2):
                datapoint_coords = tuple(map(lambda i, j: round(abs(i + j), 2), starting_coord, (dx, dy, dz)))
                abs_pressure_list.append(abs_pressure_dict[datapoint_coords])
                phase_list.append(phase_dict[datapoint_coords])

    # abs pressure and phase matrices for the scan 
    abs_pressure_matrix = np.array(abs_pressure_list).reshape(plot_shape)
    phase_matrix = np.array(phase_list).reshape(plot_shape)

    # calculate complex pressure for the scan 
    uncorrected_complex_pressure_matrix = abs_pressure_matrix*np.exp(1j*phase_matrix)
    corrected_complex_pressure_matrix = free_field_mic_correction(uncorrected_complex_pressure_matrix,
                                                                  microphone_correction_dB)

    # matrix must be inverted and flipped as the beast measures backwards...
    complex_pressure_matrix = np.flipud(corrected_complex_pressure_matrix.T)

    return complex_pressure_matrix


def interpret_beast_data(folder_path, filename, step, microphone_correction_dB, npy_save_flag=False, npy_save_path=""):

    """
    
    x
    
    args:
        csv_file_path:
        step:
        microphone_correction_dB:
        
    returns:
    
    
    """
        
    # read csv file and save the coords and voltages as lists of tuples and floats respectively
    coords, voltages = read_csv_data(folder_path, filename)
    
    # convert coord and voltage data into a complex pressure matrix
    complex_pressure_matrix = find_complex_pressure_matrix(coords, voltages, step, microphone_correction_dB)
        
    if npy_save_flag:
        os.makedirs(folder_path+"/npy/", exist_ok=True)
        np.save(folder_path+"/npy/"+filename+"_complex_pressure_matrix.npy", complex_pressure_matrix)
        
    return complex_pressure_matrix