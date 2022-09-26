import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools as it
import math as math
from scipy.special import comb


def PixPosToID(apsize, row, col):
    
    """
    
    Convert tuple of coordinates to a 1-character ID.
    
    args:
    
    returns:
    
    """
    
    return apsize[1] * row + col


def PixIDToPos(apsize, ID):
    
    """
    
    Convert 1-character ID to tuple of coordinates.
    
    args:
    
    returns:
    
    """
    row = np.mod(ID, apsize[1])
    col = (ID - np.mod(ID, apsize[1])) / apsize[1]
    return int(col), int(row)


def circular_mean(phases):
    """
    find the circular mean of a set of phases
    """
    return np.arctan2(np.sum(np.sin(phases)), np.sum(np.cos(phases)))


def cmad(phases, axis=None):
    
    """
    
    x
    
    args:
    
    returns:
    
    """

    return np.mean((1 - np.cos(phases - circular_mean(phases)))/2, axis)  


def quality_metric_flat(coalition_pixels, phasemap_vectors):

    """
    
    x
    
    args:
    
    returns:
    
    """

    Q = []
    for vec in (phasemap_vectors):
        Q.append(1 - cmad(vec[coalition_pixels]))
    return np.mean(Q)


def quality_metric_cdiffs(coalition_pixels, phasemap_vectors):

    """
    
    x
    
    args:
    
    returns:
    
    """

    norm_vecs, Q = [], []
    for vec in (phasemap_vectors):
        norm_vecs.append(vec[coalition_pixels] - min(vec[coalition_pixels]))
    for diffs_accross_patterns in np.array(norm_vecs).T:
        circ_range = circular_dist(max(diffs_accross_patterns), min(diffs_accross_patterns))
        Q.append(1 - circ_range/(2*np.pi))
    return np.mean(Q)
    

def naive_pattern_generator(pattern, elem_shape):
    
    """
    
    generate homogeneous naive phasemaps for a given pattern by naively
    combining groups of elements and taking the circular mean.
    
    args:
        pattern: the pattern to be segmented.
        elem_shape: the shape of the naive segments.
    
    returns:
    
    """
    
    naive_pattern = np.zeros(pattern.shape) 
    for row in range(int(pattern.shape[0]/elem_shape[0])):
        for col in range(int(pattern.shape[0]/elem_shape[1])):
            a1, a2, b1, b2 = row*elem_shape[0], (row+1)*elem_shape[0], col*elem_shape[1], (col+1)*elem_shape[1]
            elem_mean = circular_mean(pattern[a1:a2, b1:b2])
            naive_pattern[a1:a2, b1:b2] = np.tile(elem_mean, (elem_shape[0], elem_shape[1])) 
    return naive_pattern
    
    
def combinations_without_repetition(r, iterable=None, values=None, counts=None):
    
    """
    
    https://stackoverflow.com/questions/36429507/python-combinations-without-repetitions
    iterates over unique combinations between elements in iterable of size r without
    repeating combos of identical elements in different orders.

    args:
        r: size of the combinations.
        iterable: the set of elements to be iterated over.
        values; the unique elements present in the set.
        counts; number of instances of each uniqe value in the set.
        
    returns:
    
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
    
    This function allows the above function to work as intended even
    when some number pixels have identical phase values
       
    args:
    
    returns:
    
    """
    
    keys = np.arange(1, len(iterable) + 1)
    it_dict = dict(zip(np.arange(len(keys)), iterable))
    temp_combos = list(combinations_without_repetition(r, iterable=it_dict.keys()))
    true_combos = []
    for combo in list(temp_combos):
        true_combos.append([list(it_dict.values())[i] for i in combo])
    return true_combos
    
            
def CS_naive(shape, elem_size): 

    """
    
    x
    
    args:
    
    returns:
    
    """
    
    CS_array = np.arange(shape[0]*shape[1]).reshape(shape)
    CS = []
    for row in np.arange(0, shape[0], elem_size[0]):
        for col in np.arange(0, shape[1], elem_size[1]):
            elem = list(CS_array[row:row+elem_size[0], col:col+elem_size[1]].reshape(elem_size[0]*elem_size[1]))
            CS.append(elem)
    return(CS)
    
    
def CS_data_loader(input_phasemaps, home_path, source, dataset, datatype, suffix):

    """
    
    x
    
    args:
    
    returns:
    
    """
    
    # Load in segmented CS data
    CS_data_folder = home_path+"/output_data/"+datatype+"/pre_processed_data/"+source+"/"+dataset+"/"+suffix
    CS_data = []
    for iteration in range(1, input_phasemaps[0].size):
        current_CS = list(np.load(CS_data_folder+"/results-"+str(iteration)+".npy", allow_pickle=True))
        CS_data.append(list(current_CS))
    print("Sucessfully loaded",source, dataset, datatype, suffix, "data.")
    return CS_data


def flatten_tuple(data):
    
    """
    
    Flatten a list of tuples into a single large tuple
    
    args:
    
    returns:
    
    """
    
    if isinstance(data, tuple):
        if len(data) == 0:
            return ()
        else:
            return flatten_tuple(data[0]) + flatten_tuple(data[1:])
             
    else:
        return (data,)
        
           
    
def contiguous_combinations(num_pixels, segments):
    
    """
    
    Find all of the segments which it is valid to combine in a square array of size num_pixels
           
    args:
    
    returns:
    
    """
    
    possible_combos = []
    sidelen = int(np.sqrt(num_pixels))
    for segment_1 in segments:
        flatten_tuple(segment_1)
        new_segments = segments.copy()
        new_segments.remove(segment_1)

        for segment_2 in new_segments:
            segment_2 = flatten_tuple(segment_2)
            for bb_1 in segment_1:
                for bb_2 in segment_2:
                    if bb_1 == bb_2-1 and bb_2%sidelen != 0 \
                    or bb_1 == bb_2+1 and bb_1%sidelen != 0 \
                    or bb_1 == bb_2-sidelen \
                    or bb_1 == bb_2+sidelen:
                        seg1 = tuple(sorted(segment_1))
                        seg2 = tuple(sorted(segment_2))
                        combo = tuple(sorted((seg1, seg2)))
                        if combo not in possible_combos:
                            possible_combos.append(combo)
    return possible_combos

    
def run_AHC_algorithm(inputs, diff_datatype, output_data_folder, data_save_flag, override_flag, verbose_flag):

    """
    Performs a full run of the Aglomerative Hierarchical Clustering algorithm, saving the results for each iteration.
    If the CS_data already exists in output_data_folder, then it will load this data, rather than recalculate it.
    
    args:
        inputs: The list of unsegmented phase or difference maps (numpy arrays) which we want to segment.
        diff_datatype: string describing what type of clustering we want to perform.
        output_data_folder: folder to save the CS_data in each iteration.
        data_save_flag: if yes, CS_data is saved in the above folder in each iteration.
        verbose_flag: if yes, the function will print each iteration as it is calculated
        and provide a total calculation time when it is complete.
    
    returns:
        CS_data: returns a list of clustering structures for each iteration of the algorithm.
    
    """
    
    def AHC_initialisation(flat_phasemaps):
    
        """
        
        Initialise the CS dictionary - (keys: segment tuple, values: segment quality).
        In this initial step we return a surface in which each building block is it's own segment
        
        args:
        
        returns:
        
        """
        
        # from functions import circular_mean
        
        num_pixels = len(flat_phasemaps[0])
        segments = [[pixel] for pixel in range(num_pixels)]
        current_CS = {}
        
        for segment in segments:
            current_CS[tuple(segment)] = quality_metric_flat(segment, flat_phasemaps)
        
        return current_CS


    def run_AHC_iteration(current_CS, combination_qualities, flat_phasemaps, datatype, verbose_flag):
        
        """
        
        Run one iteration of the contiguous HC algorithm
        
        args:
        
        returns:
        
        """
        
        # calculate all combinations possible with the current CS
        if datatype == "contiguous_clustering_adjacent" or datatype == "differences_contiguous_clustering_adjacent":
            combinations = contiguous_combinations(len(flat_phasemaps[0]), list(current_CS.keys()))
        elif datatype == "noncontig_clustering":
            combinations = [tuple(combo) for combo in full_combinations_without_repetition(2, list(current_CS.keys()))]
        else:
            print(datatype, "is not a valid datatype.")
            return None
        
        # if the qualities for these combinations are not in the dictionary, add them
        for combination in combinations:
            if not combination in combination_qualities:
                combination_pixels = [pixel for sublist in combination for pixel in sublist]
                combination_qualities[combination] = quality_metric_flat(combination_pixels, flat_phasemaps)
        
        # find the highest quality combination in the dictionary - this is the combo we will make in this iteration.
        best_combination = max(combination_qualities, key=combination_qualities.get)
        best_combination_key = tuple(sorted(flatten_tuple(best_combination)))
        
        # add the new combination to the current CS
        current_CS[best_combination_key] = combination_qualities[best_combination]
        
        if verbose_flag:
            print("best combination and quality:", best_combination, current_CS[best_combination_key])
        
        # remove the segments which we combined to make it
        for seg in best_combination:
            current_CS.pop(seg)
        
        # prepare combinations which contain building blocks from the new combinations from the quality dictionary.
        removal_list = []
        for combination in list(combination_qualities):
            for bb in best_combination:
                if bb in combination and combination not in removal_list:
                    removal_list.append(combination)
        
        # remove them from the dict.
        for combination in removal_list:
    #         print(combination)
            combination_qualities.pop(combination)

        # return the CS for this iteration
        return current_CS
    
    
    # ----> comparison CS data <---
    flat_inputs = [input_map.flatten() for input_map in inputs]
    num_pixels = len(flat_inputs[0])

    # ---> initialise data storage <---
    CS_data = [] # stores each CS from each iteration as a list of tuples
    combination_qualities = {} # stores all previously calculated segment qualities
    clock = {}

    # ---> 1st iteration of the algorithm <---
    current_CS = AHC_initialisation(flat_inputs)
    CS_data.append(list(current_CS))

    if data_save_flag:
        os.makedirs(output_data_folder, exist_ok=True)
        np.save(output_data_folder+"/results-"+str(0)+".npy", list(current_CS.keys()))

    # ---> remaining iterations <---
    for iteration in range(1, num_pixels):
        if os.path.exists(output_data_folder+"/results-"+str(iteration)+".npy") and override_flag == False:
            if verbose_flag:
                print("Loading iteration: ", iteration)
            current_CS = np.load(output_data_folder+"/results-"+str(iteration)+".npy", allow_pickle=True)
            CS_data.append(list(current_CS))

        else:
            if verbose_flag:
                print()
                print("Calculating iteration: ", iteration)
            start_timer = time.perf_counter()
            current_CS = run_AHC_iteration(current_CS, combination_qualities, flat_inputs, diff_datatype, verbose_flag)
            CS_data.append(list(current_CS))
            if data_save_flag:
                np.save(output_data_folder+"/results-"+str(iteration)+".npy", list(current_CS.keys()))
            stop_clock = time.perf_counter() - start_timer
            clock[iteration] = stop_clock  

    if verbose_flag:
        print()        
        print("total calc time:", round(sum(list(clock.values())), 2), "s")
        
    return CS_data
    
def segmented_phasemaps_list_builder(CS_data, input_phasemaps, post_processed_data_folder,
                                data_save_flag, override_flag, verbose_flag):
    
    """
    
    Build segmented phasemaps from the CS_data and the original, input phasemaps and save into a list. If the data already exists in
    post_processed_data_folder, load it rather than recalculating it.
    
    args:
        CS_data: list of clustering structures of sizes={1, m*n, 1}.
        input_phasemaps: original phasemaps, accounting for the incident source waves.
        post_processed_data_folder: folder to save the list of segmented phasemaps.
        data_save_flag: if yes, the segmented phasemaps is saved in the above folder.
        override_flag: if yes, ignore data already present in post_processed_data_folder and recalculate.
        verbose_flag: if yes, the function will print each iteration as it is calculated

        
    returns:
        seg_phasemaps_list: returns a list of lists
    
    """
    
    def seg_phasemap_builder_flat(CS, input_phasemaps):

        """
        
        x
        
        args:
        
        returns:
        
        """
        
        segmented_phasemaps = []
        for i, phasemap in enumerate(input_phasemaps):
            
            flat_phasemap = phasemap.flatten() # reshape to a vector
            segmented_phasemap = np.ones_like(flat_phasemap) # initialise the new segmented phasemap
            
            for coalition_num, coalition in enumerate(CS):
                
                coalition_phases = []
                
                for pixel in coalition:
                    coalition_phases.append(flat_phasemap[pixel]) # add each phase value in the coalition to a list
               
                coalition_mean_phase = circular_mean(coalition_phases) # find the mean
                
                for pixel in coalition: # replace each pixel in our new segmented surface with the mean value
                    segmented_phasemap[pixel] = coalition_mean_phase    
            
            segmented_phasemap = np.reshape(segmented_phasemap, input_phasemaps[0].shape)
            segmented_phasemap = np.mod(segmented_phasemap, 2*np.pi) - np.pi
            
            segmented_phasemaps.append(segmented_phasemap)
        return segmented_phasemaps
    
    
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
        
        args:
        
        returns:

        """
        
        from functions import signed_circular_distance
        
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
    
    
    seg_phasemaps_list = []

    # If seg_phasemaps_list.npy already exists, then load it instead of recalculating
    if os.path.exists(post_processed_data_folder+"/seg_phasemaps_list.npy") and override_flag == False:
        
        if verbose_flag:
            print("loading segmented phasemap data from... "+post_processed_data_folder+"/seg_phasemaps_list.npy")
            
        seg_phasemaps_list = np.load(post_processed_data_folder+"/seg_phasemaps_list.npy")

    # else, perform the calculation and save to the post processing folder for this dataset   
    else:
        
        if verbose_flag:
            print("calculating segmented phasemap data...")
            
        for CS_ID, CS in enumerate(CS_data):
            seg_phasemaps = seg_phasemap_builder_constant_differences(CS, input_phasemaps)
            seg_phasemaps_list.append(seg_phasemaps)

        if data_save_flag:
            np.save(post_processed_data_folder+"/seg_phasemaps_list.npy", seg_phasemaps_list)
        
        if verbose_flag:
            print("...done.")
            
    return seg_phasemaps_list
    
    
# def segmented_props_list_builder(CS_data, seg_phasemaps_list, post_processed_data_folder,
                            # Pf, m_xy_eval, n_xy_eval, H_list,
                            # data_save_flag, override_flag, verbose_flag):
    
    # """
    
    # Segmented Props
    
    # args:
        # CS_data: list of clustering structures of sizes={1, m*n, 1}.
        # seg_phasemaps_list: list of lists of segmented phasemaps.
        # post_processed_data_folder: folder to save the list of segmented phasemaps.
        # Pf:
        # m_xy_eval, n_xy_eval:
        # H_list:
        # data_save_flag: if yes, the segmented phasemaps is saved in the above folder.
        # override_flag: if yes, ignore data already present in post_processed_data_folder and recalculate.
        # verbose_flag: if yes, the function will print each iteration as it is calculated

        
    # returns:
        # seg_props_list: returns a list of lists
    
    # """

    # from GF_functions import GF_prop

    # seg_props_list = []

    # # If seg_props_list_"+prop_plane+".npy already exists, then load it instead of recalculating
    # if os.path.exists(post_processed_data_folder+"/seg_props_list_"+prop_plane+".npy") and override_flag == False:
        
        # if verbose_flag:
            # print("loading segmented propagation data from... "+post_processed_data_folder+"/seg_props_list_"+prop_plane+".npy")
            
        # seg_props_list = np.load(post_processed_data_folder+"/seg_props_list_"+prop_plane+".npy")

    # # else, perform the calculation and save to the post processing folder for this dataset   
    # else:
        
        # if verbose_flag:
            # print("calculating segmented propagation data...")
            
        # for CS_ID, CS in enumerate(CS_data):
            
            # seg_props = []
              
            # for i, seg_phasemap in enumerate(seg_phasemaps_list[CS_ID]):

                # surface_pressure = abs(Pf)*np.exp(1j*(seg_phasemap + np.angle(Pf)))
                # prop_vec = GF_prop(surface_pressure.flatten(), H_list[i], "forward")
                # prop_mat = prop_vec.reshape((m_xy_eval, n_xy_eval))
                # seg_props.append(prop_mat)
                
            # seg_props_list.append(seg_props)

        # if data_save_flag:
            # np.save(post_processed_data_folder+"/seg_props_list_"+prop_plane+".npy", seg_props_list)
            
        # return seg_props_list
        
        
def segmented_props_list_builder(CS_data, seg_phasemaps_list, post_processed_data_folder,
                                 Pf, output_shape, H_list, prop_plane,
                                 data_save_flag, override_flag, verbose_flag):
    
    """
    
    Segmented Props
    
    args:
        CS_data: list of clustering structures of sizes={1, m*n, 1}.
        seg_phasemaps_list: list of lists of segmented phasemaps.
        post_processed_data_folder: folder to save the list of segmented phasemaps.
        Pf:
        output_shape:
        H_list:
        data_save_flag: if yes, the segmented phasemaps is saved in the above folder.
        override_flag: if yes, ignore data already present in post_processed_data_folder and recalculate.
        verbose_flag: if yes, the function will print each iteration as it is calculated

        
    returns:
        seg_props_list: returns a list of lists
    
    """

    from GF_functions import GF_prop

    seg_props_list = []

    # If seg_props_list_"+prop_plane+".npy already exists, then load it instead of recalculating
    if os.path.exists(post_processed_data_folder+"/seg_props_list_"+prop_plane+".npy") and override_flag == False:
        
        if verbose_flag:
            print("loading segmented propagation data from... "+post_processed_data_folder+"/seg_props_list_"+prop_plane+".npy")
            
        seg_props_list = np.load(post_processed_data_folder+"/seg_props_list_"+prop_plane+".npy")

    # else, perform the calculation and save to the post processing folder for this dataset   
    else:
        
        if verbose_flag:
            print("calculating segmented "+prop_plane+" propagation data...")
            
        for CS_ID, CS in enumerate(CS_data):
            
            seg_props = []
              
            for i, seg_phasemap in enumerate(seg_phasemaps_list[CS_ID]):

                surface_pressure = abs(Pf)*np.exp(1j*(seg_phasemap + np.angle(Pf)))
                prop_vec = GF_prop(surface_pressure.flatten(), H_list[i], "forward")
                prop_mat = prop_vec.reshape(output_shape)
                seg_props.append(np.flipud(prop_mat))
                
            seg_props_list.append(seg_props)

        if data_save_flag:
            np.save(post_processed_data_folder+"/seg_props_list_"+prop_plane+".npy", seg_props_list)
            
        if verbose_flag:
            print("...done")
            
        return seg_props_list
    
    
def segmented_qualities_list_builder(CS_data, input_propagations, seg_propagations_list, post_processed_data_folder,
                                threshold, data_save_flag, override_flag, verbose_flag):

    """
    
    segmented qualities
    
    args:
        CS_data: list of clustering structures of sizes={1, m*n, 1}.
        input_propagations:
        seg_propagations_list:
        post_processed_data_folder: folder to save the list of segmented phasemaps.
        data_save_flag: if yes, the segmented phasemaps is saved in the above folder.
        override_flag: if yes, ignore data already present in post_processed_data_folder and recalculate.
        verbose_flag: if yes, the function will print each iteration as it is calculated

        
    returns:
        seg_phasemaps_list: returns a list of lists
    
    """
    
    def quality(ideal_props, cmpsn_props, threshold):
    
        """

        x

        args:

        returns:

        """

        from functions import prop_thresholder, ssim_metric

        quality_list = []     

        for i in range(len(ideal_props)): 

            abs_ideal_prop = abs(ideal_props[i])/np.amax(abs(ideal_props[i]))
            thresholded_abs_ideal_prop =  prop_thresholder(abs_ideal_prop, threshold)

            abs_cmpsn_prop = abs(cmpsn_props[i])/np.amax(abs(ideal_props[i]))
            thresholded_abs_cmpsn_prop = prop_thresholder(abs_cmpsn_prop, threshold)

            quality_list.append(ssim_metric(thresholded_abs_ideal_prop, thresholded_abs_cmpsn_prop, True))

        return quality_list
    
    seg_qualities_list = []

    # If seg_qualities_list.npy already exists, then load it instead of recalculating
    if os.path.exists(post_processed_data_folder+"/seg_qualities_list.npy") and override_flag == False:
        
        if verbose_flag:
            print("loading segmented qualities data from... "+post_processed_data_folder+"/seg_qualities_list.npy")
            
        seg_qualities_list = np.load(post_processed_data_folder+"/seg_qualities_list.npy")

    # else, perform the calculation and save to the post processing folder for this dataset
    else:
        
        if verbose_flag:
            print("calculating segmented qualities data...")
            
        for CS_ID, CS in enumerate(CS_data):
            seg_qualities_list.append(quality(input_propagations, seg_propagations_list[CS_ID], threshold))
        
        if data_save_flag:
            np.save(post_processed_data_folder+"/seg_qualities_list.npy", seg_qualities_list)  

        if verbose_flag:
            print("...done.")
        
    return seg_qualities_list