import numpy as np
import math

############################################################################################################################################
############################################################################################################################################ 

def CreateWindows(element, window_size, no_overlap) :
    
    windows = [element]
        
    if len(element) > window_size :
        n_windows = math.floor(float(len(element)) / window_size)
        windows = [element[window_size * i:window_size * (i + 1)] for i in range(n_windows)]
        
        if (len(element) % window_size != 0) :
            windows.append(element[len(element) - window_size:])
            
            if no_overlap:
                remove_end = (np.random.rand() > 0.5)
                
                if (remove_end):
                    del windows[n_windows] 
                else:
                    del windows[n_windows - 1] 

    return windows 

############################################################################################################################################
############################################################################################################################################

def SplitIntoWindows(input_array, windows_indices):    
    return [
        entry[index[0]:index[-1] + 1]
        for entry, index_array in zip(input_array, windows_indices)
        for index in index_array
    ]

############################################################################################################################################
############################################################################################################################################

def GetClusterIndices(windows_indices):
    return [
        i
        for i, index_array in enumerate(windows_indices)
        for _ in index_array
    ]

############################################################################################################################################
############################################################################################################################################

def GetTruthDistanceArray(longitudinal_array, splitting_positions) :    
    truth_dist_array = [GetTruthDistanceElement(entry1, entry2) for entry1, entry2 in zip(longitudinal_array, splitting_positions)]
    return truth_dist_array

############################################################################################################################################
############################################################################################################################################

def GetTruthDistanceElement(longitudinal_element, splitting_positions) :
    
    longitudinal_element = np.asarray(longitudinal_element)
    splitting_positions = np.asarray(splitting_positions)

    # If no split positions
    if (splitting_positions.shape[0] == 0) :
        return np.full(np.asarray(longitudinal_element).shape, int(0))
    
    # work out distance from each split
    distance_from_splits = np.abs(longitudinal_element.reshape(-1, 1) - splitting_positions)

    # Pick out the separation to closest split at each position
    min_index = distance_from_splits.argmin(axis=1)
    truth_dist_element = distance_from_splits[np.arange(distance_from_splits.shape[0]), min_index]
    
    # Tokenise
    signal_mask = truth_dist_element < 0.5
    truth_dist_element[signal_mask] = int(1)
    truth_dist_element[np.logical_not(signal_mask)] = int(0)
    truth_dist_element = truth_dist_element.astype(int)
    return truth_dist_element

############################################################################################################################################
############################################################################################################################################

def ProcessTruth(is_shower_array, cluster_indices, longitudinal_array, splitting_positions, windows_indices) :
    split_point_truth = GetTruthDistanceArray(longitudinal_array, splitting_positions)
    split_point_truth = SplitIntoWindows(split_point_truth, windows_indices)
    
    contamination_truth = [
        2 if is_shower_array[cluster_indices[i]] else int(np.max(split_point_truth[i]))
        for i in range(len(split_point_truth))
    ]    
    
    return contamination_truth, split_point_truth

############################################################################################################################################
############################################################################################################################################

def ProcessFeature(feature, windows_indices) :
    # Split into windows
    feature = SplitIntoWindows(feature, windows_indices)
    # turn into numpy
    feature = np.asarray(feature)
    
    return feature

############################################################################################################################################
############################################################################################################################################