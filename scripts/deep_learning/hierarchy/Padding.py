########################################################
# Imports
########################################################

import numpy as np

########################################################
# Functions to pad our vectors
########################################################

def get_max_length(input_array) :
    
    lengths = [len(entry) for entry in input_array]
    lengths = np.array(lengths)
    
    return np.max(lengths)

def create_mask(input_array, max_len):
    
    file_mask = [True] * len(input_array)
    to_fill = [False] * (max_len - len(file_mask))
    file_mask = file_mask + to_fill
    
    return file_mask

def pad_array(input_array, max_len):
    
    pad = [0] * (max_len - len(input_array))
    input_array = list(input_array) + pad
    
    return input_array

def process_array(input_array, target_length) :
    input_array = [pad_array(entry, target_length) for entry in input_array]
    input_array = np.array(input_array)
    
    return input_array