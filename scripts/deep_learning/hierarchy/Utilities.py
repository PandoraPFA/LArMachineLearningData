import math
import numpy as np
import matplotlib.pyplot as plt

# Primary network variable limits
primaryNSpacepoints_min = 0.0
primaryNSpacepoints_max = 2000.0
primaryNuVertexSeparation_min = -50.0
primaryNuVertexSeparation_max = 500.0
primaryStartRegionNHits_min = -10.0
primaryStartRegionNHits_max = 100.0
primaryStartRegionNParticles_min = -1.0
primaryStartRegionNParticles_max = 8.0
primaryDCA_min = -60.0
primaryDCA_max = 600.0
primaryConnectionExtrapDistance_min = -700.0
primaryConnectionExtrapDistance_max = 500.0
primaryClosestParentL_min = -150.0
primaryClosestParentL_max = 150.0
primaryClosestParentT_min = -30.0
primaryClosestParentT_max = 300.0
primaryOpeningAngle_min = -0.5
primaryOpeningAngle_max = 3.14
# Later tier network variable limits
parentTrackScore_min = -1.0
parentTrackScore_max = 1.0
parentNSpacepoints_min = 0.0
parentNSpacepoints_max = 2000.0
parentTrackScore_min = -1.0
parentTrackScore_max = 1.0
parentNuVertexSeparation_min = -100
parentNuVertexSeparation_max = 750
childNuVertexSeparation_min = -100
childNuVertexSeparation_max = 750
separation3D_min = -50
separation3D_max = 700
parentEndRegionNHits_min = -10
parentEndRegionNHits_max = 80
parentEndRegionNParticles_min = -1
parentEndRegionNParticles_max = 5
parentEndRegionRToWall_min = -10
parentEndRegionRToWall_max = 400
vertexSeparation_min = -50
vertexSeparation_max = 700
doesChildConnect_min = -1
doesChildConnect_max = 1
overshootDCA_min = -700
overshootDCA_max = 700
overshootL_min = -100
overshootL_max = 700
childConnectionDCA_min = -5
childConnectionDCA_max = 50
childConnectionExtrapDistance_min = -500
childConnectionExtrapDistance_max = 500
childConnectionLRatio_min = -1
childConnectionLRatio_max = 1
parentConnectionPointNUpstreamHits_min = -10 
parentConnectionPointNUpstreamHits_max = 100
parentConnectionPointNDownstreamHits_min = -10
parentConnectionPointNDownstreamHits_max = 100
parentConnectionPointNHitRatio_min = -5
parentConnectionPointNHitRatio_max = 30
parentConnectionPointEigenValueRatio_min = -5
parentConnectionPointEigenValueRatio_max = 50 
parentConnectionPointOpeningAngle_min = -10
parentConnectionPointOpeningAngle_max = 180

############################################################################################################################################
############################################################################################################################################   

def normaliseXAxis(variable, minLimit, maxLimit) :
    interval = math.fabs(minLimit) + math.fabs(maxLimit)
    variable[variable < minLimit] = minLimit
    variable[variable > maxLimit] = maxLimit
    variable /= interval
    
############################################################################################################################################
############################################################################################################################################   

def get_max_length(input_array) :
    lengths = [len(entry) for entry in input_array]
    lengths = np.array(lengths)
    
    return np.max(lengths)

############################################################################################################################################
############################################################################################################################################   

def create_mask(input_array, max_len):
    file_mask = [True] * len(input_array)
    to_fill = [False] * (max_len - len(file_mask))
    file_mask = file_mask + to_fill
    
    return file_mask

############################################################################################################################################
############################################################################################################################################   

def process_array(input_array, target_length) :
    input_array = [pad_array(entry, target_length) for entry in input_array]
    input_array = np.array(input_array)
    
    return input_array

############################################################################################################################################
############################################################################################################################################   

def pad_array(input_array, max_len):
    pad = [0] * (max_len - len(input_array))
    input_array = list(input_array) + pad
    
    return input_array

############################################################################################################################################
############################################################################################################################################      

def drawSignalBackground(variable, truth_labels, graph_label) :
    
    signal_mask_vis = (truth_labels == 1).reshape(-1)
    background_mask_vis = (truth_labels == 0).reshape(-1)

    variable_signal = variable[signal_mask_vis].reshape(-1)
    variable_background = variable[background_mask_vis].reshape(-1)

    signal_weights = 1.0 / float(variable_signal.shape[0])
    signal_weights = np.ones(variable_signal.shape[0]) * signal_weights
    
    background_weights = 1.0 / float(variable_background.shape[0])
    background_weights = np.ones(variable_background.shape[0]) * background_weights   
        
    plt.hist(variable_signal, bins=50, color='blue', weights=signal_weights, label='signal', fill=False, histtype='step')
    plt.hist(variable_background, bins=50, color='red', weights=background_weights, label='background', fill=False, histtype='step')
    plt.title(graph_label)
    plt.xlabel(graph_label)
    plt.legend()
    plt.grid(True)
    plt.show()   
    
############################################################################################################################################
############################################################################################################################################      
    
def drawSignalBackgroundGroup(variables, y, graph_label) :

    signal_mask = (y == 1)
    wo_mask = (y == 2)
    background_mask = (y == 0)
    
    variable_signal = variables[signal_mask].reshape(-1)
    variable_wo = variables[wo_mask].reshape(-1)
    variable_background = variables[background_mask].reshape(-1)
    
    signal_weights = 1.0 / float(variable_signal.shape[0])
    signal_weights = np.ones(variable_signal.shape[0]) * signal_weights
    
    wo_weights = 1.0 / float(variable_wo.shape[0])
    wo_weights = np.ones(variable_wo.shape[0]) * wo_weights
    
    background_weights = 1.0 / float(variable_background.shape[0])
    background_weights = np.ones(variable_background.shape[0]) * background_weights  
        
    plt.hist(variable_signal, bins=50, color='blue', weights=signal_weights, label='signal', fill=False, histtype='step')
    plt.hist(variable_wo, bins=50, color='orange', weights=wo_weights, label='wrong orientation', fill=False, histtype='step')
    plt.hist(variable_background, bins=50, color='red', weights=background_weights, label='background', fill=False, histtype='step')
    plt.title(graph_label)
    plt.xlabel(graph_label)
    plt.legend()
    plt.grid(True)
    plt.show()   