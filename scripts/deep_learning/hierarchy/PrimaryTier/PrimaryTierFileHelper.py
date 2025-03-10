import numpy as np
import uproot
import math
import copy
from sklearn.utils import shuffle
import Utilities 
                    
############################################################################################################################################       
    
def ReadTreeForTraining(isTrackMode, fileName, normalise) :
    
    if (isTrackMode) :
        nLinks = 2
        treeName = "PrimaryTrackTree_TRAIN"
    else :
        nLinks = 1
        treeName = "PrimaryShowerTree_TRAIN"
    
    with uproot.open(f"{fileName}:{treeName}") as tree:    

        branches = tree.arrays()

        # Vars (put astype as they are in tree)
        primaryNSpacepoints = np.array(branches['NSpacepoints']).astype('float64').reshape(-1, nLinks)
        primaryNuVertexSeparation = np.array(branches['NuSeparation']).astype('float64').reshape(-1, nLinks)
        primaryStartRegionNHits = np.array(branches['VertexRegionNHits']).astype('float64').reshape(-1, nLinks)            
        primaryStartRegionNParticles = np.array(branches['VertexRegionNParticles']).astype('float64').reshape(-1, nLinks)            
        primaryDCA = np.array(branches['DCA']).astype('float64').reshape(-1, nLinks)            
        primaryConnectionExtrapDistance = np.array(branches['ConnectionExtrapDistance']).astype('float64').reshape(-1, nLinks)
        primaryIsPOIClosestToNu = np.array(branches['IsPOIClosestToNu']).astype('float64').reshape(-1, nLinks)
        primaryClosestParentL = np.array(branches['ParentConnectionDistance']).astype('float64').reshape(-1, nLinks)
        primaryClosestParentT = np.array(branches['ChildConnectionDistance']).astype('float64').reshape(-1, nLinks)
        # True
        isTruePrimaryLink = np.array(branches['IsTrueLink']).astype('int').reshape(-1, nLinks)
        isLinkOrientationCorrect = np.array(branches['IsOrientationCorrect']).astype('int').reshape(-1, nLinks)    
        
        # Form link truth (this will not be used in isTrackMode == False)
        y = np.zeros(isTruePrimaryLink.shape).astype('int')
        y[np.logical_and(isTruePrimaryLink,  isLinkOrientationCorrect)] = 1
        y[np.logical_and(isTruePrimaryLink,  np.logical_not(isLinkOrientationCorrect))] = 2

        # Training cuts!
        trainingCutDCA = copy.deepcopy(primaryDCA)

        # Reduce parent-child relationship vars to 1D arrays
        primaryNSpacepoints = primaryNSpacepoints[:,0]
        isTruePrimaryLink = isTruePrimaryLink[:,0]

        if (isTrackMode) :
            trainingCutDCA = trainingCutDCA[isLinkOrientationCorrect == 1]
        else :
            trainingCutDCA = trainingCutDCA[:,0]

        # How many entries do we have?      
        nEntries = isTruePrimaryLink.shape[0]   
        print('We have ', str(nEntries), ' entries to train on!')
        
        # Shuffle links for each entry
        randomIndices = np.random.rand(*(nEntries, nLinks)).argsort(axis=1)
        primaryNuVertexSeparation = np.take_along_axis(primaryNuVertexSeparation, randomIndices, axis=1) 
        primaryStartRegionNHits = np.take_along_axis(primaryStartRegionNHits, randomIndices, axis=1) 
        primaryStartRegionNParticles = np.take_along_axis(primaryStartRegionNParticles, randomIndices, axis=1) 
        primaryDCA = np.take_along_axis(primaryDCA, randomIndices, axis=1) 
        primaryConnectionExtrapDistance = np.take_along_axis(primaryConnectionExtrapDistance, randomIndices, axis=1) 
        primaryIsPOIClosestToNu = np.take_along_axis(primaryIsPOIClosestToNu, randomIndices, axis=1) 
        primaryClosestParentL = np.take_along_axis(primaryClosestParentL, randomIndices, axis=1) 
        primaryClosestParentT = np.take_along_axis(primaryClosestParentT, randomIndices, axis=1) 
        y = np.take_along_axis(y, randomIndices, axis=1)

        if (normalise) :
            Utilities.normaliseXAxis(primaryNSpacepoints, Utilities.primaryNSpacepoints_min, Utilities.primaryNSpacepoints_max)    
            Utilities.normaliseXAxis(primaryNuVertexSeparation, Utilities.primaryNuVertexSeparation_min, Utilities.primaryNuVertexSeparation_max)    
            Utilities.normaliseXAxis(primaryStartRegionNHits, Utilities.primaryStartRegionNHits_min, Utilities.primaryStartRegionNHits_max)    
            Utilities.normaliseXAxis(primaryStartRegionNParticles, Utilities.primaryStartRegionNParticles_min, Utilities.primaryStartRegionNParticles_max)    
            Utilities.normaliseXAxis(primaryDCA, Utilities.primaryDCA_min, Utilities.primaryDCA_max)
            Utilities.normaliseXAxis(primaryConnectionExtrapDistance, Utilities.primaryConnectionExtrapDistance_min, Utilities.primaryConnectionExtrapDistance_max)
            Utilities.normaliseXAxis(primaryClosestParentL, Utilities.primaryClosestParentL_min, Utilities.primaryClosestParentL_max) 
            Utilities.normaliseXAxis(primaryClosestParentT, Utilities.primaryClosestParentT_min, Utilities.primaryClosestParentT_max) 

        # Prepare output
        variables = primaryNSpacepoints.reshape(nEntries, 1)
        
        for i in range(nLinks) :
            edge_vars = np.concatenate((primaryNuVertexSeparation[:, i].reshape(nEntries, 1), \
                                        primaryStartRegionNHits[:, i].reshape(nEntries, 1), \
                                        primaryStartRegionNParticles[:, i].reshape(nEntries, 1), \
                                        primaryDCA[:, i].reshape(nEntries, 1), \
                                        primaryConnectionExtrapDistance[:, i].reshape(nEntries, 1), \
                                        primaryIsPOIClosestToNu[:, i].reshape(nEntries, 1), \
                                        primaryClosestParentL[:, i].reshape(nEntries, 1), \
                                        primaryClosestParentT[:, i].reshape(nEntries, 1)), axis=1)
            
            variables = np.concatenate((variables, edge_vars), axis=1)
            
    return nEntries, variables, y, isTruePrimaryLink, trainingCutDCA    

############################################################################################################################################ 

def ReadTreeForValidation(isTrackMode, fileName, normalise) :
    
    if (isTrackMode) :
        nLinks = 2
        treeName = "PrimaryTrackTree"
    else :
        nLinks = 1
        treeName = "PrimaryShowerTree"
    
    with uproot.open(f"{fileName}:{treeName}") as tree:    

        branches = tree.arrays()

        # ID
        particleID = np.array(branches["ParticleID"]).reshape(-1, nLinks)
        # Vars (put astype as they are in tree)
        primaryNSpacepoints = np.array(branches['NSpacepoints']).astype('float64').reshape(-1, nLinks)
        primaryNuVertexSeparation = np.array(branches['NuSeparation']).astype('float64').reshape(-1, nLinks)
        primaryStartRegionNHits = np.array(branches['VertexRegionNHits']).astype('float64').reshape(-1, nLinks)            
        primaryStartRegionNParticles = np.array(branches['VertexRegionNParticles']).astype('float64').reshape(-1, nLinks)            
        primaryDCA = np.array(branches['DCA']).astype('float64').reshape(-1, nLinks)            
        primaryConnectionExtrapDistance = np.array(branches['ConnectionExtrapDistance']).astype('float64').reshape(-1, nLinks)
        primaryIsPOIClosestToNu = np.array(branches['IsPOIClosestToNu']).astype('float64').reshape(-1, nLinks)
        primaryClosestParentL = np.array(branches['ParentConnectionDistance']).astype('float64').reshape(-1, nLinks)
        primaryClosestParentT = np.array(branches['ChildConnectionDistance']).astype('float64').reshape(-1, nLinks)
        # True
        trueVisibleGeneration = np.array(branches["TrueVisibleGeneration"]).reshape(-1, nLinks)
        trueVisibleParentID = np.array(branches['TrueVisibleParentID']).reshape(-1, nLinks)
        truePDG = np.array(branches['TruePDG']).reshape(-1, nLinks)

        # Reduce parent-child relationship vars to 1D arrays
        particleID = particleID[:,0]
        primaryNSpacepoints = primaryNSpacepoints[:,0]
        trueVisibleGeneration = trueVisibleGeneration[:,0]
        trueVisibleParentID = trueVisibleParentID[:,0]
        truePDG = truePDG[:,0]

        # How many entries do we have?      
        nEntries = particleID.shape[0]   
        print('We have ', str(nEntries), ' entries to train on!')

        if (normalise) :
            Utilities.normaliseXAxis(primaryNSpacepoints, Utilities.primaryNSpacepoints_min, Utilities.primaryNSpacepoints_max)    
            Utilities.normaliseXAxis(primaryNuVertexSeparation, Utilities.primaryNuVertexSeparation_min, Utilities.primaryNuVertexSeparation_max)    
            Utilities.normaliseXAxis(primaryStartRegionNHits, Utilities.primaryStartRegionNHits_min, Utilities.primaryStartRegionNHits_max)    
            Utilities.normaliseXAxis(primaryStartRegionNParticles, Utilities.primaryStartRegionNParticles_min, Utilities.primaryStartRegionNParticles_max)    
            Utilities.normaliseXAxis(primaryDCA, Utilities.primaryDCA_min, Utilities.primaryDCA_max)
            Utilities.normaliseXAxis(primaryConnectionExtrapDistance, Utilities.primaryConnectionExtrapDistance_min, Utilities.primaryConnectionExtrapDistance_max)
            Utilities.normaliseXAxis(primaryClosestParentL, Utilities.primaryClosestParentL_min, Utilities.primaryClosestParentL_max) 
            Utilities.normaliseXAxis(primaryClosestParentT, Utilities.primaryClosestParentT_min, Utilities.primaryClosestParentT_max) 

        # Prepare output
        variables = primaryNSpacepoints.reshape(nEntries, 1)
        
        for i in range(nLinks) :
            edge_vars = np.concatenate((primaryNuVertexSeparation[:, i].reshape(nEntries, 1), \
                                        primaryStartRegionNHits[:, i].reshape(nEntries, 1), \
                                        primaryStartRegionNParticles[:, i].reshape(nEntries, 1), \
                                        primaryDCA[:, i].reshape(nEntries, 1), \
                                        primaryConnectionExtrapDistance[:, i].reshape(nEntries, 1), \
                                        primaryIsPOIClosestToNu[:, i].reshape(nEntries, 1), \
                                        primaryClosestParentL[:, i].reshape(nEntries, 1), \
                                        primaryClosestParentT[:, i].reshape(nEntries, 1)), axis=1)
            
            variables = np.concatenate((variables, edge_vars), axis=1)
            
    return nEntries, variables, particleID, trueVisibleGeneration, trueVisibleParentID, truePDG