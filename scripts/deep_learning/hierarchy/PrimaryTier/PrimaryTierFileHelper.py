import numpy as np
import uproot
import math
from sklearn.utils import shuffle
import Normalise 
                    
############################################################################################################################################                      

def readTreeGroupLinks_track(fileName, normalise) :
        
    ###################################
    # To pull out of tree
    ###################################   
    # Link variables
    primaryTrackScore = []
    primaryNSpacepoints = []
    primaryNuVertexSeparation = [[], []]
    primaryStartRegionNHits = [[], []]
    primaryStartRegionNParticles = [[], []]
    primaryDCA = [[], []]
    primaryConnectionExtrapDistance = [[], []]
    primaryIsPOIClosestToNu = [[], []]
    primaryClosestParentL = [[], []]
    primaryClosestParentT = [[], []]    
    # Training cuts 
    trainingCutDCA = [] # only fill for correct orientation    
    # Truth
    isTruePrimaryLink = []
    isLinkOrientationCorrect = []
    y = []

    print('Reading tree: ', str(fileName),', This may take a while...')

    ####################################
    # Set tree
    ####################################    
    treeFile = uproot.open(fileName)
    tree = treeFile['PrimaryTrackTree_TRAIN']
    branches = tree.arrays()

    ####################################
    # Set tree branches
    ####################################
    primaryNSpacepoints_file = np.array(branches['NSpacepoints'])
    primaryNuVertexSeparation_file = np.array(branches['NuSeparation'])
    primaryStartRegionNHits_file = np.array(branches['VertexRegionNHits'])            
    primaryStartRegionNParticles_file = np.array(branches['VertexRegionNParticles'])            
    primaryDCA_file = np.array(branches['DCA'])            
    primaryConnectionExtrapDistance_file = np.array(branches['ConnectionExtrapDistance'])
    primaryIsPOIClosestToNu_file = np.array(branches['IsPOIClosestToNu'])
    primaryParentConnectionDistance_file = np.array(branches['ParentConnectionDistance'])
    primaryChildConnectionDistance_file = np.array(branches['ChildConnectionDistance'])
    # True
    isTruePrimaryLink_file = np.array(branches['IsTrueLink'])
    isLinkOrientationCorrect_file = np.array(branches['IsOrientationCorrect'])    
    # nLinks
    nLinks_file = isTruePrimaryLink_file.shape[0]

    ####################################
    # Now loop over loops to group them.
    ####################################
    linksMadeCounter = 0
    this_y = [0, 0]
    this_isLinkOrientationCorrect = [0, 0]
    order = [0, 1]        

    for iLink in range(nLinks_file) :

        if ((iLink % 1000) == 0) :
            print('iLink:', str(iLink) + '/' + str(nLinks_file))            

        # Set truth            
        isTruePrimaryLink_bool = math.isclose(isTruePrimaryLink_file[iLink], 1.0, rel_tol=0.001)
        isLinkOrientationCorrect_bool = math.isclose(isLinkOrientationCorrect_file[iLink], 1.0, rel_tol=0.001)

        if (isTruePrimaryLink_bool and isLinkOrientationCorrect_bool) :
            this_y[order[linksMadeCounter]] = 1 
        elif (isTruePrimaryLink_bool and (not isLinkOrientationCorrect_bool)) :
            this_y[order[linksMadeCounter]] = 2

        this_isLinkOrientationCorrect[order[linksMadeCounter]] = isLinkOrientationCorrect_file[iLink]

        # set the link information
        primaryNuVertexSeparation[order[linksMadeCounter]].append(primaryNuVertexSeparation_file[iLink])
        primaryStartRegionNHits[order[linksMadeCounter]].append(primaryStartRegionNHits_file[iLink])
        primaryStartRegionNParticles[order[linksMadeCounter]].append(primaryStartRegionNParticles_file[iLink])
        primaryDCA[order[linksMadeCounter]].append(primaryDCA_file[iLink])
        primaryConnectionExtrapDistance[order[linksMadeCounter]].append(primaryConnectionExtrapDistance_file[iLink])
        primaryIsPOIClosestToNu[order[linksMadeCounter]].append(primaryIsPOIClosestToNu_file[iLink])
        primaryClosestParentL[order[linksMadeCounter]].append(primaryParentConnectionDistance_file[iLink])
        primaryClosestParentT[order[linksMadeCounter]].append(primaryChildConnectionDistance_file[iLink])

        # Add in training cuts 
        if (isLinkOrientationCorrect_bool) :
            trainingCutDCA.append(primaryDCA_file[iLink])

        linksMadeCounter = linksMadeCounter + 1

        if (linksMadeCounter == 2) :
            # set the common vars
            primaryNSpacepoints.append(primaryNSpacepoints_file[iLink])   
            # set truth
            y.append(this_y)
            isTruePrimaryLink.append(isTruePrimaryLink_file[iLink])
            isLinkOrientationCorrect.append(this_isLinkOrientationCorrect)
            # reset                        
            linksMadeCounter = 0
            this_y = [0, 0]
            this_isLinkOrientationCorrect = [0, 0]
            order = shuffle(order)

    ###################################
    # Now turn things into numpy arrays
    ###################################
    primaryTrackScore = np.array(primaryTrackScore, dtype='float64')
    primaryNSpacepoints = np.array(primaryNSpacepoints, dtype='float64')
    primaryNuVertexSeparation = np.array(primaryNuVertexSeparation, dtype='float64')
    primaryStartRegionNHits = np.array(primaryStartRegionNHits, dtype='float64')
    primaryStartRegionNParticles = np.array(primaryStartRegionNParticles, dtype='float64')
    primaryDCA = np.array(primaryDCA, dtype='float64')
    primaryConnectionExtrapDistance = np.array(primaryConnectionExtrapDistance, dtype='float64')
    primaryIsPOIClosestToNu = np.array(primaryIsPOIClosestToNu, dtype='int')
    primaryClosestParentL = np.array(primaryClosestParentL, dtype='float64')
    primaryClosestParentT = np.array(primaryClosestParentT, dtype='float64')

    # Truth
    isTruePrimaryLink = np.array(isTruePrimaryLink)
    isLinkOrientationCorrect = np.array(isLinkOrientationCorrect)
    y = np.array(y)
        
    ###################################
    # How many links do we have?
    ###################################        
    nLinks = isTruePrimaryLink.shape[0]   
    print('We have ', str(nLinks), ' to train on!')        
          
    ###################################
    # Normalise variables
    ###################################
    if (normalise) :
        Normalise.normaliseXAxis(primaryNSpacepoints, Normalise.primaryNSpacepoints_min, Normalise.primaryNSpacepoints_max)    
        Normalise.normaliseXAxis(primaryNuVertexSeparation, Normalise.primaryNuVertexSeparation_min, Normalise.primaryNuVertexSeparation_max)    
        Normalise.normaliseXAxis(primaryStartRegionNHits, Normalise.primaryStartRegionNHits_min, Normalise.primaryStartRegionNHits_max)    
        Normalise.normaliseXAxis(primaryStartRegionNParticles, Normalise.primaryStartRegionNParticles_min, Normalise.primaryStartRegionNParticles_max)    
        Normalise.normaliseXAxis(primaryDCA, Normalise.primaryDCA_min, Normalise.primaryDCA_max)
        Normalise.normaliseXAxis(primaryConnectionExtrapDistance, Normalise.primaryConnectionExtrapDistance_min, Normalise.primaryConnectionExtrapDistance_max)
        Normalise.normaliseXAxis(primaryClosestParentL, Normalise.primaryClosestParentL_min, Normalise.primaryClosestParentL_max) 
        Normalise.normaliseXAxis(primaryClosestParentT, Normalise.primaryClosestParentT_min, Normalise.primaryClosestParentT_max) 
        
    ###################################
    # Concatenate
    ###################################
    coc0 = np.concatenate((np.concatenate((primaryNuVertexSeparation[0, :].reshape(nLinks, 1), \
                                           primaryStartRegionNHits[0, :].reshape(nLinks, 1), \
                                           primaryStartRegionNParticles[0, :].reshape(nLinks, 1), \
                                           primaryDCA[0, :].reshape(nLinks, 1), \
                                           primaryConnectionExtrapDistance[0, :].reshape(nLinks, 1), \
                                           primaryIsPOIClosestToNu[0, :].reshape(nLinks, 1), \
                                           primaryClosestParentL[0, :].reshape(nLinks, 1), \
                                           primaryClosestParentT[0, :].reshape(nLinks, 1)), axis=1), \
                           np.concatenate((primaryNuVertexSeparation[1, :].reshape(nLinks, 1), \
                                           primaryStartRegionNHits[1, :].reshape(nLinks, 1), \
                                           primaryStartRegionNParticles[1, :].reshape(nLinks, 1), \
                                           primaryDCA[1, :].reshape(nLinks, 1), \
                                           primaryConnectionExtrapDistance[1, :].reshape(nLinks, 1), \
                                           primaryIsPOIClosestToNu[1, :].reshape(nLinks, 1), \
                                           primaryClosestParentL[1, :].reshape(nLinks, 1), \
                                           primaryClosestParentT[1, :].reshape(nLinks, 1)), axis=1)), axis=1)      
    
    # concatenate variable_single and orientations
    variables = np.concatenate((primaryNSpacepoints.reshape(nLinks, 1), \
                                coc0), axis=1)

    
    return nLinks, variables, y, isTruePrimaryLink, isLinkOrientationCorrect, trainingCutDCA
    
############################################################################################################################################
############################################################################################################################################        

def readTreeGroupLinks_shower(fileName, normalise) :
        
    ###################################
    # To pull out of tree
    ###################################
    # Link variables
    primaryNSpacepoints = []
    primaryNuVertexSeparation = []
    primaryStartRegionNHits = []
    primaryStartRegionNParticles = []
    primaryDCA = []
    primaryConnectionExtrapDistance = []
    primaryIsPOIClosestToNu = []
    primaryClosestParentL = []
    primaryClosestParentT = []
    # Training cut 
    trainingCutDCA = []
    # Truth
    isTruePrimaryLink = []
    isLinkOrientationCorrect = []
    y = []
        
    print('Reading tree: ', str(fileName),', This may take a while...')

    ####################################
    # Set tree
    ####################################    
    treeFile = uproot.open(fileName)
    tree = treeFile['PrimaryShowerTree_TRAIN']
    branches = tree.arrays()

    ####################################
    # Set tree branches
    ####################################
    # Tree branches
    primaryNSpacepoints.extend(np.array(branches['NSpacepoints']))
    primaryNuVertexSeparation.extend(np.array(branches['NuSeparation']))
    primaryStartRegionNHits.extend(np.array(branches['VertexRegionNHits']))
    primaryStartRegionNParticles.extend(np.array(branches['VertexRegionNParticles']))
    primaryDCA.extend(np.array(branches['DCA']))
    primaryConnectionExtrapDistance.extend(np.array(branches['ConnectionExtrapDistance']))
    primaryIsPOIClosestToNu.extend(np.array(branches['IsPOIClosestToNu']))        
    primaryClosestParentL.extend(np.array(branches['ParentConnectionDistance']))
    primaryClosestParentT.extend(np.array(branches['ChildConnectionDistance']))
    # Training vars
    trainingCutDCA.extend(np.array(branches['DCA']))
    # True
    isTruePrimaryLink_file = np.array(branches['IsTrueLink'])
    isTruePrimaryLink.extend(isTruePrimaryLink_file)
    isLinkOrientationCorrect_file = np.array(branches['IsOrientationCorrect'])
    isLinkOrientationCorrect.extend(isLinkOrientationCorrect_file)
    # Sort out edge truth
    this_y = np.zeros(np.array(isTruePrimaryLink_file).shape)
    isTruePrimaryLink_bool = np.isclose(isTruePrimaryLink_file, np.ones(isTruePrimaryLink_file.shape), rtol=0.001)
    isLinkOrientationCorrect_bool = np.isclose(isLinkOrientationCorrect_file, np.ones(isLinkOrientationCorrect_file.shape), rtol=0.001)
    this_y[np.logical_and(isTruePrimaryLink_bool, isLinkOrientationCorrect_bool)] = 1
    this_y[np.logical_and(isTruePrimaryLink_bool, np.logical_not(isLinkOrientationCorrect_bool))] = 2
    y.extend(this_y)
            
    ###################################
    # Now turn things into numpy arrays
    ###################################
    primaryNSpacepoints = np.array(primaryNSpacepoints, dtype='float64')
    primaryNuVertexSeparation = np.array(primaryNuVertexSeparation, dtype='float64')
    primaryStartRegionNHits = np.array(primaryStartRegionNHits, dtype='float64')
    primaryStartRegionNParticles = np.array(primaryStartRegionNParticles, dtype='float64')
    primaryDCA = np.array(primaryDCA, dtype='float64')
    primaryConnectionExtrapDistance = np.array(primaryConnectionExtrapDistance, dtype='float64')
    primaryIsPOIClosestToNu = np.array(primaryIsPOIClosestToNu, dtype='int')
    primaryClosestParentL = np.array(primaryClosestParentL, dtype='float64')
    primaryClosestParentT = np.array(primaryClosestParentT, dtype='float64')
    # Truth
    isTruePrimaryLink = np.array(isTruePrimaryLink)
    isLinkOrientationCorrect = np.array(isLinkOrientationCorrect)
    y = np.array(y)
        
    ###################################
    # How many links do we have?
    ###################################        
    nLinks = isTruePrimaryLink.shape[0]   
    print('We have ', str(nLinks), ' to train on!')        
          
    ###################################
    # Normalise variables
    ###################################
    if (normalise) :
        Normalise.normaliseXAxis(primaryNSpacepoints, Normalise.primaryNSpacepoints_min, Normalise.primaryNSpacepoints_max)    
        Normalise.normaliseXAxis(primaryNuVertexSeparation, Normalise.primaryNuVertexSeparation_min, Normalise.primaryNuVertexSeparation_max)    
        Normalise.normaliseXAxis(primaryStartRegionNHits, Normalise.primaryStartRegionNHits_min, Normalise.primaryStartRegionNHits_max)    
        Normalise.normaliseXAxis(primaryStartRegionNParticles, Normalise.primaryStartRegionNParticles_min, Normalise.primaryStartRegionNParticles_max)    
        Normalise.normaliseXAxis(primaryDCA, Normalise.primaryDCA_min, Normalise.primaryDCA_max)
        Normalise.normaliseXAxis(primaryConnectionExtrapDistance, Normalise.primaryConnectionExtrapDistance_min, Normalise.primaryConnectionExtrapDistance_max) 
        Normalise.normaliseXAxis(primaryClosestParentL, Normalise.primaryClosestParentL_min, Normalise.primaryClosestParentL_max) 
        Normalise.normaliseXAxis(primaryClosestParentT, Normalise.primaryClosestParentT_min, Normalise.primaryClosestParentT_max) 
        
    ###################################
    # Concatenate
    ###################################
    variables = np.concatenate((primaryNSpacepoints.reshape(nLinks, 1), \
                           primaryNuVertexSeparation.reshape(nLinks, 1), \
                           primaryStartRegionNHits.reshape(nLinks, 1), \
                           primaryStartRegionNParticles.reshape(nLinks, 1), \
                           primaryDCA.reshape(nLinks, 1), \
                           primaryConnectionExtrapDistance.reshape(nLinks, 1), \
                           primaryIsPOIClosestToNu.reshape(nLinks, 1), \
                           primaryClosestParentL.reshape(nLinks, 1), \
                           primaryClosestParentT.reshape(nLinks, 1)), axis=1)    

    return nLinks, variables, y, isTruePrimaryLink, isLinkOrientationCorrect, trainingCutDCA

############################################################################################################################################
############################################################################################################################################        
   

    