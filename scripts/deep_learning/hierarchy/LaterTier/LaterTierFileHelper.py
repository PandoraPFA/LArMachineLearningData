import numpy as np
import uproot
import math
import Normalise

from sklearn.utils import shuffle
        
############################################################################################################################################
############################################################################################################################################        

def readTreeGroupLinks_track(fileName, normalise) :
        
    ###################################
    # To pull out of tree
    ###################################
    # Link variables
    parentTrackScore = []
    childTrackScore = []
    parentNSpacepoints = []
    childNSpacepoints = []
    separation3D = []
    parentNuVertexSep = [[], [], [], []]
    childNuVertexSep = [[], [], [], []]    
    parentEndRegionNHits = [[], [], [], []]
    parentEndRegionNParticles = [[], [], [], []]
    parentEndRegionRToWall = [[], [], [], []]
    vertexSeparation = [[], [], [], []]
    doesChildConnect = [[], [], [], []]
    overshootStartDCA = [[], [], [], []]
    overshootStartL = [[], [], [], []]
    overshootEndDCA = [[], [], [], []]
    overshootEndL = [[], [], [], []]
    childConnectionDCA = [[], [], [], []]
    childConnectionExtrapDistance = [[], [], [], []]
    childConnectionLRatio = [[], [], [], []]
    parentConnectionPointNUpstreamHits = [[], [], [], []]
    parentConnectionPointNDownstreamHits = [[], [], [], []]
    parentConnectionPointNHitRatio = [[], [], [], []]
    parentConnectionPointEigenValueRatio = [[], [], [], []]
    parentConnectionPointOpeningAngle = [[], [], [], []]
    parentIsPOIClosestToNu = [[], [], [], []]
    childIsPOIClosestToNu = [[], [], [], []]
    # Training cut variables
    trainingCutSep = []
    trainingCutL = []
    trainingCutT = []
    trainingCutDoesConnect = []
    # Truth
    trueParentChildLink = []
    isLinkOrientationCorrect = []
    trueChildVisibleGeneration = []
    y = []
    
    print('Reading tree: ', str(fileName),', This may take a while...')

    ####################################
    # Set tree
    ####################################  
    treeFile = uproot.open(fileName)
    tree = treeFile['LaterTierTrackTrackTree_TRAIN']
    branches = tree.arrays()

    ####################################
    # Set tree branches
    ####################################
    # Network vars
    parentTrackScore_file = np.array(branches['ParentTrackScore'])            
    childTrackScore_file = np.array(branches['ChildTrackScore'])
    parentNSpacepoints_file = np.array(branches['ParentNSpacepoints'])
    childNSpacepoints_file = np.array(branches['ChildNSpacepoints'])
    separation3D_file = np.array(branches['Separation3D'])
    parentNuVertexSep_file = np.array(branches['ParentNuVertexSep'])
    childNuVertexSep_file = np.array(branches['ChildNuVertexSep'])                        
    parentEndRegionNHits_file = np.array(branches['ParentEndRegionNHits'])
    parentEndRegionNParticles_file = np.array(branches['ParentEndRegionNParticles'])
    parentEndRegionRToWall_file = np.array(branches['ParentEndRegionRToWall'])
    vertexSeparation_file = np.array(branches['VertexSeparation'])        
    doesChildConnect_file = np.array(branches['DoesChildConnect'])
    overshootStartDCA_file = np.array(branches['OvershootStartDCA'])
    overshootStartL_file = np.array(branches['OvershootStartL'])
    overshootEndDCA_file = np.array(branches['OvershootEndDCA'])
    overshootEndL_file = np.array(branches['OvershootEndL'])
    childConnectionDCA_file = np.array(branches['ChildCPDCA'])
    childConnectionExtrapDistance_file = np.array(branches['ChildCPExtrapDistance'])
    childConnectionLRatio_file = np.array(branches['ChildCPLRatio'])
    parentConnectionPointNUpstreamHits_file = np.array(branches['ParentCPNUpstreamHits'])
    parentConnectionPointNDownstreamHits_file = np.array(branches['ParentCPNDownstreamHits'])
    parentConnectionPointNHitRatio_file = np.array(branches['ParentCPNHitRatio'])
    parentConnectionPointEigenValueRatio_file = np.array(branches['ParentCPEigenvalueRatio'])
    parentConnectionPointOpeningAngle_file = np.array(branches['ParentCPOpeningAngle'])
    isParentPOIClosestToNu_file = np.array(branches['ParentIsPOIClosestToNu'])
    isChildPOIClosestToNu_file = np.array(branches['ChildIsPOIClosestToNu'])
    # Truth
    trueParentChildLink_file = np.array(branches['IsTrueLink'])
    trueChildVisibleGeneration_file = np.array(branches['ChildTrueVisibleGeneration'])
    isLinkOrientationCorrect_file = np.array(branches['IsOrientationCorrect'])
    # Training cuts!
    trainingCutL_file = np.array(branches['TrainingCutL'])
    trainingCutT_file = np.array(branches['TrainingCutT'])

    # nLinks
    nLinks_file = trueParentChildLink_file.shape[0]

    ####################################
    # Now loop over loops to group them.
    ####################################
    linksMadeCounter = 0
    this_y = [0, 0, 0, 0]
    this_isLinkOrientationCorrect = [0, 0, 0, 0]
    order = [0, 1, 2, 3]

    for iLink in range(0, nLinks_file) :

        if ((iLink % 1000) == 0) :
            print('iLink:', str(iLink) + '/' + str(nLinks_file)) 

        # Set truth                                                  
        trueParentChildLink_bool = math.isclose(trueParentChildLink_file[iLink], 1.0, rel_tol=0.001)
        isLinkOrientationCorrect_bool = math.isclose(isLinkOrientationCorrect_file[iLink], 1.0, rel_tol=0.001)

        if (trueParentChildLink_bool and isLinkOrientationCorrect_bool) :
            this_y[order[linksMadeCounter]] = 1 
        elif (trueParentChildLink_bool and (not isLinkOrientationCorrect_bool)) :
            this_y[order[linksMadeCounter]] = 2

        this_isLinkOrientationCorrect[order[linksMadeCounter]] = isLinkOrientationCorrect_bool

        # set the link information
        parentNuVertexSep[order[linksMadeCounter]].append(parentNuVertexSep_file[iLink])
        childNuVertexSep[order[linksMadeCounter]].append(childNuVertexSep_file[iLink])
        parentEndRegionNHits[order[linksMadeCounter]].append(parentEndRegionNHits_file[iLink])
        parentEndRegionNParticles[order[linksMadeCounter]].append(parentEndRegionNParticles_file[iLink])
        parentEndRegionRToWall[order[linksMadeCounter]].append(parentEndRegionRToWall_file[iLink])
        vertexSeparation[order[linksMadeCounter]].append(vertexSeparation_file[iLink])
        doesChildConnect[order[linksMadeCounter]].append(doesChildConnect_file[iLink])
        overshootStartDCA[order[linksMadeCounter]].append(overshootStartDCA_file[iLink])
        overshootStartL[order[linksMadeCounter]].append(overshootStartL_file[iLink])
        overshootEndDCA[order[linksMadeCounter]].append(overshootEndDCA_file[iLink])
        overshootEndL[order[linksMadeCounter]].append(overshootEndL_file[iLink])
        childConnectionDCA[order[linksMadeCounter]].append(childConnectionDCA_file[iLink])
        childConnectionExtrapDistance[order[linksMadeCounter]].append(childConnectionExtrapDistance_file[iLink])
        childConnectionLRatio[order[linksMadeCounter]].append(childConnectionLRatio_file[iLink])
        parentConnectionPointNUpstreamHits[order[linksMadeCounter]].append(parentConnectionPointNUpstreamHits_file[iLink])
        parentConnectionPointNDownstreamHits[order[linksMadeCounter]].append(parentConnectionPointNDownstreamHits_file[iLink])
        parentConnectionPointNHitRatio[order[linksMadeCounter]].append(parentConnectionPointNHitRatio_file[iLink])
        parentConnectionPointEigenValueRatio[order[linksMadeCounter]].append(parentConnectionPointEigenValueRatio_file[iLink])
        parentConnectionPointOpeningAngle[order[linksMadeCounter]].append(parentConnectionPointOpeningAngle_file[iLink])
        isParentPOIClosestToNu_bool = math.isclose(isParentPOIClosestToNu_file[iLink], 1.0, rel_tol=0.001)
        parentIsPOIClosestToNu[order[linksMadeCounter]].append(1 if isParentPOIClosestToNu_bool else 0)
        isChildPOIClosestToNu_bool = math.isclose(isChildPOIClosestToNu_file[iLink], 1.0, rel_tol=0.001)
        childIsPOIClosestToNu[order[linksMadeCounter]].append(1 if isChildPOIClosestToNu_bool else 0)

        # Add in training cuts 
        if (isLinkOrientationCorrect_bool) :                
            trainingCutSep.append(separation3D_file[iLink])
            doesChildConnect_bool = math.isclose(doesChildConnect_file[iLink], 1.0, rel_tol=0.001)
            trainingCutDoesConnect.append(doesChildConnect_bool)
            trainingCutL.append(trainingCutL_file[iLink])
            trainingCutT.append(trainingCutT_file[iLink])                                          

        linksMadeCounter = linksMadeCounter + 1

        if (linksMadeCounter == 4) :
            # set the common vars
            parentTrackScore.append(parentTrackScore_file[iLink])
            childTrackScore.append(childTrackScore_file[iLink])
            parentNSpacepoints.append(parentNSpacepoints_file[iLink])
            childNSpacepoints.append(childNSpacepoints_file[iLink])
            separation3D.append(separation3D_file[iLink])                                                  
            # set truth                                                                          
            trueChildVisibleGeneration.append(trueChildVisibleGeneration_file[iLink])
            y.append(this_y)
            trueParentChildLink.append(trueParentChildLink_bool)
            isLinkOrientationCorrect.append(this_isLinkOrientationCorrect)                        
            # reset                        
            linksMadeCounter = 0
            this_y = [0, 0, 0, 0]
            this_isLinkOrientationCorrect = [0, 0, 0, 0]
            order = shuffle(order)
                        
    ###################################
    # Now turn things into numpy arrays
    ###################################
    # Node variables
    parentTrackScore = np.array(parentTrackScore, dtype='float64')
    childTrackScore = np.array(childTrackScore, dtype='float64')
    parentNSpacepoints = np.array(parentNSpacepoints, dtype='float64')
    childNSpacepoints = np.array(childNSpacepoints, dtype='float64')
    separation3D = np.array(separation3D, dtype='float64')
    parentNuVertexSep = np.array(parentNuVertexSep, dtype='float64')
    childNuVertexSep = np.array(childNuVertexSep, dtype='float64')    
    parentEndRegionNHits = np.array(parentEndRegionNHits, dtype='float64')
    parentEndRegionNParticles = np.array(parentEndRegionNParticles, dtype='float64')
    parentEndRegionRToWall = np.array(parentEndRegionRToWall, dtype='float64')
    vertexSeparation = np.array(vertexSeparation, dtype='float64')    
    doesChildConnect = np.array(doesChildConnect, dtype='float64')
    overshootStartDCA = np.array(overshootStartDCA, dtype='float64')
    overshootStartL = np.array(overshootStartL, dtype='float64')
    overshootEndDCA = np.array(overshootEndDCA, dtype='float64')
    overshootEndL = np.array(overshootEndL, dtype='float64')    
    childConnectionDCA = np.array(childConnectionDCA, dtype='float64')
    childConnectionExtrapDistance = np.array(childConnectionExtrapDistance, dtype='float64')
    childConnectionLRatio = np.array(childConnectionLRatio, dtype='float64')
    parentConnectionPointNUpstreamHits = np.array(parentConnectionPointNUpstreamHits, dtype='float64')
    parentConnectionPointNDownstreamHits = np.array(parentConnectionPointNDownstreamHits, dtype='float64')
    parentConnectionPointNHitRatio = np.array(parentConnectionPointNHitRatio, dtype='float64')
    parentConnectionPointEigenValueRatio = np.array(parentConnectionPointEigenValueRatio, dtype='float64')
    parentConnectionPointOpeningAngle = np.array(parentConnectionPointOpeningAngle, dtype='float64')
    parentIsPOIClosestToNu = np.array(parentIsPOIClosestToNu, dtype='float64')
    childIsPOIClosestToNu = np.array(childIsPOIClosestToNu, dtype='float64')
    # Training cut variables
    trainingCutSep = np.array(trainingCutSep, dtype='float64')
    trainingCutDoesConnect = np.array(trainingCutDoesConnect, dtype='int')
    trainingCutL = np.array(trainingCutL, dtype='float64')
    trainingCutT = np.array(trainingCutT, dtype='float64')
    # Truth 
    trueChildVisibleGeneration = np.array(trueChildVisibleGeneration, dtype='int')
    trueParentChildLink = np.array(trueParentChildLink, dtype='int')
    isLinkOrientationCorrect = np.array(isLinkOrientationCorrect, dtype='int')
    y = np.array(y, dtype='int')
    
    ###################################
    # How many links do we have?
    ###################################        
    nLinks = trueParentChildLink.shape[0]   
    print('We have ', str(nLinks), ' to train on!')
    
    ###################################
    # Normalise variables
    ###################################
    if (normalise) :
        Normalise.normaliseXAxis(parentTrackScore, Normalise.parentTrackScore_min, Normalise.parentTrackScore_max)
        Normalise.normaliseXAxis(childTrackScore, Normalise.parentTrackScore_min, Normalise.parentTrackScore_max)    
        Normalise.normaliseXAxis(parentNSpacepoints, Normalise.parentNSpacepoints_min, Normalise.parentNSpacepoints_max)   
        Normalise.normaliseXAxis(childNSpacepoints, Normalise.parentNSpacepoints_min, Normalise.parentNSpacepoints_max) 
        Normalise.normaliseXAxis(parentNuVertexSep, Normalise.parentNuVertexSeparation_min, Normalise.parentNuVertexSeparation_max) 
        Normalise.normaliseXAxis(childNuVertexSep, Normalise.parentNuVertexSeparation_min, Normalise.parentNuVertexSeparation_max)        
        Normalise.normaliseXAxis(separation3D, Normalise.separation3D_min, Normalise.separation3D_max)                                                  
        Normalise.normaliseXAxis(parentEndRegionNHits, Normalise.parentEndRegionNHits_min, Normalise.parentEndRegionNHits_max)
        Normalise.normaliseXAxis(parentEndRegionNParticles, Normalise.parentEndRegionNParticles_min, Normalise.parentEndRegionNParticles_max)
        Normalise.normaliseXAxis(parentEndRegionRToWall, Normalise.parentEndRegionRToWall_min, Normalise.parentEndRegionRToWall_max)
        Normalise.normaliseXAxis(vertexSeparation, Normalise.vertexSeparation_min, Normalise.vertexSeparation_max)
        Normalise.normaliseXAxis(doesChildConnect, Normalise.doesChildConnect_min, Normalise.doesChildConnect_max)
        Normalise.normaliseXAxis(overshootStartDCA, Normalise.overshootDCA_min, Normalise.overshootDCA_max)
        Normalise.normaliseXAxis(overshootStartL, Normalise.overshootL_min, Normalise.overshootL_max)
        Normalise.normaliseXAxis(overshootEndDCA, Normalise.overshootDCA_min, Normalise.overshootDCA_max)
        Normalise.normaliseXAxis(overshootEndL, Normalise.overshootL_min, Normalise.overshootL_max)
        Normalise.normaliseXAxis(childConnectionDCA, Normalise.childConnectionDCA_min, Normalise.childConnectionDCA_max)
        Normalise.normaliseXAxis(childConnectionExtrapDistance, Normalise.childConnectionExtrapDistance_min, Normalise.childConnectionExtrapDistance_max)
        Normalise.normaliseXAxis(childConnectionLRatio, Normalise.childConnectionLRatio_min, Normalise.childConnectionLRatio_max)
        Normalise.normaliseXAxis(parentConnectionPointNUpstreamHits, Normalise.parentConnectionPointNUpstreamHits_min, Normalise.parentConnectionPointNUpstreamHits_max)
        Normalise.normaliseXAxis(parentConnectionPointNDownstreamHits, Normalise.parentConnectionPointNDownstreamHits_min, Normalise.parentConnectionPointNDownstreamHits_max)
        Normalise.normaliseXAxis(parentConnectionPointNHitRatio, Normalise.parentConnectionPointNHitRatio_min, Normalise.parentConnectionPointNHitRatio_max)
        Normalise.normaliseXAxis(parentConnectionPointEigenValueRatio, Normalise.parentConnectionPointEigenValueRatio_min, Normalise.parentConnectionPointEigenValueRatio_max)
        Normalise.normaliseXAxis(parentConnectionPointOpeningAngle, Normalise.parentConnectionPointOpeningAngle_min, Normalise.parentConnectionPointOpeningAngle_max) 

    ###################################
    # Concatenate
    ###################################
    coc0 = np.concatenate((np.concatenate((parentNuVertexSep[0, :].reshape(nLinks, 1),
                                           childNuVertexSep[0, :].reshape(nLinks, 1),
                                           parentEndRegionNHits[0, :].reshape(nLinks, 1), \
                                           parentEndRegionNParticles[0, :].reshape(nLinks, 1), \
                                           parentEndRegionRToWall[0, :].reshape(nLinks, 1), \
                                           vertexSeparation[0, :].reshape(nLinks, 1), \
                                           doesChildConnect[0, :].reshape(nLinks, 1), \
                                           overshootStartDCA[0, :].reshape(nLinks, 1), \
                                           overshootStartL[0, :].reshape(nLinks, 1), \
                                           overshootEndDCA[0, :].reshape(nLinks, 1), \
                                           overshootEndL[0, :].reshape(nLinks, 1), \
                                           childConnectionDCA[0, :].reshape(nLinks, 1), \
                                           childConnectionExtrapDistance[0, :].reshape(nLinks, 1), \
                                           childConnectionLRatio[0, :].reshape(nLinks, 1), \
                                           parentConnectionPointNUpstreamHits[0, :].reshape(nLinks, 1), \
                                           parentConnectionPointNDownstreamHits[0, :].reshape(nLinks, 1), \
                                           parentConnectionPointNHitRatio[0, :].reshape(nLinks, 1), \
                                           parentConnectionPointEigenValueRatio[0, :].reshape(nLinks, 1), \
                                           parentConnectionPointOpeningAngle[0, :].reshape(nLinks, 1), \
                                           parentIsPOIClosestToNu[0, :].reshape(nLinks, 1), \
                                           childIsPOIClosestToNu[0, :].reshape(nLinks, 1)), axis=1), \
                           np.concatenate((parentNuVertexSep[1, :].reshape(nLinks, 1), \
                                           childNuVertexSep[1, :].reshape(nLinks, 1), \
                                           parentEndRegionNHits[1, :].reshape(nLinks, 1), \
                                           parentEndRegionNParticles[1, :].reshape(nLinks, 1), \
                                           parentEndRegionRToWall[1, :].reshape(nLinks, 1), \
                                           vertexSeparation[1, :].reshape(nLinks, 1), \
                                           doesChildConnect[1, :].reshape(nLinks, 1), \
                                           overshootStartDCA[1, :].reshape(nLinks, 1), \
                                           overshootStartL[1, :].reshape(nLinks, 1), \
                                           overshootEndDCA[1, :].reshape(nLinks, 1), \
                                           overshootEndL[1, :].reshape(nLinks, 1), \
                                           childConnectionDCA[1, :].reshape(nLinks, 1), \
                                           childConnectionExtrapDistance[1, :].reshape(nLinks, 1), \
                                           childConnectionLRatio[1, :].reshape(nLinks, 1), \
                                           parentConnectionPointNUpstreamHits[1, :].reshape(nLinks, 1), \
                                           parentConnectionPointNDownstreamHits[1, :].reshape(nLinks, 1), \
                                           parentConnectionPointNHitRatio[1, :].reshape(nLinks, 1), \
                                           parentConnectionPointEigenValueRatio[1, :].reshape(nLinks, 1), \
                                           parentConnectionPointOpeningAngle[1, :].reshape(nLinks, 1), \
                                           parentIsPOIClosestToNu[1, :].reshape(nLinks, 1), \
                                           childIsPOIClosestToNu[1, :].reshape(nLinks, 1)), axis=1), \
                           np.concatenate((parentNuVertexSep[2, :].reshape(nLinks, 1), \
                                           childNuVertexSep[2, :].reshape(nLinks, 1), \
                                           parentEndRegionNHits[2, :].reshape(nLinks, 1), \
                                           parentEndRegionNParticles[2, :].reshape(nLinks, 1), \
                                           parentEndRegionRToWall[2, :].reshape(nLinks, 1), \
                                           vertexSeparation[2, :].reshape(nLinks, 1), \
                                           doesChildConnect[2, :].reshape(nLinks, 1), \
                                           overshootStartDCA[2, :].reshape(nLinks, 1), \
                                           overshootStartL[2, :].reshape(nLinks, 1), \
                                           overshootEndDCA[2, :].reshape(nLinks, 1), \
                                           overshootEndL[2, :].reshape(nLinks, 1), \
                                           childConnectionDCA[2, :].reshape(nLinks, 1), \
                                           childConnectionExtrapDistance[2, :].reshape(nLinks, 1), \
                                           childConnectionLRatio[2, :].reshape(nLinks, 1), \
                                           parentConnectionPointNUpstreamHits[2, :].reshape(nLinks, 1), \
                                           parentConnectionPointNDownstreamHits[2, :].reshape(nLinks, 1), \
                                           parentConnectionPointNHitRatio[2, :].reshape(nLinks, 1), \
                                           parentConnectionPointEigenValueRatio[2, :].reshape(nLinks, 1), \
                                           parentConnectionPointOpeningAngle[2, :].reshape(nLinks, 1), \
                                           parentIsPOIClosestToNu[2, :].reshape(nLinks, 1), \
                                           childIsPOIClosestToNu[2, :].reshape(nLinks, 1)), axis=1), \
                           np.concatenate((parentNuVertexSep[3, :].reshape(nLinks, 1), \
                                           childNuVertexSep[3, :].reshape(nLinks, 1), \
                                           parentEndRegionNHits[3, :].reshape(nLinks, 1), \
                                           parentEndRegionNParticles[3, :].reshape(nLinks, 1), \
                                           parentEndRegionRToWall[3, :].reshape(nLinks, 1), \
                                           vertexSeparation[3, :].reshape(nLinks, 1), \
                                           doesChildConnect[3, :].reshape(nLinks, 1), \
                                           overshootStartDCA[3, :].reshape(nLinks, 1), \
                                           overshootStartL[3, :].reshape(nLinks, 1), \
                                           overshootEndDCA[3, :].reshape(nLinks, 1), \
                                           overshootEndL[3, :].reshape(nLinks, 1), \
                                           childConnectionDCA[3, :].reshape(nLinks, 1), \
                                           childConnectionExtrapDistance[3, :].reshape(nLinks, 1), \
                                           childConnectionLRatio[3, :].reshape(nLinks, 1), \
                                           parentConnectionPointNUpstreamHits[3, :].reshape(nLinks, 1), \
                                           parentConnectionPointNDownstreamHits[3, :].reshape(nLinks, 1), \
                                           parentConnectionPointNHitRatio[3, :].reshape(nLinks, 1), \
                                           parentConnectionPointEigenValueRatio[3, :].reshape(nLinks, 1), \
                                           parentConnectionPointOpeningAngle[3, :].reshape(nLinks, 1), \
                                           parentIsPOIClosestToNu[3, :].reshape(nLinks, 1), \
                                           childIsPOIClosestToNu[3, :].reshape(nLinks, 1)), axis=1)), axis=1)
    
    # concatenate variable_single and orientations    
    variables = np.concatenate((parentTrackScore.reshape(nLinks, 1), \
                                childTrackScore.reshape(nLinks, 1), \
                                parentNSpacepoints.reshape(nLinks, 1), \
                                childNSpacepoints.reshape(nLinks, 1), \
                                separation3D.reshape(nLinks, 1), \
                                coc0), axis=1)
    
    return nLinks, variables, y, trueParentChildLink, isLinkOrientationCorrect, trueChildVisibleGeneration, trainingCutSep, trainingCutDoesConnect, trainingCutL, trainingCutT

############################################################################################################################################
############################################################################################################################################

def readTreeGroupLinks_shower(fileName, normalise) :

    ###################################
    # To pull out of tree
    ###################################
    # Link variables
    parentTrackScore = []
    childTrackScore = []
    parentNSpacepoints = []
    childNSpacepoints = []
    separation3D = []
    parentNuVertexSep = [[], []]
    childNuVertexSep = [[], []]
    parentEndRegionNHits = [[], []]
    parentEndRegionNParticles = [[], []]
    parentEndRegionRToWall = [[], []]
    vertexSeparation = [[], []]
    doesChildConnect = [[], []]
    overshootStartDCA = [[], []]
    overshootStartL = [[], []]
    overshootEndDCA = [[], []]
    overshootEndL = [[], []]
    childConnectionDCA = [[], []]
    childConnectionExtrapDistance = [[], []]
    childConnectionLRatio = [[], []]
    parentConnectionPointNUpstreamHits = [[], []]
    parentConnectionPointNDownstreamHits = [[], []]
    parentConnectionPointNHitRatio = [[], []]
    parentConnectionPointEigenValueRatio = [[], []]
    parentConnectionPointOpeningAngle = [[], []]
    parentIsPOIClosestToNu = [[], []]
    childIsPOIClosestToNu = [[], []]
    # Training cut variables
    trainingCutSep = []
    trainingCutDoesConnect = []
    trainingCutL = []
    trainingCutT = []
    # Truth
    trueParentChildLink = []
    isLinkOrientationCorrect = []
    y = []
    trueChildVisibleGeneration = []
    
    print('Reading tree: ', str(fileName),', This may take a while...')

    ####################################
    # Set tree
    ####################################  
    treeFile = uproot.open(fileName)
    tree = treeFile['LaterTierTrackShowerTree_TRAIN']
    branches = tree.arrays()

    ####################################
    # Set tree branches
    ####################################
    # Network vars
    parentTrackScore_file = np.array(branches['ParentTrackScore'])            
    childTrackScore_file = np.array(branches['ChildTrackScore'])
    parentNSpacepoints_file = np.array(branches['ParentNSpacepoints'])
    childNSpacepoints_file = np.array(branches['ChildNSpacepoints'])
    separation3D_file = np.array(branches['Separation3D'])
    parentNuVertexSep_file = np.array(branches['ParentNuVertexSep'])
    childNuVertexSep_file = np.array(branches['ChildNuVertexSep'])                        
    parentEndRegionNHits_file = np.array(branches['ParentEndRegionNHits'])
    parentEndRegionNParticles_file = np.array(branches['ParentEndRegionNParticles'])
    parentEndRegionRToWall_file = np.array(branches['ParentEndRegionRToWall'])
    vertexSeparation_file = np.array(branches['VertexSeparation'])        
    doesChildConnect_file = np.array(branches['DoesChildConnect'])
    overshootStartDCA_file = np.array(branches['OvershootStartDCA'])
    overshootStartL_file = np.array(branches['OvershootStartL'])
    overshootEndDCA_file = np.array(branches['OvershootEndDCA'])
    overshootEndL_file = np.array(branches['OvershootEndL'])
    childConnectionDCA_file = np.array(branches['ChildCPDCA'])
    childConnectionExtrapDistance_file = np.array(branches['ChildCPExtrapDistance'])
    childConnectionLRatio_file = np.array(branches['ChildCPLRatio'])
    parentConnectionPointNUpstreamHits_file = np.array(branches['ParentCPNUpstreamHits'])
    parentConnectionPointNDownstreamHits_file = np.array(branches['ParentCPNDownstreamHits'])
    parentConnectionPointNHitRatio_file = np.array(branches['ParentCPNHitRatio'])
    parentConnectionPointEigenValueRatio_file = np.array(branches['ParentCPEigenvalueRatio'])
    parentConnectionPointOpeningAngle_file = np.array(branches['ParentCPOpeningAngle'])
    isParentPOIClosestToNu_file = np.array(branches['ParentIsPOIClosestToNu'])
    isChildPOIClosestToNu_file = np.array(branches['ChildIsPOIClosestToNu'])
    # Truth
    trueParentChildLink_file = np.array(branches['IsTrueLink'])
    trueVisibleGeneration_file = np.array(branches['ChildTrueVisibleGeneration'])
    isLinkOrientationCorrect_file = np.array(branches['IsOrientationCorrect'])
    # Training cuts!
    trainingCutL_file = np.array(branches['TrainingCutL'])
    trainingCutT_file = np.array(branches['TrainingCutT'])
    # nLinks
    nLinks_file = trueParentChildLink_file.shape[0]

    ####################################
    # Now loop over loops to group them.
    ####################################
    linksMadeCounter = 0
    this_y = [0, 0]
    this_isLinkOrientationCorrect = [0, 0]
    order = [0, 1]

    for iLink in range(0, nLinks_file) :

        if ((iLink % 1000) == 0) :
            print('iLink:', str(iLink) + '/' + str(nLinks_file)) 

        # Set truth                                                  
        trueParentChildLink_bool = math.isclose(trueParentChildLink_file[iLink], 1.0, rel_tol=0.001)
        isLinkOrientationCorrect_bool = math.isclose(isLinkOrientationCorrect_file[iLink], 1.0, rel_tol=0.001)

        if (trueParentChildLink_bool and isLinkOrientationCorrect_bool) :
            this_y[order[linksMadeCounter]] = 1 
        elif (trueParentChildLink_bool and (not isLinkOrientationCorrect_bool)) :
            this_y[order[linksMadeCounter]] = 2

        this_isLinkOrientationCorrect[order[linksMadeCounter]] = isLinkOrientationCorrect_bool  

        # set the link information
        parentNuVertexSep[order[linksMadeCounter]].append(parentNuVertexSep_file[iLink])
        childNuVertexSep[order[linksMadeCounter]].append(childNuVertexSep_file[iLink])
        parentEndRegionNHits[order[linksMadeCounter]].append(parentEndRegionNHits_file[iLink])
        parentEndRegionNParticles[order[linksMadeCounter]].append(parentEndRegionNParticles_file[iLink])
        parentEndRegionRToWall[order[linksMadeCounter]].append(parentEndRegionRToWall_file[iLink])
        vertexSeparation[order[linksMadeCounter]].append(vertexSeparation_file[iLink])
        doesChildConnect[order[linksMadeCounter]].append(doesChildConnect_file[iLink])
        overshootStartDCA[order[linksMadeCounter]].append(overshootStartDCA_file[iLink])
        overshootStartL[order[linksMadeCounter]].append(overshootStartL_file[iLink])
        overshootEndDCA[order[linksMadeCounter]].append(overshootEndDCA_file[iLink])
        overshootEndL[order[linksMadeCounter]].append(overshootEndL_file[iLink])
        childConnectionDCA[order[linksMadeCounter]].append(childConnectionDCA_file[iLink])
        childConnectionExtrapDistance[order[linksMadeCounter]].append(childConnectionExtrapDistance_file[iLink])
        childConnectionLRatio[order[linksMadeCounter]].append(childConnectionLRatio_file[iLink])
        parentConnectionPointNUpstreamHits[order[linksMadeCounter]].append(parentConnectionPointNUpstreamHits_file[iLink])
        parentConnectionPointNDownstreamHits[order[linksMadeCounter]].append(parentConnectionPointNDownstreamHits_file[iLink])
        parentConnectionPointNHitRatio[order[linksMadeCounter]].append(parentConnectionPointNHitRatio_file[iLink])
        parentConnectionPointEigenValueRatio[order[linksMadeCounter]].append(parentConnectionPointEigenValueRatio_file[iLink])
        parentConnectionPointOpeningAngle[order[linksMadeCounter]].append(parentConnectionPointOpeningAngle_file[iLink])
        isParentPOIClosestToNu_bool = math.isclose(isParentPOIClosestToNu_file[iLink], 1.0, rel_tol=0.001)
        parentIsPOIClosestToNu[order[linksMadeCounter]].append(1 if isParentPOIClosestToNu_bool else 0)
        isChildPOIClosestToNu_bool = math.isclose(isChildPOIClosestToNu_file[iLink], 1.0, rel_tol=0.001)
        childIsPOIClosestToNu[order[linksMadeCounter]].append(1 if isChildPOIClosestToNu_bool else 0)

        # Add in training cuts 
        if (isLinkOrientationCorrect_bool) :
            trainingCutSep.append(separation3D_file[iLink])
            doesChildConnect_bool = math.isclose(doesChildConnect_file[iLink], 1.0, rel_tol=0.001)
            trainingCutDoesConnect.append(doesChildConnect_bool)
            trainingCutL.append(trainingCutL_file[iLink])
            trainingCutT.append(trainingCutT_file[iLink])                                              

        linksMadeCounter = linksMadeCounter + 1

        if (linksMadeCounter == 2) :
            # set the common vars
            parentTrackScore.append(parentTrackScore_file[iLink])
            childTrackScore.append(childTrackScore_file[iLink])
            parentNSpacepoints.append(parentNSpacepoints_file[iLink])
            childNSpacepoints.append(childNSpacepoints_file[iLink])
            separation3D.append(separation3D_file[iLink])                                                  
            # set truth                                                                          
            trueChildVisibleGeneration.append(trueVisibleGeneration_file[iLink])        
            y.append(this_y)
            trueParentChildLink.append(trueParentChildLink_bool)
            isLinkOrientationCorrect.append(this_isLinkOrientationCorrect)                        
            # reset                        
            linksMadeCounter = 0
            this_y = [0, 0]
            this_isLinkOrientationCorrect = [0, 0]
            order = shuffle(order)

    ###################################
    # Now turn things into numpy arrays
    ###################################
    # Node variables
    parentTrackScore = np.array(parentTrackScore, dtype='float64')    
    childTrackScore = np.array(childTrackScore, dtype='float64')
    parentNSpacepoints = np.array(parentNSpacepoints, dtype='float64')
    childNSpacepoints = np.array(childNSpacepoints, dtype='float64')
    separation3D = np.array(separation3D, dtype='float64')
    parentNuVertexSep = np.array(parentNuVertexSep, dtype='float64')
    childNuVertexSep = np.array(childNuVertexSep, dtype='float64')    
    parentEndRegionNHits = np.array(parentEndRegionNHits, dtype='float64')
    parentEndRegionNParticles = np.array(parentEndRegionNParticles, dtype='float64')
    parentEndRegionRToWall = np.array(parentEndRegionRToWall, dtype='float64')
    vertexSeparation = np.array(vertexSeparation, dtype='float64')    
    doesChildConnect = np.array(doesChildConnect, dtype='float64')
    overshootStartDCA = np.array(overshootStartDCA, dtype='float64')
    overshootStartL = np.array(overshootStartL, dtype='float64')
    overshootEndDCA = np.array(overshootEndDCA, dtype='float64')
    overshootEndL = np.array(overshootEndL, dtype='float64')    
    childConnectionDCA = np.array(childConnectionDCA, dtype='float64')
    childConnectionExtrapDistance = np.array(childConnectionExtrapDistance, dtype='float64')
    childConnectionLRatio = np.array(childConnectionLRatio, dtype='float64')
    parentConnectionPointNUpstreamHits = np.array(parentConnectionPointNUpstreamHits, dtype='float64')
    parentConnectionPointNDownstreamHits = np.array(parentConnectionPointNDownstreamHits, dtype='float64')
    parentConnectionPointNHitRatio = np.array(parentConnectionPointNHitRatio, dtype='float64')
    parentConnectionPointEigenValueRatio = np.array(parentConnectionPointEigenValueRatio, dtype='float64')
    parentConnectionPointOpeningAngle = np.array(parentConnectionPointOpeningAngle, dtype='float64')
    parentIsPOIClosestToNu = np.array(parentIsPOIClosestToNu, dtype='float64')
    childIsPOIClosestToNu = np.array(childIsPOIClosestToNu, dtype='float64')
    # Training cut variables
    trainingCutSep = np.array(trainingCutSep, dtype='float64')
    trainingCutDoesConnect = np.array(trainingCutDoesConnect, dtype='int')
    trainingCutL = np.array(trainingCutL, dtype='float64')
    trainingCutT = np.array(trainingCutT, dtype='float64')
    # Truth 
    trueChildVisibleGeneration = np.array(trueChildVisibleGeneration, dtype='int')
    trueParentChildLink = np.array(trueParentChildLink, dtype='int')
    isLinkOrientationCorrect = np.array(isLinkOrientationCorrect, dtype='int')
    y = np.array(y, dtype='int')
    
    ###################################
    # How many links do we have?
    ###################################        
    nLinks = trueParentChildLink.shape[0]   
    print('We have ', str(nLinks), ' to train on!')
    
    ###################################
    # Normalise variables
    ###################################
    if (normalise) :
        Normalise.normaliseXAxis(parentTrackScore, Normalise.parentTrackScore_min, Normalise.parentTrackScore_max)
        Normalise.normaliseXAxis(childTrackScore, Normalise.parentTrackScore_min, Normalise.parentTrackScore_max)    
        Normalise.normaliseXAxis(parentNSpacepoints, Normalise.parentNSpacepoints_min, Normalise.parentNSpacepoints_max)   
        Normalise.normaliseXAxis(childNSpacepoints, Normalise.parentNSpacepoints_min, Normalise.parentNSpacepoints_max) 
        Normalise.normaliseXAxis(parentNuVertexSep, Normalise.parentNuVertexSeparation_min, Normalise.parentNuVertexSeparation_max) 
        Normalise.normaliseXAxis(childNuVertexSep, Normalise.parentNuVertexSeparation_min, Normalise.parentNuVertexSeparation_max) 
        Normalise.normaliseXAxis(separation3D, Normalise.separation3D_min, Normalise.separation3D_max)
        Normalise.normaliseXAxis(parentEndRegionNHits, Normalise.parentEndRegionNHits_min, Normalise.parentEndRegionNHits_max)
        Normalise.normaliseXAxis(parentEndRegionNParticles, Normalise.parentEndRegionNParticles_min, Normalise.parentEndRegionNParticles_max)
        Normalise.normaliseXAxis(parentEndRegionRToWall, Normalise.parentEndRegionRToWall_min, Normalise.parentEndRegionRToWall_max)
        Normalise.normaliseXAxis(vertexSeparation, Normalise.vertexSeparation_min, Normalise.vertexSeparation_max)
        Normalise.normaliseXAxis(doesChildConnect, Normalise.doesChildConnect_min, Normalise.doesChildConnect_max)
        Normalise.normaliseXAxis(overshootStartDCA, Normalise.overshootDCA_min, Normalise.overshootDCA_max)
        Normalise.normaliseXAxis(overshootStartL, Normalise.overshootL_min, Normalise.overshootL_max)
        Normalise.normaliseXAxis(overshootEndDCA, Normalise.overshootDCA_min, Normalise.overshootDCA_max)
        Normalise.normaliseXAxis(overshootEndL, Normalise.overshootL_min, Normalise.overshootL_max)
        Normalise.normaliseXAxis(childConnectionDCA, Normalise.childConnectionDCA_min, Normalise.childConnectionDCA_max)
        Normalise.normaliseXAxis(childConnectionExtrapDistance, Normalise.childConnectionExtrapDistance_min, Normalise.childConnectionExtrapDistance_max)
        Normalise.normaliseXAxis(childConnectionLRatio, Normalise.childConnectionLRatio_min, Normalise.childConnectionLRatio_max)
        Normalise.normaliseXAxis(parentConnectionPointNUpstreamHits, Normalise.parentConnectionPointNUpstreamHits_min, Normalise.parentConnectionPointNUpstreamHits_max)
        Normalise.normaliseXAxis(parentConnectionPointNDownstreamHits, Normalise.parentConnectionPointNDownstreamHits_min, Normalise.parentConnectionPointNDownstreamHits_max)
        Normalise.normaliseXAxis(parentConnectionPointNHitRatio, Normalise.parentConnectionPointNHitRatio_min, Normalise.parentConnectionPointNHitRatio_max)
        Normalise.normaliseXAxis(parentConnectionPointEigenValueRatio, Normalise.parentConnectionPointEigenValueRatio_min, Normalise.parentConnectionPointEigenValueRatio_max)
        Normalise.normaliseXAxis(parentConnectionPointOpeningAngle, Normalise.parentConnectionPointOpeningAngle_min, Normalise.parentConnectionPointOpeningAngle_max) 

    ###################################
    # Concatenate
    ###################################
    coc0 = np.concatenate((np.concatenate((parentNuVertexSep[0, :].reshape(nLinks, 1), \
                                           childNuVertexSep[0, :].reshape(nLinks, 1), \
                                           parentEndRegionNHits[0, :].reshape(nLinks, 1), \
                                           parentEndRegionNParticles[0, :].reshape(nLinks, 1), \
                                           parentEndRegionRToWall[0, :].reshape(nLinks, 1), \
                                           vertexSeparation[0, :].reshape(nLinks, 1), \
                                           doesChildConnect[0, :].reshape(nLinks, 1), \
                                           overshootStartDCA[0, :].reshape(nLinks, 1), \
                                           overshootStartL[0, :].reshape(nLinks, 1), \
                                           overshootEndDCA[0, :].reshape(nLinks, 1), \
                                           overshootEndL[0, :].reshape(nLinks, 1), \
                                           childConnectionDCA[0, :].reshape(nLinks, 1), \
                                           childConnectionExtrapDistance[0, :].reshape(nLinks, 1), \
                                           childConnectionLRatio[0, :].reshape(nLinks, 1), \
                                           parentConnectionPointNUpstreamHits[0, :].reshape(nLinks, 1), \
                                           parentConnectionPointNDownstreamHits[0, :].reshape(nLinks, 1), \
                                           parentConnectionPointNHitRatio[0, :].reshape(nLinks, 1), \
                                           parentConnectionPointEigenValueRatio[0, :].reshape(nLinks, 1), \
                                           parentConnectionPointOpeningAngle[0, :].reshape(nLinks, 1), \
                                           parentIsPOIClosestToNu[0, :].reshape(nLinks, 1), \
                                           childIsPOIClosestToNu[0, :].reshape(nLinks, 1)), axis=1), \
                           np.concatenate((parentNuVertexSep[1, :].reshape(nLinks, 1), \
                                           childNuVertexSep[1, :].reshape(nLinks, 1), \
                                           parentEndRegionNHits[1, :].reshape(nLinks, 1), \
                                           parentEndRegionNParticles[1, :].reshape(nLinks, 1), \
                                           parentEndRegionRToWall[1, :].reshape(nLinks, 1), \
                                           vertexSeparation[1, :].reshape(nLinks, 1), \
                                           doesChildConnect[1, :].reshape(nLinks, 1), \
                                           overshootStartDCA[1, :].reshape(nLinks, 1), \
                                           overshootStartL[1, :].reshape(nLinks, 1), \
                                           overshootEndDCA[1, :].reshape(nLinks, 1), \
                                           overshootEndL[1, :].reshape(nLinks, 1), \
                                           childConnectionDCA[1, :].reshape(nLinks, 1), \
                                           childConnectionExtrapDistance[1, :].reshape(nLinks, 1), \
                                           childConnectionLRatio[1, :].reshape(nLinks, 1), \
                                           parentConnectionPointNUpstreamHits[1, :].reshape(nLinks, 1), \
                                           parentConnectionPointNDownstreamHits[1, :].reshape(nLinks, 1), \
                                           parentConnectionPointNHitRatio[1, :].reshape(nLinks, 1), \
                                           parentConnectionPointEigenValueRatio[1, :].reshape(nLinks, 1), \
                                           parentConnectionPointOpeningAngle[1, :].reshape(nLinks, 1), \
                                           parentIsPOIClosestToNu[1, :].reshape(nLinks, 1), \
                                           childIsPOIClosestToNu[1, :].reshape(nLinks, 1)), axis=1)), axis=1)

    # concatenate variable_single and orientations
    variables = np.concatenate((parentTrackScore.reshape(nLinks, 1), \
                                childTrackScore.reshape(nLinks, 1), \
                                parentNSpacepoints.reshape(nLinks, 1), \
                                childNSpacepoints.reshape(nLinks, 1), \
                                separation3D.reshape(nLinks, 1), \
                                coc0), axis=1)
    
    return nLinks, variables, y, trueParentChildLink, isLinkOrientationCorrect, trueChildVisibleGeneration, trainingCutSep, trainingCutDoesConnect, trainingCutL, trainingCutT

############################################################################################################################################
############################################################################################################################################    

    