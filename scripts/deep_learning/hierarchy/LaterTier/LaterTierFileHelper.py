import numpy as np
import uproot
import Utilities
import copy
from sklearn.utils import shuffle

############################################################################################################################################

def ReadTreeForTraining(isTrackMode, fileName, normalise) :
        
    if (isTrackMode) :
        nLinks = 4
        treeName = "LaterTierTrackTrackTree"
    else :
        nLinks = 2
        treeName = "LaterTierTrackShowerTree"
    
    with uproot.open(f"{fileName}:{treeName}") as tree:

        branches = tree.arrays()

        # Vars (put astype as they are in tree)
        parentTrackScore = np.array(branches['ParentTrackScore']).astype('float64')            
        childTrackScore = np.array(branches['ChildTrackScore']).astype('float64')
        parentNSpacepoints = np.array(branches['ParentNSpacepoints']).astype('float64')
        childNSpacepoints = np.array(branches['ChildNSpacepoints']).astype('float64')
        separation3D = np.array(branches['Separation3D']).astype('float64')
        parentNuVertexSep = np.array(branches['ParentNuVertexSep']).astype('float64')
        childNuVertexSep = np.array(branches['ChildNuVertexSep']).astype('float64')                        
        parentEndRegionNHits = np.array(branches['ParentEndRegionNHits']).astype('float64')
        parentEndRegionNParticles = np.array(branches['ParentEndRegionNParticles']).astype('float64')
        parentEndRegionRToWall = np.array(branches['ParentEndRegionRToWall']).astype('float64')
        vertexSeparation = np.array(branches['VertexSeparation']).astype('float64')        
        doesChildConnect = np.array(branches['DoesChildConnect']).astype('float64')
        overshootStartDCA = np.array(branches['OvershootStartDCA']).astype('float64')
        overshootStartL = np.array(branches['OvershootStartL']).astype('float64')
        overshootEndDCA = np.array(branches['OvershootEndDCA']).astype('float64')
        overshootEndL = np.array(branches['OvershootEndL']).astype('float64')
        childConnectionDCA = np.array(branches['ChildCPDCA']).astype('float64')
        childConnectionExtrapDistance = np.array(branches['ChildCPExtrapDistance']).astype('float64')
        childConnectionLRatio = np.array(branches['ChildCPLRatio']).astype('float64')
        parentConnectionPointNUpstreamHits = np.array(branches['ParentCPNUpstreamHits']).astype('float64')
        parentConnectionPointNDownstreamHits = np.array(branches['ParentCPNDownstreamHits']).astype('float64')
        parentConnectionPointNHitRatio = np.array(branches['ParentCPNHitRatio']).astype('float64')
        parentConnectionPointEigenValueRatio = np.array(branches['ParentCPEigenvalueRatio']).astype('float64')
        parentConnectionPointOpeningAngle = np.array(branches['ParentCPOpeningAngle']).astype('float64')
        isParentPOIClosestToNu = np.array(branches['ParentIsPOIClosestToNu']).astype('float64')
        isChildPOIClosestToNu = np.array(branches['ChildIsPOIClosestToNu']).astype('float64')
        # Truth
        trueParentChildLink = np.array(branches['IsTrueLink']).astype('int')                        # int
        trueChildVisibleGeneration = np.array(branches['ChildTrueVisibleGeneration']).astype('int') # int
        isLinkOrientationCorrect = np.array(branches['IsOrientationCorrect']).astype('int')         # int
        # isTraining        
        isTrainingLink = np.array(branches['IsTrainingLink']).astype('int')
        
        # Pick out training links and reshape
        parentTrackScore = parentTrackScore[isTrainingLink == 1].reshape(-1, nLinks)
        childTrackScore = childTrackScore[isTrainingLink == 1].reshape(-1, nLinks)
        parentNSpacepoints = parentNSpacepoints[isTrainingLink == 1].reshape(-1, nLinks)
        childNSpacepoints = childNSpacepoints[isTrainingLink == 1].reshape(-1, nLinks)
        separation3D = separation3D[isTrainingLink == 1].reshape(-1, nLinks)
        parentNuVertexSep = parentNuVertexSep[isTrainingLink == 1].reshape(-1, nLinks)
        childNuVertexSep = childNuVertexSep[isTrainingLink == 1].reshape(-1, nLinks)
        parentEndRegionNHits = parentEndRegionNHits[isTrainingLink == 1].reshape(-1, nLinks)
        parentEndRegionNParticles = parentEndRegionNParticles[isTrainingLink == 1].reshape(-1, nLinks)
        parentEndRegionRToWall = parentEndRegionRToWall[isTrainingLink == 1].reshape(-1, nLinks)
        vertexSeparation = vertexSeparation[isTrainingLink == 1].reshape(-1, nLinks)
        doesChildConnect = doesChildConnect[isTrainingLink == 1].reshape(-1, nLinks)
        overshootStartDCA = overshootStartDCA[isTrainingLink == 1].reshape(-1, nLinks)
        overshootStartL = overshootStartL[isTrainingLink == 1].reshape(-1, nLinks)
        overshootEndDCA = overshootEndDCA[isTrainingLink == 1].reshape(-1, nLinks)
        overshootEndL = overshootEndL[isTrainingLink == 1].reshape(-1, nLinks)
        childConnectionDCA = childConnectionDCA[isTrainingLink == 1].reshape(-1, nLinks)
        childConnectionExtrapDistance = childConnectionExtrapDistance[isTrainingLink == 1].reshape(-1, nLinks)
        childConnectionLRatio = childConnectionLRatio[isTrainingLink == 1].reshape(-1, nLinks)
        parentConnectionPointNUpstreamHits = parentConnectionPointNUpstreamHits[isTrainingLink == 1].reshape(-1, nLinks)
        parentConnectionPointNDownstreamHits = parentConnectionPointNDownstreamHits[isTrainingLink == 1].reshape(-1, nLinks)
        parentConnectionPointNHitRatio = parentConnectionPointNHitRatio[isTrainingLink == 1].reshape(-1, nLinks)
        parentConnectionPointEigenValueRatio = parentConnectionPointEigenValueRatio[isTrainingLink == 1].reshape(-1, nLinks)
        parentConnectionPointOpeningAngle = parentConnectionPointOpeningAngle[isTrainingLink == 1].reshape(-1, nLinks)
        isParentPOIClosestToNu = isParentPOIClosestToNu[isTrainingLink == 1].reshape(-1, nLinks)
        isChildPOIClosestToNu = isChildPOIClosestToNu[isTrainingLink == 1].reshape(-1, nLinks)
        trueParentChildLink = trueParentChildLink[isTrainingLink == 1].reshape(-1, nLinks)
        trueChildVisibleGeneration = trueChildVisibleGeneration[isTrainingLink == 1].reshape(-1, nLinks)
        isLinkOrientationCorrect = isLinkOrientationCorrect[isTrainingLink == 1].reshape(-1, nLinks)
        
        # Form link truth
        y = np.zeros(trueParentChildLink.shape).astype('int')
        y[np.logical_and(trueParentChildLink,  isLinkOrientationCorrect)] = 1
        y[np.logical_and(trueParentChildLink,  np.logical_not(isLinkOrientationCorrect))] = 2

        # Training cuts!
        trainingCutL = np.array(branches['TrainingCutL']).astype('float64')[isTrainingLink == 1].reshape(-1, nLinks)
        trainingCutT = np.array(branches['TrainingCutT']).astype('float64')[isTrainingLink == 1].reshape(-1, nLinks)
        trainingCutSep = copy.deepcopy(separation3D)
        trainingCutDoesConnect = np.isclose(doesChildConnect, 1.0).astype('int')
        
        # Reduce parent-child relationship vars to 1D arrays
        parentTrackScore = parentTrackScore[:,1]
        childTrackScore = childTrackScore[:,1]
        parentNSpacepoints = parentNSpacepoints[:,1]
        childNSpacepoints = childNSpacepoints[:,1]
        separation3D = separation3D[:,1]        
        trueParentChildLink = trueParentChildLink[:,1]
        trueChildVisibleGeneration = trueChildVisibleGeneration[:,1]            
        trainingCutL = trainingCutL[isLinkOrientationCorrect == 1]
        trainingCutT = trainingCutT[isLinkOrientationCorrect == 1]
        trainingCutSep = trainingCutSep[isLinkOrientationCorrect == 1]
        trainingCutDoesConnect = trainingCutDoesConnect[isLinkOrientationCorrect == 1]
        
        # How many entries do we have?      
        nEntries = trueParentChildLink.shape[0]   
        print('We have ', str(nEntries), ' entries to train on!')
        
        # Shuffle links for each entry
        randomIndices = np.random.rand(*(nEntries, nLinks)).argsort(axis=1)
        parentNuVertexSep = np.take_along_axis(parentNuVertexSep, randomIndices, axis=1)        
        childNuVertexSep = np.take_along_axis(childNuVertexSep, randomIndices, axis=1)
        parentEndRegionNHits = np.take_along_axis(parentEndRegionNHits, randomIndices, axis=1)
        parentEndRegionNParticles = np.take_along_axis(parentEndRegionNParticles, randomIndices, axis=1)
        parentEndRegionRToWall = np.take_along_axis(parentEndRegionRToWall, randomIndices, axis=1)
        vertexSeparation = np.take_along_axis(vertexSeparation, randomIndices, axis=1)
        doesChildConnect = np.take_along_axis(doesChildConnect, randomIndices, axis=1)
        overshootStartDCA = np.take_along_axis(overshootStartDCA, randomIndices, axis=1)
        overshootStartL = np.take_along_axis(overshootStartL, randomIndices, axis=1)
        overshootEndDCA = np.take_along_axis(overshootEndDCA, randomIndices, axis=1)
        overshootEndL = np.take_along_axis(overshootEndL, randomIndices, axis=1)
        childConnectionDCA = np.take_along_axis(childConnectionDCA, randomIndices, axis=1)
        childConnectionExtrapDistance = np.take_along_axis(childConnectionExtrapDistance, randomIndices, axis=1)
        childConnectionLRatio = np.take_along_axis(childConnectionLRatio, randomIndices, axis=1)
        parentConnectionPointNUpstreamHits = np.take_along_axis(parentConnectionPointNUpstreamHits, randomIndices, axis=1)
        parentConnectionPointNDownstreamHits = np.take_along_axis(parentConnectionPointNDownstreamHits, randomIndices, axis=1)
        parentConnectionPointNHitRatio = np.take_along_axis(parentConnectionPointNHitRatio, randomIndices, axis=1)
        parentConnectionPointEigenValueRatio = np.take_along_axis(parentConnectionPointEigenValueRatio, randomIndices, axis=1)
        parentConnectionPointOpeningAngle = np.take_along_axis(parentConnectionPointOpeningAngle, randomIndices, axis=1)
        isParentPOIClosestToNu = np.take_along_axis(isParentPOIClosestToNu, randomIndices, axis=1)
        isChildPOIClosestToNu = np.take_along_axis(isChildPOIClosestToNu, randomIndices, axis=1)
        y = np.take_along_axis(y, randomIndices, axis=1)
        
        if (normalise) :
            Utilities.normaliseXAxis(parentTrackScore, Utilities.parentTrackScore_min, Utilities.parentTrackScore_max)
            Utilities.normaliseXAxis(childTrackScore, Utilities.parentTrackScore_min, Utilities.parentTrackScore_max)    
            Utilities.normaliseXAxis(parentNSpacepoints, Utilities.parentNSpacepoints_min, Utilities.parentNSpacepoints_max)   
            Utilities.normaliseXAxis(childNSpacepoints, Utilities.parentNSpacepoints_min, Utilities.parentNSpacepoints_max) 
            Utilities.normaliseXAxis(parentNuVertexSep, Utilities.parentNuVertexSeparation_min, Utilities.parentNuVertexSeparation_max) 
            Utilities.normaliseXAxis(childNuVertexSep, Utilities.parentNuVertexSeparation_min, Utilities.parentNuVertexSeparation_max)        
            Utilities.normaliseXAxis(separation3D, Utilities.separation3D_min, Utilities.separation3D_max)                                                  
            Utilities.normaliseXAxis(parentEndRegionNHits, Utilities.parentEndRegionNHits_min, Utilities.parentEndRegionNHits_max)
            Utilities.normaliseXAxis(parentEndRegionNParticles, Utilities.parentEndRegionNParticles_min, Utilities.parentEndRegionNParticles_max)
            Utilities.normaliseXAxis(parentEndRegionRToWall, Utilities.parentEndRegionRToWall_min, Utilities.parentEndRegionRToWall_max)
            Utilities.normaliseXAxis(vertexSeparation, Utilities.vertexSeparation_min, Utilities.vertexSeparation_max)
            Utilities.normaliseXAxis(doesChildConnect, Utilities.doesChildConnect_min, Utilities.doesChildConnect_max)
            Utilities.normaliseXAxis(overshootStartDCA, Utilities.overshootDCA_min, Utilities.overshootDCA_max)
            Utilities.normaliseXAxis(overshootStartL, Utilities.overshootL_min, Utilities.overshootL_max)
            Utilities.normaliseXAxis(overshootEndDCA, Utilities.overshootDCA_min, Utilities.overshootDCA_max)
            Utilities.normaliseXAxis(overshootEndL, Utilities.overshootL_min, Utilities.overshootL_max)
            Utilities.normaliseXAxis(childConnectionDCA, Utilities.childConnectionDCA_min, Utilities.childConnectionDCA_max)
            Utilities.normaliseXAxis(childConnectionExtrapDistance, Utilities.childConnectionExtrapDistance_min, Utilities.childConnectionExtrapDistance_max)
            Utilities.normaliseXAxis(childConnectionLRatio, Utilities.childConnectionLRatio_min, Utilities.childConnectionLRatio_max)
            Utilities.normaliseXAxis(parentConnectionPointNUpstreamHits, Utilities.parentConnectionPointNUpstreamHits_min, Utilities.parentConnectionPointNUpstreamHits_max)
            Utilities.normaliseXAxis(parentConnectionPointNDownstreamHits, Utilities.parentConnectionPointNDownstreamHits_min, Utilities.parentConnectionPointNDownstreamHits_max)
            Utilities.normaliseXAxis(parentConnectionPointNHitRatio, Utilities.parentConnectionPointNHitRatio_min, Utilities.parentConnectionPointNHitRatio_max)
            Utilities.normaliseXAxis(parentConnectionPointEigenValueRatio, Utilities.parentConnectionPointEigenValueRatio_min, Utilities.parentConnectionPointEigenValueRatio_max)
            Utilities.normaliseXAxis(parentConnectionPointOpeningAngle, Utilities.parentConnectionPointOpeningAngle_min, Utilities.parentConnectionPointOpeningAngle_max)         
                
        # Prepare output
        variables = np.concatenate((parentTrackScore.reshape(nEntries, 1), \
                                    childTrackScore.reshape(nEntries, 1), \
                                    parentNSpacepoints.reshape(nEntries, 1), \
                                    childNSpacepoints.reshape(nEntries, 1), \
                                    separation3D.reshape(nEntries, 1)), axis=1)
        
        for i in range(nLinks) :
            edge_vars = np.concatenate((parentNuVertexSep[:, i].reshape(-1, 1), \
                                        childNuVertexSep[:, i].reshape(-1, 1), \
                                        parentEndRegionNHits[:, i].reshape(-1, 1), \
                                        parentEndRegionNParticles[:, i].reshape(-1, 1), \
                                        parentEndRegionRToWall[:, i].reshape(-1, 1), \
                                        vertexSeparation[:, i].reshape(-1, 1), \
                                        doesChildConnect[:, i].reshape(-1, 1), \
                                        overshootStartDCA[:, i].reshape(-1, 1), \
                                        overshootStartL[:, i].reshape(-1, 1), \
                                        overshootEndDCA[:, i].reshape(-1, 1), \
                                        overshootEndL[:, i].reshape(-1, 1), \
                                        childConnectionDCA[:, i].reshape(-1, 1), \
                                        childConnectionExtrapDistance[:, i].reshape(-1, 1), \
                                        childConnectionLRatio[:, i].reshape(-1, 1), \
                                        parentConnectionPointNUpstreamHits[:, i].reshape(-1, 1), \
                                        parentConnectionPointNDownstreamHits[:, i].reshape(-1, 1), \
                                        parentConnectionPointNHitRatio[:, i].reshape(-1, 1), \
                                        parentConnectionPointEigenValueRatio[:, i].reshape(-1, 1), \
                                        parentConnectionPointOpeningAngle[:, i].reshape(-1, 1), \
                                        isParentPOIClosestToNu[:, i].reshape(-1, 1), \
                                        isChildPOIClosestToNu[:, i].reshape(-1, 1)), axis=1)
        
            variables = np.concatenate((variables, edge_vars), axis=1)
            
    return nEntries, variables, y, trueParentChildLink, trueChildVisibleGeneration, trainingCutSep, trainingCutDoesConnect, trainingCutL, trainingCutT

############################################################################################################################################

def ReadTreeForValidation(isTrackMode, fileName, normalise) :
        
    if (isTrackMode) :
        nLinks = 4
        treeName = "LaterTierTrackTrackTree"
    else :
        nLinks = 2
        treeName = "LaterTierTrackShowerTree"
    
    with uproot.open(f"{fileName}:{treeName}") as tree:

        branches = tree.arrays()

        # ID stuff
        childID = np.array(branches["ChildParticleID"]).astype('int').reshape(-1, nLinks)
        parentID = np.array(branches["ParentParticleID"]).astype('int').reshape(-1, nLinks)
        # Vars (put astype as they are in tree)
        parentTrackScore = np.array(branches['ParentTrackScore']).astype('float64').reshape(-1, nLinks)            
        childTrackScore = np.array(branches['ChildTrackScore']).astype('float64').reshape(-1, nLinks)
        parentNSpacepoints = np.array(branches['ParentNSpacepoints']).astype('float64').reshape(-1, nLinks)
        childNSpacepoints = np.array(branches['ChildNSpacepoints']).astype('float64').reshape(-1, nLinks)
        separation3D = np.array(branches['Separation3D']).astype('float64').reshape(-1, nLinks)
        parentNuVertexSep = np.array(branches['ParentNuVertexSep']).astype('float64').reshape(-1, nLinks)
        childNuVertexSep = np.array(branches['ChildNuVertexSep']).astype('float64').reshape(-1, nLinks)                        
        parentEndRegionNHits = np.array(branches['ParentEndRegionNHits']).astype('float64').reshape(-1, nLinks)
        parentEndRegionNParticles = np.array(branches['ParentEndRegionNParticles']).astype('float64').reshape(-1, nLinks)
        parentEndRegionRToWall = np.array(branches['ParentEndRegionRToWall']).astype('float64').reshape(-1, nLinks)
        vertexSeparation = np.array(branches['VertexSeparation']).astype('float64').reshape(-1, nLinks)        
        doesChildConnect = np.array(branches['DoesChildConnect']).astype('float64').reshape(-1, nLinks)
        overshootStartDCA = np.array(branches['OvershootStartDCA']).astype('float64').reshape(-1, nLinks)
        overshootStartL = np.array(branches['OvershootStartL']).astype('float64').reshape(-1, nLinks)
        overshootEndDCA = np.array(branches['OvershootEndDCA']).astype('float64').reshape(-1, nLinks)
        overshootEndL = np.array(branches['OvershootEndL']).astype('float64').reshape(-1, nLinks)
        childConnectionDCA = np.array(branches['ChildCPDCA']).astype('float64').reshape(-1, nLinks)
        childConnectionExtrapDistance = np.array(branches['ChildCPExtrapDistance']).astype('float64').reshape(-1, nLinks)
        childConnectionLRatio = np.array(branches['ChildCPLRatio']).astype('float64').reshape(-1, nLinks)
        parentConnectionPointNUpstreamHits = np.array(branches['ParentCPNUpstreamHits']).astype('float64').reshape(-1, nLinks)
        parentConnectionPointNDownstreamHits = np.array(branches['ParentCPNDownstreamHits']).astype('float64').reshape(-1, nLinks)
        parentConnectionPointNHitRatio = np.array(branches['ParentCPNHitRatio']).astype('float64').reshape(-1, nLinks)
        parentConnectionPointEigenValueRatio = np.array(branches['ParentCPEigenvalueRatio']).astype('float64').reshape(-1, nLinks)
        parentConnectionPointOpeningAngle = np.array(branches['ParentCPOpeningAngle']).astype('float64').reshape(-1, nLinks)
        isParentPOIClosestToNu = np.array(branches['ParentIsPOIClosestToNu']).astype('float64').reshape(-1, nLinks)
        isChildPOIClosestToNu = np.array(branches['ChildIsPOIClosestToNu']).astype('float64').reshape(-1, nLinks)
        
        # Reduce parent-child relationship vars to 1D arrays
        childID = childID[:,0]
        parentID = parentID[:,0]
        parentTrackScore = parentTrackScore[:,0]
        childTrackScore = childTrackScore[:,0]
        parentNSpacepoints = parentNSpacepoints[:,0]
        childNSpacepoints = childNSpacepoints[:,0]
        separation3D = separation3D[:,0]
        
        # Need to retain some not normalised copys
        minNuVertexSep_notNorm = np.min(childNuVertexSep, axis=1)
        separation3D_notNorm = copy.deepcopy(separation3D)
        
        # How many entries do we have?      
        nEntries = childID.shape[0]   
        print('We have ', str(nEntries), ' entries to train on!')
        
        if (normalise) :
            Utilities.normaliseXAxis(parentTrackScore, Utilities.parentTrackScore_min, Utilities.parentTrackScore_max)
            Utilities.normaliseXAxis(childTrackScore, Utilities.parentTrackScore_min, Utilities.parentTrackScore_max)    
            Utilities.normaliseXAxis(parentNSpacepoints, Utilities.parentNSpacepoints_min, Utilities.parentNSpacepoints_max)   
            Utilities.normaliseXAxis(childNSpacepoints, Utilities.parentNSpacepoints_min, Utilities.parentNSpacepoints_max) 
            Utilities.normaliseXAxis(parentNuVertexSep, Utilities.parentNuVertexSeparation_min, Utilities.parentNuVertexSeparation_max) 
            Utilities.normaliseXAxis(childNuVertexSep, Utilities.parentNuVertexSeparation_min, Utilities.parentNuVertexSeparation_max)        
            Utilities.normaliseXAxis(separation3D, Utilities.separation3D_min, Utilities.separation3D_max)                                                  
            Utilities.normaliseXAxis(parentEndRegionNHits, Utilities.parentEndRegionNHits_min, Utilities.parentEndRegionNHits_max)
            Utilities.normaliseXAxis(parentEndRegionNParticles, Utilities.parentEndRegionNParticles_min, Utilities.parentEndRegionNParticles_max)
            Utilities.normaliseXAxis(parentEndRegionRToWall, Utilities.parentEndRegionRToWall_min, Utilities.parentEndRegionRToWall_max)
            Utilities.normaliseXAxis(vertexSeparation, Utilities.vertexSeparation_min, Utilities.vertexSeparation_max)
            Utilities.normaliseXAxis(doesChildConnect, Utilities.doesChildConnect_min, Utilities.doesChildConnect_max)
            Utilities.normaliseXAxis(overshootStartDCA, Utilities.overshootDCA_min, Utilities.overshootDCA_max)
            Utilities.normaliseXAxis(overshootStartL, Utilities.overshootL_min, Utilities.overshootL_max)
            Utilities.normaliseXAxis(overshootEndDCA, Utilities.overshootDCA_min, Utilities.overshootDCA_max)
            Utilities.normaliseXAxis(overshootEndL, Utilities.overshootL_min, Utilities.overshootL_max)
            Utilities.normaliseXAxis(childConnectionDCA, Utilities.childConnectionDCA_min, Utilities.childConnectionDCA_max)
            Utilities.normaliseXAxis(childConnectionExtrapDistance, Utilities.childConnectionExtrapDistance_min, Utilities.childConnectionExtrapDistance_max)
            Utilities.normaliseXAxis(childConnectionLRatio, Utilities.childConnectionLRatio_min, Utilities.childConnectionLRatio_max)
            Utilities.normaliseXAxis(parentConnectionPointNUpstreamHits, Utilities.parentConnectionPointNUpstreamHits_min, Utilities.parentConnectionPointNUpstreamHits_max)
            Utilities.normaliseXAxis(parentConnectionPointNDownstreamHits, Utilities.parentConnectionPointNDownstreamHits_min, Utilities.parentConnectionPointNDownstreamHits_max)
            Utilities.normaliseXAxis(parentConnectionPointNHitRatio, Utilities.parentConnectionPointNHitRatio_min, Utilities.parentConnectionPointNHitRatio_max)
            Utilities.normaliseXAxis(parentConnectionPointEigenValueRatio, Utilities.parentConnectionPointEigenValueRatio_min, Utilities.parentConnectionPointEigenValueRatio_max)
            Utilities.normaliseXAxis(parentConnectionPointOpeningAngle, Utilities.parentConnectionPointOpeningAngle_min, Utilities.parentConnectionPointOpeningAngle_max)         
                
        # Prepare output
        variables = np.concatenate((parentTrackScore.reshape(nEntries, 1), \
                                    childTrackScore.reshape(nEntries, 1), \
                                    parentNSpacepoints.reshape(nEntries, 1), \
                                    childNSpacepoints.reshape(nEntries, 1), \
                                    separation3D.reshape(nEntries, 1)), axis=1)
        
        for i in range(nLinks) :
            edge_vars = np.concatenate((parentNuVertexSep[:, i].reshape(-1, 1), \
                                        childNuVertexSep[:, i].reshape(-1, 1), \
                                        parentEndRegionNHits[:, i].reshape(-1, 1), \
                                        parentEndRegionNParticles[:, i].reshape(-1, 1), \
                                        parentEndRegionRToWall[:, i].reshape(-1, 1), \
                                        vertexSeparation[:, i].reshape(-1, 1), \
                                        doesChildConnect[:, i].reshape(-1, 1), \
                                        overshootStartDCA[:, i].reshape(-1, 1), \
                                        overshootStartL[:, i].reshape(-1, 1), \
                                        overshootEndDCA[:, i].reshape(-1, 1), \
                                        overshootEndL[:, i].reshape(-1, 1), \
                                        childConnectionDCA[:, i].reshape(-1, 1), \
                                        childConnectionExtrapDistance[:, i].reshape(-1, 1), \
                                        childConnectionLRatio[:, i].reshape(-1, 1), \
                                        parentConnectionPointNUpstreamHits[:, i].reshape(-1, 1), \
                                        parentConnectionPointNDownstreamHits[:, i].reshape(-1, 1), \
                                        parentConnectionPointNHitRatio[:, i].reshape(-1, 1), \
                                        parentConnectionPointEigenValueRatio[:, i].reshape(-1, 1), \
                                        parentConnectionPointOpeningAngle[:, i].reshape(-1, 1), \
                                        isParentPOIClosestToNu[:, i].reshape(-1, 1), \
                                        isChildPOIClosestToNu[:, i].reshape(-1, 1)), axis=1)
        
            variables = np.concatenate((variables, edge_vars), axis=1)
            
    return nEntries, childID, parentID, variables, minNuVertexSep_notNorm, separation3D_notNorm