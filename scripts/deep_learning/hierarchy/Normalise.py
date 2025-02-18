import math

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



    
    
    

