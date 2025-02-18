import numpy as np
import matplotlib.pyplot as plt

BOGUS_INT = -999

####################################################################################################################################### 
#######################################################################################################################################

def calculateHierarchyMetrics_trainValidation(nEntries, particleMask_in, trackShowerScore_in, trueVisibleGeneration_in, trueVisibleParentID_in, new_parentID, new_gen) :
    for isTrack in [True, False] :
        n_two_d = 0
        n_true_primary = 0
        n_true_secondary = 0
        n_true_tertiary = 0
        n_true_higher = 0

        # NEW!
        n_correct_parent_correct_tier_primary = 0
        n_correct_parent_wrong_tier_primary = 0
        n_tagged_as_primary_primary = 0
        n_incorrect_parent_primary = 0
        n_not_tagged_primary = 0 

        n_correct_parent_correct_tier_secondary = 0
        n_correct_parent_wrong_tier_secondary = 0
        n_tagged_as_primary_secondary = 0
        n_incorrect_parent_secondary = 0
        n_not_tagged_secondary = 0 

        n_correct_parent_correct_tier_tertiary = 0
        n_correct_parent_wrong_tier_tertiary = 0
        n_correct_parent_tertiary = 0
        n_tagged_as_primary_tertiary = 0
        n_incorrect_parent_tertiary = 0
        n_not_tagged_tertiary = 0 

        n_correct_parent_correct_tier_higher = 0
        n_correct_parent_wrong_tier_higher = 0
        n_correct_parent_higher = 0
        n_tagged_as_primary_higher = 0
        n_incorrect_parent_higher = 0
        n_not_tagged_higher = 0 

        for iEvent in range(nEntries) : 

            # Particle mask
            particle_mask = np.array(particleMask_in[iEvent])
            # PFP info
            trackShowerScore_np = np.array(trackShowerScore_in[iEvent])[particle_mask]
            # Truth
            trueVisibleGeneration_np = np.array(trueVisibleGeneration_in[iEvent])[particle_mask]
            trueVisibleParentID_np = np.array(trueVisibleParentID_in[iEvent])[particle_mask]
            # New
            newParentID_np = np.array(new_parentID[iEvent])[particle_mask]
            newGen_np = np.array(new_gen[iEvent])[particle_mask]

            #########################
            # Get tier masks
            #########################
            trackShower_mask = (trackShowerScore_np > 0.5) if isTrack else np.logical_not(trackShowerScore_np > 0.5)
            true_primary_mask = np.logical_and(trackShower_mask, trueVisibleGeneration_np == 2)
            true_secondary_mask = np.logical_and(trackShower_mask, trueVisibleGeneration_np == 3)
            true_tertiary_mask = np.logical_and(trackShower_mask, trueVisibleGeneration_np == 4)
            true_higher_mask = np.logical_and(trackShower_mask, np.logical_not(np.logical_or(true_primary_mask, np.logical_or(true_secondary_mask, true_tertiary_mask))))

            #############################################
            # Get metrics for this event - debugging
            #############################################
            # Totals
            this_two_d = 0
            this_true_primary = np.count_nonzero(true_primary_mask)
            this_true_secondary = np.count_nonzero(true_secondary_mask)
            this_true_tertiary = np.count_nonzero(true_tertiary_mask)
            this_true_higher = np.count_nonzero(true_higher_mask)

            # Primary
            this_correct_parent_correct_tier_primary = np.count_nonzero(newGen_np[true_primary_mask] == 2)
            this_correct_parent_wrong_tier_primary = 0
            this_tagged_as_primary_primary = 0
            this_not_tagged_primary = np.count_nonzero(newGen_np[true_primary_mask] == BOGUS_INT)
            this_incorrect_parent_primary = np.count_nonzero(np.logical_and(newGen_np[true_primary_mask] != 2, \
                                                                            newGen_np[true_primary_mask] != BOGUS_INT)) 
            # Secondary
            this_correct_parent_correct_tier_secondary = np.count_nonzero(np.logical_and(newParentID_np[true_secondary_mask] == trueVisibleParentID_np[true_secondary_mask], \
                                                                                         newGen_np[true_secondary_mask] == 3))
            this_correct_parent_wrong_tier_secondary = np.count_nonzero(np.logical_and(newParentID_np[true_secondary_mask] == trueVisibleParentID_np[true_secondary_mask], \
                                                                                       np.logical_and(newGen_np[true_secondary_mask] != 3, \
                                                                                                      newGen_np[true_secondary_mask] != BOGUS_INT)))
            this_tagged_as_primary_secondary = np.count_nonzero(newGen_np[true_secondary_mask] == 2)
            this_not_tagged_secondary = np.count_nonzero(newGen_np[true_secondary_mask] == BOGUS_INT)
            this_incorrect_parent_secondary = np.count_nonzero(np.logical_not(np.logical_or(newParentID_np[true_secondary_mask] == trueVisibleParentID_np[true_secondary_mask], \
                                                                                            np.logical_or(newGen_np[true_secondary_mask] == 2, \
                                                                                                          newGen_np[true_secondary_mask] == BOGUS_INT))))
            # Tertiary
            this_correct_parent_correct_tier_tertiary = np.count_nonzero(np.logical_and(newParentID_np[true_tertiary_mask] == trueVisibleParentID_np[true_tertiary_mask], \
                                                                                        newGen_np[true_tertiary_mask] == 4))
            this_correct_parent_wrong_tier_tertiary = np.count_nonzero(np.logical_and(newParentID_np[true_tertiary_mask] == trueVisibleParentID_np[true_tertiary_mask], \
                                                                                      np.logical_and(newGen_np[true_tertiary_mask] != 4, \
                                                                                                     newGen_np[true_tertiary_mask] != BOGUS_INT)))
            this_tagged_as_primary_tertiary = np.count_nonzero(newGen_np[true_tertiary_mask] == 2)
            this_not_tagged_tertiary = np.count_nonzero(newGen_np[true_tertiary_mask] == BOGUS_INT)
            this_incorrect_parent_tertiary = np.count_nonzero(np.logical_not(np.logical_or(newParentID_np[true_tertiary_mask] == trueVisibleParentID_np[true_tertiary_mask], \
                                                                                           np.logical_or(newGen_np[true_tertiary_mask] == 2, \
                                                                                                         newGen_np[true_tertiary_mask] == BOGUS_INT))))
            # Higher
            this_correct_parent_correct_tier_higher = 0
            this_correct_parent_wrong_tier_higher = np.count_nonzero(newParentID_np[true_higher_mask] == trueVisibleParentID_np[true_higher_mask])
            this_tagged_as_primary_higher = np.count_nonzero(newGen_np[true_higher_mask] == 2)
            this_not_tagged_higher = np.count_nonzero(newGen_np[true_higher_mask] == BOGUS_INT)
            this_incorrect_parent_higher = np.count_nonzero(np.logical_not(np.logical_or(newParentID_np[true_higher_mask] == trueVisibleParentID_np[true_higher_mask], \
                                                                                         np.logical_or(newGen_np[true_higher_mask] == 2, \
                                                                                                       newGen_np[true_higher_mask] == BOGUS_INT))))


            #############################################
            # Add metrics to global
            #############################################
            n_two_d += this_two_d
            n_true_primary += this_true_primary
            n_true_secondary += this_true_secondary
            n_true_tertiary += this_true_tertiary
            n_true_higher += this_true_higher

            n_correct_parent_correct_tier_primary += this_correct_parent_correct_tier_primary
            n_correct_parent_wrong_tier_primary += this_correct_parent_wrong_tier_primary
            n_tagged_as_primary_primary += this_tagged_as_primary_primary
            n_incorrect_parent_primary += this_incorrect_parent_primary
            n_not_tagged_primary += this_not_tagged_primary
            n_correct_parent_correct_tier_secondary += this_correct_parent_correct_tier_secondary
            n_correct_parent_wrong_tier_secondary += this_correct_parent_wrong_tier_secondary
            n_tagged_as_primary_secondary += this_tagged_as_primary_secondary
            n_incorrect_parent_secondary += this_incorrect_parent_secondary
            n_not_tagged_secondary += this_not_tagged_secondary
            n_correct_parent_correct_tier_tertiary += this_correct_parent_correct_tier_tertiary
            n_correct_parent_wrong_tier_tertiary += this_correct_parent_wrong_tier_tertiary
            n_tagged_as_primary_tertiary += this_tagged_as_primary_tertiary
            n_incorrect_parent_tertiary += this_incorrect_parent_tertiary
            n_not_tagged_tertiary += this_not_tagged_tertiary
            n_correct_parent_correct_tier_higher += this_correct_parent_correct_tier_higher
            n_correct_parent_wrong_tier_higher += this_correct_parent_wrong_tier_higher
            n_tagged_as_primary_higher += this_tagged_as_primary_higher
            n_incorrect_parent_higher += this_incorrect_parent_higher
            n_not_tagged_higher += this_not_tagged_higher

        #############################################
        # Print metrics
        #############################################   

        n_correct_parent_correct_tier_primary_frac = round(0.0 if n_true_primary == 0 else float(n_correct_parent_correct_tier_primary) / float(n_true_primary), 2)
        n_correct_parent_wrong_tier_primary_frac = round(0.0 if n_true_primary == 0 else float(n_correct_parent_wrong_tier_primary) / float(n_true_primary), 2)
        n_tagged_as_primary_primary_frac = round(0.0 if n_true_primary == 0 else float(n_tagged_as_primary_primary) / float(n_true_primary), 2)
        n_incorrect_parent_primary_frac = round(0.0 if n_true_primary == 0 else float(n_incorrect_parent_primary) / float(n_true_primary), 2)
        n_not_tagged_primary_frac = round(0.0 if n_true_primary == 0 else float(n_not_tagged_primary) / float(n_true_primary), 2)

        n_correct_parent_correct_tier_secondary_frac = round(0.0 if n_true_secondary == 0 else float(n_correct_parent_correct_tier_secondary) / float(n_true_secondary), 2)
        n_correct_parent_wrong_tier_secondary_frac = round(0.0 if n_true_secondary == 0 else float(n_correct_parent_wrong_tier_secondary) / float(n_true_secondary), 2)
        n_tagged_as_primary_secondary_frac = round(0.0 if n_true_secondary == 0 else float(n_tagged_as_primary_secondary) / float(n_true_secondary), 2)
        n_incorrect_parent_secondary_frac = round(0.0 if n_true_secondary == 0 else float(n_incorrect_parent_secondary) / float(n_true_secondary), 2)
        n_not_tagged_secondary_frac = round(0.0 if n_true_secondary == 0 else float(n_not_tagged_secondary) / float(n_true_secondary), 2)

        n_correct_parent_correct_tier_tertiary_frac = round(0.0 if n_true_tertiary == 0 else float(n_correct_parent_correct_tier_tertiary) / float(n_true_tertiary), 2)
        n_correct_parent_wrong_tier_tertiary_frac = round(0.0 if n_true_tertiary == 0 else float(n_correct_parent_wrong_tier_tertiary) / float(n_true_tertiary), 2)
        n_tagged_as_primary_tertiary_frac = round(0.0 if n_true_tertiary == 0 else float(n_tagged_as_primary_tertiary) / float(n_true_tertiary), 2)
        n_incorrect_parent_tertiary_frac = round(0.0 if n_true_tertiary == 0 else float(n_incorrect_parent_tertiary) / float(n_true_tertiary), 2)
        n_not_tagged_tertiary_frac = round(0.0 if n_true_tertiary == 0 else float(n_not_tagged_tertiary) / float(n_true_tertiary), 2)

        n_correct_parent_correct_tier_higher_frac = round(0.0 if n_true_higher == 0 else float(n_correct_parent_correct_tier_higher) / float(n_true_higher), 2)
        n_correct_parent_wrong_tier_higher_frac = round(0.0 if n_true_higher == 0 else float(n_correct_parent_wrong_tier_higher) / float(n_true_higher), 2)
        n_tagged_as_primary_higher_frac = round(0.0 if n_true_higher == 0 else float(n_tagged_as_primary_higher) / float(n_true_higher), 2)
        n_incorrect_parent_higher_frac = round(0.0 if n_true_higher == 0 else float(n_incorrect_parent_higher) / float(n_true_higher), 2)
        n_not_tagged_higher_frac = round(0.0 if n_true_higher == 0 else float(n_not_tagged_higher) / float(n_true_higher), 2)


        print('------------------------------------------------------------')
        print(('TRACK' if isTrack else 'SHOWER'))
        print('------------------------------------------------------------')
        print('NEW - True Gen   | Primary | Secondary | Tertiary | Higher |')
        print('------------------------------------------------------------')
        print('Correct parent CT |' + str(n_correct_parent_correct_tier_primary_frac) + str(' '* (9 - len(str(n_correct_parent_correct_tier_primary_frac)))) + \
                                '|' + str(n_correct_parent_correct_tier_secondary_frac) + str(' '* (11 - len(str(n_correct_parent_correct_tier_secondary_frac)))) + \
                                '|' + str(n_correct_parent_correct_tier_tertiary_frac) + str(' '* (10 - len(str(n_correct_parent_correct_tier_tertiary_frac)))) + \
                                '|' + str(n_correct_parent_correct_tier_higher_frac) + str(' '* (8 - len(str(n_correct_parent_correct_tier_higher_frac)))) + \
                                '|')
        print('Correct parent WT |' + str(n_correct_parent_wrong_tier_primary_frac) + str(' '* (9 - len(str(n_correct_parent_wrong_tier_primary_frac)))) + \
                                '|' + str(n_correct_parent_wrong_tier_secondary_frac) + str(' '* (11 - len(str(n_correct_parent_wrong_tier_secondary_frac)))) + \
                                '|' + str(n_correct_parent_wrong_tier_tertiary_frac) + str(' '* (10 - len(str(n_correct_parent_wrong_tier_tertiary_frac)))) + \
                                '|' + str(n_correct_parent_wrong_tier_higher_frac) + str(' '* (8 - len(str(n_correct_parent_wrong_tier_higher_frac)))) + \
                                '|')
        print('False primary     |' + str(n_tagged_as_primary_primary_frac) + str(' '* (9 - len(str(n_tagged_as_primary_primary_frac)))) + \
                                '|' + str(n_tagged_as_primary_secondary_frac) + str(' '* (11 - len(str(n_tagged_as_primary_secondary_frac)))) + \
                                '|' + str(n_tagged_as_primary_tertiary_frac) + str(' '* (10 - len(str(n_tagged_as_primary_tertiary_frac)))) + \
                                '|' + str(n_tagged_as_primary_higher_frac) + str(' '* (8 - len(str(n_tagged_as_primary_higher_frac)))) + \
                                '|')
        print('Incorrect parent  |' + str(n_incorrect_parent_primary_frac) + str(' '* (9 - len(str(n_incorrect_parent_primary_frac)))) + \
                                '|' + str(n_incorrect_parent_secondary_frac) + str(' '* (11 - len(str(n_incorrect_parent_secondary_frac)))) + \
                                '|' + str(n_incorrect_parent_tertiary_frac) + str(' '* (10 - len(str(n_incorrect_parent_tertiary_frac)))) + \
                                '|' + str(n_incorrect_parent_higher_frac) + str(' '* (8 - len(str(n_incorrect_parent_higher_frac)))) + \
                                '|')
        print('Not tagged        |' + str(n_not_tagged_primary_frac) + str(' '* (9 - len(str(n_not_tagged_primary_frac)))) + \
                                '|' + str(n_not_tagged_secondary_frac) + str(' '* (11 - len(str(n_not_tagged_secondary_frac)))) + \
                                '|' + str(n_not_tagged_tertiary_frac) + str(' '* (10 - len(str(n_not_tagged_tertiary_frac)))) + \
                                '|' + str(n_not_tagged_higher_frac) + str(' '* (8 - len(str(n_not_tagged_higher_frac)))) + \
                                '|')
        print('------------------------------------------------------------')
        print('Total             |' + str(n_true_primary) + str(' '* (9 - len(str(n_true_primary)))) + \
                                '|' + str(n_true_secondary) + str(' '* (11 - len(str(n_true_secondary)))) + \
                                '|' + str(n_true_tertiary) + str(' '* (10 - len(str(n_true_tertiary)))) + \
                                '|' + str(n_true_higher) + str(' '* (8 - len(str(n_true_higher)))) + \
                                '|')
        print('n_two_d (not included in metrics):', n_two_d)
        print('------------------------------------------------------------')
        print('')
        
####################################################################################################################################### 
#######################################################################################################################################

