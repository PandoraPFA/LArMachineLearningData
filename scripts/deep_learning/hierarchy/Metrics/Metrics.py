import numpy as np
import matplotlib.pyplot as plt

BOGUS_INT = -999

####################################################################################################################################### 
#######################################################################################################################################

def calculateHierarchyMetrics_trainValidation(particleMask_in, trackShowerScore_in, trueVisibleGeneration_in, trueVisibleParentID_in, new_parentID, new_gen) :
    
    for isTrack in [True, False] :
        
        if isTrack :
            trackShowerMask = np.logical_and(particleMask_in, trackShowerScore_in > 0.5)
        else :
            trackShowerMask = np.logical_and(particleMask_in, trackShowerScore_in < 0.5)
            
        # Get masks
        true_primary_mask = np.logical_and(trackShowerMask, trueVisibleGeneration_in == 2)
        true_secondary_mask = np.logical_and(trackShowerMask, trueVisibleGeneration_in == 3)
        true_tertiary_mask = np.logical_and(trackShowerMask, trueVisibleGeneration_in == 4)
        true_higher_mask = np.logical_and(trackShowerMask, trueVisibleGeneration_in > 4)
        
        # Do counts  
        n_true_primary = np.count_nonzero(true_primary_mask)
        n_true_secondary = np.count_nonzero(true_secondary_mask)
        n_true_tertiary = np.count_nonzero(true_tertiary_mask)
        n_true_higher = np.count_nonzero(true_higher_mask)

        n_correct_parent_correct_tier_primary = np.count_nonzero(new_gen[true_primary_mask] == 2)
        n_correct_parent_wrong_tier_primary = 0
        n_tagged_as_primary_primary = 0
        n_not_tagged_primary = np.count_nonzero(new_gen[true_primary_mask] == BOGUS_INT)
        n_incorrect_parent_primary = np.count_nonzero(np.logical_and(new_gen[true_primary_mask] != 2, \
                                                                     new_gen[true_primary_mask] != BOGUS_INT))
        n_correct_parent_correct_tier_secondary = np.count_nonzero(np.logical_and(new_parentID[true_secondary_mask] == trueVisibleParentID_in[true_secondary_mask], \
                                                                                  new_gen[true_secondary_mask] == 3))
        n_correct_parent_wrong_tier_secondary = np.count_nonzero(np.logical_and(new_parentID[true_secondary_mask] == trueVisibleParentID_in[true_secondary_mask], \
                                                                                np.logical_and(new_gen[true_secondary_mask] != 3, \
                                                                                               new_gen[true_secondary_mask] != BOGUS_INT)))
        n_tagged_as_primary_secondary = np.count_nonzero(new_gen[true_secondary_mask] == 2)
        n_not_tagged_secondary = np.count_nonzero(new_gen[true_secondary_mask] == BOGUS_INT)
        n_incorrect_parent_secondary = np.count_nonzero(np.logical_not(np.logical_or(new_parentID[true_secondary_mask] == trueVisibleParentID_in[true_secondary_mask], \
                                                                                     np.logical_or(new_gen[true_secondary_mask] == 2, \
                                                                                                   new_gen[true_secondary_mask] == BOGUS_INT))))
        n_correct_parent_correct_tier_tertiary = np.count_nonzero(np.logical_and(new_parentID[true_tertiary_mask] == trueVisibleParentID_in[true_tertiary_mask], \
                                                                                 new_gen[true_tertiary_mask] == 4))
        n_correct_parent_wrong_tier_tertiary = np.count_nonzero(np.logical_and(new_parentID[true_tertiary_mask] == trueVisibleParentID_in[true_tertiary_mask], \
                                                                               np.logical_and(new_gen[true_tertiary_mask] != 4, \
                                                                                              new_gen[true_tertiary_mask] != BOGUS_INT)))
        n_tagged_as_primary_tertiary = np.count_nonzero(new_gen[true_tertiary_mask] == 2)
        n_not_tagged_tertiary = np.count_nonzero(new_gen[true_tertiary_mask] == BOGUS_INT)
        n_incorrect_parent_tertiary = np.count_nonzero(np.logical_not(np.logical_or(new_parentID[true_tertiary_mask] == trueVisibleParentID_in[true_tertiary_mask], \
                                                                                    np.logical_or(new_gen[true_tertiary_mask] == 2, \
                                                                                                  new_gen[true_tertiary_mask] == BOGUS_INT))))
        n_correct_parent_correct_tier_higher = 0
        n_correct_parent_wrong_tier_higher = np.count_nonzero(new_parentID[true_higher_mask] == trueVisibleParentID_in[true_higher_mask])
        n_tagged_as_primary_higher = np.count_nonzero(new_gen[true_higher_mask] == 2)
        n_not_tagged_higher = np.count_nonzero(new_gen[true_higher_mask] == BOGUS_INT)
        n_incorrect_parent_higher = np.count_nonzero(np.logical_not(np.logical_or(new_parentID[true_higher_mask] == trueVisibleParentID_in[true_higher_mask], \
                                                                                  np.logical_or(new_gen[true_higher_mask] == 2, \
                                                                                                new_gen[true_higher_mask] == BOGUS_INT))))
        
        # Calc fractions
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
        print('------------------------------------------------------------')
        print('')
        
####################################################################################################################################### 
#######################################################################################################################################
         
def calculateHierarchyMetrics_leptonValidation(particleMask_in, trueVisibleGeneration_in, truePDG_in, new_gen) :

    # Get masks
    true_primary_mask = np.logical_and(particleMask_in, trueVisibleGeneration_in == 2)
    target_muon_mask = np.logical_and(true_primary_mask, np.abs(truePDG_in) == 13)
    target_proton_mask = np.logical_and(true_primary_mask, np.abs(truePDG_in) == 2212)
    target_pion_mask = np.logical_and(true_primary_mask, np.abs(truePDG_in) == 211)
    target_electron_mask = np.logical_and(true_primary_mask, np.abs(truePDG_in) == 11)
    target_photon_mask = np.logical_and(true_primary_mask, np.abs(truePDG_in) == 22)
     
    # Do sums
    n_true_muon = np.count_nonzero(target_muon_mask)
    n_true_electron = np.count_nonzero(target_electron_mask)
    n_true_pion = np.count_nonzero(target_pion_mask)
    n_true_photon = np.count_nonzero(target_photon_mask)
    n_true_proton = np.count_nonzero(target_proton_mask)
    n_tagged_as_primary_muon = np.count_nonzero(new_gen[target_muon_mask] == 2)
    n_incorrect_parent_muon = np.count_nonzero(new_gen[target_muon_mask] != 2)
    n_tagged_as_primary_electron = np.count_nonzero(new_gen[target_electron_mask] == 2)
    n_incorrect_parent_electron = np.count_nonzero(new_gen[target_electron_mask] != 2)
    n_tagged_as_primary_proton = np.count_nonzero(new_gen[target_proton_mask] == 2)
    n_incorrect_parent_proton = np.count_nonzero(new_gen[target_proton_mask] != 2)
    n_tagged_as_primary_pion = np.count_nonzero(new_gen[target_pion_mask] == 2)
    n_incorrect_parent_pion = np.count_nonzero(new_gen[target_pion_mask] != 2)
    n_tagged_as_primary_photon = np.count_nonzero(new_gen[target_photon_mask] == 2)
    n_incorrect_parent_photon = np.count_nonzero(new_gen[target_photon_mask] != 2)
         
    # Calc fractions
    n_tagged_as_primary_muon_frac = round(0.0 if n_true_muon == 0 else float(n_tagged_as_primary_muon) / float(n_true_muon), 2)
    n_incorrect_parent_muon_frac = round(0.0 if n_true_muon == 0 else float(n_incorrect_parent_muon) / float(n_true_muon), 2)
    n_tagged_as_primary_electron_frac = round(0.0 if n_true_electron == 0 else float(n_tagged_as_primary_electron) / float(n_true_electron), 2)
    n_incorrect_parent_electron_frac = round(0.0 if n_true_electron == 0 else float(n_incorrect_parent_electron) / float(n_true_electron), 2)
    n_tagged_as_primary_proton_frac = round(0.0 if n_true_proton == 0 else float(n_tagged_as_primary_proton) / float(n_true_proton), 2)
    n_incorrect_parent_proton_frac = round(0.0 if n_true_proton == 0 else float(n_incorrect_parent_proton) / float(n_true_proton), 2)
    n_tagged_as_primary_pion_frac = round(0.0 if n_true_pion == 0 else float(n_tagged_as_primary_pion) / float(n_true_pion), 2)
    n_incorrect_parent_pion_frac = round(0.0 if n_true_pion == 0 else float(n_incorrect_parent_pion) / float(n_true_pion), 2)
    n_tagged_as_primary_photon_frac = round(0.0 if n_true_photon == 0 else float(n_tagged_as_primary_photon) / float(n_true_photon), 2)
    n_incorrect_parent_photon_frac = round(0.0 if n_true_photon == 0 else float(n_incorrect_parent_photon) / float(n_true_photon), 2)

    print('')
    print('-------------------------------------------------------------')
    print('    True PDG     | Muon | Electron | Photon | Pion | Proton |')
    print('-------------------------------------------------------------')
    print('Correct primary  |' + str(n_tagged_as_primary_muon_frac) + str(' '* (6 - len(str(n_tagged_as_primary_muon_frac)))) + \
                           '|' + str(n_tagged_as_primary_electron_frac) + str(' '* (10 - len(str(n_tagged_as_primary_electron_frac)))) + \
                           '|' + str(n_tagged_as_primary_proton_frac) + str(' '* (8 - len(str(n_tagged_as_primary_proton_frac)))) + \
                           '|' + str(n_tagged_as_primary_pion_frac) + str(' '* (6 - len(str(n_tagged_as_primary_pion_frac)))) + \
                           '|' + str(n_tagged_as_primary_photon_frac) + str(' '* (8 - len(str(n_tagged_as_primary_photon_frac)))) + \
                           '|')
    print('Incorrect parent |' + str(n_incorrect_parent_muon_frac) + str(' '* (6 - len(str(n_incorrect_parent_muon_frac)))) + \
                           '|' + str(n_incorrect_parent_electron_frac) + str(' '* (10 - len(str(n_incorrect_parent_electron_frac)))) + \
                           '|' + str(n_incorrect_parent_proton_frac) + str(' '* (8 - len(str(n_incorrect_parent_proton_frac)))) + \
                           '|' + str(n_incorrect_parent_pion_frac) + str(' '* (6 - len(str(n_incorrect_parent_pion_frac)))) + \
                           '|' + str(n_incorrect_parent_photon_frac) + str(' '* (8 - len(str(n_incorrect_parent_photon_frac)))) + \
                           '|')
    print('-------------------------------------------------------------')
    print('Total             |' + str(n_true_muon) + str(' '* (6 - len(str(n_true_muon)))) + \
                            '|' + str(n_true_electron) + str(' '* (10 - len(str(n_true_electron)))) + \
                            '|' + str(n_true_proton) + str(' '* (8 - len(str(n_true_proton)))) + \
                            '|' + str(n_true_pion) + str(' '* (6 - len(str(n_true_pion)))) + \
                            '|' + str(n_true_photon) + str(' '* (8 - len(str(n_true_photon)))) + \
                            '|')
    print('-------------------------------------------------------------')
    print('')

####################################################################################################################################### 
#######################################################################################################################################        