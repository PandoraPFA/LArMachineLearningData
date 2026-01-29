import numpy as np
import pickle
import uproot
from input_track import InputTrack

def extract_tracks_from_root_file(input_file_path, tree_name, output_path, output_file_name, hits_cut, exit_only):
    """
        Function to extract tracks from the input root file and store them in a pickle file

        Args:
            input_file_path: location of the input root file
            tree_name: name of the ROOT tree inside the input file
            output_path: path where to write the pickle file
            output_file_name: file name of the output pickle file
            hits_cut: minimum number of hits per view to consider a track
            exit_only: a flag to keep only exiting tracks and all kaons
    """

    all_tracks = []
    
    with uproot.open(input_file_path+":"+tree_name) as tree:

        # Run in batches of 100 events and only load the branches that we need
        for b, batch in enumerate(tree.iterate(['uViewX', 'uViewWire', 'uViewQ', 'uViewNHits',
                                                'vViewX', 'vViewWire', 'vViewQ', 'vViewNHits',
                                                'wViewX', 'wViewWire', 'wViewQ', 'wViewNHits',
                                                'truePdg', 'nTrackChildren', 'nShowerChildren', 'nTotalDescendants', 'nDescendantHits', 
                                                'recoTier', 'isExiting'], library='np', step_size=100)):

            if not b % 100:
                print('Processing batch', b)

            for t in range(0, len(batch['uViewX'])):

                this_pdg = batch['truePdg'][t]
                if abs(this_pdg) != 13 and abs(this_pdg) != 211 and this_pdg != 2212 and abs(this_pdg) !=321:
                    continue

                if exit_only and batch['isExiting'][t] == 0:
                    if abs(this_pdg) != 321:
                        continue

                if batch['uViewNHits'][t] == 0 or batch['vViewNHits'][t] == 0 or batch['wViewNHits'][t] == 0:
                    continue

                if hits_cut > 0:
                    if batch['uViewNHits'][t] < hits_cut or batch['vViewNHits'][t] < hits_cut or batch['wViewNHits'][t] < hits_cut:
                        continue

                this_track = InputTrack(batch['uViewX'][t], batch['uViewWire'][t], batch['uViewQ'][t], batch['uViewNHits'][t],
                                        batch['vViewX'][t], batch['vViewWire'][t], batch['vViewQ'][t], batch['vViewNHits'][t],
                                        batch['wViewX'][t], batch['wViewWire'][t], batch['wViewQ'][t], batch['wViewNHits'][t],
                                        batch['nTrackChildren'][t], batch['nShowerChildren'][t], batch['nTotalDescendants'][t],
                                        batch['nDescendantHits'][t], batch['recoTier'][t], batch['truePdg'][t])

                sample_dir = 'contained'
                if batch['isExiting'][t] == 1:
                    sample_dir = 'exiting'

                this_output_file_path = output_path + "/" + sample_dir + "/" + output_file_name + '_' + str(b) + '_' + str(t) + '.pkl'
                with open(this_output_file_path, 'wb') as f:
                    pickle.dump(this_track, f)

if __name__ == '__main__':
    input_file_name = 'track_pid_training_6.root'
    tree_name = 'tracks'
   
    # For files 3 to 5 just get the exiting tracks
    extract_tracks_from_root_file('track_pid_training_1.root', tree_name, "dataset/2d/", "track_1", 10, False)
    extract_tracks_from_root_file('track_pid_training_2.root', tree_name, "dataset/2d/", "track_2", 10, False)
    extract_tracks_from_root_file('track_pid_training_3.root', tree_name, "dataset/2d/", "track_3", 10, True)
    extract_tracks_from_root_file('track_pid_training_4.root', tree_name, "dataset/2d/", "track_4", 10, True)
    extract_tracks_from_root_file('track_pid_training_5.root', tree_name, "dataset/2d/", "track_5", 10, True)
    extract_tracks_from_root_file('track_pid_training_6.root', tree_name, "dataset/2d/", "track_6", 10, True)
    extract_tracks_from_root_file('track_pid_training_7.root', tree_name, "dataset/2d/", "track_7", 10, False)
    extract_tracks_from_root_file('track_pid_training_8.root', tree_name, "dataset/2d/", "track_8", 10, False)
    
