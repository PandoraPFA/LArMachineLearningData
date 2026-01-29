import math
import pickle
import torch
import csv
from input_track import InputTrack

class TrackDataset2d(torch.utils.data.Dataset):
    def __init__(self, file_path, file_name, sequence_length):
        """
            Track dataset class

            Args:
                file_path: location of the text file listing all of the pickle file locations
                file_name: name of the text file listing the pickle file locations
                sequence_length: number of hits to include in the sequence
        """
        self.file_path = file_path
        self.input_files = []
        self.get_input_files(file_name)
        self.sequence_length = sequence_length

    def get_input_files(self, file_name):
        """
            Read the list of pickle files from the text file

            Args:
                file_name: name of the input text file
        """
        with open(self.file_path + file_name, 'r') as input_file_list:
            self.input_files = [row[0] for row in csv.reader(input_file_list)]
        print('Found',len(self.input_files),'tracks')

    # Label values:
    # 0 = muon, 1 = pion, 2 = proton, 3 = kaon
    def convert_pdg_to_label(self, pdg):
        """
            Convert particle type into the true labels

            Args:
                pdg: the true PDG code of the track
        """
        if abs(pdg) == 13:
            return 0
        elif abs(pdg) == 211:
            return 1
        elif abs(pdg) == 2212:
            return 2
        elif abs(pdg) == 321:
            return 3
        else:
            print('Unknown pdg code. Giving up')
            exit()

    def pad_or_crop_view(self, n_hits, x, wire, q):
        """
            Pad or crop the input sequences to be of length sequence_length

            Args:
                n_hits: length of the input sequence
                x: the x coordinates of the track hits
                wire: the wire coordinates of the track hits
                q: the charge of the track hits

            Returns:
                the padded or cropped (x, wire, q) tensors
        """
        # Pad the tensors if we need to
        if n_hits < self.sequence_length:
            x = self.pad_sequence(x)
            wire = self.pad_sequence(wire)
            q = self.pad_sequence(q)
        # Crop the tensors if we need to
        elif n_hits > self.sequence_length:
            x = x[(n_hits - self.sequence_length):]
            wire = wire[(n_hits - self.sequence_length):]
            q = q[(n_hits - self.sequence_length):]

        return x, wire, q

    def pad_sequence(self, sequence):
        """
            Apply padding to short sequences to bring them up to sequence_length

            Args:
                sequence: the input sequence

            Returns:
                the padded sequence
        """
        n_padding = self.sequence_length - sequence.size(0)
        sequence = torch.nn.functional.pad(sequence, (n_padding, 0), value=0)
        return sequence

    def __getitem__(self, index):
        """
            Get the requested track from the dataset

            Args:
                index: the index of the track in the dataset

            Returns:
                The three view sequences, the auxillary variables and the truth label
        """
        track = None
        with open(self.file_path + self.input_files[index], 'rb') as f:
            track = pickle.load(f)

        # u view
        u_view_x = torch.tensor(track.u_view_x, dtype=torch.float)
        u_view_x = u_view_x - u_view_x[0]
        u_view_wire = torch.tensor(track.u_view_wire, dtype=torch.float)
        u_view_wire = u_view_wire - u_view_wire[0]
        u_view_q = torch.tensor(track.u_view_q, dtype=torch.float)
        u_view_n_hits = track.u_view_n_hits
        u_view_x, u_view_wire, u_view_q = self.pad_or_crop_view(u_view_n_hits, u_view_x, u_view_wire, u_view_q)

        # v view
        v_view_x = torch.tensor(track.v_view_x, dtype=torch.float)
        v_view_x = v_view_x - v_view_x[0]
        v_view_wire = torch.tensor(track.v_view_wire, dtype=torch.float)
        v_view_wire = v_view_wire - v_view_wire[0]
        v_view_q = torch.tensor(track.v_view_q, dtype=torch.float)
        v_view_n_hits = track.v_view_n_hits
        v_view_x, v_view_wire, v_view_q = self.pad_or_crop_view(v_view_n_hits, v_view_x, v_view_wire, v_view_q)

        # w view
        w_view_x = torch.tensor(track.w_view_x, dtype=torch.float)
        w_view_x = w_view_x - w_view_x[0]
        w_view_wire = torch.tensor(track.w_view_wire, dtype=torch.float)
        w_view_wire = w_view_wire - w_view_wire[0]
        w_view_q = torch.tensor(track.w_view_q, dtype=torch.float)
        w_view_n_hits = track.w_view_n_hits
        w_view_x, w_view_wire, w_view_q = self.pad_or_crop_view(w_view_n_hits, w_view_x, w_view_wire, w_view_q)

        class_label = torch.tensor(self.convert_pdg_to_label(track.pdg), dtype=torch.long)

        u_view_sequence = torch.stack([u_view_x / 1000., u_view_wire / 1000., u_view_q])
        v_view_sequence = torch.stack([v_view_x / 1000., v_view_wire / 1000., v_view_q])
        w_view_sequence = torch.stack([w_view_x / 1000., w_view_wire / 1000., w_view_q])

        if u_view_n_hits != 0:
            u_view_n_hits = math.log10(u_view_n_hits) / 5.0
        if v_view_n_hits != 0:
            v_view_n_hits = math.log10(v_view_n_hits) / 5.0
        if w_view_n_hits != 0:
            w_view_n_hits = math.log10(w_view_n_hits) / 5.0
        n_descendant_hits = track.n_descendant_hits
        if n_descendant_hits != 0:
            n_descendant_hits = math.log10(n_descendant_hits) / 5.0

        auxillary = torch.tensor([track.n_child_trk, track.n_child_shw, track.n_descendants, n_descendant_hits,
                                  track.reco_tier, u_view_n_hits, v_view_n_hits, w_view_n_hits], dtype=torch.float32)

        return (u_view_sequence.transpose(0,1), v_view_sequence.transpose(0,1), w_view_sequence.transpose(0,1), auxillary), class_label 

    def __len__(self):
        """
            Length of the dataset

            Returns:
                length of the dataset
        """
        return len(self.input_files)

