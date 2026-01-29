import csv
import torch
import torchvision as tv
import torchvision.io as tvio

class NeutrinoDatasetWithVertex(torch.utils.data.Dataset):
    """
        Neutrino image dataset

        Args:
            truth_file: name of the summary file containing the image path and truth information
            file_path: path containing truth_file
    """
    def __init__(self, truth_file, file_path):
        self.targets = None

        # Row format in csv files: 'image_path', nuRecoVertexDriftBin, nuRecoVertexWireBin, nu_type, n_tracks, n_showers
        with open(file_path + truth_file) as csvfile:
            self.targets = [row for row in csv.reader(csvfile)]
            self.targets = [(row[0], [float(x) for x in row[1:3]], [int(x) for x in row[3:]]) for row in self.targets]
        print('Loaded', len(self.targets), 'images')
        self.transform = tv.transforms.Compose([
                            tv.transforms.Lambda(lambda x: x.float() / 255.0),  # Normalize after loading
                         ])
        n_flv = [0, 0, 0]
        n_trk = [0, 0, 0, 0, 0, 0]
        n_shw = [0, 0, 0, 0, 0, 0]

        for e in range(len(self.targets)):
            n_flv[self.targets[e][2][0]] += 1

            if self.targets[e][2][1] > 4:
                self.targets[e][2][1] = 5
            n_trk[self.targets[e][2][1]] += 1

            if self.targets[e][2][2] > 4:
                self.targets[e][2][2] = 5
            n_shw[self.targets[e][2][2]] += 1

        print("- Flavour:", n_flv)
        print("- # Tracks:", n_trk)
        print("- # Showers:", n_shw)

    def calc_weights(self, idx, n_elements):
        """
            Calculate weights for class balancing

            Args:
                idx (int): the output to calculate the weight for
                n_elements: the number of outputs of the network

            Returns:
                The weights for the classes of the given output
        """
        weights = torch.zeros(n_elements)
        for event in self.targets:
            weights[event[2][idx]] += 1
        weights = float(len(self.targets)) / weights        
        weights = weights / weights.sum()
        return weights * n_elements

    def get_flavour_weights(self):
        # This corresponds to the zeroth output and has three classes
        return self.calc_weights(0, 3)

    def get_n_tracks_weights(self):
        # This corresponds to the first output and has six classes
        return self.calc_weights(1, 6)

    def get_n_showers_weights(self):
        # This corresponds to the second output and has six classes
        return self.calc_weights(2, 6)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        """
            Load the requested image

            Args:
                index (int): the image position in the dataset

            Returns:
                torch.tensors for the image, vertex position and labels
        """
        image_path = self.targets[index][0]
        image = tvio.read_image(image_path)
        image = self.transform(image)
        vtx = self.targets[index][1]
        vtx = torch.tensor(vtx, dtype=torch.float)
        labels = self.targets[index][2]
        labels = torch.tensor(labels, dtype=torch.long)
        return (image, vtx) , labels

