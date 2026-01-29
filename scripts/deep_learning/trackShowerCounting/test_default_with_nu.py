import torch
from neutrino_dataset import NeutrinoDatasetWithVertex
import official_models
import usefulUtils
import numpy as np
from sklearn.metrics import confusion_matrix

nu_classes = 3
trk_classes = 6
shw_classes = 6
my_model = official_models.ConvNeXtV2WithVertexAndNuClass(1, nu_classes, trk_classes, shw_classes, depths=[3, 3, 9, 3], dims=[32, 64, 128, 128], drop_path_rate=0.3, head_init_scale=1.)
my_model.load_state_dict(torch.load('five_hits_with_nu/model_state_dict_large_epoch_37.pt'))

# Print the number of model parameters
model_parameters = filter(lambda p: p.requires_grad, my_model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Total parameters",params)

# Load the dataset and divide into train, validation and test samles
dataset = NeutrinoDatasetWithVertex('truth_info_full.csv','images/trk_5_hits/')
np.random.seed(42)
indices = np.arange(len(dataset))
np.random.shuffle(indices)

# Define split points
train_idx, val_idx, test_idx = np.split(indices, [int(0.7*len(indices)), int(0.9*len(indices))])

# Create samplers
train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

# Create data loaders
batch_size = 16
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

print("Number of images (train, validation, test):", len(train_idx), len(val_idx), len(test_idx))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device', device)

my_model.to(device)

n_outputs = 3

# Accuracy
running_acc = np.zeros(n_outputs, dtype='float32')

# Confusion matrices
total_cm_nu = np.zeros((3,3), dtype=int)
total_cm_tracks = np.zeros((6,6), dtype=int)
total_cm_showers = np.zeros((6,6), dtype=int)

# Nu classification scores
arr_nc_score = np.empty((0,1), dtype=float)
arr_cc_numu_score = np.empty((0,1), dtype=float)
arr_cc_nue_score = np.empty((0,1), dtype=float)

# Track classification scores
n_trk_classes = 6
arr_trk_scores = []
for i in range(n_trk_classes):
    arr_trk_scores.append(np.empty((0,1), dtype=float))

# Shower classification scores
n_shw_classes = 6
arr_shw_scores = []
for i in range(n_shw_classes):
    arr_shw_scores.append(np.empty((0,1), dtype=float))

arr_true_pid = np.empty((0,1), dtype=float)
arr_true_trk = np.empty((0,1), dtype=float)
arr_true_shw = np.empty((0,1), dtype=float)

# Softmax layer since it is usually applied by the loss function
softmax = torch.nn.Softmax(dim=1)

my_model.eval()
with torch.no_grad():
    for batch_no, ((images, vertices), labels) in enumerate(test_loader):
        images = images.to(device)
        vertices = vertices.to(device)
        labels = labels.to(device)

        # Split the labels into shape [events_per_batch, 1] i.e. one for each output
        individual_labels = [labels[:, i] for i in range(labels.size(1))]

        # Make the predictions and apply the softmax activation
        outputs = my_model(images, vertices)
        outputs[0] = softmax(outputs[0])
        outputs[1] = softmax(outputs[1])
        outputs[2] = softmax(outputs[2])

        arr_nc_score = np.concatenate((arr_nc_score, outputs[0][:,0].cpu().unsqueeze(1).numpy()))
        arr_cc_numu_score = np.concatenate((arr_cc_numu_score, outputs[0][:,1].cpu().unsqueeze(1).numpy()))
        arr_cc_nue_score = np.concatenate((arr_cc_nue_score, outputs[0][:,2].cpu().unsqueeze(1).numpy()))

        for i in range(n_trk_classes):
            arr_trk_scores[i] = np.concatenate((arr_trk_scores[i], outputs[1][:,i].cpu().unsqueeze(1).numpy()))

        for i in range(n_shw_classes):
            arr_shw_scores[i] = np.concatenate((arr_shw_scores[i], outputs[2][:,i].cpu().unsqueeze(1).numpy()))

        arr_true_pid = np.concatenate((arr_true_pid, labels[:,0].cpu().unsqueeze(1).numpy()))
        arr_true_trk = np.concatenate((arr_true_trk, labels[:,1].cpu().unsqueeze(1).numpy()))
        arr_true_shw = np.concatenate((arr_true_shw, labels[:,2].cpu().unsqueeze(1).numpy()))

        batch_acc = np.zeros(n_outputs, dtype='float32')
        for a in range(len(batch_acc)):
            batch_acc[a] += (usefulUtils.calculate_accuracy(outputs[a], individual_labels[a]))[0]
        running_acc += batch_acc

        total_cm_nu += confusion_matrix(outputs[0].argmax(dim=1).cpu().numpy(), individual_labels[0].cpu().numpy(), labels=np.arange(3))
        total_cm_tracks += confusion_matrix(outputs[1].argmax(dim=1).cpu().numpy(), individual_labels[1].cpu().numpy(), labels=np.arange(6))
        total_cm_showers += confusion_matrix(outputs[2].argmax(dim=1).cpu().numpy(), individual_labels[2].cpu().numpy(), labels=np.arange(6))

print("Testing complete!")
print(total_cm_nu)
print(total_cm_tracks)
print(total_cm_showers)

np.savez("testing_outputs.npz", arr_nc_score, arr_cc_numu_score, arr_cc_nue_score, \
                                arr_trk_scores[0], arr_trk_scores[1], arr_trk_scores[2], arr_trk_scores[3], arr_trk_scores[4], arr_trk_scores[5], \
                                arr_shw_scores[0], arr_shw_scores[1], arr_shw_scores[2], arr_shw_scores[3], arr_shw_scores[4], arr_shw_scores[5], \
                                arr_true_pid, arr_true_trk, arr_true_shw)
