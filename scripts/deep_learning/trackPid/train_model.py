import math
import torch
from track_dataset import TrackDataset2d
from models import TrackPIDNetwork2d
from input_track import InputTrack
import numpy as np
from sklearn.metrics import confusion_matrix
import copy

sample = 'contained'

def GetDatasets(batch_size, sequence_length):
    """
        Load the datasets

        Args:
            batch_size: the number of tracks that will be processed in parallel
            sequence_length: the number of hits included in the sequences

        Returns:
            Dataloaders containing the training, validation and testing datasets
    """
    # Load the dataset and divide into train, validation and test samles
    dataset = TrackDataset2d('dataset/2d/'+sample+'/',sample+'_track_files.txt', sequence_length)
    indices = np.arange(len(dataset))
    np.random.seed(42)
    np.random.shuffle(indices)

    # Define split points
    train_idx, val_idx, test_idx = np.split(indices, [int(0.7*len(indices)), int(0.9*len(indices))])
    print(len(train_idx), len(val_idx), len(test_idx))

    # Create samplers
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    test_subset = torch.utils.data.Subset(dataset, test_idx)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

# =============================================================================

def count_particle_types(data_loader, n_classes):
    """
        Count the number of tracks within each class

        Args:
            data_loader: the data sample to count
            n_classes: the number of true classes

        Returns:
            number of counts per true class
    """
    n_counts = torch.zeros(n_classes)
    for (_, labels) in data_loader:
        for l in labels:
            n_counts[l] += 1
    return n_counts

# =============================================================================

def margin_hinge_loss(logits, target, margin=1.0):
    """
        Hinge loss term

        Args:
            logits: raw prediction values from the model
            target: true classes of the predicted events
            margin: margin term in the hinge loss formula

        Returns:
            the hinge loss value
    """
    batch_size, num_classes = logits.shape

    # Select the logit of the correct class
    true_logits = logits[torch.arange(batch_size), target]

    # Compute margin violations for all other classes
    # hinge = max(0, logits_j - true_logit + margin)
    margins = logits - true_logits.unsqueeze(1) + margin

    # Zero out the correct class margin
    margins[torch.arange(batch_size), target] = 0.0

    # Apply relu for hinge loss
    hinge_loss = torch.clamp(margins, min=0.0)

    # Average over classes and batch
    return hinge_loss.sum(dim=1).mean()

# =============================================================================

def one_epoch(epoch, model, data_loader, pid_lossfn, optimiser, initial_lr, lr_scheduler, use_plateau_not_cosine, is_training, n_warmup, n_freeze_enc, n_freeze_cross):
    """
        Iterate over one entire dataloader, either training, testing or validating

        Args:
            epoch: the epoch number
            model: the model being trained
            data_loader: the training, testing or validation sample
            pid_lossfn: the loss function associated with the model
            optmiser: the optimiser used for training
            initial_lr: the initial learning rate
            lr_scheduler: the learning rate scheduler used
            use_plateau_not_cosine: whether we are using the ReduceLROnPlateau or CosineAnnealingLR
            is_training: whether we are training or inferring
            n_warmup: number of warm-up epochs
            n_freeze_encoder: number of epochs to wait before learning the self-attention encoder weights
            n_freeze_cross: number of epochs to wait before learning the cross-attention encoder weights

        Returns:
            the average loss per batch
    """
    epoch_loss = 0.0

    for name, p in model.named_parameters():
        if "cross_attention" in name:
            p.requires_grad = (epoch >= n_freeze_cross)
        if "encoder" in name:
            p.requires_grad = (epoch >= n_freeze_enc)

    # Set the learning rate for this epoch
    if is_training:
        model.train()
        if epoch < n_warmup:
            lr = initial_lr * (epoch + 1) / float(n_warmup)
            print("Learning rate warmup:", lr)
            param_groups = optimiser.param_groups
            param_groups[0]['lr'] = lr
            param_groups[1]['lr'] = 0.1 * lr
#            for param_group in optimiser.param_groups:
#                param_group['lr'] = lr
        elif use_plateau_not_cosine == False:
            for param_group in optimiser.param_groups:
                print("Learning rate cosine annealing:", param_group['lr'])
        else:
            for param_group in optimiser.param_groups:
                print("Learning rate plateau:", param_group['lr'])
    else:
        model.eval()

    for b, (data, labels) in enumerate(data_loader): 
        data = (data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device))
        labels = labels.to(device)
        batch_loss = 0.0

        hinge_weight = max(0.0, 0.5 - epoch / 20)
        if is_training:
            optimiser.zero_grad()
            outputs = model(data[0], data[1], data[2], data[3])
            batch_loss = pid_lossfn(outputs, labels) + hinge_weight * margin_hinge_loss(outputs, labels, 1.0)
            batch_loss.backward()
            optimiser.step()
        else:
            with torch.no_grad():
                outputs = model(data[0], data[1], data[2], data[3])
                batch_loss = pid_lossfn(outputs, labels) + hinge_weight * margin_hinge_loss(outputs, labels, 1.0)

        epoch_loss += batch_loss.item()

    # Step the cosine scheduler if training
    if is_training and not use_plateau_not_cosine and epoch >= n_warmup:
        lr_scheduler.step()
    
    # Step the plateau scheduler on the validation loss only
    if use_plateau_not_cosine and not is_training and epoch >= n_warmup:
        lr_scheduler.step(epoch_loss / len(data_loader))

    return epoch_loss / len(data_loader)

# =============================================================================

n_features = 3
n_classes = 4 # 0 = muon, 1 = pion, 2 = proton, 3 = kaon
sequence_length = 128
model_depth = 64
n_heads = 8
feed_forward_depth = model_depth * 2
n_encoder_layers = 6
n_cross_attention_layers = 2
dropout = 0.2 #NB: classification uses 2x this value
n_auxillary = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device', device)
my_model = TrackPIDNetwork2d(n_features, n_classes, sequence_length, model_depth, n_heads, feed_forward_depth, n_encoder_layers, n_cross_attention_layers, dropout, n_auxillary)


model_parameters = filter(lambda p: p.requires_grad, my_model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Total parameters",params)

# Dataset split into three dataloaders
batch_size = 128
train_loader, val_loader, test_loader = GetDatasets(batch_size, sequence_length)
print(len(train_loader), len(val_loader), len(test_loader))

n_counts = count_particle_types(train_loader, n_classes)
pid_weights = torch.ones(n_classes).to(device)
pid_weights[1] = n_counts[0] / n_counts[1]
pid_weights[2] = n_counts[0] / n_counts[2]

# Loss functions
use_plateau_not_cosine = False
pid_loss_fn = torch.nn.CrossEntropyLoss(weight = pid_weights)
initial_lr = 1e-4
warmup_epochs = 10
xatten_params = []
other_params = []
for name, p in my_model.named_parameters():
    (xatten_params if "cross" in name else other_params).append(p)
optimiser = torch.optim.AdamW([
    {"params": other_params, "lr": initial_lr},
    {"params": xatten_params, "lr": 0.1 * initial_lr}], weight_decay=0.01)

lr_scheduler = None
if not use_plateau_not_cosine:
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=25, eta_min = 1e-7)
else:
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=2)

my_model.to(device)

n_epochs = 100
end_patience = 10
n_freeze_enc = 2
n_freeze_cross = 10
training_losses = {}
validation_losses = {}
best_epoch_loss = 1e9
best_epoch = -1
best_epoch_state = None
n_epoch_no_improvement = 0
for e in range(0, n_epochs):
    one_epoch_train_loss = one_epoch(e, my_model, train_loader, pid_loss_fn, optimiser, initial_lr, lr_scheduler, use_plateau_not_cosine, True, warmup_epochs, n_freeze_enc, n_freeze_cross)
    one_epoch_valid_loss = one_epoch(e, my_model, val_loader, pid_loss_fn, optimiser, initial_lr, lr_scheduler, use_plateau_not_cosine, False, warmup_epochs, n_freeze_enc, n_freeze_cross)
    one_epoch_test_loss = one_epoch(e, my_model, test_loader, pid_loss_fn, optimiser, initial_lr, lr_scheduler, use_plateau_not_cosine, False, warmup_epochs, n_freeze_enc, n_freeze_cross)
    print('Epoch', e, 'training loss =', one_epoch_train_loss, ',validation loss', one_epoch_valid_loss, 'and test loss', one_epoch_test_loss)
    training_losses[e] = one_epoch_train_loss
    validation_losses[e] = one_epoch_valid_loss
    if one_epoch_valid_loss < best_epoch_loss:
        best_epoch = e
        best_epoch_loss = one_epoch_valid_loss
        best_epoch_state = copy.deepcopy(my_model.state_dict())
        n_epoch_no_improvement = 0
    else:
        n_epoch_no_improvement += 1

    if n_epoch_no_improvement >= end_patience:
        print("No improvement in validation loss for", end_patience, "epochs, stopping training")
        break

################################## TESTING ####################################

from sklearn.metrics import confusion_matrix

print("Loading model state from best epoch (", best_epoch, ")")
my_model.load_state_dict(best_epoch_state)

# Run the test sample
test_loss = 0.0

my_model.eval()
test_confusion_matrix = np.zeros((n_classes,n_classes), dtype=int)
with torch.no_grad():
    for _, (data, labels) in enumerate(test_loader): 
        data = (data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device))
        labels = labels.to(device)
        batch_loss = None
    
        outputs = my_model(data[0], data[1], data[2], data[3])
        hinge_weight = max(0.0, 0.5 - best_epoch / 20)
        batch_loss = pid_loss_fn(outputs, labels) + hinge_weight * margin_hinge_loss(outputs, labels, 1.0)
        test_loss += batch_loss.item()
    
        outputs_as_class_cpu = outputs.argmax(dim=1).cpu()
        labels_cpu = labels.cpu()
    
        test_confusion_matrix += confusion_matrix(outputs_as_class_cpu.numpy(), labels_cpu.numpy(), labels=np.arange(n_classes))

test_loss = test_loss / len(test_loader)

print(test_loss)
print(test_confusion_matrix)

torch.save(best_epoch_state, 'best_'+sample+'_model_'+str(sequence_length)+'_'+str(model_depth)+'_'+str(n_heads)+'_'+str(n_encoder_layers)+'_'+str(feed_forward_depth)+'.pth')
