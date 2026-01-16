import torch
from neutrino_dataset import NeutrinoDatasetWithVertex
import official_models
import usefulUtils
import numpy as np

nu_classes = 3
trk_classes = 6
shw_classes = 6
my_model = official_models.ConvNeXtV2WithVertexAndNuClass(1, nu_classes, trk_classes, shw_classes, depths=[3, 3, 9, 3], dims=[32, 64, 128, 128], drop_path_rate=0.3, head_init_scale=1.)

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

# Calculate the class weights so that we can deal with class imbalance
event_flv_weights = dataset.get_flavour_weights().to(device)
trk_weights = dataset.get_n_tracks_weights().to(device)
shw_weights = dataset.get_n_showers_weights().to(device)
print("Loss function weights:")
print(event_flv_weights)
print(trk_weights)
print(shw_weights)

# Loss functions for the two outputs. N.B. these include an implicit softmax
smoothing = 0.
nu_lossfn = torch.nn.CrossEntropyLoss(weight=event_flv_weights, label_smoothing=smoothing)
trk_lossfn = torch.nn.CrossEntropyLoss(weight=trk_weights, label_smoothing=smoothing)
shw_lossfn = torch.nn.CrossEntropyLoss(weight=shw_weights, label_smoothing=smoothing)

initial_lr = 1e-4
optimizer = torch.optim.AdamW(my_model.parameters(), lr=initial_lr, weight_decay=1e-2)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
my_model.to(device)

n_outputs = 3
n_epochs=50
for epoch in range(0, n_epochs):
    if epoch < 10:
        lr = initial_lr * (epoch + 1) / 10
        print("Learning rate warmup:", lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            break
    else:
        cosine_scheduler.step()
        for param_group in optimizer.param_groups:
            print("Learning rate cosine annealing:", param_group['lr'])
            break

    # Training
    running_loss = 0.0
    running_acc = np.zeros(n_outputs, dtype='float32')
    my_model.train()
    for batch_no, ((images, vertices), labels) in enumerate(train_loader):
        images = images.to(device)
        vertices = vertices.to(device)
        labels = labels.to(device)

        # Split the labels into shape [events_per_batch, 1] i.e. one for each output
        individual_labels = [labels[:, i] for i in range(labels.size(1))]

        # Forward pass
        outputs = my_model(images, vertices)
        loss = nu_lossfn(outputs[0], individual_labels[0])
        loss += trk_lossfn(outputs[1], individual_labels[1])
        loss += shw_lossfn(outputs[2], individual_labels[2])

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        running_loss += batch_loss

        batch_acc = np.zeros(n_outputs, dtype='float32')
        for a in range(len(batch_acc)):
            batch_acc[a] += (usefulUtils.calculate_accuracy(outputs[a], individual_labels[a]))[0]
        running_acc += batch_acc

    # Validation
    running_val_loss = 0.0
    running_val_acc = np.zeros(n_outputs, dtype='float32')
    my_model.eval()
    with torch.no_grad():
        for batch_no, ((images, vertices), labels) in enumerate(val_loader):
            images = images.to(device)
            vertices = vertices.to(device)
            labels = labels.to(device)

            # Split the labels into shape [events_per_batch, 1] i.e. one for each output
            individual_labels = [labels[:, i] for i in range(labels.size(1))]

            # Make the predictions
            outputs = my_model(images, vertices)
            loss = nu_lossfn(outputs[0], individual_labels[0])
            loss += trk_lossfn(outputs[1], individual_labels[1])
            loss += shw_lossfn(outputs[2], individual_labels[2])

            batch_loss = loss.item()
            running_val_loss += batch_loss

            batch_acc = np.zeros(n_outputs, dtype='float32')
            for a in range(len(batch_acc)):
                batch_acc[a] += (usefulUtils.calculate_accuracy(outputs[a], individual_labels[a]))[0]
            running_val_acc += batch_acc

    epoch_acc = running_acc / len(train_loader)
    epoch_val_acc = running_val_acc / len(val_loader)

    print("Epoch", epoch, "training loss:", running_loss/len(train_loader))
    print("Epoch", epoch, "training accuracy", epoch_acc)
    print("Epoch", epoch, "validation loss:", running_val_loss/len(val_loader))
    print("Epoch", epoch, "validation accuracy", epoch_val_acc)

    if epoch > 5:
        torch.save(my_model.state_dict(), 'five_hits_with_nu/model_state_dict_large_epoch_' + str(epoch) + '.pt')

print("Training complete!")
