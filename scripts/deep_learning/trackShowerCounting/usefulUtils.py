import torch

def calculate_accuracy(outputs, labels):
    """
        Calculate accuracy given model outputs and true labels.

    Args:
        outputs (torch.Tensor): Model outputs (logits or probabilities) of shape (batch_size, num_classes).
        labels (torch.Tensor): True labels of shape (batch_size).

    Returns:
        Accuracy as a percentage, both total and for all classes individually
    """

    num_classes = outputs.size(1)

    # Get the predicted class indices (class with highest score)
    predicted = torch.argmax(outputs, dim=1)

    # Initialize arrays to hold correct predictions and total samples per class
    correct_per_class = torch.zeros(num_classes)
    total_per_class = torch.zeros(num_classes)

    # Calculate the number of correct predictions per class
    for i in range(num_classes):
        correct_per_class[i] = ((predicted == i) & (labels == i)).sum().item()
        total_per_class[i] = (labels == i).sum().item()

    # Overall accuracy
    total_correct = correct_per_class.sum().item()
    total_samples = labels.size(0)
    overall_accuracy = total_correct / total_samples

    # Per-class accuracy (handle cases where there might be no samples for a class)
    class_accuracy = correct_per_class / (total_per_class + 1e-6)

    accuracy_tensor = 100.0 * torch.cat((torch.tensor([overall_accuracy]), class_accuracy))

    return accuracy_tensor
