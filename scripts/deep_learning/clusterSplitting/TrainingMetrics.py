import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

################################################################################################################################################
################################################################################################################################################

contamination_labels = [0, 1, 2]
contamination_strings = { 0 : 'Track Not Contaminated', 1 : 'Track Contaminated', 2 : 'Shower'}
contamination_colours = { 0 : 'red', 1 : 'blue', 2 : 'green'}

split_point_labels = [0, 1]
split_point_strings = { 0 : 'False', 1 : 'True'}
split_point_colours = { 0 : 'red', 1 : 'blue'}

################################################################################################################################################
################################################################################################################################################

def plot_scores(scores_train, scores_test, truth_train, truth_test, is_contamination) :
    
    class_indices = contamination_labels if is_contamination else split_point_labels
    class_strings = contamination_strings if is_contamination else split_point_strings
    class_colours = contamination_colours if is_contamination else split_point_colours
    
    for class_index in class_indices :
        class_scores_train = scores_train[truth_train == class_index]
        class_scores_test = scores_test[truth_test == class_index]

        plotting_weights_train = 1.0 / float(class_scores_train.shape[0])
        plotting_weights_train = torch.ones(class_scores_train.shape) * plotting_weights_train
        plotting_weights_test = 1.0 / float(class_scores_test.shape[0])
        plotting_weights_test = torch.ones(class_scores_test.shape) * plotting_weights_test    
    
        plt.hist(class_scores_train, bins=50, range=(0, 1.0), color=class_colours[class_index], label=(f'{class_strings[class_index]} (train)'), weights=plotting_weights_train, histtype='step', linestyle='solid')
        plt.hist(class_scores_test, bins=50, range=(0, 1.0), color=class_colours[class_index], label=(f'{class_strings[class_index]} (test)'), weights=plotting_weights_test, histtype='step', linestyle='dashed')

    plt.yscale("log")
    plt.xlabel('Classification Score')
    plt.ylabel('log(Proportion of Clusters)')
    plt.legend(loc='best')
    plt.show()
    
################################################################################################################################################
################################################################################################################################################       

def draw_confusion(pred, labels, thresholds, is_contamination):  
    
    class_indices = contamination_labels if is_contamination else split_point_labels
    n_classes = len(contamination_labels) + 1
    
    if (len(thresholds) != len(class_indices)) :
        print('Wrong number of thresholds!')
        return
    
    scores = pred.copy()
    
    # Modify if binary input
    if (len(class_indices) == 2) :
        scores = np.hstack((1 - scores, scores))

    selected = np.argmax(scores, axis=1)
    selected_scores = scores.max(axis=1)
    
    for class_index in class_indices :
        below_threshold = (selected == class_index) & (selected_scores < thresholds[class_index])
        selected[below_threshold] = (n_classes - 1) # optional reject case
        
    conf_matrix = confusion_matrix(labels, selected)    
    
    # redefine n_classes if needs be
    n_classes = conf_matrix.shape[0]
    
    true_sums = np.sum(conf_matrix, axis=1)
    pred_sums = np.sum(conf_matrix, axis=0)

    true_normalised = np.zeros(shape=(n_classes, n_classes))
    pred_normalised = np.zeros(shape=(n_classes, n_classes))

    for true_index in range(n_classes) : 
        for pred_index in range(n_classes) :
            n_entries = conf_matrix[true_index][pred_index]
            if true_sums[true_index] > 0 :
                true_normalised[true_index][pred_index] = float(n_entries) / float(true_sums[true_index])
            if pred_sums[pred_index] > 0 :
                pred_normalised[true_index][pred_index] = float(n_entries) / float(pred_sums[pred_index])

    display_labels = list(contamination_strings.values()) if is_contamination else list(split_point_strings.values())
    
    # Handle our reject case
    if (n_classes > len(display_labels)) :
        display_labels.append('Other')
                
    displayTrue_norm = ConfusionMatrixDisplay(confusion_matrix=true_normalised, display_labels=display_labels)
    displayTrue_norm.plot()
    displayPred_norm = ConfusionMatrixDisplay(confusion_matrix=pred_normalised, display_labels=display_labels)
    displayPred_norm.plot()