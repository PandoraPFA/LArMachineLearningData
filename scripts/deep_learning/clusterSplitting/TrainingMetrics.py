import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

################################################################################################################################################
# Encoder
################################################################################################################################################

def plot_scores_class(scores_train, scores_test, truth_train, truth_test, score_class_index) :
    
    class_scores_train = scores_train[:, score_class_index]
    class_scores_test = scores_test[:, score_class_index]
    
    for class_index in [0, 1, 2] :
    
        this_scores_train = class_scores_train[truth_train == class_index]
        this_scores_test = class_scores_test[truth_test == class_index]
    
        plotting_weights_train = 1.0 / float(this_scores_train.shape[0])
        plotting_weights_train = torch.ones(this_scores_train.shape) * plotting_weights_train
        
        plotting_weights_test = 1.0 / float(this_scores_test.shape[0])
        plotting_weights_test = torch.ones(this_scores_test.shape) * plotting_weights_test

        legend_string = 'False' if class_index == 0 else 'True' if class_index == 1 else 'Shower'
        graph_color = 'red' if class_index == 0 else 'blue' if class_index == 1 else 'green'
        
        plt.hist(this_scores_train, bins=50, range=(0, 1.0), color=graph_color, label=(legend_string + ' train'), weights=plotting_weights_train, histtype='step', linestyle='solid')
        plt.hist(this_scores_test, bins=50, range=(0, 1.0), color=graph_color, label=(legend_string + ' test'), weights=plotting_weights_test, histtype='step', linestyle='dashed')

    plt.yscale("log")
    plt.xlabel(('Classification Score For Class: ' + str(score_class_index)))
    plt.ylabel('log(Proportion of Clusters)')
    plt.legend(loc='best')
    plt.show() 
    
################################################################################################################################################
################################################################################################################################################      
    
def draw_confusion_class(scores, labels, threshold):
    
    n_classes = scores.shape[1] + 1
    pred = np.argmax(scores, axis=1)
    pred[scores.max(axis=1) < threshold] = (n_classes - 1) # optional reject case
    confMatrix = confusion_matrix(labels, pred)
    
    trueSums = np.sum(confMatrix, axis=1)
    predSums = np.sum(confMatrix, axis=0)

    trueNormalised = np.zeros(shape=(n_classes, n_classes))
    predNormalised = np.zeros(shape=(n_classes, n_classes))

    for trueIndex in range(n_classes) : 
        for predIndex in range(n_classes) :
            nEntries = confMatrix[trueIndex][predIndex]
            if trueSums[trueIndex] > 0 :
                trueNormalised[trueIndex][predIndex] = float(nEntries) / float(trueSums[trueIndex])
            if predSums[predIndex] > 0 :
                predNormalised[trueIndex][predIndex] = float(nEntries) / float(predSums[predIndex])

    displayTrueNorm = ConfusionMatrixDisplay(confusion_matrix=trueNormalised, display_labels=range(n_classes))
    displayTrueNorm.plot()

    displayPredNorm = ConfusionMatrixDisplay(confusion_matrix=predNormalised, display_labels=range(n_classes))
    displayPredNorm.plot()     
    plt.show()        

################################################################################################################################################
# Encoder-decoder
################################################################################################################################################

def plot_scores(scores_train, scores_test, truth_train, truth_test) :

    true_scores_train = scores_train[truth_train == 1]
    false_scores_train = scores_train[truth_train == 0]
    true_scores_test = scores_test[truth_test == 1]
    false_scores_test = scores_test[truth_test == 0]
    
    true_plotting_weights_train = 1.0 / float(true_scores_train.shape[0])
    true_plotting_weights_train = torch.ones(true_scores_train.shape) * true_plotting_weights_train
    false_plotting_weights_train = 1.0 / float(false_scores_train.shape[0])
    false_plotting_weights_train = torch.ones(false_scores_train.shape) * false_plotting_weights_train
    true_plotting_weights_test = 1.0 / float(true_scores_test.shape[0])
    true_plotting_weights_test = torch.ones(true_scores_test.shape) * true_plotting_weights_test
    false_plotting_weights_test = 1.0 / float(false_scores_test.shape[0])
    false_plotting_weights_test = torch.ones(false_scores_test.shape) * false_plotting_weights_test    
    
    plt.hist(true_scores_train, bins=50, range=(0, 1.0), color='blue', label='signal_train', weights=true_plotting_weights_train, histtype='step', linestyle='solid')
    plt.hist(false_scores_train, bins=50, range=(0, 1.0), color='red', label='background_train', weights=false_plotting_weights_train, histtype='step', linestyle='solid')
    plt.hist(true_scores_test, bins=50, range=(0, 1.0), color='blue', label='signal_test', weights=true_plotting_weights_test, histtype='step', linestyle='dashed')
    plt.hist(false_scores_test, bins=50, range=(0, 1.0), color='red', label='background_test', weights=false_plotting_weights_test, histtype='step', linestyle='dashed')    
    
    plt.yscale("log")
    plt.xlabel('Classification Score')
    plt.ylabel('log(Proportion of Clusters)')
    plt.legend(loc='best')
    plt.show()
    
################################################################################################################################################
################################################################################################################################################      

def draw_confusion(pred, labels, threshold):
    
    n_classes = 2
    scores = pred.copy()
    predicted_true_mask = scores > threshold
    predicted_false_mask = np.logical_not(predicted_true_mask)
    scores[predicted_true_mask] = 1
    scores[predicted_false_mask] = 0
    confMatrix = confusion_matrix(labels, scores)
    
    trueSums = np.sum(confMatrix, axis=1)
    predSums = np.sum(confMatrix, axis=0)

    trueNormalised = np.zeros(shape=(n_classes, n_classes))
    predNormalised = np.zeros(shape=(n_classes, n_classes))

    for trueIndex in range(n_classes) : 
        for predIndex in range(n_classes) :
            nEntries = confMatrix[trueIndex][predIndex]
            if trueSums[trueIndex] > 0 :
                trueNormalised[trueIndex][predIndex] = float(nEntries) / float(trueSums[trueIndex])
            if predSums[predIndex] > 0 :
                predNormalised[trueIndex][predIndex] = float(nEntries) / float(predSums[predIndex])

    displayTrueNorm = ConfusionMatrixDisplay(confusion_matrix=trueNormalised, display_labels=["False", "True"])
    displayTrueNorm.plot()

    displayPredNorm = ConfusionMatrixDisplay(confusion_matrix=predNormalised, display_labels=["False", "True"])
    displayPredNorm.plot()  
    
################################################################################################################################################
################################################################################################################################################    

 