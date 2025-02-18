import torch
import numpy as np

import matplotlib.pyplot as plt

import sklearn 
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, roc_curve, auc, RocCurveDisplay, ConfusionMatrixDisplay

#################################################################################################################################
#################################################################################################################################

def plot_scores_classifier(pos_score_train, neg_score_train, pos_score_test, neg_score_test):
    
    pos_score_train = np.array(pos_score_train)
    neg_score_train = np.array(neg_score_train)
    pos_score_test = np.array(pos_score_test)
    neg_score_test = np.array(neg_score_test)    
    
    pos_plotting_weights_train = 1.0 / float(len(pos_score_train))
    pos_plotting_weights_train = np.ones(len(pos_score_train)) * pos_plotting_weights_train
    neg_plotting_weights_train = 1.0 / float(len(neg_score_train))
    neg_plotting_weights_train = np.ones(len(neg_score_train)) * neg_plotting_weights_train
    pos_plotting_weights_test = 1.0 / float(len(pos_score_test))
    pos_plotting_weights_test = np.ones(len(pos_score_test)) * pos_plotting_weights_test
    neg_plotting_weights_test = 1.0 / float(len(neg_score_test))
    neg_plotting_weights_test = np.ones(len(neg_score_test)) * neg_plotting_weights_test    
    
    plt.hist(pos_score_train, bins=50, range=(0, 1.0), color='blue', label='signal_train', weights=pos_plotting_weights_train, histtype='step', linestyle='solid')
    plt.hist(neg_score_train, bins=50, range=(0, 1.0), color='red', label='background_train', weights=neg_plotting_weights_train, histtype='step', linestyle='solid')
    plt.hist(pos_score_test, bins=50, range=(0, 1.0), color='blue', label='signal_test', weights=pos_plotting_weights_test, histtype='step', linestyle='dashed')
    plt.hist(neg_score_test, bins=50, range=(0, 1.0), color='red', label='background_test', weights=neg_plotting_weights_test, histtype='step', linestyle='dashed')    

    
    #plt.ylim(0, 0.8)
    plt.yscale("log")
    
    plt.xlabel('Classification Score')
    plt.ylabel('log(Proportion of Showers)')
    plt.legend(loc='best')
    plt.show()    
    
#################################################################################################################################
#################################################################################################################################
    
def plot_scores_branch(background_score_train, true_score_train, wrong_orientation_score_train, \
                       background_score_test, true_score_test, wrong_orientation_score_test, target_class):
    
    background_score_train = np.array(background_score_train)[:,target_class]
    true_score_train = np.array(true_score_train)[:,target_class]
    wrong_orientation_score_train = np.array(wrong_orientation_score_train)[:,target_class]
    background_score_test = np.array(background_score_test)[:,target_class]
    true_score_test = np.array(true_score_test)[:,target_class]
    wrong_orientation_score_test = np.array(wrong_orientation_score_test)[:,target_class]    
    
    background_plotting_weights_train = 1.0 / float(len(background_score_train))
    background_plotting_weights_train = np.ones(len(background_score_train)) * background_plotting_weights_train
    true_plotting_weights_train = 1.0 / float(len(true_score_train))
    true_plotting_weights_train = np.ones(len(true_score_train)) * true_plotting_weights_train    
    wrong_orientation_plotting_weights_train = 1.0 / float(len(wrong_orientation_score_train))
    wrong_orientation_plotting_weights_train = np.ones(len(wrong_orientation_score_train)) * wrong_orientation_plotting_weights_train
    background_plotting_weights_test = 1.0 / float(len(background_score_test))
    background_plotting_weights_test = np.ones(len(background_score_test)) * background_plotting_weights_test
    true_plotting_weights_test = 1.0 / float(len(true_score_test))
    true_plotting_weights_test = np.ones(len(true_score_test)) * true_plotting_weights_test    
    wrong_orientation_plotting_weights_test = 1.0 / float(len(wrong_orientation_score_test))
    wrong_orientation_plotting_weights_test = np.ones(len(wrong_orientation_score_test)) * wrong_orientation_plotting_weights_test   

    plt.hist(true_score_train, bins=50, range=(0, 1.1), color='blue', label='true_score_train', weights=true_plotting_weights_train, histtype='step', linestyle='solid')
    plt.hist(background_score_train, bins=50, range=(0, 1.1), color='red', label='background_score_train', weights=background_plotting_weights_train, histtype='step', linestyle='solid')
    plt.hist(wrong_orientation_score_train, bins=50, range=(0, 1.1), color='orange', label='wrong_orientation_score_train', weights=wrong_orientation_plotting_weights_train, histtype='step', linestyle='solid')    
    plt.hist(true_score_test, bins=50, range=(0, 1.1), color='blue', label='true_score_test', weights=true_plotting_weights_test, histtype='step', linestyle='dashed')
    plt.hist(background_score_test, bins=50, range=(0, 1.1), color='red', label='background_score_test', weights=background_plotting_weights_test, histtype='step', linestyle='dashed')
    plt.hist(wrong_orientation_score_test, bins=50, range=(0, 1.1), color='orange', label='wrong_orientation_score_test', weights=wrong_orientation_plotting_weights_test, histtype='step', linestyle='dashed')     
    
    plt.title('Background Score' if target_class == 0 else ('True Score' if target_class == 1 else 'Wrong Orientation Score'))
    plt.legend(loc='upper center')
    plt.show()        

#################################################################################################################################
#################################################################################################################################

def plot_loss_evolution(epochs, training_loss, test_loss, label):

    plt.plot(epochs, training_loss, label='Training Loss', color='b')
    plt.plot(epochs, test_loss, label='Validation Loss', color='g')

    plt.xlabel('Epochs')
    plt.title(label)
    plt.ylabel('Loss')
    plt.tick_params('y')
    plt.legend(loc='upper right')

    plt.show()
    
#################################################################################################################################
#################################################################################################################################
    
def calculate_accuracy(pos_test_score, neg_test_score):
    
    scores = torch.cat([pos_test_score, neg_test_score])
    true_labels = torch.cat([torch.ones(pos_test_score.shape[0]), torch.zeros(neg_test_score.shape[0])]).numpy()

    thresholds = torch.arange(0.05, 1.0, 0.05)
    predictions = []
    accuracy = []

    for threshold in thresholds:
        prediction = (scores >= threshold).float()
        accuracy.append(balanced_accuracy_score(true_labels, prediction.numpy()))
        predictions.append(prediction)

    max_accuracy = torch.max(torch.tensor(accuracy))
    max_accuracy_index = torch.argmax(torch.tensor(accuracy))
    optimal_threshold = thresholds[max_accuracy_index].item()

    return optimal_threshold, max_accuracy

#################################################################################################################################
#################################################################################################################################

def plot_edge_rate(epochs, correct_edge_train, incorrect_edge_train, correct_edge_test, incorrect_edge_test, is_true_positive):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    correct_label = 'positive_as_positive' if is_true_positive else 'negative_as_negative'
    incorrect_label = 'positive_as_negative' if is_true_positive else 'negative_as_positive'
    
    ax.plot(epochs, correct_edge_train, label=(correct_label + '_train'), color='g')
    ax.plot(epochs, incorrect_edge_train, label=(incorrect_label + '_train'), color='r')
    
    ax.plot(epochs, correct_edge_test, label=(correct_label + '_test'), color='g', linestyle='dashed')
    ax.plot(epochs, incorrect_edge_test, label=(incorrect_label + '_test'), color='r', linestyle='dashed')

    # Add Pandora performance    
    if is_true_positive :
        plt.axhline(y = 0.97, color = 'green', linestyle = '-') 
        plt.axhline(y = 0.03, color = 'red', linestyle = '-') 
    else :
        plt.axhline(y = 0.36, color = 'green', linestyle = '-') 
        plt.axhline(y = 0.64, color = 'red', linestyle = '-') 
    
    ax.set_title(str('Edge rate for ' + ('positive' if is_true_positive else 'negative') + ' edges'))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Proportion of Edges')
    plt.legend()
    ax.set_ylim((0,1))
    plt.show()

#################################################################################################################################
#################################################################################################################################    
    
def plot_roc_curve(pos_test_score, neg_test_score):
    scores = torch.cat([pos_test_score, neg_test_score])
    true_labels = torch.cat([torch.ones(pos_test_score.shape[0]), torch.zeros(neg_test_score.shape[0])]).numpy()    
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
    display.plot()
    plt.show()
    
#################################################################################################################################
#################################################################################################################################    
    
def draw_confusion_with_threshold(scores, true_labels, threshold):
    
    nClasses = 2
    
    scores = np.copy(scores)
    true_labels = np.copy(true_labels)
    
    scores = scores.reshape(-1)
    true_labels = true_labels.reshape(-1)
    
    predicted_true_mask = scores > threshold
    predicted_false_mask = np.logical_not(predicted_true_mask)
    scores[predicted_true_mask] = 1
    scores[predicted_false_mask] = 0

    confMatrix = confusion_matrix(true_labels, scores)
    
    print(confMatrix)
    
    trueSums = np.sum(confMatrix, axis=1)
    predSums = np.sum(confMatrix, axis=0)

    trueNormalised = np.zeros(shape=(nClasses, nClasses))
    predNormalised = np.zeros(shape=(nClasses, nClasses))

    for trueIndex in range(nClasses) : 
        for predIndex in range(nClasses) :
            nEntries = confMatrix[trueIndex][predIndex]
            if trueSums[trueIndex] > 0 :
                trueNormalised[trueIndex][predIndex] = float(nEntries) / float(trueSums[trueIndex])
            if predSums[predIndex] > 0 :
                predNormalised[trueIndex][predIndex] = float(nEntries) / float(predSums[predIndex])

    displayTrueNorm = ConfusionMatrixDisplay(confusion_matrix=trueNormalised, display_labels=["False", "True"])
    displayTrueNorm.plot()

    displayPredNorm = ConfusionMatrixDisplay(confusion_matrix=predNormalised, display_labels=["False", "True"])
    displayPredNorm.plot()    