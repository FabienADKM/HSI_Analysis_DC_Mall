# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 22:23:37 2021

@author: fabmi
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import imshow_stretch, gen_train_test, morph_prof
import seaborn as sns
import pandas as pd


def PCA(data_array):
    """
    Realise the principal components analysis
    Inputs::
        data_array (array) : array containing the HyperSpectral Image (HSI)
    Outputs::
        Dp (array) : PCA matrix
        S (list): eigenvalues list
    """
    Sigma = np.cov(data_array, rowvar=False)  # covariance matrix
    [U, S, V] = np.linalg.svd(Sigma)  # U: eigenvectors of Sigma, S: eigenvalues of Sigma
    Dp = np.dot(data_array, U)  # linear transformation
    return Dp,S


def plot_eigenvalues_PCA(k,data_array):
    _,S = PCA(data_array)
    plt.figure()
    plt.plot(S[:k])
    plt.title('Eigenvalues')
    plt.show()

    
def SVC_data_split(data_array,labels,orig_labels,n_samples_per_class = 40):
    
    classes = np.unique(labels)
    #n_classes = classes.shape[0]
    tr_idx, ts_idx = gen_train_test(orig_labels, labels, classes, n_samples_per_class)

    X_train, y_train = data_array[tr_idx].astype('double'), orig_labels[tr_idx].ravel()
    X_test, y_test = data_array[ts_idx].astype('double'), orig_labels[ts_idx].ravel()
    # data scaling: they will have zero mean and unit standard deviation
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train,y_train,X_test,y_test,scaler


def SVC_fit(data_array,X_train,y_train,X_test,y_test,scaler,gamma,kernel = 'rbf',C=100):
    #seed = np.random.seed(42)
    # Classification with SVM
    # gamma = 1/data_array.shape[1]
    svm = SVC(kernel=kernel, C=C, gamma=gamma)
    svm.fit(X_train, y_train)
    # Inference and accuracy calculation
    pred = svm.predict(scaler.transform(data_array))  # classification of the total image
    acc = svm.score(X_test, y_test)  # TEST mean accuracy (n. correctly classified test samples / total n. test samples)
    #print('Mean TEST Overall Accuracy (OA):', acc)
    return svm, acc, pred


def display_metrics(svm,cl,X_test,y_test):
    """
    Computes three differents metrics for the Svm classifier and for each class:
    Recall = TP/(TP+FN)
    Precision = TP/(TP+FP)
    F1-Score = 2/(1/P+1/R)
    
    Inputs::
        svm (sklearn.svm): trained model of svm classifier
        cl (list): list of all the classes of the dataset
        
    Outputs::
        cm (array): confusion matrix
        df_metrics(Pandas.DataFrame): Dataframe containing the values of the three metrics 
                                      for each classes
    """
    #cl = ['Roof', 'Street', 'Path', 'Grass', 'Trees', 'Water', 'Shadow']
    y_predicted = svm.predict(X_test)
    cm =confusion_matrix(y_test, y_predicted)
    p = precision_score(y_test, y_predicted, average=None)
    r = recall_score(y_test, y_predicted, average=None)
    fs = f1_score(y_test, y_predicted, average=None)
    
    df_metrics = pd.DataFrame({'Precision':p,
                               'Recall':r,
                               'F1 Score': fs},
                            index = cl)
    
    return cm, df_metrics


def plot_confusion_matrix(cl,X_test,y_test):
    cm,_ = display_metrics(cl,X_test,y_test)
    df_cm = pd.DataFrame(cm, index = [i for i in cl],
                  columns = [i for i in cl])
    plt.figure()
    sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted classes')
    plt.ylabel('Actual classes')
    plt.title('Confusion matrix')
    plt.show()


