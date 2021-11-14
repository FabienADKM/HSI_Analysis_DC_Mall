# -*- coding: utf-8 -*-
"""
Remote Sensing - Hyperspectral image analysis
@author: Fabien Adokpo Migan 
"""

"""
The purpose of this script is to analyze some common techniques for the analysis of remote 
sensing on hyperspectral images. We will analyze a hyperspectral image acquired on Washington 
DC Mall a try to classify each of its pixels with a SVM classifier. To assess the performances
of each methods we will have a closer look at well chosen metrics for this tasks.
"""
import time
import scipy.io
import numpy as np


import Classifier

def compute_HSI_seg(data_array,labels,orig_labels,cl,gamma):
    X_train,y_train,X_test,y_test,scaler = Classifier.SVC_data_split(data_array,labels,orig_labels)
    svm, acc, pred = Classifier.SVC_fit(data_array,X_train,y_train,X_test,y_test,scaler,gamma,kernel = 'rbf',C=100)
    cm, df_metrics = Classifier.display_metrics(svm,cl,X_test,y_test)

    return cm, df_metrics, acc


def main_segmentation(file = 'dcmall.mat',methods = 'HSI'):
    
    cl = ['Roof', 'Street', 'Path', 'Grass', 'Trees', 'Water', 'Shadow']
    
    data = scipy.io.loadmat(file)['data']
    labels = scipy.io.loadmat(file)['test']

    
    # images are reshaped in a vectorized fashion 
    n_rows, n_columns, n_bands = data.shape
    data_array = data.reshape((n_rows * n_columns, n_bands))
    labels = labels.reshape((n_rows * n_columns, 1))  # ground truth for the classification
    
    # classes from 1 to 7 are taken for classification
    orig_labels=np.copy(labels)
    valid_indices = labels != 0
    labels = labels[valid_indices]
    
    if methods == 'HSI': 
        gamma = 1/data_array.shape[1]
        
        cm,df,acc = compute_HSI_seg(data_array,labels,orig_labels,cl,gamma)
    
    elif methods == 'PCA':
        Dp,S = Classifier.PCA(data_array)
        #Plot the eigenvalue to make sure how many axis are needed for the segmentation 
        #Here we only need two new axis
        data_PCA = Dp[:,(0,1)]
        cm,df,acc = compute_HSI_seg(data_PCA,labels,orig_labels,cl,1/2)
        
    elif methods == 'RGB':
        bR = 53
        bG = 31
        bB = 20
        cm,df,acc = compute_HSI_seg(data_array[:,(bR, bG, bB)],labels,orig_labels,cl,1/3)
        
    elif methods == 'NIR':
        bR = 53
        bG = 31
        NIR = 121
        cm,df,acc = compute_HSI_seg(data_array[:,(NIR,bR, bG)],labels,orig_labels,cl,1/3)
        
    return cm, df, acc

if __name__ == "__main__":
    #np.random.seed(42)
    methods_diff = ['HSI','PCA','RGB','NIR']
    
    for methods in methods_diff:
        start = time.time()
        cm, df, acc = main_segmentation(methods = methods)
        end = time.time()
        print(f'________________{methods}____________________')
        print('')
        print(df)
        print('')
        print(f'time of computation {end - start}')
        print('')
        print('Mean TEST Overall Accuracy (OA):', acc)
        print(f'')
        
        
        



