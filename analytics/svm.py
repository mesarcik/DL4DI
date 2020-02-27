"""
    This file performs the SVM based analytics after training 
    Misha Mesarcik 2020
"""
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from config import *

def get_svm_analytics(embedding_train,
                      embedding_test,
                      y_train,
                      y_test,
                      verbose=False):
    """
        This function returns the confusion matrix a classification_report from linear SVM
        embedding_train (np.array) 2D numpy array of the embedding of training data
        embedding_test  (np.array) 2D numpy array of the embedding of test data
        y_train (np.array) 1D numpy array of the labels for training data
        y_test  (np.array) 1D numpy array of the labels for training data
    """
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(embedding_train, y_train)
    y_pred = svclassifier.predict(embedding_test)

    return confusion_matrix(y_test,y_pred),pd.DataFrame(classification_report(y_test,
                                                                              y_pred,
                                                                              output_dict=True)).transpose()
