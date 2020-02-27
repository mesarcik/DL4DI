'''
    This file contains the analytics to be automatically done after training
    Misha Mesarcik 2020
'''
import analytics 
from config import config

def get_analytics(model,X_train, X_test, y_train, y_test):
    """
        This returns the analytics defined by the architecture as specified in config
        model (keras.Model) the model from train.py
        X_train (np.array) the preprocessed data segmented into training data
        X_test (np.array) the preprocessed data segmented into test data
        y_train (np.array) the labels of the training data 
        y_test (np.array) the labels of the test data 
    """

    print('getting svm analytics')
    embedding_train,_,_ = model.get_layer('encoder').predict(X_train)
    embedding_test,_,_ = model.get_layer('encoder').predict(X_test)
    return analytics.get_svm_analytics(embedding_train, embedding_test, y_train, y_test)

