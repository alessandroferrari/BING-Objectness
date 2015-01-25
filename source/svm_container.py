'''
Created on Jan 8, 2015

@author: alessandro
'''
import os
import cv2
import numpy as np
from py_liblinear import PyLibLinear
from random import shuffle
from functools import partial
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from sklearn.svm import LinearSVC, SVC
from sklearn.grid_search import GridSearchCV
from multiprocessing import Pool as ProcessesPool
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score


CUSTOM_LIBLINEAR_WRAPPER = 0
SKLEARN_LIBLINEAR_WRAPPER = 1

#solver types svm
L2R_LR = 0 
L2R_L2LOSS_SVC_DUAL = 1 
L2R_L2LOSS_SVC = 2 
L2R_L1LOSS_SVC_DUAL = 3 
MCSVM_CS = 4 
L1R_L2LOSS_SVC = 5 
L1R_LR = 6 
L2R_LR_DUAL = 7 
L2R_L2LOSS_SVR = 11 


def shuffle_dataset(X,y):
    
    nr = X.shape[0]
    
    idxs = range(nr)
    shuffled_idxs = shuffle(idxs)
    
    shuffled_X = X[shuffled_idxs,:]
    shuffled_y = y[shuffled_idxs]
    
    return shuffled_X, shuffled_y


def predict(X, weights, bias, intercept=0.0):
    
    res = np.sum(X*weights, axis=1) + bias
    
    return (res>intercept).astype(int)


def dtype_ensure(X_train, y_train, X_test = None,  y_test = None, dtype_ref = np.float32):
    
    if X_train.dtype != np.float32:
        X_train = X_train.astype(np.float32)
    if y_train.dtype != np.float32:
        y_train = y_train.astype(np.float32)
    if not X_test is None and X_test.dtype != np.float32:
        X_test = X_test.astype(np.float32)
    if not y_test is None and y_test.dtype != np.float32:
        y_test = y_test.astype(np.float32)
        
    return X_train, y_train, X_test, y_test


def fit_svm_custom_wrapper_liblinear_with_cv(C, X_train, X_test, y_train, y_test):
                    
    X_train, y_train, X_test, y_test = dtype_ensure(X_train, y_train, X_test, y_test)
            
    p = PyLibLinear()
    w = p.trainSVM(X_train,y_train,L1R_L2LOSS_SVC,C,1.0,0.0001)
    
    weights = w[0,:-1].ravel()
    bias = w[0,-1]
    y_pred = predict(X_test, weights, bias, intercept = 0.5)
    
    y_tr_pred = predict(X_train, weights, bias, intercept = 0.5)
    
    accuracy, recall, precision, f1, tr_err = classifier_statistics(y_train, y_test, y_tr_pred, y_pred)
    
    return C, accuracy, recall, precision, f1, tr_err


def fit_svm_sklearn_liblinear_with_cv(C, X_train, X_test, y_train, y_test):
        
    X_train, y_train, X_test, y_test = dtype_ensure(X_train, y_train, X_test, y_test)
            
    clf = LinearSVC(C=C, dual=False,penalty='l1',loss='l2', class_weight='auto')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    y_tr_pred = clf.predict(X_train)
    
    accuracy, recall, precision, f1, tr_err = classifier_statistics(y_train, y_test, y_tr_pred, y_pred)
    
    return C, accuracy, recall, precision, f1, tr_err


def fit_svm_with_cv(C, X_train, X_test, y_train, y_test, wrapper_type = CUSTOM_LIBLINEAR_WRAPPER):

    if wrapper_type == CUSTOM_LIBLINEAR_WRAPPER:
        return fit_svm_custom_wrapper_liblinear_with_cv(C, X_train, X_test, y_train, y_test)
    elif wrapper_type == SKLEARN_LIBLINEAR_WRAPPER:
        return fit_svm_sklearn_liblinear_with_cv(C, X_train, X_test, y_train, y_test)
    
    raise Exception("Invalid wrapper type selected!")


def fit_svm_sklearn_liblinear(X, y, C = 1.0):
    
    clf = LinearSVC(C=C, dual=False,penalty='l1',loss='l2', class_weight='auto')
    clf.fit(X, y)
    weights = clf.coef_
    bias = clf.intercept_
    
    return weights, bias


def fit_svm_custom_wrapper_liblinear(X, y, C = 1.0):
    
    p = Prova()
    w = p.trainSVM(X,y,L1R_L2LOSS_SVC,C,1.0,0.01)
    weights = w[0,:-1].ravel()
    bias = w[0,-1]
    
    return weights, bias


def fit_svm(X, y, C = 1.0, wrapper_type = CUSTOM_LIBLINEAR_WRAPPER):

    X, y, _, _ = dtype_ensure(X, y)
    
    if wrapper_type == CUSTOM_LIBLINEAR_WRAPPER:
        return fit_svm_custom_wrapper_liblinear(X, y, C = C)
    elif wrapper_type == SKLEARN_LIBLINEAR_WRAPPER:
        return fit_svm_sklearn_liblinear(X, y, C = C)
    
    raise Exception("Invalid wrapper_type selected!")


def classifier_statistics(y_train, y_test, y_tr_pred, y_pred):
    
    tr_err = accuracy_score(y_train, y_tr_pred)
    
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return accuracy, recall, precision, f1, tr_err


def plot_2d( x, y, xlabel = "log10(C)" , ylabel = "accuracy", title = "Accuracy by C"):

    fig = plt.figure()
    fig.canvas.set_window_title('{0}'.format(title))
    ax = fig.add_subplot(111)

    ax_xlabel = ax.set_xlabel(xlabel)
    ax_ylabel = ax.set_ylabel(ylabel)

    x=np.log10(x)
    surf = ax.plot(x, y)
    ax.set_ylim(0, 1.01)
    
    
def weights_representation(weights, gradient_edge, repr_edge = 400):
    
    w = np.reshape(weights, (8,8))
    wmin = np.min(w)
    wmax = np.max(w)
    w = ((w - wmin)/(wmax-wmin))*255
    w = w.astype(np.uint8)
        
    return cv2.resize(w,(repr_edge, repr_edge), interpolation = cv2.INTER_NEAREST)

def model_selection(X, y, C_list = None, wrapper_type = CUSTOM_LIBLINEAR_WRAPPER, results_dir = None, gradient_edge = None):
                    
    if C_list is None:
        C_list = [10.0**(-4),10.0**(-3),10.0**(-2),10.0**(-1),10.0,10.0**2,10.0**3,10.0**4]
    
    if len(C_list)>1:
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
        
        func = partial(fit_svm_with_cv, X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, wrapper_type = wrapper_type)
        
        num_cpus = cpu_count()
        
        num_processes = min(len(C_list),num_cpus-1)
        
        pool = ProcessesPool(num_processes)
        cv_results = pool.map(func, C_list)
        
        C_array = np.array(C_list)
        accuracy_array = np.zeros(len(C_list))
        precision_array = np.zeros(len(C_list))
        recall_array = np.zeros(len(C_list))
        f1_score_array = np.zeros(len(C_list))
        tr_err_array = np.zeros(len(C_list))
        
        for cv_result in cv_results:
            C, accuracy, recall, precision, f1, tr_err = cv_result
            idx = C_list.index(C)
            accuracy_array[idx] = accuracy
            recall_array[idx] = recall
            precision_array[idx] = precision
            f1_score_array[idx] = f1
            tr_err_array[idx] = tr_err
            
        plot_2d(C_array,accuracy_array,xlabel = "log10(C)" , ylabel = "accuracy", title = "Accuracy by C")
        plot_2d(C_array,precision_array,xlabel = "log10(C)" , ylabel = "precision", title = "Precision by C")
        plot_2d(C_array,recall_array,xlabel = "log10(C)" , ylabel = "recall", title = "Recall by C")
        plot_2d(C_array,f1_score_array,xlabel = "log10(C)" , ylabel = "f1 score", title = "f1 score by C")
        plot_2d(C_array,tr_err_array,xlabel = "log10(C)" , ylabel = "training error", title = "Training error by C")
        plt.show()
        
        max_score_idx = np.argmax(f1_score_array)
        
        selected_C = C_array[max_score_idx]
        rec = recall_array[max_score_idx]
        acc = accuracy_array[max_score_idx]
        prec = precision_array[max_score_idx]
        f1 = f1_score_array[max_score_idx]
        tr_err = tr_err_array[max_score_idx]
        
        print "Selected C = {0}. Accuracy = {1:.2f}. Precision = {2:.2f}. Recall = {3:.2f}. f1 score = {4:.2f}. Training error = {5:.2f}.".format(selected_C,acc*100,prec*100,rec*100,f1*100,tr_err*100)
    
    elif len(C_list) == 1:
        
        selected_C = C_list[0]
    
    else:
        
        selected_C = 10.0        
    
    weights, bias = fit_svm(X,y,C=selected_C, wrapper_type = wrapper_type)
    
    if not results_dir is None:
        if not os.path.exists(results_dir):
            raise Exception("The destination path %s suggested to save the averaged dataset does not exist!"%results_dir)
        w = weights_representation(weights, gradient_edge)
        cv2.imwrite(os.path.join(results_dir, "weights.png"),w)
        
    return weights, bias