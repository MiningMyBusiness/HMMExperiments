import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

def SetConfMat(num_states):
    conf_mat = np.eye(num_states) + 0.68*np.random.rand(num_states,num_states)
    for i,row in enumerate(conf_mat):
        conf_mat[i,:] = row/np.sum(row)
    return conf_mat

def SampDist(prob_array):
    array_cdf = np.zeros(len(prob_array))
    array_cdf[0] = prob_array[0]
    for i in range(1,len(prob_array)):
        array_cdf[i] = prob_array[i] + array_cdf[i-1]
    state = np.min(np.where(array_cdf >= np.random.rand(1)))
    return state

num_classifiers = 2

for k in range(num_classifiers):
    ## load ground truth data
    all_gt = np.load('all_gt.npy')

    ## set parameters
    num_states = 12
    conf_mat = SetConfMat(num_states)

    ## get predictions
    all_pred = []

    for j,case in enumerate(all_gt):
        print('Processing case ' + str(j+1) + ' of ' + str(len(all_gt)) + '.')
        preds = np.zeros((len(case),), dtype=int)
        for i,state in enumerate(case):
            preds[i] = int(SampDist(conf_mat[state,:]))
        all_pred.append(preds)

    ## save predictions
    fileName = 'Acc_25/all_pred_' + str(k) + '.npy'
    np.save(fileName, all_pred)
