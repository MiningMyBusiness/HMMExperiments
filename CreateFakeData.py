import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# set initial probability distribution of semi-Markov Process 
def SetInitProb(num_states):
    init_prob = np.random.rand(num_states)
    init_prob = init_prob/np.sum(init_prob)
    return init_prob

def SetTransMat(num_states):
    trans_mat = sp.rand(num_states, num_states,
                            density=0.35).toarray()
    for i in range(num_states):
        trans_mat[i,i] = 0
    for i,row in enumerate(trans_mat):
        trans_mat[i,:] = row/np.sum(row)
    return trans_mat

def SetStateDur(num_states, avg_case_dur):
    avg_state_dur = avg_case_dur/num_states
    state_dur_mean = 8*np.sqrt(avg_state_dur)*np.random.randn(num_states) + avg_state_dur
    state_dur_mean_sqt = 8*np.sqrt(state_dur_mean)
    state_dur_std = 3*np.sqrt(np.mean(state_dur_mean_sqt))*np.random.randn(num_states) + state_dur_mean_sqt
    return state_dur_mean, state_dur_std

def SampDist(prob_array):
    array_cdf = np.zeros(len(prob_array))
    array_cdf[0] = prob_array[0]
    for i in range(1,len(prob_array)):
        array_cdf[i] = prob_array[i] + array_cdf[i-1]
    state = np.min(np.where(array_cdf >= np.random.rand(1)))
    return state

## set parameters
num_states = 12
avg_case_dur = 12000
init_prob = SetInitProb(num_states)
trans_mat = SetTransMat(num_states)
state_dur_mean, state_dur_std = SetStateDur(num_states, avg_case_dur)

## generate samples
all_gt = []
num_cases = 100

for i in range(0,num_cases):
    this_case_dur = int(8000*np.random.rand(1) + 8000)
    this_case = []
    iter = 0
    while len(this_case) < this_case_dur:
        if iter == 0:
            state = int(SampDist(init_prob))
        else:
            state = int(SampDist(trans_mat[state,:]))
        iter += 1
        this_state_dur = int(state_dur_std[state]*np.random.randn(1) + state_dur_mean[state])
        sub_list = [state]*this_state_dur
        this_case.extend(sub_list)
    this_case_arr = np.array(this_case)
    all_gt.append(this_case_arr)

np.save('all_gt.npy', all_gt)
