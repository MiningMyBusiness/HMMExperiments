import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm

## load data
print('Loading data...')
all_gt = np.load('all_gt.npy')
all_pred = np.load('Acc_65/all_pred_0.npy')

num_sims = 5
num_states = 12
save_diagonals = np.zeros((num_sims,num_states))

for qqq in range(0,num_sims):
    print('DOING SIMULATION ' + str(qqq+1)  + ' OF ' + str(num_sims) + '.')
    ## define how to split data
    print(' Defining data splits...')
    ignore = 70 # to test how little data we need
    est_trans = 80
    est_conf = 0
    fit_params = 10
    re_order = np.random.permutation(len(all_gt))

    ## split data
    print(' Splitting data...')
    gt_est_trans = all_gt[re_order[ignore:est_trans]]
    gt_est_conf = all_gt[re_order[est_trans:est_trans+est_conf]]
    gt_fit = all_gt[re_order[est_trans+est_conf:est_trans+est_conf+fit_params]]
    gt_test = all_gt[re_order[est_trans+est_conf+fit_params:]]

    pred_est_trans = all_pred[re_order[ignore:est_trans]]
    pred_est_conf = all_pred[re_order[est_trans:est_trans+est_conf]]
    pred_fit = all_pred[re_order[est_trans+est_conf:est_trans+est_conf+fit_params]]
    pred_test = all_pred[re_order[est_trans+est_conf+fit_params:]]

    ## estimate init prob
    print(' Estimating start probabilities...')
    init_prob = np.ones(num_states)
    for case in all_gt[re_order[ignore:est_trans+fit_params]]:
        first_state = case[0]
        init_prob[first_state] += 1
    init_prob = init_prob/np.sum(init_prob)

    ## estimate transition matrix
    print(' Estimating transition probabilities...')
    trans_mat = np.ones((num_states,num_states))
    for case in all_gt[re_order[ignore:est_trans+fit_params]]:
        for i in range(1,len(case)):
            prev_state = case[i-1]
            curr_state = case[i]
            trans_mat[prev_state,curr_state] += 1
    for i,row in enumerate(trans_mat):
        trans_mat[i,:] = row/np.sum(row)

    ## estimate emission matrix
    print(' Estimating emission probabilities...')
    emiss_mat = np.ones((num_states, num_states))
    for i in range(0,len(gt_fit)):
        case = gt_fit[i]
        preds = pred_fit[i]
        for gt,pred in zip(case,preds):
            emiss_mat[gt,pred] += 1
    for i,row in enumerate(emiss_mat):
        emiss_mat[i,:] = row/np.sum(row)
    acc_oa = np.nanmean(np.diagonal(emiss_mat))

    ## define hmm model
    print(' Defining HMM model...')
    model = hmm.MultinomialHMM(n_components=num_states, n_iter=10,
                               params='ste', init_params='')
    model.startprob_=init_prob
    model.transmat_=trans_mat
    model.emissionprob_=emiss_mat

    ## organize examples to fit model on
    print(' Organizing data to refine HMM parameters...')
    lengths = np.zeros((len(pred_fit),), dtype=int)
    all_x = np.array([pred_fit[0]]).T
    lengths[0] = len(pred_fit[0])
    for i in range(1,len(pred_fit)):
        this_x = np.array([pred_fit[i]]).T
        all_x = np.concatenate([all_x, this_x])
        lengths[i] = len(pred_fit[i])

    # fit model
    print(' Fitting HMM model...')
    model = model.fit(all_x, lengths)

    # decode test runs
    print(' Decoding test runs...')
    decode_test = []
    for case in pred_test:
        decode_state = model.decode(np.array([case]).T)
        decode_test.append(decode_state[1])

    # get confusion mat of decoded vals
    decoded_conf_mat = np.zeros((num_states,num_states))
    for case_gt,case_decode in zip(gt_test,decode_test):
        for gt,decode in zip(case_gt,case_decode):
            decoded_conf_mat[gt,decode] +=1
    for i,row in enumerate(decoded_conf_mat):
        decoded_conf_mat[i,:] = row/np.sum(row)

    save_diagonals[qqq,:] = np.diagonal(decoded_conf_mat)
