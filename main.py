#-*- coding: utf-8 -*-


import gzip
import cPickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from multi_layer_km import test_SdC
from cluster_acc import acc

K = 10
trials = 1

filename = 'data.pkl.gz'
path = './data/'
dataset = path+filename



# perform DCN

#  need to train with 250 epochs of layerwise, and 250 epochs of end-end SAE
#  to get the initialization file with the following setting, takes a while


config = {'Init': 'data.pkl.gz',
          'lbd':  1,  # reconstruction
          'beta': 1,
          'output_dir': 'Xiang_results',
          'save_file': 'Xiang_10.pkl.gz',
          'pretraining_epochs': 250,
          'pretrain_lr_base': 0.0001,
          'mu': 0.9,
          'finetune_lr': 0.0001,
          'training_epochs': 50,
          'dataset': dataset,
          'batch_size': 256,
          'nClass': K,
          'hidden_dim': [500, 500, 2000, 10],
          'diminishing': False}

results = []
for i in range(trials):
    res_metrics = test_SdC(**config)
    results.append(res_metrics)

results_SAEKM = np.zeros((trials, 3))
results_DCN = np.zeros((trials, 3))

N = config['training_epochs']/5
for i in range(trials):
    results_SAEKM[i] = results[i][0]
    results_DCN[i] = results[i][N]
SAEKM_mean = np.mean(results_SAEKM, axis=0)
SAEKM_std = np.std(results_SAEKM, axis=0)
DCN_mean = np.mean(results_DCN, axis=0)
DCN_std = np.std(results_DCN, axis=0)

results_all = np.concatenate((DCN_mean, DCN_std, SAEKM_mean, SAEKM_std),
                             axis=0)
print(results_all)
np.savetxt('Xiang_results.txt', results_all, fmt='%.3f')

