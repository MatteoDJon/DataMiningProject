#!/usr/bin/env python

"""
load_MITBIH.py

Download .csv files and annotations from:
    kaggle.com/mondejar/mitbih-database

VARPA, University of Coruna
Mondejar Guerra, Victor M.
23 Oct 2017
"""

import csv
import gc
import os
import pickle as pickle
import time

import numpy as np
import scipy.stats
from features_ECG import *
from scipy.signal import medfilt
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from feature_selection import *


def create_features_labels_name(DS, winL, winR, do_preprocess, maxRR, use_RR, norm_RR, compute_morph, db_path, reduced_DS, leads_flag):
    
    features_labels_name = db_path + 'features/' + 'w_' + str(winL) + '_' + str(winR) + '_' + DS 

    if do_preprocess:
        features_labels_name += '_rm_bsline'

    if maxRR:
        features_labels_name += '_maxRR'

    if use_RR:
        features_labels_name += '_RR'
    
    if norm_RR:
        features_labels_name += '_norm_RR'

    for descp in compute_morph:
        features_labels_name += '_' + descp

    if reduced_DS:
        features_labels_name += '_reduced'
        
    if leads_flag[0] == 1:
        features_labels_name += '_MLII'

    if leads_flag[1] == 1:
        features_labels_name += '_V1'

    features_labels_name += '.p'

    return features_labels_name


def save_wvlt_PCA(PCA, pca_k, family, level):
    f = open('Wvlt_' + family + '_' + str(level) + '_PCA_' + str(pca_k) + '.p', 'wb')
    pickle.dump(PCA, f, 2)
    f.close


def load_wvlt_PCA(pca_k, family, level):
    f = open('Wvlt_' + family + '_' + str(level) + '_PCA_' + str(pca_k) + '.p', 'rb')
    # disable garbage collector       
    gc.disable()# this improve the required loading time!
    PCA = pickle.load(f)
    gc.enable()
    f.close()

    return PCA
# Load the data with the configuration and features selected
# Params:
# - leads_flag = [MLII, V1] set the value to 0 or 1 to reference if that lead is used
# - reduced_DS = load DS1, DS2 patients division (Chazal) or reduced version, 
#                i.e., only patients in common that contains both MLII and V1 
def load_mit_db(DS, winL, winR, do_preprocess, maxRR, use_RR, norm_RR, compute_morph, db_path, reduced_DS, leads_flag):

    features_labels_name = create_features_labels_name(DS, winL, winR, do_preprocess, maxRR, use_RR, norm_RR, compute_morph, db_path, reduced_DS, leads_flag) 

    if os.path.isfile(features_labels_name):
        print("Loading pickle: " + features_labels_name + "...")
        f = open(features_labels_name, 'rb')
        # disable garbage collector       
        gc.disable()# this improve the required loading time!
        features, labels, patient_num_beats = pickle.load(f)
        gc.enable()
        f.close()


    else:
        print("Loading MIT BIH arr (" + DS + ") ...")

        # ML-II
        if reduced_DS == False:
            #DS1 = [101,106]
            DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230,100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
            DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

        # ML-II + V1
        else:
            DS1 = [101, 106, 108, 109, 112, 115, 118, 119, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
            DS2 = [105, 111, 113, 121, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

        mit_pickle_name = db_path + 'python_mit'
        if reduced_DS:
            mit_pickle_name = mit_pickle_name + '_reduced_'

        if do_preprocess:
            mit_pickle_name = mit_pickle_name + '_rm_bsline'

        mit_pickle_name = mit_pickle_name + '_wL_' + str(winL) + '_wR_' + str(winR)
        mit_pickle_name = mit_pickle_name + '_' + DS + '.p'

        # If the data with that configuration has been already computed Load pickle
        if os.path.isfile(mit_pickle_name):
            f = open(mit_pickle_name, 'rb')
            # disable garbage collector       
            gc.disable()# this improve the required loading time!
            my_db = pickle.load(f)
            gc.enable()
            f.close()
        else: # Load data and compute de RR features 
            if DS == 'DS1':
                my_db = load_signal(DS1, winL, winR, do_preprocess)
            else:
                my_db = load_signal(DS2, winL, winR, do_preprocess)

            print("Saving signal processed data ...")
            # Save data
            # Protocol version 0 itr_features_balanceds the original ASCII protocol and is backwards compatible with earlier versions of Python.
            # Protocol version 1 is the old binary format which is also compatible with earlier versions of Python.
            # Protocol version 2 was introduced in Python 2.3. It provides much more efficient pickling of new-style classes.
            f = open(mit_pickle_name, 'wb')
            #pickle.dump(my_db, f, 2)
            f.close
        '''
        RR_total = [[] for i in range(len(my_db.beat))]
        lbp_total = [[] for i in range(len(my_db.beat))]
        myMorph_total = [[] for i in range(len(my_db.beat))]
        HOS_total = [[] for i in range(len(my_db.beat))]
        wvlt_total = [[] for i in range(len(my_db.beat))]
        hbf_total = [[] for i in range(len(my_db.beat))]
        '''
        features = np.array([], dtype=float)
        #labels = np.array([], dtype=np.int32)

        '''

        features_N = np.array([], dtype=float)
        features_SVEB = np.array([], dtype=float)
        features_VEB = np.array([], dtype=float)
        features_F = np.array([], dtype=float)

        labels_N = np.array([], dtype=np.int32)
        labels_SVEB = np.array([], dtype=np.int32)
        labels_VEB = np.array([], dtype=np.int32)
        labels_F = np.array([], dtype=np.int32)
        '''
        # This array contains the number of beats for each patient (for cross_val)
        patient_num_beats = np.array([], dtype=np.int32)
        for p in range(len(my_db.beat)):
            patient_num_beats = np.append(patient_num_beats, len(my_db.beat[p]))

        # Compute RR features
        #f_RR = np.empty((0, 4))
        if use_RR or norm_RR:
            if DS == 'DS1':
                RR = [RR_intervals() for i in range(len(DS1))]
            else:
                RR = [RR_intervals() for i in range(len(DS2))]

            print("Computing RR intervals ...")

            for p in range(len(my_db.beat)):
                if maxRR:
                    RR[p] = compute_RR_intervals(my_db.R_pos[p])
                else:
                    RR[p] = compute_RR_intervals(my_db.orig_R_pos[p])
                    
                RR[p].pre_R = RR[p].pre_R[(my_db.valid_R[p] == 1)]
                RR[p].post_R = RR[p].post_R[(my_db.valid_R[p] == 1)]
                RR[p].local_R = RR[p].local_R[(my_db.valid_R[p] == 1)]
                RR[p].global_R = RR[p].global_R[(my_db.valid_R[p] == 1)]

                #RR_total[p].append(total_features)

                '''
                dim=0
                if maxRR:
                    dim=len(my_db.R_pos[p])
                else:
                    dim=len(my_db.orig_R_pos[p])

                for s in range(0,dim-1):
                    single_pre = RR[p].pre_R[s]
                    single_post = RR[p].post_R[s]
                    single_local = RR[p].local_R[s]
                    single_global = RR[p].global_R[s]

                    single_row = np.column_stack((single_pre,single_post,single_local,single_global))

                    RR_total[p].append(single_row)
                 '''
        f_RR_N = np.empty((0, 4))
        f_RR_SVEB = np.empty((0, 4))
        f_RR_VEB = np.empty((0, 4))
        f_RR_F = np.empty((0, 4))
        #patient_index=0
        #beat_index=0
        if use_RR:
            f_RR = np.empty((0,4))
            total_index=0
            #for p in range(len(RR)):
            for p in range(len(RR)):
                    row = np.column_stack((RR[p].pre_R, RR[p].post_R, RR[p].local_R, RR[p].global_R))
                    f_RR = np.vstack((f_RR, row))
                    '''
                    if (my_db.class_ID[p][index] == 0):
                        f_RR_N = np.vstack((f_RR_N, row))
                    elif (my_db.class_ID[p][index] == 1):
                        f_RR_SVEB = np.vstack((f_RR_SVEB, row))
                    elif (my_db.class_ID[p][index] == 2):
                        f_RR_VEB = np.vstack((f_RR_VEB, row))
                    elif (my_db.class_ID[p][index] == 3):
                        f_RR_F = np.vstack((f_RR_F, row))
                    else:
                        print("features not assigned")
                    index+=1
                    total_index+=1
                    
                    '''
                    #beat_index+=1
                    #if(beat_index>=len(my_db.beat[patient_index])):
                    #    beat_index=0
                    #   patient_index+=1

            features = np.column_stack((features, f_RR)) if features.size else f_RR

            '''
            features_N = np.column_stack((features_N, f_RR_N)) if features_N.size else f_RR_N
            features_SVEB = np.column_stack((features_SVEB, f_RR_SVEB)) if features_SVEB.size else f_RR_SVEB
            features_VEB = np.column_stack((features_VEB, f_RR_VEB)) if features_VEB.size else f_RR_VEB
            features_F = np.column_stack((features_F, f_RR_F)) if features_F.size else f_RR_F
            '''
        
        if norm_RR:
            f_RR_norm = np.empty((0,4))
            for p in range(len(RR)):
                # Compute avg values!
                avg_pre_R = np.average(RR[p].pre_R)
                avg_post_R = np.average(RR[p].post_R)
                avg_local_R = np.average(RR[p].local_R)
                avg_global_R = np.average(RR[p].global_R)

                row = np.column_stack((RR[p].pre_R / avg_pre_R, RR[p].post_R / avg_post_R, RR[p].local_R / avg_local_R, RR[p].global_R / avg_global_R))
                f_RR_norm = np.vstack((f_RR_norm, row))

            features = np.column_stack((features, f_RR_norm))  if features.size else f_RR_norm

        #########################################################################################
        # Compute morphological features
        print("Computing morphological features (" + DS + ") ...")

        num_leads = np.sum(leads_flag)

        # Raw
        if 'resample_10' in compute_morph:
            print("Resample_10 ...")
            start = time.time()

            f_raw = np.empty((0, 10 * num_leads))

            for p in range(len(my_db.beat)):
                for beat in my_db.beat[p]:
                    f_raw_lead = np.empty([])
                    for s in range(2):
                        if leads_flag[s] == 1:
                            resamp_beat = scipy.signal.resample(beat[s], 10)
                            if f_raw_lead.size == 1:
                                f_raw_lead =  resamp_beat
                            else:
                                f_raw_lead = np.hstack((f_raw_lead, resamp_beat))
                    f_raw = np.vstack((f_raw, f_raw_lead))

            features = np.column_stack((features, f_raw))  if features.size else f_raw

            end = time.time()
            print("Time resample: " + str(format(end - start, '.2f')) + " sec" )

        if 'raw' in compute_morph:
            print("Raw ...")
            start = time.time()

            f_raw = np.empty((0, (winL + winR) * num_leads))

            for p in range(len(my_db.beat)):
                for beat in my_db.beat[p]:
                    f_raw_lead = np.empty([])
                    for s in range(2):
                        if leads_flag[s] == 1:
                            if f_raw_lead.size == 1:
                                f_raw_lead =  beat[s]
                            else:
                                f_raw_lead = np.hstack((f_raw_lead, beat[s]))
                    f_raw = np.vstack((f_raw, f_raw_lead))

            features = np.column_stack((features, f_raw))  if features.size else f_raw

            end = time.time()
            print("Time raw: " + str(format(end - start, '.2f')) + " sec" )
        # LBP 1D
        # 1D-local binary pattern based feature extraction for classification of epileptic EEG signals: 2014, unas 55 citas, Q2-Q1 Matematicas
        # https://ac.els-cdn.com/S0096300314008285/1-s2.0-S0096300314008285-main.pdf?_tid=8a8433a6-e57f-11e7-98ec-00000aab0f6c&acdnat=1513772341_eb5d4d26addb6c0b71ded4fd6cc23ed5

        # 1D-LBP method, which derived from implementation steps of 2D-LBP, was firstly proposed by Chatlani et al. for detection of speech signals that is non-stationary in nature [23]

        # From Raw signal

        # TODO: Some kind of preprocesing or clean high frequency noise?

        # Compute 2 Histograms: LBP or Uniform LBP
        # LBP 8 = 0-255
        # U-LBP 8 = 0-58
        # Uniform LBP are only those pattern wich only presents 2 (or less) transitions from 0-1 or 1-0
        # All the non-uniform patterns are asigned to the same value in the histogram

        if 'u-lbp' in compute_morph:
            print("u-lbp ...")

            f_lbp = np.empty((0, 59 * num_leads))
            for p in range(len(my_db.beat)):
                for beat in my_db.beat[p]:
                    f_lbp_lead = np.empty([])
                    for s in range(2):
                        if leads_flag[s] == 1:
                            if f_lbp_lead.size == 1:
                                f_lbp_lead = compute_Uniform_LBP(beat[s], 8)
                            else:
                                f_lbp_lead = np.hstack((f_lbp_lead, compute_Uniform_LBP(beat[s], 8)))
                    f_lbp = np.vstack((f_lbp, f_lbp_lead))

            features = np.column_stack((features, f_lbp))  if features.size else f_lbp
            print(features.shape)

        f_lbp = np.empty((0, 16 * num_leads))


        f_lbp_N = np.empty((0, 16 * num_leads))
        f_lbp_SVEB = np.empty((0, 16 * num_leads))
        f_lbp_VEB = np.empty((0, 16 * num_leads))
        f_lbp_F = np.empty((0, 16 * num_leads))


        if 'lbp' in compute_morph:
            print("lbp ...")

            #f_lbp = np.empty((0, 16 * num_leads))

            for p in range(len(my_db.beat)):
                index=0
                for beat in my_db.beat[p]:
                    f_lbp_lead = np.empty([])
                    for s in range(2):
                        if leads_flag[s] == 1:
                            if f_lbp_lead.size == 1:

                                f_lbp_lead = compute_LBP(beat[s], 4)
                            else:
                                f_lbp_lead = np.hstack((f_lbp_lead, compute_LBP(beat[s], 4)))
                    '''
                    if (my_db.class_ID[p][index] == 0):
                        f_lbp_N = np.vstack((f_lbp_N, f_lbp_lead))
                        np.append(labels_N,my_db.class_ID[p][index])
                    elif (my_db.class_ID[p][index] == 1):
                        f_lbp_SVEB = np.vstack((f_lbp_SVEB, f_lbp_lead))
                        np.append(labels_SVEB,my_db.class_ID[p][index])
                    elif (my_db.class_ID[p][index] == 2):
                        f_lbp_VEB = np.vstack((f_lbp_VEB, f_lbp_lead))
                        np.append(labels_VEB,my_db.class_ID[p][index])
                    elif (my_db.class_ID[p][index] == 3):
                        f_lbp_F = np.vstack((f_lbp_F, f_lbp_lead))
                        np.append(labels_F,my_db.class_ID[p][index])
                    else:
                        print("features not assigned")
                    '''
                    index+=1
                    f_lbp = np.vstack((f_lbp, f_lbp_lead))
                    #lbp_total[p].append(f_lbp)

            features = np.column_stack((features, f_lbp))  if features.size else f_lbp

            '''
            features_N = np.column_stack((features_N, f_lbp_N)) if features_N.size else f_lbp_N
            features_SVEB = np.column_stack((features_SVEB, f_lbp_SVEB)) if features_SVEB.size else f_lbp_SVEB
            features_VEB = np.column_stack((features_VEB, f_lbp_VEB)) if features_VEB.size else f_lbp_VEB
            features_F = np.column_stack((features_F, f_lbp_F)) if features_F.size else f_lbp_F
            '''
            print(features.shape)

        f_hbf = np.empty((0, 15 * num_leads))

        f_hbf_N = np.empty((0, 15 * num_leads))
        f_hbf_SVEB = np.empty((0, 15 * num_leads))
        f_hbf_VEB = np.empty((0, 15 * num_leads))
        f_hbf_F = np.empty((0, 15 * num_leads))

        if 'hbf5' in compute_morph:
            print("hbf ...")

            #f_hbf = np.empty((0, 15 * num_leads))

            for p in range(len(my_db.beat)):
                index=0
                for beat in my_db.beat[p]:
                    f_hbf_lead = np.empty([])
                    for s in range(2):
                        if leads_flag[s] == 1:
                            if f_hbf_lead.size == 1:
                                f_hbf_lead = compute_HBF(beat[s])
                            else:
                                f_hbf_lead = np.hstack((f_hbf_lead, compute_HBF(beat[s])))
                    '''
                    if (my_db.class_ID[p][index] == 0):
                        f_hbf_N = np.vstack((f_hbf_N, f_hbf_lead))
                    elif (my_db.class_ID[p][index] == 1):
                        f_hbf_SVEB = np.vstack((f_hbf_SVEB, f_hbf_lead))
                    elif (my_db.class_ID[p][index] == 2):
                        f_hbf_VEB = np.vstack((f_hbf_VEB, f_hbf_lead))
                    elif (my_db.class_ID[p][index] == 3):
                        f_hbf_F = np.vstack((f_hbf_F, f_hbf_lead))
                    else:
                        print("features not assigned")
                    '''
                    index+=1


                    f_hbf = np.vstack((f_hbf, f_hbf_lead))

                    #hbf_total[p].append(f_hbf)

            features = np.column_stack((features, f_hbf))  if features.size else f_hbf

            '''

            features_N = np.column_stack((features_N, f_hbf_N)) if features_N.size else f_hbf_N
            features_SVEB = np.column_stack((features_SVEB, f_hbf_SVEB)) if features_SVEB.size else f_hbf_SVEB
            features_VEB = np.column_stack((features_VEB, f_hbf_VEB)) if features_VEB.size else f_hbf_VEB
            features_F = np.column_stack((features_F, f_hbf_F)) if features_F.size else f_hbf_F
            '''
            print(features.shape)

        # Wavelets
        f_wav = np.empty((0, 23 * num_leads))

        f_wav_N = np.empty((0, 23 * num_leads))
        f_wav_SVEB = np.empty((0, 23 * num_leads))
        f_wav_VEB = np.empty((0, 23 * num_leads))
        f_wav_F = np.empty((0, 23 * num_leads))


        if 'wvlt' in compute_morph:
            print("Wavelets ...")

            #f_wav = np.empty((0, 23 * num_leads))

            for p in range(len(my_db.beat)):
                index=0
                for b in my_db.beat[p]:
                    f_wav_lead = np.empty([])
                    for s in range(2):
                        if leads_flag[s] == 1:
                            if f_wav_lead.size == 1:
                                f_wav_lead =  compute_wavelet_descriptor(b[s], 'db1', 3)
                            else:
                                f_wav_lead = np.hstack((f_wav_lead, compute_wavelet_descriptor(b[s], 'db1', 3)))
                    f_wav = np.vstack((f_wav, f_wav_lead))
                    '''
                    if (my_db.class_ID[p][index] == 0):
                        f_wav_N = np.vstack((f_wav_N, f_wav_lead))
                    elif (my_db.class_ID[p][index] == 1):
                        f_wav_SVEB = np.vstack((f_wav_SVEB, f_wav_lead))
                    elif (my_db.class_ID[p][index] == 2):
                        f_wav_VEB = np.vstack((f_wav_VEB, f_wav_lead))
                    elif (my_db.class_ID[p][index] == 3):
                        f_wav_F = np.vstack((f_wav_F, f_wav_lead))
                    else:
                        print("features not assigned")
                    '''
                    index+=1

                    #wvlt_total[p].append(f_wav)
                    #f_wav = np.vstack((f_wav, compute_wavelet_descriptor(b,  'db1', 3)))

            features = np.column_stack((features, f_wav))  if features.size else f_wav

            '''
            features_N = np.column_stack((features_N, f_wav_N)) if features_N.size else f_wav_N
            features_SVEB = np.column_stack((features_SVEB, f_wav_SVEB)) if features_SVEB.size else f_wav_SVEB
            features_VEB = np.column_stack((features_VEB, f_wav_VEB)) if features_VEB.size else f_wav_VEB
            features_F = np.column_stack((features_F, f_wav_F)) if features_F.size else f_wav_F
            '''

        # Wavelets
        if 'wvlt+pca' in compute_morph:
            pca_k = 7
            print("Wavelets + PCA ("+ str(pca_k) + "...")
            
            family = 'db1'
            level = 3

            f_wav = np.empty((0, 23 * num_leads))

            for p in range(len(my_db.beat)):
                for b in my_db.beat[p]:
                    f_wav_lead = np.empty([])
                    for s in range(2):
                        if leads_flag[s] == 1:
                            if f_wav_lead.size == 1:
                                f_wav_lead =  compute_wavelet_descriptor(b[s], family, level)
                            else:
                                f_wav_lead = np.hstack((f_wav_lead, compute_wavelet_descriptor(b[s], family, level)))
                    f_wav = np.vstack((f_wav, f_wav_lead))
                    #f_wav = np.vstack((f_wav, compute_wavelet_descriptor(b,  'db1', 3)))


            if DS == 'DS1':
                # Compute PCA
                #PCA = sklearn.decomposition.KernelPCA(pca_k) # gamma_pca
                IPCA = IncrementalPCA(n_components=pca_k, batch_size=10)# NOTE: due to memory errors, we employ IncrementalPCA
                IPCA.fit(f_wav) 

                # Save PCA
                save_wvlt_PCA(IPCA, pca_k, family, level)
            else:
                # Load PCAfrom sklearn.decomposition import PCA, IncrementalPCA
                IPCA = load_wvlt_PCA( pca_k, family, level)
            # Extract the PCA
            #f_wav_PCA = np.empty((0, pca_k * num_leads))
            f_wav_PCA = IPCA.transform(f_wav)
            f_wav=f_wav_PCA
            features = np.column_stack((features, f_wav_PCA))  if features.size else f_wav_PCA

        # HOS
        n_intervals = 6
        lag = int(round((winL + winR) / n_intervals))

        f_HOS = np.empty((0, (n_intervals - 1) * 2 * num_leads))

        f_HOS_N = np.empty((0, (n_intervals - 1) * 2 * num_leads))
        f_HOS_SVEB = np.empty((0, (n_intervals - 1) * 2 * num_leads))
        f_HOS_VEB = np.empty((0, (n_intervals - 1) * 2 * num_leads))
        f_HOS_F = np.empty((0, (n_intervals - 1) * 2 * num_leads))

        if 'HOS' in compute_morph:
            print("HOS ...")
            n_intervals = 6
            lag = int(round( (winL + winR )/ n_intervals))

            #f_HOS = np.empty((0, (n_intervals-1) * 2 * num_leads))
            for p in range(len(my_db.beat)):
                index=0
                for b in my_db.beat[p]:
                    f_HOS_lead = np.empty([])
                    for s in range(2):
                        if leads_flag[s] == 1:
                            if f_HOS_lead.size == 1:
                                f_HOS_lead =  compute_hos_descriptor(b[s], n_intervals, lag)
                            else:
                                f_HOS_lead = np.hstack((f_HOS_lead, compute_hos_descriptor(b[s], n_intervals, lag)))
                    f_HOS = np.vstack((f_HOS, f_HOS_lead))
                    '''
                    if (my_db.class_ID[p][index] == 0):
                        f_HOS_N = np.vstack((f_HOS_N, f_HOS_lead))
                    elif (my_db.class_ID[p][index] == 1):
                        f_HOS_SVEB = np.vstack((f_HOS_SVEB, f_HOS_lead))
                    elif (my_db.class_ID[p][index] == 2):
                        f_HOS_VEB = np.vstack((f_HOS_VEB, f_HOS_lead))
                    elif (my_db.class_ID[p][index] == 3):
                        f_HOS_F = np.vstack((f_HOS_F, f_HOS_lead))
                    else:
                        print("features not assigned")
                    '''
                    index+=1

                    #HOS_total[p].append(f_HOS)
                    #f_HOS = np.vstack((f_HOS, compute_hos_descriptor(b, n_intervals, lag)))

            features = np.column_stack((features, f_HOS))  if features.size else f_HOS
            '''
            features_N = np.column_stack((features_N, f_HOS_N)) if features_N.size else f_HOS_N
            features_SVEB = np.column_stack((features_SVEB, f_HOS_SVEB)) if features_SVEB.size else f_HOS_SVEB
            features_VEB = np.column_stack((features_VEB, f_HOS_VEB)) if features_VEB.size else f_HOS_VEB
            features_F = np.column_stack((features_F, f_HOS_F)) if features_F.size else f_HOS_F
            '''
            print(features.shape)

        # My morphological descriptor
        f_myMorhp = np.empty((0, 4 * num_leads))

        f_myMorhp_N = np.empty((0, 4 * num_leads))
        f_myMorhp_SVEB = np.empty((0, 4 * num_leads))
        f_myMorhp_VEB = np.empty((0, 4 * num_leads))
        f_myMorhp_F = np.empty((0, 4 * num_leads))

        if 'myMorph' in compute_morph:
            print("My Descriptor ...")
            #f_myMorhp = np.empty((0,4 * num_leads))
            for p in range(len(my_db.beat)):
                index=0
                for b in my_db.beat[p]:
                    f_myMorhp_lead = np.empty([])
                    for s in range(2):
                        if leads_flag[s] == 1:
                            if f_myMorhp_lead.size == 1:
                                f_myMorhp_lead =  compute_my_own_descriptor(b[s], winL, winR)
                            else:
                                f_myMorhp_lead = np.hstack((f_myMorhp_lead, compute_my_own_descriptor(b[s], winL, winR)))
                    f_myMorhp = np.vstack((f_myMorhp, f_myMorhp_lead))
                    '''
                    if (my_db.class_ID[p][index] == 0):
                        f_myMorhp_N = np.vstack((f_myMorhp_N, f_myMorhp_lead))
                    elif (my_db.class_ID[p][index] == 1):
                        f_myMorhp_SVEB = np.vstack((f_myMorph_SVEB, f_myMorhp_lead))
                    elif (my_db.class_ID[p][index] == 2):
                        f_myMorhp_VEB = np.vstack((f_myMorhp_VEB, f_myMorhp_lead))
                    elif (my_db.class_ID[p][index] == 3):
                        f_myMorhp_F = np.vstack((f_myMorhp_F, f_myMorhp_lead))
                    else:
                        print("features not assigned")
                    '''
                    index+=1

                    #myMorph_total[p].append(f_myMorhp)

                    #f_myMorhp = np.vstack((f_myMorhp, compute_my_own_descriptor(b, winL, winR)))
                    
            features = np.column_stack((features, f_myMorhp))  if features.size else f_myMorhp
            '''
            features_N = np.column_stack((features_N, f_myMorhp_N)) if features_N.size else f_myMorhp_N
            features_SVEB = np.column_stack((features_SVEB, f_myMorhp_SVEB)) if features_SVEB.size else f_myMorhp_SVEB
            features_VEB = np.column_stack((features_VEB, f_myMorhp_VEB)) if features_VEB.size else f_myMorhp_VEB
            features_F = np.column_stack((features_F, f_myMorhp_F)) if features_F.size else f_myMorhp_F
            '''
            print(features.shape)


        #features_N = np.array([], dtype=float)
        #features_SVEB = np.array([], dtype=float)
        #features_VEB = np.array([], dtype=float)
        #features_F = np.array([],dtype=float)

        labels = []

        labels_N = []
        labels_SVEB = []
        labels_VEB = []
        labels_F = []

        index=0

        #dim_N=0
        #dim_SVEB=0
        #dim_VEB=0
        #dim_F=0

        

        for patient_index in range(0, len(my_db.beat)):
            for beat_index in range(0, len(my_db.beat[patient_index])):
                #row=features[index]
                #f_myMorhp = np.vstack((f_myMorhp, f_myMorhp_lead))
                if(my_db.class_ID[patient_index][beat_index] == 0):
                    #features_N = np.vstack((features_N, row))
                    #features_N = np.vstack((features_N, row)) if features_N.size else row
                    #labels_N.append(my_db.class_ID[patient_index][beat_index])
                    labels.append(0)
                    labels_N.append(0)
                    labels_SVEB.append(1)
                    labels_VEB.append(1)
                    labels_F.append(1)
                elif (my_db.class_ID[patient_index][beat_index] == 1):
                    #features_SVEB = np.vstack((features_SVEB, row))
                    #features_SVEB = np.vstack((features_SVEB, row)) if features_SVEB.size else row
                    #labels_SVEB.append(my_db.class_ID[patient_index][beat_index])
                    labels.append(1)
                    labels_N.append(1)
                    labels_SVEB.append(0)
                    labels_VEB.append(1)
                    labels_F.append(1)
                elif (my_db.class_ID[patient_index][beat_index] == 2):
                    #features_VEB = np.vstack((features_VEB, row))
                    #features_VEB = np.vstack((features_VEB, row)) if features_VEB.size else row
                    #labels_VEB.append(my_db.class_ID[patient_index][beat_index])
                    labels.append(2)
                    labels_N.append(1)
                    labels_SVEB.append(1)
                    labels_VEB.append(0)
                    labels_F.append(1)
                elif (my_db.class_ID[patient_index][beat_index] == 3):
                    #features_F = np.vstack((features_F, row))
                    #features_F = np.vstack((features_F, row)) if features_F.size else row
                    #labels_F.append(my_db.class_ID[patient_index][beat_index])
                    labels.append(3)
                    labels_N.append(1)
                    labels_SVEB.append(1)
                    labels_VEB.append(1)
                    labels_F.append(0)
                else:
                    print("features missing assignment")
                index+=1
        #labels_N=np.full((dim_N,1),0)
        #labels_SVEB=np.full((dim_SVEB,1),1)
        #labels_VEB=np.full((dim_VEB,1),2)
        #labels_F=np.full((dim_F,1),3)

        scaler = StandardScaler()
        scaler.fit(features)

        features_scaled=scaler.transform(features)

        #features_scaled = features

        '''

        print(features_N.shape)
        print(features_SVEB.shape)
        print(features_VEB.shape)
        print(features_F.shape)

        scalerN = StandardScaler()
        scalerN.fit(features_N)
        scalerSVEB = StandardScaler()
        scalerSVEB.fit(features_SVEB)
        scalerVEB = StandardScaler()
        scalerVEB.fit(features_VEB)
        scalerF = StandardScaler()
        scalerF.fit(features_F)

        features_N_scaled = scalerN.transform(features_N)
        features_SVEB_scaled = scalerSVEB.transform(features_SVEB)
        features_VEB_scaled = scalerVEB.transform(features_VEB)
        features_F_scaled = scalerF.transform(features_F)
        '''
        best_features = []

        print("generic features selection")
        print("info gain")
        # run_feature_selection(features_N_scaled, labels_N, "info_Gain", best_features)
        run_feature_selection(features_scaled, labels, "info_Gain", best_features)
        print("select_K_Best")
        # run_feature_selection(features_N_scaled, labels_N, "select_K_Best", best_features)
        #run_feature_selection(features_scaled, labels, "select_chi", best_features)
        print("slct_percentile")
        # run_feature_selection(features_N_scaled, labels_N, "slct_percentile", best_features)
        run_feature_selection(features_scaled, labels, "select_fclassif", best_features)

        print("class N features selection")
        print("info gain")
        #run_feature_selection(features_N_scaled, labels_N, "info_Gain", best_features)
        run_feature_selection(features_scaled, labels_N, "info_Gain", best_features)
        print("select_K_Best")
        #run_feature_selection(features_N_scaled, labels_N, "select_K_Best", best_features)
        #run_feature_selection(features_scaled, labels_N, "select_chi", best_features)
        print("slct_percentile")
        #run_feature_selection(features_N_scaled, labels_N, "slct_percentile", best_features)
        run_feature_selection(features_scaled, labels_N, "sselect_fclassif", best_features)

        print("class SVEB features selection")
        print("info gain")
        #run_feature_selection(features_SVEB_scaled, labels_SVEB, "info_Gain", best_features)
        run_feature_selection(features_scaled, labels_SVEB, "info_Gain", best_features)
        print("select_K_Best")
        #run_feature_selection(features_SVEB_scaled, labels_SVEB, "select_K_Best", best_features)
        #run_feature_selection(features_scaled, labels_SVEB, "select_chi", best_features)
        print("slct_percentile")
        #run_feature_selection(features_SVEB_scaled, labels_SVEB, "slct_percentile", best_features)
        run_feature_selection(features_scaled, labels_SVEB, "select_fclassif", best_features)

        print("class VEB features selection")
        #run_feature_selection(features_VEB_scaled, labels_VEB, "info_Gain", best_features)
        run_feature_selection(features_scaled, labels_VEB, "info_Gain", best_features)
        print("select_K_Best")
        #run_feature_selection(features_VEB_scaled, labels_VEB, "select_K_Best", best_features)
        #run_feature_selection(features_scaled, labels_VEB, "select_chi", best_features)
        print("slct_percentile")
        #run_feature_selection(features_VEB_scaled, labels_VEB, "slct_percentile", best_features)
        run_feature_selection(features_scaled, labels_VEB, "select_fclassif", best_features)

        print("class F features selection")
        #run_feature_selection(features_F_scaled, labels_F, "info_Gain", best_features)
        run_feature_selection(features_scaled, labels_F, "info_Gain", best_features)
        print("select_K_Best")
        #run_feature_selection(features_F_scaled, labels_F, "select_K_Best", best_features)
        #run_feature_selection(features_scaled, labels_F, "select_chi", best_features)
        print("slct_percentile")
        #run_feature_selection(features_F_scaled, labels_F, "slct_percentile", best_features)
        run_feature_selection(features_scaled, labels_F, "select_fclassif", best_features)


        '''
        
        modified_features=[]

        best_features=[]

        with open("C:/Users/Matteo/Desktop/database_extracted_with_features_modified.csv", 'w') as myfile:
            # fieldnames = ['beat','RR','lbp','hbf5','wvlt','HOS','myMorph','class']
            fieldnames = ['beat', 'features', 'class']
            wr = csv.DictWriter(myfile, fieldnames=fieldnames)
            wr.writeheader()
            index = 0
            for patient_index in range(0, len(my_db.beat)):
                for beat_index in range(0, len(my_db.beat[patient_index])):
                    myBeat = my_db.beat[patient_index][beat_index]
                    myClass = my_db.class_ID[patient_index][beat_index]
                    row=np.array([f_RR[index],f_lbp[index],f_hbf[index],f_wav[index],f_HOS[index],f_myMorhp[index]])
                    modified_features.append(row)
                    wr.writerow({'beat': myBeat, 'features': row, 'class': myClass})

                    # row=features[index]
                    # print(len(row))
                    # wr.writerow({'beat': myBeat, 'RR': RR_row, 'lbp': lbp_row, 'hbf5': hbf_row, 'wvlt': wvlt_row, 'HOS': hos_row,'myMorph': myMorph_row, 'class': myClass})
                    # wr.writerow({'beat': myBeat,'RR':row[0],'lbp':row[1],'hbf5':row[2],'wvlt':row[3],'HOS':row[4],'myMorph':row[5],'class': myClass})
                    index += 1
                    # wr.writerow({'beat': myBeat,'RR':RR_total[patient_index][beat_index],'lbp':lbp_total[patient_index][beat_index],'hbf5':hbf_total[patient_index][beat_index],
                    #             'wvlt':wvlt_total[patient_index][beat_index],'HOS':HOS_total[patient_index][beat_index],'myMorph':myMorph_total[patient_index][beat_index],'class': myClass})

        '''

        #run_feature_selection(modified_features, my_db.class_ID, "info_Gain", best_features)

        '''
        
        with open("C:/Users/Matteo/Desktop/database_extracted_RR.csv", 'w') as myfile:
            # fieldnames = ['beat','RR','lbp','hbf5','wvlt','HOS','myMorph','class']
            fieldnames = ['feature', 'class']
            wr = csv.DictWriter(myfile, fieldnames=fieldnames)
            wr.writeheader()
            index = 0
            for patient_index in range(0, len(my_db.beat)):
                for beat_index in range(0, len(my_db.beat[patient_index])):
                    myBeat = my_db.beat[patient_index][beat_index]
                    myClass = my_db.class_ID[patient_index][beat_index]
                    wr.writerow({'feature': f_RR[index], 'class': myClass})
                    index += 1

        with open("C:/Users/Matteo/Desktop/database_extracted_lbp.csv", 'w') as myfile:
            # fieldnames = ['beat','RR','lbp','hbf5','wvlt','HOS','myMorph','class']
            fieldnames = ['feature', 'class']
            wr = csv.DictWriter(myfile, fieldnames=fieldnames)
            wr.writeheader()
            index = 0
            for patient_index in range(0, len(my_db.beat)):
                for beat_index in range(0, len(my_db.beat[patient_index])):
                    myBeat = my_db.beat[patient_index][beat_index]
                    myClass = my_db.class_ID[patient_index][beat_index]
                    wr.writerow({'feature': f_lbp[index], 'class': myClass})

                    index += 1

        with open("C:/Users/Matteo/Desktop/database_extracted_hbf.csv", 'w') as myfile:
            # fieldnames = ['beat','RR','lbp','hbf5','wvlt','HOS','myMorph','class']
            fieldnames = ['feature', 'class']
            wr = csv.DictWriter(myfile, fieldnames=fieldnames)
            wr.writeheader()
            index = 0
            for patient_index in range(0, len(my_db.beat)):
                for beat_index in range(0, len(my_db.beat[patient_index])):
                    myBeat = my_db.beat[patient_index][beat_index]
                    myClass = my_db.class_ID[patient_index][beat_index]
                    wr.writerow({'feature': f_hbf[index], 'class': myClass})

                    index += 1

        with open("C:/Users/Matteo/Desktop/database_extracted_wvlt.csv", 'w') as myfile:
            # fieldnames = ['beat','RR','lbp','hbf5','wvlt','HOS','myMorph','class']
            fieldnames = ['feature', 'class']
            wr = csv.DictWriter(myfile, fieldnames=fieldnames)
            wr.writeheader()
            index = 0
            for patient_index in range(0, len(my_db.beat)):
                for beat_index in range(0, len(my_db.beat[patient_index])):
                    myBeat = my_db.beat[patient_index][beat_index]
                    myClass = my_db.class_ID[patient_index][beat_index]
                    wr.writerow({'feature': f_wav[index], 'class': myClass})

                    index += 1
        with open("C:/Users/Matteo/Desktop/database_extracted_HOS.csv", 'w') as myfile:
            # fieldnames = ['beat','RR','lbp','hbf5','wvlt','HOS','myMorph','class']
            fieldnames = ['feature', 'class']
            wr = csv.DictWriter(myfile, fieldnames=fieldnames)
            wr.writeheader()
            index = 0
            for patient_index in range(0, len(my_db.beat)):
                for beat_index in range(0, len(my_db.beat[patient_index])):
                    myBeat = my_db.beat[patient_index][beat_index]
                    myClass = my_db.class_ID[patient_index][beat_index]
                    wr.writerow({'feature': f_HOS[index], 'class': myClass})

                    index += 1

        with open("C:/Users/Matteo/Desktop/database_extracted_myMorph.csv", 'w') as myfile:
            # fieldnames = ['beat','RR','lbp','hbf5','wvlt','HOS','myMorph','class']
            fieldnames = ['feature', 'class']
            wr = csv.DictWriter(myfile, fieldnames=fieldnames)
            wr.writeheader()
            index = 0
            for patient_index in range(0, len(my_db.beat)):
                for beat_index in range(0, len(my_db.beat[patient_index])):
                    myBeat = my_db.beat[patient_index][beat_index]
                    myClass = my_db.class_ID[patient_index][beat_index]
                    wr.writerow({'feature': f_myMorhp[index], 'class': myClass})

                    index += 1
        
        '''

        '''
        with open("C:/Users/Matteo/Desktop/database_extracted_with_features.csv", 'w') as myfile:
            #fieldnames = ['beat','RR','lbp','hbf5','wvlt','HOS','myMorph','class']
            fieldnames = ['beat', 'features', 'class']
            wr = csv.DictWriter(myfile, fieldnames=fieldnames)
            wr.writeheader()
            index=0
            for patient_index in range(0, len(my_db.beat)):
                for beat_index in range(0, len(my_db.beat[patient_index])):
                    myBeat = my_db.beat[patient_index][beat_index]
                    myClass = my_db.class_ID[patient_index][beat_index]
                    wr.writerow({'beat': myBeat, 'features': features[index], 'class': myClass})
                    
                    #row=features[index]
                    #print(len(row))
                    #wr.writerow({'beat': myBeat, 'RR': RR_row, 'lbp': lbp_row, 'hbf5': hbf_row, 'wvlt': wvlt_row, 'HOS': hos_row,'myMorph': myMorph_row, 'class': myClass})
                    #wr.writerow({'beat': myBeat,'RR':row[0],'lbp':row[1],'hbf5':row[2],'wvlt':row[3],'HOS':row[4],'myMorph':row[5],'class': myClass})
                    index+=1
                    #wr.writerow({'beat': myBeat,'RR':RR_total[patient_index][beat_index],'lbp':lbp_total[patient_index][beat_index],'hbf5':hbf_total[patient_index][beat_index],
                    #             'wvlt':wvlt_total[patient_index][beat_index],'HOS':HOS_total[patient_index][beat_index],'myMorph':myMorph_total[patient_index][beat_index],'class': myClass})
        '''

        labels = np.array(sum(my_db.class_ID, [])).flatten()
        print("labels")

        # Set labels array!
        print('writing pickle: ' +  features_labels_name + '...')
        f = open(features_labels_name, 'wb')
        pickle.dump([features, labels, patient_num_beats], f, 2)
        f.close

    return features, labels, patient_num_beats


# DS: contains the patient list for load
# winL, winR: indicates the size of the window centred at R-peak at left and right side
# do_preprocess: indicates if preprocesing of remove baseline on signal is performed
def load_signal(DS, winL, winR, do_preprocess):

    class_ID = [[] for i in range(len(DS))]
    beat = [[] for i in range(len(DS))] # record, beat, lead
    R_poses = [ np.array([]) for i in range(len(DS))]
    Original_R_poses = [ np.array([]) for i in range(len(DS))]
    valid_R = [ np.array([]) for i in range(len(DS))]
    my_db = mit_db()
    patients = []

    # Lists
    # beats = []
    # classes = []
    # valid_R = np.empty([])
    # R_poses = np.empty([])
    # Original_R_poses = np.empty([])

    size_RR_max = 20

    #pathDB = '/home/mondejar/dataset/ECG/'
    pathDB='C:/Users/Matteo/Desktop/data_mining_prog/'
    DB_name = 'mit-bih-database'
    fs = 360
    jump_lines = 1

    # Read files: signal (.csv )  annotations (.txt)
    fRecords = list()
    fAnnotations = list()

    lst = os.listdir(pathDB + DB_name + "/csv")
    lst.sort()
    for file in lst:
        if file.endswith(".csv"):
            if int(file[0:3]) in DS:
                fRecords.append(file)
        elif file.endswith(".txt"):
            if int(file[0:3]) in DS:
                fAnnotations.append(file)

    MITBIH_classes = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F']#, 'P', '/', 'f', 'u']
    AAMI_classes = []
    AAMI_classes.append(['N', 'L', 'R'])                    # N
    AAMI_classes.append(['A', 'a', 'J', 'S', 'e', 'j'])     # SVEB
    AAMI_classes.append(['V', 'E'])                         # VEB
    AAMI_classes.append(['F'])                              # F
    #AAMI_classes.append(['P', '/', 'f', 'u'])              # Q

    RAW_signals = []
    r_index = 0

    #for r, a in zip(fRecords, fAnnotations):
    for r in range(0, len(fRecords)):

        print("Processing signal " + str(r) + " / " + str(len(fRecords)) + "...")

        # 1. Read signalR_poses
        filename = pathDB + DB_name + "/csv/" + fRecords[r]
        print(filename)
        f = open(filename, 'r')
        reader = csv.reader(f, delimiter=',')
        next(reader) # skip first line!
        MLII_index = 1
        V1_index = 2
        if int(fRecords[r][0:3]) == 114:
            MLII_index = 2
            V1_index = 1

        MLII = []
        V1 = []
        for row in reader:
            MLII.append((int(row[MLII_index])))
            V1.append((int(row[V1_index])))
        f.close()


        RAW_signals.append((MLII, V1)) ## NOTE a copy must be created in order to preserve the original signal
        # display_signal(MLII)

        # 2. Read annotations
        filename = pathDB + DB_name + "/csv/" + fAnnotations[r]
        print(filename)
        f = open(filename, 'rb')
        next(f) # skip first line!

        annotations = []
        for line in f:
            annotations.append(line)
        f.close
        # 3. Preprocessing signal!
        if do_preprocess:
            #scipy.signal
            # median_filter1D
            baseline = medfilt(MLII, 71)
            baseline = medfilt(baseline, 215)

            # Remove Baseline
            for i in range(0, len(MLII)):
                MLII[i] = MLII[i] - baseline[i]

            # TODO Remove High Freqs

            # median_filter1D
            baseline = medfilt(V1, 71)
            baseline = medfilt(baseline, 215)

            # Remove Baseline
            for i in range(0, len(V1)):
                V1[i] = V1[i] - baseline[i]


        # Extract the R-peaks from annotations
        for a in annotations:
            aS = a.split()

            pos = int(aS[1])
            originalPos = int(aS[1])
            classAnttd = aS[2]
            if pos > size_RR_max and pos < (len(MLII) - size_RR_max):
                index, value = max(enumerate(MLII[pos - size_RR_max : pos + size_RR_max]), key=operator.itemgetter(1))
                pos = (pos - size_RR_max) + index

            peak_type = 0
            #pos = pos-1
            classType=str(classAnttd)
            classType=classType.replace('b','')
            classType=classType.replace("'","")
            if classType in MITBIH_classes:
                if(pos > winL and pos < (len(MLII) - winR)):
                    beat[r].append( (MLII[pos - winL : pos + winR], V1[pos - winL : pos + winR]))
                    for i in range(0,len(AAMI_classes)):
                        if classType in AAMI_classes[i]:
                            class_AAMI = i
                            break #exit loop
                    #convert class
                    class_ID[r].append(class_AAMI)

                    valid_R[r] = np.append(valid_R[r], 1)
                else:
                    valid_R[r] = np.append(valid_R[r], 0)
            else:
                valid_R[r] = np.append(valid_R[r], 0)

            R_poses[r] = np.append(R_poses[r], pos)
            Original_R_poses[r] = np.append(Original_R_poses[r], originalPos)
        #R_poses[r] = R_poses[r][(valid_R[r] == 1)]
        #Original_R_poses[r] = Original_R_poses[r][(valid_R[r] == 1)]


    # Set the data into a bigger struct that keep all the records!
    my_db.filename = fRecords

    my_db.raw_signal = RAW_signals
    my_db.beat = beat # record, beat, lead
    my_db.class_ID = class_ID
    my_db.valid_R = valid_R
    my_db.R_pos = R_poses
    my_db.orig_R_pos = Original_R_poses

    return my_db


    '''
    
        print("Start export")

        with open("C:/Users/Matteo/Desktop/database_extracted_with_features.csv", 'w') as myfile:
            fieldnames = ['beat','RR','lbp','hbf5','wvlt','HOS','myMorph','class']
            #fieldnames = ['beat', 'features', 'class']
            wr = csv.DictWriter(myfile, fieldnames=fieldnames)
            wr.writeheader()
            index = 0
            RR = [RR_intervals() for i in range(len(DS1))]
            #f_lbp = np.empty((0, 16 * num_leads))
            for p in range(0, len(my_db.beat)):

                if maxRR:
                    RR[p] = compute_RR_intervals(my_db.R_pos[p])
                else:
                    RR[p] = compute_RR_intervals(my_db.orig_R_pos[p])

                RR[p].pre_R = RR[p].pre_R[(my_db.valid_R[p] == 1)]
                RR[p].post_R = RR[p].post_R[(my_db.valid_R[p] == 1)]
                RR[p].local_R = RR[p].local_R[(my_db.valid_R[p] == 1)]
                RR[p].global_R = RR[p].global_R[(my_db.valid_R[p] == 1)]

    
                f_RR = np.empty((0, 4))
                for c in range(len(RR)):
                    row = np.column_stack((RR[c].pre_R, RR[c].post_R, RR[c].local_R, RR[c].global_R))
                    f_RR = np.vstack((f_RR, row))
                    
                for b in range(0, len(my_db.beat[p])):

                    myBeat = my_db.beat[p][b]

                    RR_row = np.column_stack((RR[p].pre_R[b], RR[p].post_R[b], RR[p].local_R[b], RR[p].global_R[b]))
                   
                    f_lbp = np.empty((0, 16 * num_leads))

                    for p in range(len(my_db.beat)):
                        for beat in my_db.beat[p]:
                            f_lbp_lead = np.empty([])
                            for s in range(2):
                                if leads_flag[s] == 1:
                                    if f_lbp_lead.size == 1:

                                        f_lbp_lead = compute_LBP(beat[s], 4)
                                    else:
                                        f_lbp_lead = np.hstack((f_lbp_lead, compute_LBP(beat[s], 4)))
                            f_lbp = np.vstack((f_lbp, f_lbp_lead))
                            # lbp_total[p].append(f_lbp)

                    f_lbp_lead = np.empty([])

                    for s in range(2):
                        if leads_flag[s] == 1:
                            if f_lbp_lead.size == 1:

                                f_lbp_lead = compute_LBP(myBeat, 4)
                            else:
                                f_lbp_lead = np.hstack((f_lbp_lead, compute_LBP(myBeat, 4)))

                    lbp_row=f_lbp_lead

                    f_hbf_lead = np.empty([])
                    for s in range(2):
                        if leads_flag[s] == 1:
                            if f_hbf_lead.size == 1:
                                f_hbf_lead = compute_HBF(myBeat)
                            else:
                                f_hbf_lead = np.hstack((f_hbf_lead, compute_HBF(myBeat)))

                    hbf_row=f_hbf_lead

                    f_wav_lead = np.empty([])
                    for s in range(2):
                        if leads_flag[s] == 1:
                            if f_wav_lead.size == 1:
                                f_wav_lead = compute_wavelet_descriptor(myBeat, 'db1', 3)
                            else:
                                f_wav_lead = np.hstack((f_wav_lead, compute_wavelet_descriptor(myBeat, 'db1', 3)))

                    wvlt_row=f_wav_lead

                    f_HOS_lead = np.empty([])
                    for s in range(2):
                        if leads_flag[s] == 1:
                            if f_HOS_lead.size == 1:
                                f_HOS_lead = compute_hos_descriptor(myBeat, n_intervals, lag)
                            else:
                                f_HOS_lead = np.hstack((f_HOS_lead, compute_hos_descriptor(myBeat, n_intervals, lag)))

                    hos_row=f_HOS_lead

                    f_myMorhp_lead = np.empty([])
                    for s in range(2):
                        if leads_flag[s] == 1:
                            if f_myMorhp_lead.size == 1:
                                f_myMorhp_lead = compute_my_own_descriptor(myBeat, winL, winR)
                            else:
                                f_myMorhp_lead = np.hstack(
                                    (f_myMorhp_lead, compute_my_own_descriptor(myBeat, winL, winR)))

                    myMorph_row=f_myMorhp_lead

                    
                    if maxRR:
                        RR[p] = compute_RR_intervals(my_db.R_pos[p])
                    else:
                        RR[p] = compute_RR_intervals(my_db.orig_R_pos[p])

                    RR[p].pre_R = RR[p].pre_R[(my_db.valid_R[p] == 1)]
                    RR[p].post_R = RR[p].post_R[(my_db.valid_R[p] == 1)]
                    RR[p].local_R = RR[p].local_R[(my_db.valid_R[p] == 1)]
                    RR[p].global_R = RR[p].global_R[(my_db.valid_R[p] == 1)]

                    f_RR = np.empty((0, 4))
                    for p in range(len(RR)):
                        row = np.column_stack((RR[p].pre_R, RR[p].post_R, RR[p].local_R, RR[p].global_R))
                        f_RR = np.vstack((f_RR, row))

                    myClass = my_db.class_ID[p][b]
                    #wr.writerow({'beat': myBeat, 'features': features[index], 'class': myClass})
                    wr.writerow({'beat': myBeat, 'RR': RR_row, 'lbp': lbp_row, 'hbf5': hbf_row, 'wvlt': wvlt_row, 'HOS': hos_row,'myMorph': myMorph_row, 'class': myClass})

        print("Export finished")

    
    
    
    
    
    '''
