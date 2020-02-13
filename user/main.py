#### application started from the user

import pickle as pickle
import gc as gc
import os
from features_ECG import *

from header import *

from basic_fusion import *

# get a new ecg file path

# perform feature selection on the new file

# perform classification using the trained svm model

def main():

#open  the svm model trained with the training set

    svm1, svm2, svm3, svm4 = load_svm_models()
    print("APPLICATION STARTED SUCCESSFULLY")
    print("Application commands are:")
    print("!quit - quit the application")
    print("!ecg_predict - perform arythmia detection on a specific ecg_signal")

    while( 1 ):
        cmd = input("Enter your command: ")
        if cmd=="!quit" :
            print( "See you soon!")
            exit(1)

        if cmd=="!ecg_predict":
            print("The ecg-waves listed in the database are:")
            ecg_list = os.listdir( pathDB )
            ecg_list.sort()

            for file in ecg_list:
                if file.endswith( ".csv"):
                    print(file)

            print("Note: to insert a new ecg-wave put its csv and annotation in: " + pathDB)

            ecg_name = input("Enter ecg-wave's name: ")

            ## open the raw_signal

            if ( not os.path.isfile( pathDB +"csv/"+ ecg_name + ".csv" ) ) & ( not os.path.isfile( pathDB +"csv/"+ ecg_name + ".txt" )):
                print("Uncorrect ecg entered, please retry")
                break

            print( "Loading " + ecg_name + "...")

            signal = upload_my_signal( ecg_name )

            #### TODO: signal plotting

            ####

            print( "Computing features: RR, HOS, Wavelet and Our Morph")

            print("RR ...")

            f_RR = np.array([], dtype=float)

            RR=RR_intervals()

            if maxRR:
                RR = compute_RR_intervals(signal.R_pos)
            else:
                RR = compute_RR_intervals(signal.orig_R_pos)

            RR.pre_R = RR.pre_R[(signal.valid_R == 1)]
            RR.post_R = RR.post_R[(signal.valid_R == 1)]
            RR.local_R = RR.local_R[(signal.valid_R == 1)]
            RR.global_R = RR.global_R[(signal.valid_R == 1)]

            f_RR = np.empty((0, 4))
            row = np.column_stack((RR.pre_R, RR.post_R, RR.local_R, RR.global_R))
            f_RR = np.vstack((f_RR, row))

            features_RR = np.array([], dtype=float)

            features_RR = np.column_stack((features_RR, f_RR))  if features_RR.size else f_RR


            #features_RR = compute_RR_intervals(signal.R_pos)

            print("HOS ...")

            f_HOS = np.empty((0, (n_intervals - 1) * 2 * num_leads))
            for b in signal.beat:
                f_HOS_lead = np.empty([])
                f_HOS_lead = compute_hos_descriptor(b, n_intervals, lag)
                f_HOS = np.vstack((f_HOS, f_HOS_lead))

            features_HOS = np.array([], dtype=float)

            features_HOS = np.column_stack((features_HOS, f_HOS))  if features_HOS.size else f_HOS

            leads_flag = [1, 0]

            print("Wavelet ...")
            f_wav = np.empty((0, 23 * num_leads))

            for b in signal.beat:
                f_wav_lead = np.empty([])
                for s in range(2):
                    if leads_flag[s] == 1:
                        if f_wav_lead.size == 1:
                            f_wav_lead = compute_wavelet_descriptor(b[s], 'db1', 3)
                        else:
                            f_wav_lead = np.hstack((f_wav_lead, compute_wavelet_descriptor(b[s], 'db1', 3)))
                f_wav = np.vstack((f_wav, f_wav_lead))
                # f_wav = np.vstack((f_wav, compute_wavelet_descriptor(b,  'db1', 3)))

            features_WVL = np.array([], dtype=float)

            features_WVL = np.column_stack((features_WVL, f_wav))  if features_WVL.size else f_wav

            print("My Descriptor ...")


            f_myMorhp = np.empty((0, 4 * num_leads))
            for b in signal.beat:
                f_myMorhp_lead = np.empty([])
                for s in range(2):
                    if leads_flag[s] == 1:
                        if f_myMorhp_lead.size == 1:
                            f_myMorhp_lead = compute_my_own_descriptor(b[s], winL, winR)
                        else:
                            f_myMorhp_lead = np.hstack((f_myMorhp_lead, compute_my_own_descriptor(b[s], winL, winR)))
                f_myMorhp = np.vstack((f_myMorhp, f_myMorhp_lead))
                # f_myMorhp = np.vstack((f_myMorhp, compute_my_own_descriptor(b, winL, winR)))

            features_myMorhp = np.array([], dtype=float)

            features_myMorhp = np.column_stack((features_myMorhp, f_myMorhp))  if features_myMorhp.size else f_myMorhp

            print("Result Prediction ...")

            decision_ovo1 = svm1.decision_function(features_RR)
            decision_ovo2 = svm2.decision_function(features_HOS)
            decision_ovo3 = svm3.decision_function(features_WVL)
            decision_ovo4 = svm4.decision_function(features_myMorhp)

            predict_RR, prob_ovo_RR_sig = ovo_voting_exp(decision_ovo1,4)
            predict_HOS, prob_ovo_HOS_sig = ovo_voting_exp(decision_ovo2, 4)
            predict_WVL, prob_ovo_WVL_sig = ovo_voting_exp(decision_ovo3, 4)
            predict_MyMorph, prob_ovo_MyMorph_sig = ovo_voting_exp(decision_ovo4, 4)

            probs_ensemble = np.stack((prob_ovo_RR_sig, prob_ovo_WVL_sig, prob_ovo_HOS_sig, prob_ovo_MyMorph_sig))

            n_ensembles, n_instances, n_classes = probs_ensemble.shape

            predictions_prob_rule = basic_rules(probs_ensemble, 0)

            print(predictions_prob_rule)

            '''
            res1 = svm1.predict(features_RR)
            res2 = svm2.predict(features_HOS)
            res3 = svm3.predict(features_WVL)
            res4 = svm4.predict(features_myMorhp)
            
            '''



            #print("Combining Prediction ...")

        else:
            print("Command not found!")
            print("Application commands are:")
            print("!quit - quit the application")
            print("!ecg_predict - get an ecg path and perform arythmia detection")


if __name__ == '__main__':

    import sys

    main()
