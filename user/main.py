#### application started from the user

import pickle as pickle
import gc as gc
import os
import mit_db
from header_user import *


# get a new ecg file path

# perform feature selection on the new file

# perform classification using the trained svm model

# open  the svm model trained with the training set
    svm1, svm2, svm3, svm4 = load_svm_models()

    print("APPLICATION STARTED SUCCESSFULLY")
    print("Application commands are:")
    print("!quit - quit the application")
    print("!ecg_predict - perform arythmia detection on a specific ecg_signal")

while( 1 )
    cmd = input("Enter your command: ")
    if cmd.strcmp("!quit") :
        print( "See you soon!")
        exit(1)

    if cmd.strcmp("!ecg_predict"):
        print("The ecg-waves listed in the database are:")
        ecg_list = os.listdir( pathDB )
        ecg_list.sort()

        for file in ecg_list
            if file.endswith( ".csv")
                print(file)

        print("Note: to insert a new ecg-wave put its csv and annotation in: " + pathDB)

        ecg_name = input("Enter ecg-wave's name: ")

        ## open the raw_signal

        if ( not os.path.isfile( pathDB + ecg_name + ".csv" ) ) & ( not os.path.isfile( pathDB + ecg_name + ".txt" )):
            print("Uncorrect ecg entered, please retry")
            break

        print( "Loading " + ecg_name + "...")

        signal = upload_my_signal( ecg_name )

        #### TODO: signal plotting

        ####

        print( "Computing features: RR, HOS, Wavelet and Our Morph")

        print("RR ...")

        features_RR = compute_RR_intervals(signal.R_pos)

        print("HOS ...")

        f_HOS = np.empty((0, (n_intervals - 1) * 2 * num_leads))
        for b in signal.beat[p]:
            f_HOS_lead = np.empty([])
            f_HOS_lead = compute_hos_descriptor(b[s], n_intervals, lag)
            f_HOS = np.vstack((f_HOS, f_HOS_lead))

        features_HOS = np.column_stack((features_HOS, f_HOS))

        print("Wavelet ...")
        f_wav = np.empty((0, 23 * num_leads))

        for b in signal.beat:
            f_wav_lead = np.empty([])
            f_wav_lead = compute_wavelet_descriptor(b, 'db1', 3)
            f_wav = np.vstack((f_wav, f_wav_lead))

        features_WVL = np.column_stack((features_WVL, f_wav))

        print("My Descriptor ...")

        f_myMorhp = np.empty((0, 4 * num_leads))
        for b in signal.beat:
            f_myMorhp_lead = np.empty([])
            f_myMorhp_lead = compute_my_own_descriptor(b, winL, winR)
            f_myMorhp = np.vstack((f_myMorhp, f_myMorhp_lead))

        features_myMorhp = np.column_stack((features_myMorhp, f_myMorhp))

        print("Result Prediction ...")
            res1 = svm1.predict(features_RR)
            res2 = svm2.predict(features_HOS)
            res3 = svm3.predict(features_WVL)
            res4 = svm4.predict(features_myMorhp)

        print("Combining Prediction ...")




    else
        print("Command not found!")
        print("Application commands are:")
        print("!quit - quit the application")
        print("!ecg_predict - get an ecg path and perform arythmia detection")
