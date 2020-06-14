#### application started from the user

import pickle as pickle
import gc as gc
import os
import numpy
from features_ECG import *

from header import *

#from basic_fusion import *

# get a new ecg file path

# perform feature selection on the new file

# perform classification using the trained svm model

def main():

#open  the svm model trained with the training set

    model = load_model()
    print("APPLICATION STARTED SUCCESSFULLY")
    print("Application commands are:")
    print("!quit - quit the application")
    print("!ecg_predict - perform arythmia detection on a specific ecg_signal")

    while( 1 ):
        cmd = input("Enter your command: ")
        if cmd=="!quit" :
            print( "See you soon!")
            exit(1)

        elif cmd=="!ecg_predict":
            print("The ecg-waves listed in the database are:")
            ecg_list = os.listdir( pathDB +"csv/")
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

            visualize_ecg(signal)

            ####

            features = compute_features(signal)

            features_scaled = fit_features(features)

            prediction = do_prediction(model,features_scaled)

            print("Result Prediction ...")
            printRawPrediction(prediction,ecg_name)

        else:
            print("Command not found!")
            print("Application commands are:")
            print("!quit - quit the application")
            print("!ecg_predict - get an ecg path and perform arythmia detection")


if __name__ == '__main__':

    import sys

    main()
