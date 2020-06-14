import gc

import csv

from basic_fusion import *
from scipy.signal import *
import joblib as pickle
from tkinter import *
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import mutual_info_classif

model_path = "../models_18/info_svm"

pathDB = "../db/"

MLII_index = 1
V1_index = 2

MITBIH_classes = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F'] # non viene considerata Q = '/', 'f', 'Q'
Macro_classes = ['N','SVEB','VEB','F']
AAMI_classes = []

size_RR_max = 20

maxRR=True

### TODO: scegliere come settare i seguenti parametri
winL = 90
winR = 90
num_leads = 1
n_intervals = 6
lag = int(round((winL + winR) / n_intervals))
leads_flag = [1, 0]
n_intervals = 6
lag = int(round((winL + winR) / n_intervals))
###

def load_model():
    f = open(model_path, "rb")
    gc.disable()
    model = pickle.load(f)
    gc.enable()
    f.close()
    return model

class my_signal:
    def __INIT__(self):
        self.RAW_SIGNAL = []
        self.MLII = []
        self.V1 = []
        self.beat = []
        self.valid_R = []
        self.R_pos = []
        self.orig_R_pos = []

def upload_my_signal( ecg_name ):
    s = my_signal()

    s.RAW_SIGNAL, s.MLII, s.V1 = load_ecg_signal(pathDB+"csv/" +ecg_name + ".csv")
    s.MLII, s.V1 = preprocess_signal(s.MLII, s.V1)

    s_ann = load_annotation(pathDB+"csv/"+ecg_name + "annotations.txt", s.RAW_SIGNAL)

    s.beat, s.valid_R, s.R_pos, s.orig_R_pos = R_peaks_extraction(s.MLII, s.V1, s_ann)

    return s

def load_ecg_signal( path_to_ecg ):
    # 1. Load signal

    # 1.1 data record
    f = open(path_to_ecg , 'rt')
    reader = csv.reader(f, delimiter=',')
    next(reader)  # skip first line!

    MLII = []
    V1 = []
    RAW_SIGNAL = []

    for row in reader:
        MLII.append((int(row[MLII_index])))
        V1.append((int(row[V1_index])))
    f.close()

    RAW_SIGNAL.append((MLII, V1)) # copy of the raw signal

    return RAW_SIGNAL, MLII, V1

def load_annotation( path_to_ecg, RAW_SIGNAL):
    f = open(path_to_ecg , 'rb')
    next(f) # skip first line

    annotations = []
    for line in f:
        annotations.append(line)
    f.close()

    return annotations

def preprocess_signal(MLII, V1):
    # 2. Preprocess the signal

    # median_filter1D
    baseline = medfilt(MLII, 71)
    baseline = medfilt(baseline, 215)

    # Remove Baseline
    for i in range(0, len(MLII)):
        MLII[i] = MLII[i] - baseline[i]

    baseline = medfilt(V1, 71)
    baseline = medfilt(baseline, 215)

    # Remove Baseline
    for i in range(0, len(V1)):
        V1[i] = V1[i] - baseline[i]

        # 1.2 data annotation
        beat = []  # record, beat, lead
        valid_R = []
        R_pos = []
        orig_R_pos = []

    return MLII, V1

def R_peaks_extraction(MLII, V1, ANNOTATIONS):
    beat = []
    valid_R = []
    R_pos = []
    orig_R_pos = []
    # Extract the R-peaks from annotations
    for a in ANNOTATIONS:
        aS = a.split()
        pos = int(aS[1])
        originalPos = int(aS[1])
        if pos > size_RR_max and pos < (len(MLII) - size_RR_max):
            index, value = max(enumerate(MLII[pos - size_RR_max : pos + size_RR_max]), key=operator.itemgetter(1))
            pos = (pos - size_RR_max) + index
        peak_type = 0
        if (pos > winL and pos < (len(MLII) - winR)):
            beat.append((MLII[pos - winL: pos + winR], V1[pos - winL: pos + winR]))
            valid_R = np.append(valid_R, 1)
        else:
            valid_R = np.append(valid_R, 0)
        R_pos = np.append(R_pos, pos)
        orig_R_pos = np.append(orig_R_pos, originalPos)

    return beat, valid_R, R_pos, orig_R_pos

def compute_features(signal):

    features = np.array([], dtype=float)

    patient_num_beats = np.array([], dtype=np.int32)
    for p in range(len(signal.beat)):
        patient_num_beats = np.append(patient_num_beats, len(signal.beat[p]))

    print("Computing features: RR, HOS, Wavelet, Our Morph, Lbp and Hbf ")

    print("RR ...")

    f_RR = np.array([], dtype=float)

    RR = RR_intervals()

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

    features = np.column_stack((features, f_RR)) if features.size else f_RR

    print("lbp ...")

    f_lbp = np.empty((0, 16 * num_leads))

    for beat in signal.beat:
        f_lbp_lead = np.empty([])
        for s in range(2):
            if leads_flag[s] == 1:
                if f_lbp_lead.size == 1:
                    f_lbp_lead = compute_LBP(beat[s], 4)
                else:
                    f_lbp_lead = np.hstack((f_lbp_lead, compute_LBP(beat[s], 4)))
        f_lbp = np.vstack((f_lbp, f_lbp_lead))

    features = np.column_stack((features, f_lbp)) if features.size else f_lbp

    print("hbf ...")

    f_hbf = np.empty((0, 15 * num_leads))

    for beat in signal.beat:
        f_hbf_lead = np.empty([])
        for s in range(2):
            if leads_flag[s] == 1:
                if f_hbf_lead.size == 1:
                    f_hbf_lead = compute_HBF(beat[s])
                else:
                    f_hbf_lead = np.hstack((f_hbf_lead, compute_HBF(beat[s])))

        f_hbf = np.vstack((f_hbf, f_hbf_lead))

    features = np.column_stack((features, f_hbf)) if features.size else f_hbf

    f_wav = np.empty((0, 23 * num_leads))

    print("Wavelets ...")

    for b in signal.beat:
        f_wav_lead = np.empty([])
        for s in range(2):
            if leads_flag[s] == 1:
                if f_wav_lead.size == 1:
                    f_wav_lead = compute_wavelet_descriptor(b[s], 'db1', 3)
                else:
                    f_wav_lead = np.hstack((f_wav_lead, compute_wavelet_descriptor(b[s], 'db1', 3)))
        f_wav = np.vstack((f_wav, f_wav_lead))

    features = np.column_stack((features, f_wav)) if features.size else f_wav

    print("HOS ...")

    f_HOS = np.empty((0, (n_intervals - 1) * 2 * num_leads))

    for b in signal.beat:
        f_HOS_lead = np.empty([])
        for s in range(2):
            if leads_flag[s] == 1:
                if f_HOS_lead.size == 1:
                    f_HOS_lead = compute_hos_descriptor(b[s], n_intervals, lag)
                else:
                    f_HOS_lead = np.hstack((f_HOS_lead, compute_hos_descriptor(b[s], n_intervals, lag)))
        f_HOS = np.vstack((f_HOS, f_HOS_lead))

    features = np.column_stack((features, f_HOS)) if features.size else f_HOS

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

    features = np.column_stack((features, f_myMorhp)) if features.size else f_myMorhp

    return features

def fit_features(features):

    scaler = StandardScaler()
    scaler.fit(features)
    tr_features_scaled = scaler.transform(features)

    return tr_features_scaled

def features_reduction( tr_features ):

    #selection_indexes = [0, 1, 40, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 70]
    selection_indexes = [0, 1, 40, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 70]
    reduction_indexes = []

    '''
    if type_features == 0 : selection_indexes = get_info_selection(tr_features, tr_labels, num_features)
    elif type_features == 1: selection_indexes = get_f_selection(tr_features, tr_labels, num_features)
    else:
        print("ERRORE NEL TIPO DI FEATURES DA SELEZIONARE")
        exit(-1)
    '''

    for i in range( len( tr_features[0, :] ) ):
        present = False
        for s in selection_indexes:
            if s == i: present = True
        if present == False : reduction_indexes.append( i )

    tr_features = np.delete( tr_features, reduction_indexes  , axis = 1)

    return tr_features

def do_prediction(model,features):

    features_to_select = features_reduction(features)

    prediction = model.predict(features_to_select)

    return prediction



def macro_classes_assignment(assigned_class):
    if(assigned_class>=0 and assigned_class<=2):
        return 0
    elif(assigned_class>3 and assigned_class<=8):
        return 1
    elif(assigned_class>8 and assigned_class<=10):
        return 2
    else:
        return 3

def printRawPrediction(predictions_prob_rule,num_file):

    dimClasses=len(Macro_classes)
    countArray=np.zeros(dimClasses)
    dim=len(predictions_prob_rule)
    completeName=(pathDB+"results/"+num_file+"result.txt")
    f=open(completeName,"w")
    for i in range(0, dim):
        macro_class = macro_classes_assignment(int(predictions_prob_rule[i]))
        countArray[macro_class]=(countArray[macro_class]+1)
        line=(str(i+1)+","+Macro_classes[macro_class]+"\n")
        f.write(line)
    f.close()
    for j in range(0,dimClasses):
        if(countArray[j]>0):
            print(str((countArray[j]/dim)*100)+"% of the beats have been classified as belonging to the class "+Macro_classes[j])
    print("The complete prediction has been saved into the file "+num_file+"result.txt in the directory results")

def visualize_ecg(signal):

    tk=Tk()
    MLII_X = signal.MLII
    V5_X = signal.V1
    dim = len(MLII_X)
    timeArray = np.arange(dim) / 360
    fig, axs = plt.subplots(2, 1, 'all')
    plt.subplots_adjust(hspace=0.5)
    ax1 = axs[0]
    ax2 = axs[1]
    ax1.set_xlabel("time in s")
    ax1.set_ylabel("ECG in mV")
    ax1.set_title("MLII")
    ax2.set_xlabel("time in s")
    ax2.set_ylabel("ECG in mV")
    ax2.set_title("V5")
    index = 0
    fig.show()
    try:
        while (index < dim - 1):
            ys1 = MLII_X[index * 360:(index + 8) * 360]
            ys2 = V5_X[index * 360:(index + 8) * 360]
            xs1 = timeArray[index * 360:(index + 8) * 360]
            xs2 = timeArray[index * 360:(index + 8) * 360]
            ax1.plot(xs1, ys1)
            ax2.plot(xs2, ys2)
            fig.canvas.draw()
            leftVal = timeArray[index * 360]
            rightVal = timeArray[(index + 8) * 360]
            ax1.set_xlim(left=leftVal, right=rightVal)
            ax2.set_xlim(left=leftVal, right=rightVal)
            time.sleep(0.2)
            plt.pause(0.2)
            index += 1
    except:
        tk.destroy()