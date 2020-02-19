import gc

import csv

from scipy.signal import *
from features_ECG import *
from basic_fusion import *
import joblib as pickle

svm_path1 = "ovo_rbf_MLII_rm_bsln_maxRR_RR__weighted_C_0.001.joblib.pkl"
svm_path2 = "ovo_rbf_MLII_rm_bsln_maxRR_HOS_weighted_C_0.001.joblib.pkl"
svm_path3 = "ovo_rbf_MLII_rm_bsln_maxRR_wvlt_weighted_C_0.001.joblib.pkl"
svm_path4 = "ovo_rbf_MLII_rm_bsln_maxRR_myMorph_weighted_C_0.001.joblib.pkl"

pathDB = "C:/Users/Matteo/Desktop/data_mining_prog/db/"

MLII_index = 1
V1_index = 2

MITBIH_classes = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F'] # non viene considerata Q = '/', 'f', 'Q'
AAMI_classes = []

size_RR_max = 20

maxRR=True

### TODO: scegliere come settare i seguenti parametri
winL = 90
winR = 90
num_leads = 1
n_intervals = 6
lag = int(round((winL + winR) / n_intervals))
###

def load_svm_models():
    AAMI_classes.append(['N', 'L', 'R'])  # N
    AAMI_classes.append(['A', 'a', 'J', 'S', 'e', 'j'])  # SVEB
    AAMI_classes.append(['V', 'E'])  # VEB
    AAMI_classes.append(['F'])  # F

    print("Loading pickle " + svm_path1 + " of the models ...")
    f = open(pathDB + "svm_models/" + svm_path1, "rb")
    gc.disable()
    svm1 = pickle.load(f)
    gc.enable()
    f.close()

    print("Loading pickle " + svm_path2 + " of the models ...")
    f = open(pathDB + "svm_models/" + svm_path2, "rb")
    gc.disable()
    svm2 = pickle.load(f)
    gc.enable()
    f.close()

    print("Loading pickle " + svm_path3 + " of the models ...")
    f = open(pathDB + "svm_models/" + svm_path3, "rb")
    gc.disable()
    svm3 = pickle.load(f)
    gc.enable()
    f.close()

    print("Loading pickle " + svm_path4 + " of the models ...")
    f = open(pathDB + "svm_models/" + svm_path4, "rb")
    gc.disable()
    svm4 = pickle.load(f)
    gc.enable()
    f.close()

    return svm1, svm2, svm3, svm4

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
        #pos = pos-1
        if (pos > winL and pos < (len(MLII) - winR)):
            beat.append((MLII[pos - winL: pos + winR], V1[pos - winL: pos + winR]))
            valid_R = np.append(valid_R, 1)
        else:
            valid_R = np.append(valid_R, 0)
        R_pos = np.append(R_pos, pos)
        orig_R_pos = np.append(orig_R_pos, originalPos)

    return beat, valid_R, R_pos, orig_R_pos

def compute_features(signal):

    print("Computing features: RR, HOS, Wavelet and Our Morph")

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

    features_RR = np.array([], dtype=float)

    features_RR = np.column_stack((features_RR, f_RR)) if features_RR.size else f_RR

    # features_RR = compute_RR_intervals(signal.R_pos)

    print("HOS ...")

    f_HOS = np.empty((0, (n_intervals - 1) * 2 * num_leads))
    for b in signal.beat:
        f_HOS_lead = np.empty([])
        f_HOS_lead = compute_hos_descriptor(b, n_intervals, lag)
        f_HOS = np.vstack((f_HOS, f_HOS_lead))

    features_HOS = np.array([], dtype=float)

    features_HOS = np.column_stack((features_HOS, f_HOS)) if features_HOS.size else f_HOS

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

    features_WVL = np.column_stack((features_WVL, f_wav)) if features_WVL.size else f_wav

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

    features_myMorhp = np.column_stack((features_myMorhp, f_myMorhp)) if features_myMorhp.size else f_myMorhp

    return features_RR,features_HOS,features_WVL,features_myMorhp

def fit_features(features_RR,features_HOS,features_WVL,features_myMorhp):
    scalerRR = StandardScaler()
    scalerRR.fit(features_RR)
    tr_features_scaled_RR = scalerRR.transform(features_RR)

    scalerHOS = StandardScaler()
    scalerHOS.fit(features_HOS)
    tr_features_scaled_HOS = scalerHOS.transform(features_HOS)

    scalerWVLT = StandardScaler()
    scalerWVLT.fit(features_WVL)
    tr_features_scaled_WVLT = scalerWVLT.transform(features_WVL)

    scalerMy = StandardScaler()
    scalerMy.fit(features_myMorhp)
    tr_features_scaled_MyMorph = scalerMy.transform(features_myMorhp)

    return tr_features_scaled_RR,tr_features_scaled_HOS,tr_features_scaled_WVLT,tr_features_scaled_MyMorph

def prediction(svm1,svm2,svm3,svm4,tr_features_scaled_RR,tr_features_scaled_HOS,tr_features_scaled_WVLT,tr_features_scaled_MyMorph):

    decision_ovo1 = svm1.decision_function(tr_features_scaled_RR)
    decision_ovo2 = svm2.decision_function(tr_features_scaled_HOS)
    decision_ovo3 = svm3.decision_function(tr_features_scaled_WVLT)
    decision_ovo4 = svm4.decision_function(tr_features_scaled_MyMorph)

    predict_RR, prob_ovo_RR_sig = ovo_voting_exp(decision_ovo1, 4)
    predict_HOS, prob_ovo_HOS_sig = ovo_voting_exp(decision_ovo2, 4)
    predict_WVL, prob_ovo_WVL_sig = ovo_voting_exp(decision_ovo3, 4)
    predict_MyMorph, prob_ovo_MyMorph_sig = ovo_voting_exp(decision_ovo4, 4)

    probs_ensemble = np.stack((prob_ovo_RR_sig, prob_ovo_WVL_sig, prob_ovo_HOS_sig, prob_ovo_MyMorph_sig))

    predictions_prob_rule = basic_rules(probs_ensemble, 0)

    return predictions_prob_rule

def printRawPrediction(predictions_prob_rule):
    dimClasses=len(MITBIH_classes)
    countArray=np.zeros(dimClasses)
    dim=len(predictions_prob_rule)
    for i in range(0, dim):
        countArray[int(predictions_prob_rule[i])]=(countArray[int(predictions_prob_rule[i])]+1)
    for j in range(0,dimClasses):
        if(countArray[j]>0):
            print(str((countArray[j]/dim)*100)+"% of the beats have been classified as belonging to the class "+MITBIH_classes[j])

def visualize_ecg(signal,predictions_prob_rule):

    MLII_X = signal.MLII
    V5_X = signal.V1
    positions=signal.R_pos
    dimPos=len(positions)
    dim = len(MLII_X)
    fs = 360
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
    # ax2.get_lines.set_color_cycle('r')
    index = 0
    fig.show()
    while (index < dim - 1):
        xs1 = MLII_X[index * 360:(index + 8) * 360]
        xs2 = V5_X[index * 360:(index + 8) * 360]
        ys1 = timeArray[index * 360:(index + 8) * 360]
        ys2 = timeArray[index * 360:(index + 8) * 360]
        ax1.plot(ys1, xs1)
        ax2.plot(ys2, xs2)

        for j in range(0,dimPos):
            if(positions[j]>=(index * 360) and positions[j]<=((index + 8) * 360)):
                ax1.annotate(MITBIH_classes[int(predictions_prob_rule[j])],xy=(positions[j],xs1[j]))
                ax2.annotate(MITBIH_classes[int(predictions_prob_rule[j])], xy=(positions[j],xs2[j]))

        fig.canvas.draw()
        leftVal = timeArray[index * 360]
        rightVal = timeArray[(index + 8) * 360]
        ax1.set_xlim(left=leftVal, right=rightVal)
        ax2.set_xlim(left=leftVal, right=rightVal)
        time.sleep(0.2)
        plt.pause(0.2)

        index += 1