import gc
import pickle

from scipy.signal import *
from features_ECG import *

svm_path1 = ""
svm_path2 = ""
svm_path3 = ""
svm_path4 = ""

pathDB = "/Users/guido/Desktop/DataMining/mitbih-database/csv"

MLII_index = 1
V1_index = 2

MITBIH_classes = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F'] # non viene considerata Q = '/', 'f', 'Q'
AAMI_classes = []

size_RR_max = 20

### TODO: scegliere come settare i seguenti parametri
winL = 1
winR = 1
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
    f = open( svm_path1, "rb" )
    gc.disable()
    svm1 = pickle.load(f)
    gc.enable()
    f.close()

    print("Loading pickle " + svm_path2 + " of the models ...")
    f = open( svm_path2, "rb" )
    gc.disable()
    svm2 = pickle.load(f)
    gc.enable()
    f.close()

    print("Loading pickle " + svm_path3 + " of the models ...")
    f = open( svm_path3, "rb" )
    gc.disable()
    svm3 = pickle.load(f)
    gc.enable()
    f.close()

    print("Loading pickle " + svm_path3 + " of the models ...")
    f = open( svm_path4, "rb" )
    gc.disable()
    svm4 = pickle.load(f)
    gc.enable()
    f.close()

    return svm1,svm2,svm3,svm4

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

    s.RAW_SIGNAL = load_ecg_signal( ecg_name + ".csv")
    s.MLII, s.V1 = preprocess_signal(s.RAW_SIGNAL)

    s_ann = load_annotation(ecg_name + ".txt", s.RAW_SIGNAL)

    s.beat, s.valid_R, s.R_pos, s.orig_R_pos = R_peaks_extraction(s.MLII, s.V1, s_ann)

    return s

def load_ecg_signal( path_to_ecg ):
    # 1. Load signal

    # 1.1 data record
    f = open(path_to_ecg + ".csv", 'rt')
    reader = csv.reader(f, delimiter=',')
    next(reader)  # skip first line!

    MLII = []
    V1 = []
    RAW_SIGNAL = []

    for row in reader:
        MLII.append((int(row[MLII_index])))
        V1.append((int(row[V1_index])))
    f.close()

    RAW_SIGNAL.append(MLII, V1) # copy of the raw signal

    return RAW_SIGNAL

def load_annotation( path_to_ecg, RAW_SIGNAL):
    f = open(path_to_ecg + ".txt", 'rb')
    next(f) # skip first line

    annotations = []
    for line in f:
        annotations.append(line)
    f.close()

    return annotations

def preprocess_signal(RAW_SIGNAL):
    MLII, V1 = RAW_SIGNAL

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
        classAnttd = aS[2]

        if pos > size_RR_max and pos < (len(MLII) - size_RR_max):
            index, value = max(enumerate(MLII[pos - size_RR_max: pos + size_RR_max]), key=operator.itemgetter(1))
            pos = (pos - size_RR_max) + index

        if classAnttd in MITBIH_classes: # non considero la classe AAMI 'Q' per questo filtro
            if (pos > winL and pos < (len(MLII) - winR)):
                beat.append((MLII[pos - winL: pos + winR], V1[pos - winL: pos + winR]))

                valid_R = np.append(valid_R, 1)
            else:
                valid_R  = np.append(valid_R, 0)
        else:
            valid_R = np.append(valid_R, 0)

        R_pos = np.append(R_pos, pos)
        orig_R_pos = np.append(orig_R_pos, originalPos)

    return beat, valid_R, R_pos, orig_R_pos



