#   data extracion 
#
#
#

from load_MITBIH import * 
import csv


DS = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230, 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

winL = 90
winR = 90

do_preprocess = True


def main():
    my_db = load_signal(DS, winL, winR, do_preprocess )

    print("Saving database into csv file")

    with open( "/home/gagliardi/data_extracted.csv", 'w') as myfile:
        fieldnames = ['beat', 'class']
        wr = csv.DictWriter(myfile, fieldnames = fieldnames)
        wr.writeheader()

        for patient_index in range( 0, len( my_db.beat ) ): 
            for beat_index in range (0, len(my_db.beat[ patient_index ] )):  
                myBeat = my_db.beat[patient_index][beat_index] 
                myClass =  my_db.class_ID[patient_index][beat_index]

                wr.writerow({'beat' : myBeat, 'class' : myClass})

if __name__ == "__main__":
    main()