import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(numberFile):

    path_directory="C:/Users/Matteo/Desktop/data_mining_prog/mit-bih-database/csv/"
    path_file=path_directory+numberFile+".csv"
    signal=pd.read_csv(path_file)
    MLII_X=signal[["'MLII'"]]
    V5_X=signal[["'V5'"]]
    dim=len(MLII_X)
    fs=360
    timeArray=np.arange(dim)/360
    fig,axs=plt.subplots(2,1,'all')
    plt.subplots_adjust(hspace=0.5)
    ax1=axs[0]
    ax2=axs[1]
    #ax1=fig.add_subplot(1,1,1)
    ax1.set_xlabel("time in s")
    ax1.set_ylabel("ECG in mV")
    ax1.set_title("MLII")
    #ax1.get_lines.set_color_cycle('r')
    #ax2=fig.add_subplot(2,1,1)
    ax2.set_xlabel("time in s")
    ax2.set_ylabel("ECG in mV")
    ax2.set_title("V5")
     #ax2.get_lines.set_color_cycle('r')
    index=0
    fig.show()
    while(index<dim-1):
        xs1 = MLII_X[index*360:(index + 8)*360]
        xs2 = V5_X[index * 360:(index + 8) * 360]
        ys1 = timeArray[index*360:(index + 8)*360]
        ys2 = timeArray[index * 360:(index + 8) * 360]
        ax1.plot(ys1,xs1)
        ax2.plot(ys2,xs2)
        fig.canvas.draw()
        leftVal=timeArray[index*360]
        rightVal=timeArray[(index+8)*360]
        ax1.set_xlim(left=leftVal,right=rightVal)
        ax2.set_xlim(left=leftVal, right=rightVal)
        time.sleep(0.2)
        plt.pause(0.2)

        index+=1


if __name__ == '__main__':


    main("100")
