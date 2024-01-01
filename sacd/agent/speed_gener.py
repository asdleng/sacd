import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from statsmodels.tsa.arima.model import ARIMA
def normalize_speed(speed_seq,upper,lower):
    speed_seq = (speed_seq-lower)/(upper-lower)
    speed_seq = np.clip(speed_seq,0,1)
    return speed_seq

def load_spd_data():
    spd = []
    spd_sv = []
    for i in range(6):
        spd_file = "/home/i/sacd/ref_spd_data/speed"+str(i+1)+".csv"
        spd.append(np.genfromtxt(spd_file, delimiter=','))
        spd_sv.append(convert_tv_to_sv(spd[i]))
    return spd_sv
    
def random_select_spd(spd_seqs,l):
    start_p = []
    for i in range(len(spd_seqs)):
        start_p.append(len(spd_seqs[i])-l-1)
    index_seg = np.random.choice(range(len(spd_seqs)))
    index_start_p = np.random.choice(range(start_p[index_seg]))
    return np.array(spd_seqs[index_seg][index_start_p:index_start_p+l])

def convert_tv_to_sv(spd):
    dis = np.cumsum(spd)
    new_spd = []
    for x in range(int(dis[-1])):
        new_spd.append(np.interp(x,dis,spd))
    return new_spd
def main():
    spd_sv = load_spd_data()
    select_spd = random_select_spd(spd_sv,300)

    plt.figure(figsize=(10, 6))
    plt.plot(select_spd, label='Random Selected Trajectory')
    plt.xlabel('Distance (m)')
    plt.ylabel('Speed (m/s)')
    plt.title('Random Selected Trajectory')
    plt.legend()
    plt.show()

if __name__=="__main__":
    main()
