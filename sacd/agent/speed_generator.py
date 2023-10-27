import numpy as np
import matplotlib.pyplot as plt
import random
def generate_speed_with_random_acc(init_speed,mean,std,length):
    accelerations = [random.gauss(mean, std) for _ in range(length)]
    speed_seq = [init_speed]
    accel = 0
    k = 0
    for i in range(length-1):
        if i%10 == 0:
            accel = accelerations[k]
            k = k+1
        else:
            accel = accelerations[k]
        new_velocity = speed_seq[-1] + accel
        speed_seq.append(new_velocity)
    speed_seq = np.array(speed_seq)
    speed_seq = np.clip(speed_seq,8,20)
    return speed_seq
def normalize_speed(speed_seq):
    speed_seq = (speed_seq-8)/20
    speed_seq = np.clip(speed_seq,0,1)
    return speed_seq
def main():
    init_speed = np.random.uniform(12,18)
    mean = 0
    std = 0.1
    length = 300
    speed_seq = generate_speed_with_random_acc(init_speed,mean,std,length)
    plt.figure(1)
    plt.plot(speed_seq,color='blue')
    plt.show()

if __name__=="__main__":
    main()