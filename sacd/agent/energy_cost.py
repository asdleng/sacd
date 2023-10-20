import numpy as np
import matplotlib.pyplot as plt
m = 1200
A = 2.5
umax = 3
Cd = 0.32
pa = 1.184
b0 = 0.1569
b1 = 0.0245
b2 = -7.415e-4
b3 = 5.975e-5
c0 = 0.0722
c1 = 0.0968
c2 = 0.0011
Fd = 0.1000
g = 0.81
miu = 0.0012
def energy_cost(v,acc):
    energy = 0
    for i in range(len(v)):
        if v[i] < 0.1 or acc[i]<0:
            z = 1
        else:
            z = 0
        f_cruise = b0+b1*v[i]+b2*v[i]**2+b3*v[i]**3
        f_acc = acc[i]*(c0+c1*v[i]+c2*v[i]**2)
        f = (1-z)*(f_cruise + f_acc)+z*Fd
        energy+=f
    return energy
def true_segment_length(my_list):
    current_segment_length = 0
    max_segment_length = 0

    for value in my_list:
        if value:
            current_segment_length += 1
        else:
            max_segment_length = max(max_segment_length, current_segment_length)
            current_segment_length = 0

    # Check for the max segment length one more time in case it ends with True values
    max_segment_length = max(max_segment_length, current_segment_length)

    return max_segment_length
def main():
    scenario_num = 6
    dis_file = "/home/i/sacd/ref_spd_data/distance"+str(scenario_num)+".csv"
    intersection_file = "/home/i/sacd/ref_spd_data/intersections"+str(scenario_num)+".csv"
    spd_file = "/home/i/sacd/ref_spd_data/speed"+str(scenario_num)+".csv"
    save_file = "/home/i/sacd/ref_spd_data/scenario"+str(scenario_num)
    dis = np.genfromtxt(dis_file, delimiter=',')
    intersection = np.genfromtxt(intersection_file, delimiter=',')
    spd = np.genfromtxt(spd_file, delimiter=',')

    plt.plot(dis,spd,color='green')
    inter_dis = 50
    spd_segs = []
    dis_segs = []
    last_inter = -100
    target_times = []
    for inter in intersection:
        index = np.array((dis<inter-inter_dis)*(dis>last_inter+inter_dis))
        target_t = true_segment_length(index)
        target_times.append(target_t)
        spd_segs.append(spd[index])
        dis_segs.append(dis[index])
        plt.plot(dis[index],spd[index],color='blue')
        plt.plot([inter,inter],[0,20],color='red')
        last_inter = inter
    plt.show()
    plt.pause(0.001)
    # 保存每段目标时间
    with open(save_file+'/target_times.txt', "w") as file:
            np.savetxt(file, target_times,newline="\n",fmt="%.2f")
    # 保存速度-距离
    with open(save_file+'/spd_seg.txt', "w") as file:
        for spd_seg in spd_segs:
            np.savetxt(file, spd_seg.reshape(1, -1), delimiter=",",newline="\n",fmt="%.2f")
    with open(save_file+'/dis_seg.txt', "w") as file:
        for dis_seg in dis_segs:
            np.savetxt(file, dis_seg.reshape(1, -1), delimiter=",",newline="\n",fmt="%.2f")
    
    # 插值到每米一个参考速度
    with open(save_file+'/spd_seg_lin.txt', "w") as file:
        for i in range(len(dis_segs)):
            x = dis_segs[i][0]
            vs = []
            while x<dis_segs[i][-1]:
                v = np.interp(x,dis_segs[i],spd_segs[i])
                vs.append(v)
                x+=1
            vs = np.array(vs)
            np.savetxt(file, vs.reshape(1, -1), delimiter=",",newline="\n",fmt="%.2f")

    acc = np.diff(spd)
    spd = spd[:-1]
    energy = energy_cost(spd,acc)
    print(energy)
    
    return
if __name__=="__main__":
    main()