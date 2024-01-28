import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

import scipy.io as scio
dataFile = '/home/i/sacd/ref_spd_data/ele_para.mat'
par = scio.loadmat(dataFile)
par = par['par']
mas = par['mas'][0][0][0][0]
wlr = par['wlr'][0][0][0][0]
fdg = par['fdg'][0][0][0][0]
gav = par['gav'][0][0][0][0]
Cr1 = par['Cr1'][0][0][0][0]
Cr2 = par['Cr2'][0][0][0][0]
rho = par['rho'][0][0][0][0]
ACd = par['ACd'][0][0][0][0]
Mot_maxbr = par['Mot_maxbr'][0][0][0][0]
Mot_Tindx = par['Mot_Tindx'][0][0][0]
Mot_Sindx = par['Mot_Sindx'][0][0][0]
Mot_maxtq = par['Mot_maxtq'][0][0][0]
Mot_map = par['Mot_map'][0][0]
Mot_map = np.nan_to_num(Mot_map, nan=0.0)
Discharge_eff = par['Discharge_eff'][0][0][0][0]
Charge_eff = par['Charge_eff'][0][0][0][0]
Trans_eff = par['Trans_eff'][0][0][0][0]

def energy_cost_motor(v,acc):
    interp_eff = interp2d(Mot_Sindx, Mot_Tindx, Mot_map, kind='linear')
    energy = 0
    mot_spd = v/wlr*fdg
    Ft = mas*acc + 0.5 * ACd * rho * v * v + (Cr1+ Cr2*v) * mas * gav
    mot_tq = Ft*wlr/Trans_eff/fdg;
    mot_tqb = mot_tq.copy()
    for i in range(len(mot_tq)):
        if mot_tq[i]<0.0:
            mot_tq[i] = 0.0
        if mot_tqb[i]>0.0:
            mot_tqb[i] = 0.0
    eff_T = interp_eff(mot_spd, mot_tq).diagonal()
    eff_B = interp_eff(mot_spd, -mot_tqb).diagonal()
    Energy_cost = mot_spd*mot_tq/eff_T/Discharge_eff
    Brake_cost = mot_spd*mot_tqb*eff_B *Charge_eff
    energy = sum(Energy_cost+Brake_cost)
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

def gen_spd_segs(dis_file,intersection_file,spd_file,save_file):
    dis = np.genfromtxt(dis_file, delimiter=',')
    intersection = np.genfromtxt(intersection_file, delimiter=',')
    spd = np.genfromtxt(spd_file, delimiter=',')
    acc = np.diff(spd)
    acc = np.append(acc,np.array([0]))
    energy = energy_cost_motor(spd,acc)
    print(energy)

    plt.plot(dis,spd,color='green')
    inter_dis = 00
    spd_segs = []
    dis_segs = []
    last_inter = -inter_dis*2
    target_times = []
    for inter in intersection:
        index = np.array((dis<=inter-inter_dis)*(dis>=last_inter+inter_dis))
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

def main():
    scenario_num = 6

    intersection_file = "/home/i/sacd/ref_spd_data/intersections"+str(scenario_num)+".csv"
    # 3-stage
    dis_file = "/home/i/sacd/ref_spd_data/distance"+str(scenario_num)+".csv"
    spd_file = "/home/i/sacd/ref_spd_data/speed"+str(scenario_num)+".csv"
    save_file = "/home/i/sacd/ref_spd_data/scenario"+str(scenario_num)
    # IDM
    idm_dis_file = "/home/i/sacd/ref_spd_data/idm_distance"+str(scenario_num)+".csv"
    idm_spd_file = "/home/i/sacd/ref_spd_data/idm_speed"+str(scenario_num)+".csv"
    save_idm_file = "/home/i/sacd/ref_spd_data/idm_scenario"+str(scenario_num)



    gen_spd_segs(dis_file,intersection_file,spd_file,save_file)
    gen_spd_segs(idm_dis_file,intersection_file,idm_spd_file,save_idm_file)
    
    return
if __name__=="__main__":
    main()