# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:19:11 2024

@author: daizh
"""

#%% import pakage
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

#%% import data
# import function
os.chdir(os.path.dirname(sys.argv[0]))
from import_data import *
#  work path
wd_ = 'D:/sheared_granular_gouge_different_evironment_condition'
os.chdir(wd_)

# set data path
path_FrictionState_ = 'particle_info/particle_info_friction_state.txt'
path_ParticleInfo_ = 'particle_info/particle_info_%d.txt' # note this path need to replace the number
path_HistoryData_ = 'particle_info/history_data.his'

path_NormalStress_ = ['1MPa', '10MPa', '20MPa', '30MPa', '40MPa']
path_ShearVelocity_ = ['p01mps', 'p05mps', 'p1mps', 'p5mps', '1mps']
path_LoadStiffness_ = ['1k', '2k', '4k', '20k', '80k']

path_Conditions_ = ['normal_stress', 'shear_velocity', 'k']

# read FrictionState and HistoryData (normal stress)
# table head: 0. Load pointer displacement, 1. Friction, 2. Top plate displacement, 
# 3. Bottom plate displacement, 4. Top plate velocity, 5. Bottom plate velocity
FrictionState_1MPa_ = read_FrictionState(os.path.join(path_Conditions_[0], path_NormalStress_[0], path_FrictionState_))
FrictionState_10MPa_ = read_FrictionState(os.path.join(path_Conditions_[0], path_NormalStress_[1], path_FrictionState_))
FrictionState_20MPa_ = read_FrictionState(os.path.join(path_Conditions_[0], path_NormalStress_[2], path_FrictionState_))
FrictionState_30MPa_ = read_FrictionState(os.path.join(path_Conditions_[0], path_NormalStress_[3], path_FrictionState_))
FrictionState_40MPa_ = read_FrictionState(os.path.join(path_Conditions_[0], path_NormalStress_[4], path_FrictionState_))

# table head: 0. step, 1. displacement, 2. gouge_thickness, 3. shear_strain,
# 4. shear_stress, 5. normal_stress, 6. friction,  7. plate_displacement,
# 8. plate_velocity, 9. plate_displacement_bottom, 10. plate_velocity_bottom,
# 11. shear_strain_rate
HistoryData_1MPa_ = read_HistoryData(os.path.join(path_Conditions_[0], path_NormalStress_[0], path_HistoryData_))
HistoryData_10MPa_ = read_HistoryData(os.path.join(path_Conditions_[0], path_NormalStress_[1], path_HistoryData_))
HistoryData_20MPa_ = read_HistoryData(os.path.join(path_Conditions_[0], path_NormalStress_[2], path_HistoryData_))
HistoryData_30MPa_ = read_HistoryData(os.path.join(path_Conditions_[0], path_NormalStress_[3], path_HistoryData_))
HistoryData_40MPa_ = read_HistoryData(os.path.join(path_Conditions_[0], path_NormalStress_[4], path_HistoryData_))

# read FrictionState and HistoryData (shear velocity)
# table head: 0. Load pointer displacement, 1. Friction, 2. Top plate displacement, 
# 3. Bottom plate displacement, 4. Top plate velocity, 5. Bottom plate velocity
FrictionState_p01mps_ = read_FrictionState(os.path.join(path_Conditions_[1], path_ShearVelocity_[0], path_FrictionState_))
FrictionState_p05mps_ = read_FrictionState(os.path.join(path_Conditions_[1], path_ShearVelocity_[1], path_FrictionState_))
FrictionState_p1mps_ = read_FrictionState(os.path.join(path_Conditions_[1], path_ShearVelocity_[2], path_FrictionState_))
FrictionState_p5mps_ = read_FrictionState(os.path.join(path_Conditions_[1], path_ShearVelocity_[3], path_FrictionState_))
FrictionState_1mps_ = read_FrictionState(os.path.join(path_Conditions_[1], path_ShearVelocity_[4], path_FrictionState_))

# table head: 0. step, 1. displacement, 2. gouge_thickness, 3. shear_strain,
# 4. shear_stress, 5. normal_stress, 6. friction,  7. plate_displacement,
# 8. plate_velocity, 9. plate_displacement_bottom, 10. plate_velocity_bottom,
# 11. shear_strain_rate
HistoryData_p01mps_ = read_HistoryData(os.path.join(path_Conditions_[1], path_ShearVelocity_[0], path_HistoryData_))
HistoryData_p05mps_ = read_HistoryData(os.path.join(path_Conditions_[1], path_ShearVelocity_[1], path_HistoryData_))
HistoryData_p1mps_ = read_HistoryData(os.path.join(path_Conditions_[1], path_ShearVelocity_[2], path_HistoryData_))
HistoryData_p5mps_ = read_HistoryData(os.path.join(path_Conditions_[1], path_ShearVelocity_[3], path_HistoryData_))
HistoryData_1mps_ = read_HistoryData(os.path.join(path_Conditions_[1], path_ShearVelocity_[4], path_HistoryData_))

# read FrictionState and HistoryData (load stiffness)
# table head: 0. Load pointer displacement, 1. Friction, 2. Top plate displacement, 
# 3. Bottom plate displacement, 4. Top plate velocity, 5. Bottom plate velocity
FrictionState_1k_ = read_FrictionState(os.path.join(path_Conditions_[2], path_LoadStiffness_[0], path_FrictionState_))
FrictionState_2k_ = read_FrictionState(os.path.join(path_Conditions_[2], path_LoadStiffness_[1], path_FrictionState_))
FrictionState_4k_ = read_FrictionState(os.path.join(path_Conditions_[2], path_LoadStiffness_[2], path_FrictionState_))
FrictionState_20k_ = read_FrictionState(os.path.join(path_Conditions_[2], path_LoadStiffness_[3], path_FrictionState_))
FrictionState_80k_ = read_FrictionState(os.path.join(path_Conditions_[2], path_LoadStiffness_[4], path_FrictionState_))

# table head: 0. step, 1. displacement, 2. gouge_thickness, 3. shear_strain,
# 4. shear_stress, 5. normal_stress, 6. friction,  7. plate_displacement,
# 8. plate_velocity, 9. plate_displacement_bottom, 10. plate_velocity_bottom,
# 11. shear_strain_rate
HistoryData_1k_ = read_HistoryData(os.path.join(path_Conditions_[2], path_LoadStiffness_[0], path_HistoryData_))
HistoryData_2k_ = read_HistoryData(os.path.join(path_Conditions_[2], path_LoadStiffness_[1], path_HistoryData_))
HistoryData_4k_ = read_HistoryData(os.path.join(path_Conditions_[2], path_LoadStiffness_[2], path_HistoryData_))
HistoryData_20k_ = read_HistoryData(os.path.join(path_Conditions_[2], path_LoadStiffness_[3], path_HistoryData_))
HistoryData_80k_ = read_HistoryData(os.path.join(path_Conditions_[2], path_LoadStiffness_[4], path_HistoryData_))

t_ = HistoryData_80k_[:,4]
t_index_ = []
for i_ in np.where(t_ < 0)[0]:
    t_index_.append(np.arange(i_ - 100, i_ + 100))
t_index_ = np.array(t_index_)    
t_index_ = np.ravel(t_index_)
t_index_ = np.unique(t_index_)
t_index_ = np.delete(t_index_, np.where(t_index_ > HistoryData_80k_.shape[0] - 1))
HistoryData_80k_ = np.delete(HistoryData_80k_, t_index_, axis = 0)

#%% distinguish and slice the events' data
########### define functions

# 1. get the part which large than the threshold
def initial_extract(history_, thresh_v_):
    '''
    Parameters: 
        history_--the history of simulated fault
        thresh_v_--the threshold of velocity to define a slip event
        
    Returns: 
        list_event_slice_--record the start and end index of each event
        
    '''
    list_event_slice_ = []
    t_ = np.full([2, 1], np.nan)
    in_slip_ = False
    for i_ in range(history_.shape[0]):
        if in_slip_:
            if np.abs((history_[i_, 8] - history_[i_, 10]) / 2) < thresh_v_:
                t_[1] = i_
                list_event_slice_.append(t_.astype('int'))
                if t_[1] < t_[0]:
                    print('顺序错误')
                in_slip_ = False
        else:
            if np.abs((history_[i_, 8] - history_[i_, 10]) / 2) > thresh_v_:
                t_[0] = i_
                in_slip_ = True
        
    print('Extract successful!')
    return list_event_slice_

# list_event_slice_20MPa_ = initial_extract(HistoryData_20MPa_, thresh_v_ = 0.1)

# plt.plot(np.abs((HistoryData_20MPa_[54596:54599, 8] - HistoryData_20MPa_[54570:54605, 10]) / 2))
    

# 2. connect the neibor events which are in a certain distance  
def connect_neighbor(history_, list_event_slice_, thresh_v_):
    '''
    Parameters: 
        history_--the history of simulated fault
        list_event_slice_--record the start and end index of each event (get from function initial_extract)
        thresh_v_--the threshold of velocity to define a slip event
    Returns: 
        list_conncted_event_--record the start and end index of connected events
        
    '''
    list_conncted_event_ = []
    
    interval_displacement_ = history_[1, 1] - history_[0, 1]
    plate_velocity_ = np.abs((history_[:, 8] - history_[:, 10]) / 2)
    i_events_ = 0
    # find the peak points 
    for start_end_ in list_event_slice_:
        peak_v_ = -1 # record the maximum
        peak_ = 0  # record the position
        for i_ in range(start_end_[0][0], start_end_[1][0]):
            if plate_velocity_[i_] > peak_v_:
                peak_v_ = plate_velocity_[i_]
                peak_ = i_
        
        # extend the index 
        k_1_ = (plate_velocity_[peak_] - plate_velocity_[start_end_[0][0] - 1]) / ((peak_-(start_end_[0][0] - 1)) * interval_displacement_)
        k_2_ = (plate_velocity_[peak_] - plate_velocity_[start_end_[1][0]] + 1) / ((-peak_ + (start_end_[1][0] + 1)) * interval_displacement_)
        list_event_slice_[i_events_][0] -= int(thresh_v_ / k_1_ / interval_displacement_ + 1) 
  
        list_event_slice_[i_events_][1] += int(thresh_v_ / k_2_ / interval_displacement_ + 1)
        i_events_ += 1
    # conect the event
    count_connect_ = 0 # record the connect times
    connect_state_ = False # judge if the connecting is going
    t_ = np.full([2, 1], np.nan)
    for i_ in range(len(list_event_slice_) - 1):
        if connect_state_:
            
            if list_event_slice_[i_][1] >=  list_event_slice_[i_ + 1][0]:
                continue
            else:
                t_[1] = list_event_slice_[i_][1]
                list_conncted_event_.append(t_.astype('int'))
                
                connect_state_ = False
                if t_[1] < t_[0]:
                    print('顺序错误')
        else:
            if list_event_slice_[i_][1] >=  list_event_slice_[i_ + 1][0]:
                t_[0] = list_event_slice_[i_][0]
                count_connect_ += 1
                connect_state_ = True
                
            else:
                t_[0] = list_event_slice_[i_][0]
                t_[1] = list_event_slice_[i_][1]
                list_conncted_event_.append(t_.astype('int'))
                if t_[1] < t_[0]:
                    print('顺序错误')
    if connect_state_:
        t_[1] = list_event_slice_[-1][1]
        list_conncted_event_.append(t_.astype('int'))
    if list_conncted_event_[-1][1] > history_.shape[0] - 1:
        list_conncted_event_[-1][1][0] = int(history_.shape[0] - 1)
    print('Connected successful! %d events have been connected!!!' % count_connect_)

    return list_conncted_event_

# list_event_slice_20MPa_ = initial_extract(HistoryData_20MPa_, thresh_v_ = 0.1)
# list_connected_events_20MPa_ = connect_neighbor(HistoryData_20MPa_, list_event_slice_20MPa_, 0.1)

# 3. find the stable period
def find_stable(history_, list_conncted_event_, thresh_d_):
    '''
    Parameters: 
        history_--the history of simulated fault
        list_conncted_event_--record the start and end index of connected events (get from function connect_neighbor)
        thresh_d_--the threshold of displacement to define stable period
    Returns: 
        list_stable_event_--record the start and end index of events in stable period
        
    '''
    list_stable_event_ = []
    count_deleted_events_ = 0
    for start_end_ in list_conncted_event_:
        if history_[start_end_[1][0], 1] < thresh_d_:
            count_deleted_events_ += 1
            continue
        else:
            list_stable_event_.append(start_end_.copy())
    print('Select the events in stable period successful! %d events are deleted!' % count_deleted_events_)
    return list_stable_event_
# list_event_slice_20MPa_ = initial_extract(HistoryData_20MPa_, thresh_v_ = 0.1)
# list_connected_events_20MPa_ = connect_neighbor(HistoryData_20MPa_, list_event_slice_20MPa_, thresh_v_ = 0.1)
# list_stable_event_20MPa_ = find_stable(HistoryData_20MPa_, list_connected_events_20MPa_, thresh_d_ = 2e-3)



#%% ######### do the calculation
# 1MPa
list_event_slice_1MPa_ = initial_extract(HistoryData_1MPa_, thresh_v_ = 0.1)
list_connected_events_1MPa_ = connect_neighbor(HistoryData_1MPa_, list_event_slice_1MPa_, thresh_v_ = 0.1)
list_stable_event_1MPa_ = find_stable(HistoryData_1MPa_, list_connected_events_1MPa_, thresh_d_ = 2e-3)
filename_ = open('python_event_data/events_1MPa_.pkl', 'wb')
pickle.dump(list_stable_event_1MPa_, filename_)
filename_.close()
print('%d events are saved to %s' % (len(list_stable_event_1MPa_), filename_))
# 10MPa
list_event_slice_10MPa_ = initial_extract(HistoryData_10MPa_, thresh_v_ = 0.1)
list_connected_events_10MPa_ = connect_neighbor(HistoryData_10MPa_, list_event_slice_10MPa_, thresh_v_ = 0.1)
list_stable_event_10MPa_ = find_stable(HistoryData_10MPa_, list_connected_events_10MPa_, thresh_d_ = 2e-3)
filename_ = open('python_event_data/events_10MPa_.pkl', 'wb')
pickle.dump(list_stable_event_10MPa_, filename_)
filename_.close()
print('%d events are saved to %s' % (len(list_stable_event_10MPa_), filename_))
# 20MPa
list_event_slice_20MPa_ = initial_extract(HistoryData_20MPa_, thresh_v_ = 0.1)
list_connected_events_20MPa_ = connect_neighbor(HistoryData_20MPa_, list_event_slice_20MPa_, thresh_v_ = 0.1)
list_stable_event_20MPa_ = find_stable(HistoryData_20MPa_, list_connected_events_20MPa_, thresh_d_ = 2e-3)
filename_ = open('python_event_data/events_20MPa_.pkl', 'wb')
pickle.dump(list_stable_event_20MPa_, filename_)
filename_.close()
print('%d events are saved to %s' % (len(list_stable_event_20MPa_), filename_))
# 30MPa
list_event_slice_30MPa_ = initial_extract(HistoryData_30MPa_, thresh_v_ = 0.1)
list_connected_events_30MPa_ = connect_neighbor(HistoryData_30MPa_, list_event_slice_30MPa_, thresh_v_ = 0.1)
list_stable_event_30MPa_ = find_stable(HistoryData_30MPa_, list_connected_events_30MPa_, thresh_d_ = 2e-3)
filename_ = open('python_event_data/events_30MPa_.pkl', 'wb')
pickle.dump(list_stable_event_30MPa_, filename_)
filename_.close()
print('%d events are saved to %s' % (len(list_stable_event_30MPa_), filename_))
# 40MPa
list_event_slice_40MPa_ = initial_extract(HistoryData_40MPa_, thresh_v_ = 0.1)
list_connected_events_40MPa_ = connect_neighbor(HistoryData_40MPa_, list_event_slice_40MPa_, thresh_v_ = 0.1)
list_stable_event_40MPa_ = find_stable(HistoryData_40MPa_, list_connected_events_40MPa_, thresh_d_ = 2e-3)
filename_ = open('python_event_data/events_40MPa_.pkl', 'wb')
pickle.dump(list_stable_event_40MPa_, filename_)
filename_.close()
print('%d events are saved to %s' % (len(list_stable_event_40MPa_), filename_))
# p01mps
list_event_slice_p01mps_ = initial_extract(HistoryData_p01mps_, thresh_v_ = 0.1)
list_connected_events_p01mps_ = connect_neighbor(HistoryData_p01mps_, list_event_slice_p01mps_, thresh_v_ = 0.1)
list_stable_event_p01mps_ = find_stable(HistoryData_p01mps_, list_connected_events_p01mps_, thresh_d_ = 2e-3)
filename_ = open('python_event_data/events_p01mps_.pkl', 'wb')
pickle.dump(list_stable_event_p01mps_, filename_)
filename_.close()
print('%d events are saved to %s' % (len(list_stable_event_p01mps_), filename_))
# p05mps
list_event_slice_p05mps_ = initial_extract(HistoryData_p05mps_, thresh_v_ = 0.1)
list_connected_events_p05mps_ = connect_neighbor(HistoryData_p05mps_, list_event_slice_p05mps_, thresh_v_ = 0.1)
list_stable_event_p05mps_ = find_stable(HistoryData_p05mps_, list_connected_events_p05mps_, thresh_d_ = 2e-3)
filename_ = open('python_event_data/events_p05mps_.pkl', 'wb')
pickle.dump(list_stable_event_p05mps_, filename_)
filename_.close()
print('%d events are saved to %s' % (len(list_stable_event_p05mps_), filename_))
# p5mps
list_event_slice_p5mps_ = initial_extract(HistoryData_p5mps_, thresh_v_ = 0.1)
list_connected_events_p5mps_ = connect_neighbor(HistoryData_p5mps_, list_event_slice_p5mps_, thresh_v_ = 0.1)
list_stable_event_p5mps_ = find_stable(HistoryData_p5mps_, list_connected_events_p5mps_, thresh_d_ = 2e-3)
filename_ = open('python_event_data/events_p5mps_.pkl', 'wb')
pickle.dump(list_stable_event_p5mps_, filename_)
filename_.close()
print('%d events are saved to %s' % (len(list_stable_event_p5mps_), filename_))
# 1mps
list_event_slice_1mps_ = initial_extract(HistoryData_1mps_, thresh_v_ = 0.1)
list_connected_events_1mps_ = connect_neighbor(HistoryData_1mps_, list_event_slice_1mps_, thresh_v_ = 0.1)
list_stable_event_1mps_ = find_stable(HistoryData_1mps_, list_connected_events_1mps_, thresh_d_ = 2e-3)
filename_ = open('python_event_data/events_1mps_.pkl', 'wb')
pickle.dump(list_stable_event_1mps_, filename_)
filename_.close()
print('%d events are saved to %s' % (len(list_stable_event_1mps_), filename_))

# 2k
list_event_slice_2k_ = initial_extract(HistoryData_2k_, thresh_v_ = 0.1)
list_connected_events_2k_ = connect_neighbor(HistoryData_2k_, list_event_slice_2k_, thresh_v_ = 0.1)
list_stable_event_2k_ = find_stable(HistoryData_2k_, list_connected_events_2k_, thresh_d_ = 2e-3)
filename_ = open('python_event_data/events_2k_.pkl', 'wb')
pickle.dump(list_stable_event_2k_, filename_)
filename_.close()
print('%d events are saved to %s' % (len(list_stable_event_2k_), filename_))

# 4k
list_event_slice_4k_ = initial_extract(HistoryData_4k_, thresh_v_ = 0.1)
list_connected_events_4k_ = connect_neighbor(HistoryData_4k_, list_event_slice_4k_, thresh_v_ = 0.1)
list_stable_event_4k_ = find_stable(HistoryData_4k_, list_connected_events_4k_, thresh_d_ = 2e-3)
filename_ = open('python_event_data/events_4k_.pkl', 'wb')
pickle.dump(list_stable_event_4k_, filename_)
filename_.close()
print('%d events are saved to %s' % (len(list_stable_event_4k_), filename_))

# 20k
list_event_slice_20k_ = initial_extract(HistoryData_20k_, thresh_v_ = 0.1)
list_connected_events_20k_ = connect_neighbor(HistoryData_20k_, list_event_slice_20k_, thresh_v_ = 0.1)
list_stable_event_20k_ = find_stable(HistoryData_20k_, list_connected_events_20k_, thresh_d_ = 2e-3)
filename_ = open('python_event_data/events_20k_.pkl', 'wb')
pickle.dump(list_stable_event_20k_, filename_)
filename_.close()
print('%d events are saved to %s' % (len(list_stable_event_20k_), filename_))

# 80k
list_event_slice_80k_ = initial_extract(HistoryData_80k_, thresh_v_ = 0.1)
list_connected_events_80k_ = connect_neighbor(HistoryData_80k_, list_event_slice_80k_, thresh_v_ = 0.1)
list_stable_event_80k_ = find_stable(HistoryData_80k_, list_connected_events_80k_, thresh_d_ = 2e-3)
filename_ = open('python_event_data/events_80k_.pkl', 'wb')
pickle.dump(list_stable_event_80k_, filename_)
filename_.close()
print('%d events are saved to %s' % (len(list_stable_event_80k_), filename_))

# read
# filename_ = open('list_correlation_o_20220708.pkl', 'rb')
# list_correlation_o_ = pickle.load(filename_)
# print(list_correlation_o_)
# filename_.close()