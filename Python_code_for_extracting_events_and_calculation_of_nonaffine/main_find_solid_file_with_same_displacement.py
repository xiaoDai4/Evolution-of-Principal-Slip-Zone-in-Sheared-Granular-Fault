# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:59:09 2024

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

#%% find the corresponding index of solid file
t_ = 1.5e-3
print('%f mm' % t_)
for i_ in range(FrictionState_1MPa_.shape[0]):
    if FrictionState_1MPa_[i_ ,2] - FrictionState_1MPa_[i_ ,3] > t_:
        print('The index for 1MPa is %d' % i_)
        break
    
for i_ in range(FrictionState_10MPa_.shape[0]):
    if FrictionState_10MPa_[i_ ,2] - FrictionState_10MPa_[i_ ,3] > t_:
        print('The index for 10MPa is %d' % i_)
        break
    
for i_ in range(FrictionState_20MPa_.shape[0]):
    if FrictionState_20MPa_[i_ ,2] - FrictionState_20MPa_[i_ ,3] > t_:
        print('The index for 20MPa is %d' % i_)
        break
    
for i_ in range(FrictionState_30MPa_.shape[0]):
    if FrictionState_30MPa_[i_ ,2] - FrictionState_30MPa_[i_ ,3] > t_:
        print('The index for 30MPa is %d' % i_)
        break
    
for i_ in range(FrictionState_40MPa_.shape[0]):
    if FrictionState_40MPa_[i_ ,2] - FrictionState_40MPa_[i_ ,3] > t_:
        print('The index for 40MPa is %d' % i_)
        break

for i_ in range(FrictionState_p01mps_.shape[0]):
    if FrictionState_p01mps_[i_ ,2] - FrictionState_p01mps_[i_ ,3] > t_:
        print('The index for p01mps is %d' % i_)
        break

for i_ in range(FrictionState_p05mps_.shape[0]):
    if FrictionState_p05mps_[i_ ,2] - FrictionState_p05mps_[i_ ,3] > t_:
        print('The index for p05mps is %d' % i_)
        break
    
for i_ in range(FrictionState_p5mps_.shape[0]):
    if FrictionState_p5mps_[i_ ,2] - FrictionState_p5mps_[i_ ,3] > t_:
        print('The index for p5mps is %d' % i_)
        break

for i_ in range(FrictionState_1mps_.shape[0]):
    if FrictionState_1mps_[i_ ,2] - FrictionState_1mps_[i_ ,3] > t_:
        print('The index for 1mps is %d' % i_)
        break
    
for i_ in range(FrictionState_2k_.shape[0]):
    if FrictionState_2k_[i_ ,2] - FrictionState_2k_[i_ ,3] > t_:
        print('The index for 2k is %d' % i_)
        break
    
for i_ in range(FrictionState_4k_.shape[0]):
    if FrictionState_4k_[i_ ,2] - FrictionState_4k_[i_ ,3] > t_:
        print('The index for 4k is %d' % i_)
        break
    
for i_ in range(FrictionState_20k_.shape[0]):
    if FrictionState_20k_[i_ ,2] - FrictionState_20k_[i_ ,3] > t_:
        print('The index for 20k is %d' % i_)
        break
    
for i_ in range(FrictionState_80k_.shape[0]):
    if FrictionState_80k_[i_ ,2] - FrictionState_80k_[i_ ,3] > t_:
        print('The index for 80k is %d' % i_)
        break
    
#%% find the corresponding index of solid file when slip start

print('start solid file')
t_ = HistoryData_1MPa_[:,4].argmax()
for i_ in range(FrictionState_1MPa_.shape[0]):
    if FrictionState_1MPa_[i_ ,0 ] > HistoryData_1MPa_[t_,1]:
        print('The index for 1MPa is %d' % (i_ - 2))
        break
    
t_ = HistoryData_10MPa_[:,4].argmax()
for i_ in range(FrictionState_10MPa_.shape[0]):
    if FrictionState_10MPa_[i_ ,0 ] > HistoryData_10MPa_[t_,1]:
        print('The index for 10MPa is %d' % (i_ - 2))
        break
    
t_ = HistoryData_20MPa_[:,4].argmax()
for i_ in range(FrictionState_20MPa_.shape[0]):
    if FrictionState_20MPa_[i_ ,0 ] > HistoryData_20MPa_[t_,1]:
        print('The index for 20MPa is %d' % (i_ - 2))
        break
    
t_ = HistoryData_30MPa_[:,4].argmax()
for i_ in range(FrictionState_30MPa_.shape[0]):
    if FrictionState_30MPa_[i_ ,0 ] > HistoryData_30MPa_[t_,1]:
        print('The index for 30MPa is %d' % (i_ - 2))
        break
    
t_ = HistoryData_40MPa_[:,4].argmax()
for i_ in range(FrictionState_40MPa_.shape[0]):
    if FrictionState_40MPa_[i_ ,0 ] > HistoryData_40MPa_[t_,1]:
        print('The index for 40MPa is %d' % (i_ - 2))
        break
    
t_ = HistoryData_p01mps_[:,4].argmax()
for i_ in range(FrictionState_p01mps_.shape[0]):
    if FrictionState_p01mps_[i_ ,0 ] > HistoryData_p01mps_[t_,1]:
        print('The index for p01mps is %d' % (i_ - 2))
        break
    
t_ = HistoryData_p05mps_[:,4].argmax()
for i_ in range(FrictionState_p05mps_.shape[0]):
    if FrictionState_p05mps_[i_ ,0 ] > HistoryData_p05mps_[t_,1]:
        print('The index for p05mps is %d' % (i_ - 2))
        break
    
t_ = HistoryData_p5mps_[:,4].argmax()
for i_ in range(FrictionState_p5mps_.shape[0]):
    if FrictionState_p5mps_[i_ ,0 ] > HistoryData_p5mps_[t_,1]:
        print('The index for p5mps is %d' % (i_ - 2))
        break  
    
t_ = HistoryData_1mps_[:,4].argmax()
for i_ in range(FrictionState_1mps_.shape[0]):
    if FrictionState_1mps_[i_ ,0 ] > HistoryData_1mps_[t_,1]:
        print('The index for 1mps is %d' % (i_ - 2))
        break  

    
t_ = HistoryData_2k_[:,4].argmax()
for i_ in range(FrictionState_2k_.shape[0]):
    if FrictionState_2k_[i_ ,0 ] > HistoryData_2k_[t_,1]:
        print('The index for 2k is %d' % (i_ - 2))
        break  
    
t_ = HistoryData_4k_[:,4].argmax()
for i_ in range(FrictionState_4k_.shape[0]):
    if FrictionState_4k_[i_ ,0 ] > HistoryData_4k_[t_,1]:
        print('The index for 4k is %d' % (i_ - 2))
        break  
    
t_ = HistoryData_20k_[:,4].argmax()
for i_ in range(FrictionState_20k_.shape[0]):
    if FrictionState_20k_[i_ ,0 ] > HistoryData_20k_[t_,1]:
        print('The index for 20k is %d' % (i_ - 2))
        break  
    
t_ = HistoryData_80k_[:,4].argmax()
for i_ in range(FrictionState_80k_.shape[0]):
    if FrictionState_80k_[i_ ,0 ] > HistoryData_80k_[t_,1]:
        print('The index for 80k is %d' % (i_ - 2))
        break      