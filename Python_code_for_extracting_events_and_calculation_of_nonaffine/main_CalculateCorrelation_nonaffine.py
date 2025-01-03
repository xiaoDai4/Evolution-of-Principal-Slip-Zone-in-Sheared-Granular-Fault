#%% import pakages
import pandas as pd
import numpy as np
import os
import sys
import math

#%% set parameter
# file name to imported
file_names_ = ['D:/sheared_granular_gouge_different_evironment_condition/python_nonaffine_displacement_data/nonaffine_of_1MPa_0p5_from_129_to_209.npy',
               'D:/sheared_granular_gouge_different_evironment_condition/python_nonaffine_displacement_data/nonaffine_of_10MPa_0p5_from_168_to_248.npy',
               'D:/sheared_granular_gouge_different_evironment_condition/python_nonaffine_displacement_data/nonaffine_of_20MPa_0p5_from_217_to_297.npy',
               'D:/sheared_granular_gouge_different_evironment_condition/python_nonaffine_displacement_data/nonaffine_of_30MPa_0p5_from_259_to_339.npy',
               'D:/sheared_granular_gouge_different_evironment_condition/python_nonaffine_displacement_data/nonaffine_of_40MPa_0p5_from_292_to_372.npy',
               'D:/sheared_granular_gouge_different_evironment_condition/python_nonaffine_displacement_data/nonaffine_of_p01mps_0p5_from_210_to_290.npy',
               'D:/sheared_granular_gouge_different_evironment_condition/python_nonaffine_displacement_data/nonaffine_of_p05mps_0p5_from_208_to_288.npy',
               'D:/sheared_granular_gouge_different_evironment_condition/python_nonaffine_displacement_data/nonaffine_of_p5mps_0p5_from_212_to_292.npy',
               'D:/sheared_granular_gouge_different_evironment_condition/python_nonaffine_displacement_data/nonaffine_of_1mps_0p5_from_216_to_296.npy',
               'D:/sheared_granular_gouge_different_evironment_condition/python_nonaffine_displacement_data/nonaffine_of_2k_0p5_from_173_to_253.npy',
               'D:/sheared_granular_gouge_different_evironment_condition/python_nonaffine_displacement_data/nonaffine_of_4k_0p5_from_147_to_227.npy',
               'D:/sheared_granular_gouge_different_evironment_condition/python_nonaffine_displacement_data/nonaffine_of_20k_0p5_from_129_to_209.npy',
               'D:/sheared_granular_gouge_different_evironment_condition/python_nonaffine_displacement_data/nonaffine_of_80k_0p5_from_126_to_206.npy']  
# the coordinate of particles to import
path_ParticleInfo_ = ['D:/sheared_granular_gouge_different_evironment_condition/normal_stress/1MPa/particle_info/particle_info_%d.txt',
               'D:/sheared_granular_gouge_different_evironment_condition/normal_stress/10MPa/particle_info/particle_info_%d.txt',
               'D:/sheared_granular_gouge_different_evironment_condition/normal_stress/20MPa/particle_info/particle_info_%d.txt',
               'D:/sheared_granular_gouge_different_evironment_condition/normal_stress/30MPa/particle_info/particle_info_%d.txt',
               'D:/sheared_granular_gouge_different_evironment_condition/normal_stress/40MPa/particle_info/particle_info_%d.txt',
               'D:/sheared_granular_gouge_different_evironment_condition/shear_velocity/p01mps/particle_info/particle_info_%d.txt',
               'D:/sheared_granular_gouge_different_evironment_condition/shear_velocity/p05mps/particle_info/particle_info_%d.txt',
               'D:/sheared_granular_gouge_different_evironment_condition/shear_velocity/p5mps/particle_info/particle_info_%d.txt',
               'D:/sheared_granular_gouge_different_evironment_condition/shear_velocity/1mps/particle_info/particle_info_%d.txt',
               'D:/sheared_granular_gouge_different_evironment_condition/k/2k/particle_info/particle_info_%d.txt',
               'D:/sheared_granular_gouge_different_evironment_condition/k/4k/particle_info/particle_info_%d.txt',
               'D:/sheared_granular_gouge_different_evironment_condition/k/20k/particle_info/particle_info_%d.txt',
               'D:/sheared_granular_gouge_different_evironment_condition/k/80k/particle_info/particle_info_%d.txt'] 

index_ = [129, 168, 217, 259, 292, 210, 208, 212, 216, 173, 147, 129, 126]

a_ = 2.5e-5

r_ = np.linspace(0, 25e-5, 30)  # mutual distance to be calculated

# define the range of particles we concerned
l_xlim_ = 0.5e-3
u_xlim_ = 3.5e-3
l_ylim_ = 1e-3
u_ylim_ = 3e-3

# path_FrictionState_ = '../particle_info/particle_info_friction_state.txt'
# path_ParticleInfo_ = '../particle_info/particle_info_%d.txt' # note this path need to replace the number
# path_HistoryData_ = '../particle_info/history_data.his'

#%% change work directory
os.chdir(os.path.dirname(sys.argv[0]))
# import function
from import_data import *
from calculate_distance import *
from calculate_correlation import *
from filter_particles import *


#%% correlation
steps_ = len(file_names_)
print('Start Correlation!')
for i_ in range(steps_):
    # import data 
    print('\n-----------------------------------------------------\n')
    ParticleInfo_ = read_ParticleInfo(path_ParticleInfo_[i_], index_[i_])
    n_particles_ = ParticleInfo_.shape[0]
    values_ = np.load(file_names_[i_])
    print('\nRead %s\n' % file_names_[i_])
    
    # filter particles
    c_ = ParticleInfo_[:,[1,2]] # the coordinate of particles
    c_filtered_, v_filtered_ = filter_particles(l_xlim_, u_xlim_, l_ylim_, u_ylim_, c_, values_)
    
    # distance
    print('\n-----------------------------------------------------\n')
    
    distance_ = calculate_distance(c_filtered_) 
    
    # calculate correlation
    print('\n-----------------------------------------------------\n')
    correlation_ = calculate_correlation(distance_, v_filtered_, r_, a_)
    
    # save data
    print('\n-----------------------------------------------------\n')
    correlation_ = np.vstack([r_, correlation_])
    # correlation_: row 0, i.e. correlation_[0, :] is the prescribed mutual distance
    # correlation_: row 1, i.e. correlation_[1, :] is the prescribed mutual distance
    np.save('D:/sheared_granular_gouge_different_evironment_condition/python_correlation_data/correlation_of_%s' % os.path.basename(file_names_[i_]), correlation_)
    
    print('Save data done')
print('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
print('Correlation saved. ALL DONE.')