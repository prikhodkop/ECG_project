import numpy as np
import logging

import sys
sys.path.append('..')

try:
  import user_project_config as conf
except:
  import project_config as conf

try:
  import psutil
  sys_utils = True
except:
  logging.warning('psutil package is not available')
  sys_utils = False


from IO import data_loading as dl
from utils import logg 
from utils import data_processing as dp
from utils import system_utils as su
from filters import data_filtering as df
from objectives import objectives as obj
from features import features as fea


if __name__ == '__main__':
  
  ###########################################################
  # Settings
  FIXED_GIDNS_LIST = None # If specified, only these GIDNS are considered
  MAX_PATIENTS_NUMBER = None # Note that None related ALL PATIENTS
  
  RR_filtering_params = df.get_default_RR_filtering_params()
  logging.warning('Filtering parameters:\n%s'%RR_filtering_params)
  
  pulse_features_params = fea.get_default_pulse_features_params()
  logging.warning('Pulse features parameters:\n%s'%pulse_features_params)

  stat_features_names = []  # e.g. ['Sex', 'BMIgr']

  OBJECTIVE_NAME = 'cl_sleep_interval' # e.g. 'BMIgr', 'Sex', 'cl_sleep_interval'
  sample_name = OBJECTIVE_NAME + '_3g' # train-test filename
  SEED = 0
  
  MAX_NUMBER_OF_CHUNKS_PER_PATIENT = None # Note that None related ALL CHUNKS
  CHUNK_TYPE = 'Fixed Time' # 'Fixed beats number' 

  if CHUNK_TYPE == 'Fixed Time':
    CHUNK_TIME_INTERVAL = 5.0 # minutes
    MIN_NUMBER_OF_BEATS_IN_CHUNK = 10 # beats

  elif CHUNK_TYPE == 'Fixed beats number':
    NUMBER_OF_BEATS_IN_CHUNK = 1000

  TEST_PARTITION = 'random' # method for train / test separation
  TEST_PORTION = 0.3
  
  ###############################################################
  # Initial configuration
  np.random.seed(SEED)
  logg.configure_logging() # For more details use logg.configure_logging(console_level=logging.DEBUG)

  stat_info = { 'mortality':   dl.read_dta('mortality_SAHR_ver101214', data_folder=conf.path_to_dta),
                'selected_pp': dl.read_dta('selected_pp', data_folder=conf.path_to_dta),
                'sleep':       dl.get_sleep_time(conf.path_to_dta) 
              }

  #################################################################
  
  # Specify patients GIDNS for study 
  if FIXED_GIDNS_LIST is not None:
    logging.warning('Only %s patients are used in this study: %s'%(len(FIXED_GIDNS_LIST), FIXED_GIDNS_LIST))
    asked_GIDNS = FIXED_GIDNS_LIST
  else:
    asked_GIDNS = dl.get_GIDNS(path_to_dta=conf.path_to_dta)
    if MAX_PATIENTS_NUMBER is not None:
      logging.warning('Only first %s patients are used in this study'%MAX_PATIENTS_NUMBER)
      asked_GIDNS = asked_GIDNS[:MAX_PATIENTS_NUMBER]  

  logging.debug('asked_GIDNS: %s'%asked_GIDNS)
  logging.info('Searching for %s patients'%len(asked_GIDNS))
  

  ######## Load and process data for specified GIDNS ################
  abcent_patients = []
  filtered_patients = []
  split_problem_patients = []
  objective_problem_patients = [] 
  loaded_GIDNS = [] # successfully loaded patients in ascending order
  X = {}
  y = {}
  logging.info('Starting data preparation')
  for patient_number, GIDN in enumerate(asked_GIDNS):
    
    msg = 'GIDN %s. Patient %s out of %s processing.'%(GIDN, patient_number+1, len(asked_GIDNS))
    if patient_number%50==0:
      logging.info(msg)
      su.check_memory(verbose=True)
    else:
      logging.debug(msg)
   
    ###### Load pulse data for the patient #######
    data_RR = dl.load_RR_data(GIDN, path=conf.path_to_RR)
    # data_RR (np.array) in format (time since midnight [ms], interval [ms], beat type)
    if data_RR is None:
      abcent_patients.append(GIDN)
      continue
    
    ####### Filter intervals of given types and lengths #######

    filtered_data_RR, filtration_info = df.filter_data_RR(data_RR, RR_filtering_params)
    # filtered_data_RR (np.array) in format (time since midnight [ms], intervals [ms])
    if filtered_data_RR is None:
      filtered_patients.append(GIDN)
      continue
    msg = 'Filtration of intervals: %s'%filtration_info
    logging.debug(msg)

    ###### Splitting data into chunks ###########

    if CHUNK_TYPE == 'Fixed Time':
      splitted_data_RR = dp.split_time_chunks(filtered_data_RR, CHUNK_TIME_INTERVAL, MIN_NUMBER_OF_BEATS_IN_CHUNK, 
        MAX_NUMBER_OF_CHUNKS_PER_PATIENT) # splitted_data_RR (list of np.arrays)
    
    elif CHUNK_TYPE == 'Fixed beats number':
      splitted_data_RR = dp.split_beats_chunks(filtered_data_RR, NUMBER_OF_BEATS_IN_CHUNK, 
        MAX_NUMBER_OF_CHUNKS_PER_PATIENT)
    
    if splitted_data_RR is None:
      split_problem_patients.append(GIDN)
      continue
    logging.debug('Splitting intervals')
    
    GIDN_y, objective_classes_names = obj.generate_examples(OBJECTIVE_NAME, splitted_data_RR, stat_info, GIDN) # np.array of objective values (float)
    if GIDN_y is None:
      objective_problem_patients.append(GIDN)
      continue
    msg = 'Objective values for chunks: %s'%GIDN_y.T
    logging.debug(msg)

    splitted_data_RR, initial_times = dp.time_initialization(splitted_data_RR) # splitted_data_RR (list of np.array): 
    logging.debug('Set start time for each chunk equal zero') # np.array of np.int64 in format (time [ms], intervals [ms])

    # Light-features #!!!
    # pulse_features_params =  {'use initial_times':  True,
    #                           'sampling rate':      1000, #Hz 
    #                           'vizualization':      False, # demonstration of sample features in graphic manner  
    #                           'save pic':           False, # saving pics of sample features in graphic manner  
    #                           'time features':      { 'step SDANN': [60000], #ms, window size for computation SDANN, see [1]
    #                                                   'step SDNNind': [60000], #ms, window size for computation SDNNind, see [1]
    #                                                   'step sdHR': [600000], #ms, window size for coputation sdHR, see [1]
    #                                                   'threshold NN': [20, 50], #ms, threshold for coputation NNx ans pNNx, see [1]
    #                                                   'threshold outlier': 0.2, #ms, threshold for coputation outlier, see [1]
    #                                                   'triangular features': False,
    #                                                 },     

    #                           'frequency features': { 'frequency bounds': [0., 0.0033, 0.04, 0.15, 0.4], #Hz bounds for ULF, VLF, LF and HF, see[1]
    #                                                   'frequency range': [0.001, 0.4, 200] #Hz lowest and highest frequencies, see[1]
    #                                                 },

    #                           'nonlinear features': None 
    #                          }


    
    splitted_pulse_features, pulse_features_names = fea.generate_pulse_features(splitted_data_RR, 
      pulse_features_params, initial_times=initial_times) # (list of np.array of floats)
    
    stat_features = fea.get_stat_features(stat_features_names, stat_info, GIDN) #TODO

    GIDN_features = fea.get_features_matrix(splitted_pulse_features) # (np.array) [chunks * features]

    sample_info = { 'Objective classes names': objective_classes_names, 
                    'Filtering params':        RR_filtering_params,
                    'Features params':         pulse_features_params,
                    'Features names':          pulse_features_names,
                    'CHUNK TIME INTERVAL':     CHUNK_TIME_INTERVAL
                  }

    X[GIDN] = GIDN_features
    y[GIDN] = GIDN_y
    loaded_GIDNS.append(GIDN)
    su.check_memory()
  
  #### Loading statistics ##############

  if abcent_patients:
    logging.info('Failed to load .RR for %s patients out of %s'%(len(abcent_patients), len(asked_GIDNS)))
    logging.info('Abcent patients: %s'%abcent_patients)
  
  if filtered_patients:
    logging.info('No data are available after filtration for %s patients out of %s'%(len(filtered_patients), len(asked_GIDNS)))
    logging.info('Filtration problem patients: %s'%filtered_patients)
  
  if split_problem_patients:
    logging.info('No data after splitting for %s patients out of %s'%(len(split_problem_patients), len(asked_GIDNS)))
    logging.info('Split problem patients: %s'%split_problem_patients)

  if objective_problem_patients:
    logging.info('No objective for %s patients out of %s'%(len(objective_problem_patients), len(asked_GIDNS)))
    logging.info('Objective problem patients: %s'%objective_problem_patients)

    
  ### Prepare training / test samples ##########

  train_GIDNS, test_GIDNS, partition_info = dp.split_GIDNS_for_test(loaded_GIDNS, TEST_PARTITION, 
                                                            TEST_PORTION)
  logging.info('%s test/train splitting: %s'%(TEST_PARTITION, partition_info))


  def combine_splitted_dict(d, GIDNS):
    combined = [d[GIDN] for GIDN in GIDNS]
    return np.vstack(combined)
 
  trainX = combine_splitted_dict(X, train_GIDNS)
  trainY = combine_splitted_dict(y, train_GIDNS)
  testX = combine_splitted_dict(X, test_GIDNS)
  testY = combine_splitted_dict(y, test_GIDNS)
  su.check_memory()

  dp.training_stats(trainX, trainY, testX, testY)

  path = dl.save_hdf5_sample(sample_name, sample_info, trainX, trainY, testX, testY)
  logging.info('Training and test samples are saved in hdf5 format: '+path)
  
  su.check_memory(verbose=True)
  logging.info('Data preparation is finished')
  
