import numpy as np
import logging

import sys
sys.path.append('..')

try:
  import user_project_config as conf
except:
  import project_config as conf


from IO import data_loading as dl
from utils import logg 
from utils import data_processing as dp
from filters import data_filtering as df
from objectives import objectives as obj
from features import features as fea


if __name__ == '__main__':
  
  ###########################################################
  # Settings
  FIXED_GIDNS_LIST = None # If specified, only these GIDNS are considered
  
  MAX_PATIENTS_NUMBER = 30 # or None #!!!
  
  RR_filtering_params = df.get_default_RR_filtering_params()
  
  #pulse_features_params = fea.get_default_pulse_features_params()
  pulse_features_params =  {'time features':      {},     
                            'frequency features': None,
                            'nonlinear features': None
                           }


  OBJECTIVE_NAME = 'BMIgr' #'Sex' # 'cl_sleep_interval'
  SEED = 0
  
  MAX_NUMBER_OF_CHUNKS_PER_PATIENT = None
  CHUNK_TYPE = 'Fixed Time' # 'Fixed beats number' 

  if CHUNK_TYPE == 'Fixed Time':
    CHUNK_TIME_INTERVAL = 5.0 # minutes 
    MIN_NUMBER_OF_BEATS_IN_CHUNK = 10 # beats

  elif CHUNK_TYPE == 'Fixed beats number':
    NUMBER_OF_BEATS_IN_CHUNK = 1000
  
  ###############################################################
  # Initial configuration
  np.random.seed(SEED)
  logg.configure_logging(console_level=logging.DEBUG) # logging.INFO
  #logg.configure_logging()

  stat_info = { 'mortality':   dl.read_dta('mortality_SAHR_ver101214', data_folder=conf.path_to_dta),
                'selected_pp': dl.read_dta('selected_pp', data_folder=conf.path_to_dta),
                'sleep':       dl.read_dta('sleep', data_folder=conf.path_to_dta) 
              } # dl.get_sleep_time(path_to_sleep_csv=conf.path_to_dta)

  #################################################################
  
  # Specify patients GIDNS for study 
  if FIXED_GIDNS_LIST is not None:
    logging.warning('Only %s patients are used in this study: %s'%(len(FIXED_GIDNS_LIST), FIXED_GIDNS_LIST))
    GIDNS = FIXED_GIDNS_LIST
  else:
    GIDNS = dl.get_GIDNS(path_to_dta=conf.path_to_dta)
    if MAX_PATIENTS_NUMBER is not None:
      logging.warning('Only first %s patients are used in this study'%MAX_PATIENTS_NUMBER)
      GIDNS = GIDNS[:MAX_PATIENTS_NUMBER] #!!!  

  logging.debug('GIDNS: %s'%GIDNS)
  logging.info('Searching for %s patients'%len(GIDNS))
  

  ######## Load and process data for specified GIDNS ################
  abcent_patients = []
  filtered_patients = []
  splitproblem_patients = []
  X = []
  y = []
  for patient_number, GIDN in enumerate(GIDNS):
    
    msg = 'GIDN %s. Patient %s out of %s processing.'%(GIDN, patient_number+1, len(GIDNS))
    if patient_number%10==0:
      logging.info(msg)
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
        MAX_NUMBER_OF_CHUNKS_PER_PATIENT)
    
    elif CHUNK_TYPE == 'Fixed beats number':
      splitted_data_RR = dp.split_beats_chunks(filtered_data_RR, NUMBER_OF_BEATS_IN_CHUNK, 
        MAX_NUMBER_OF_CHUNKS_PER_PATIENT)
    
    if splitted_data_RR is None:
      splitproblem_patients.append(GIDN)
      continue
    msg = 'Splitting intervals'
    logging.debug(msg)
    # splitted_data_RR (list of np.arrays)

    GIDN_y = obj.generate_examples(OBJECTIVE_NAME, splitted_data_RR, stat_info, GIDN)
    # list of objective values (float)
    msg = 'Objective values for chunks: %s'%GIDN_y
    logging.debug(msg)

    splitted_data_RR, initial_times = dp.time_initialization(splitted_data_RR)
    logging.debug('Set start time for each chunk equal zero')
    # splitted_data_RR (list of np.array): np.array of np.int64 in format (time [ms], intervals [ms])

    splitted_features, names = fea.generate_pulse_features(splitted_data_RR, pulse_features_params)
  
    zxc

    X += GIDN_features
    y += GIDN_y


    

    #print '\ndata_RR:\n', data_RR, '\n'

  logging.info('Failed to load .RR for %s patients out of %s'%(len(abcent_patients), len(GIDNS)))
  logging.info('Abcent patients: %s'%abcent_patients)

  logging.info('No data are available after filtration for %s patients out of %s'%(len(filtered_patients), len(GIDNS)))
  logging.info('Filtered patients: %s'%filtered_patients)
  
  logging.info('No data after splitting for %s patients out of %s'%(len(splitproblem_patients), len(GIDNS)))
  logging.info('Filtered patients: %s'%splitproblem_patients)
  
  zxc

  split_GIDNS_for_test(GIDNS, method)
  
  shufle_training_data() #!!!

  save_train_test_data()



  trainX, trainY, testX, testY = load_train_test_data()

  model = build_model(trainX, trainY, model_configuration, sample_configuration)
  save_model(model_name, model)
  report = validate_models(models_list, testX, testY, draw_pictures=True)
  


  #find_model(settings) # based on hash



  def split_GIDNS_for_test(GIDNS, method):
    pass

  #!!!
  #d = dict(globals())
  #d.update(locals())
  #print d