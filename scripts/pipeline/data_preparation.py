import numpy as np
import logging

import sys
sys.path.append('..')
import project_config as conf
from IO import data_loading as dl
from utils import logg 
from filters import data_filtering as df


if __name__ == '__main__':
  
  #!!!
  TEST_MODE = True # Use small portion of patients to speed up data processing
  logg.configure_logging()
  
  if TEST_MODE:
    max_patient_number = 30
    logging.warning('TEST_MODE to speed up data processing is activated')
  else:
    max_patient_number = 1800

  RR_filtering_params = df.get_default_RR_filtering_params()


  GIDNS = dl.get_GIDNS(path_to_dta=conf.path_to_dta)
  logging.debug('GIDNS: %s'%GIDNS)
  logging.info('Searching for %s patients'%len(GIDNS))


  GIDNS = GIDNS[:max_patient_number] #!!! TEST_MODE  
  abcent_patients = []
  #all_filtered_data_RR = {}
  for patient_number, GIDN in enumerate(GIDNS):
    
    data_RR = dl.load_RR_data(GIDN, path=conf.path_to_RR)
    if data_RR is None:
      abcent_patients.append(GIDN)
      continue
      
    filtered_data_RR = df.filter_data_RR(data_RR, RR_filtering_params)
    #all_filtered_data_RR[GIDN] = filtered_data_RR
    #print '\ndata_RR:\n', data_RR, '\n'

  logging.info('Failed to load .RR for %s patients out of %s'%(len(abcent_patients), len(GIDNS)))
  logging.info('Abcent patients: %s'%abcent_patients)



  #mortality = dl.read_dta('mortality_SAHR_ver101214', data_folder=conf.path_to_dta)
  #selected_pp = dl.read_dta('selected_pp', data_folder=conf.path_to_dta)