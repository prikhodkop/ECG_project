import numpy as np
import logging

import sys
sys.path.append('..')
import project_config as conf
from utils import logg 



def generate_examples(OBJECTIVE_NAME, splitted_data_RR, stat_info, GIDN):
  """
  Get objective values for prediction.

  Args:
    OBJECTIVE_NAME (str): unique objective name
    splitted_data_RR (list of np.arrays): arrays of int64 in format (time since midnight [ms], intervals [ms])
    stat_info (dict): patients info, incuding .dta files
    GIDN (int): unique identifier of the patient

  Returns
    1) y (np.array of floats): objective, e.g. sex, age, sleep status
    objective_classes_names (dict or None)

    or

    2) None, None if objective is not available 
  """
  objective_classes_names = None

  if OBJECTIVE_NAME == 'cl_sleep_interval':
    y, objective_classes_names = get_sleep_interval_objective(splitted_data_RR, stat_info, GIDN)

  elif OBJECTIVE_NAME in ['Sex', 'BMIgr']: #TODO
    y = get_info_objective(OBJECTIVE_NAME, splitted_data_RR, stat_info, GIDN)

  elif OBJECTIVE_NAME == 'patients_ver1':
    objective_classes_names = {}
    objective_classes_names['targets'] = ['cl_sleep_interval', 'Sex', 'BMIgr', 'Age', 'MRT_CVD'] # , 'SelfHealth2'

    sleep_y, sleep_classes_names = get_sleep_interval_objective(splitted_data_RR, stat_info, GIDN)
    
    sex_y   = get_info_objective('Sex', splitted_data_RR, stat_info, GIDN)
    bmigr_y = get_info_objective('BMIgr', splitted_data_RR, stat_info, GIDN)
    age_y = get_info_objective('Age', splitted_data_RR, stat_info, GIDN)
    cvd_y = get_info_objective('MRT_CVD', splitted_data_RR, stat_info, GIDN)
    #health_y = get_info_objective('SelfHealth2', splitted_data_RR, stat_info, GIDN)
    
    objective_classes_names['cl_sleep_interval'] = sleep_classes_names

    y = np.vstack((sleep_y, sex_y, bmigr_y, age_y, cvd_y)).T #  health_y,
  
  else:
    msg = 'Not implemented objective type' #TODO
    logging.critical(msg)
    raise Exception(msg)

  if y is None:
    return None
  else:
    return y, objective_classes_names


def get_info_objective(OBJECTIVE_NAME, splitted_data_RR, stat_info, GIDN):
  number_of_chunks = len(splitted_data_RR)

  if OBJECTIVE_NAME in ['MRT_CVD']:
    data = stat_info['mortality']
  else:
    data = stat_info['selected_pp']
  
  GIDN_info = data[data['GIDN']==GIDN]
  obj_value = float(data[data['GIDN']==GIDN][OBJECTIVE_NAME])   
  y = [obj_value for i in xrange(number_of_chunks)]
  
  return y



def get_sleep_interval_objective(splitted_data_RR, stat_info, GIDN):
  """
  Get objective for classification problem related to sleep identification for short pulse intervals
  We suppose that 'sleep' == 1.0, 'awake' == 0.0.
  """
  sleep = stat_info['sleep'] # from .dta file

  if not (GIDN in sleep['start'].keys()):
    return None, None
  else:
    start_sleep = sleep['start'][GIDN]
    end_sleep = sleep['end'][GIDN]

    y = []    
    for data_RR in splitted_data_RR:
      beat_times = data_RR[:, 0] # np.array
      indixes_of_sleep_beats = (beat_times > start_sleep) * (beat_times < end_sleep)
      if sum(indixes_of_sleep_beats) / float(len(indixes_of_sleep_beats)) > 0.5:
        y.append(1.0)
      else:
        y.append(0.0)
    
    objective_classes_names = {1.0:'sleep', 0.0:'awake'}
    return y, objective_classes_names



if __name__ == '__main__':
  pass
  