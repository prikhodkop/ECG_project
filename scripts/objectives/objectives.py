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
    objective_classes_names (dict)

    or

    2) None, None if objective is not available 
  """

  if OBJECTIVE_NAME == 'cl_sleep_interval':
    y, objective_classes_names = get_sleep_interval_objective(splitted_data_RR, stat_info, GIDN)

  elif OBJECTIVE_NAME in ['Sex', 'BMIgr']: #TODO
    y, objective_classes_names = get_info_objective(OBJECTIVE_NAME, splitted_data_RR, stat_info, GIDN)

  else:
    msg = 'Not implemented objective type' #TODO
    logging.critical(msg)
    raise Exception(msg)

  if y is None:
    return None, None
  else:
    return np.array([y]).T, objective_classes_names


def get_info_objective(OBJECTIVE_NAME, splitted_data_RR, stat_info, GIDN):
  number_of_chunks = len(splitted_data_RR)
  pp = stat_info['selected_pp']
  GIDN_info = pp[pp['GIDN']==GIDN]
  
  obj_value = float(pp[pp['GIDN']==GIDN][OBJECTIVE_NAME])   
  y = [obj_value for i in xrange(number_of_chunks)]
  # TODO !!!
  #!!! what if not available?
  return y, None



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
  