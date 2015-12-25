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
    y (list of floats): objective, e.g. sex, age, sleep status
  """

  if OBJECTIVE_NAME == 'cl_sleep_interval':
    y = get_sleep_interval_objective(splitted_data_RR, stat_info, GIDN)

  elif OBJECTIVE_NAME in ['Sex', 'BMIgr']: #TODO
    y = get_info_objective(OBJECTIVE_NAME, splitted_data_RR, stat_info, GIDN)

  else:
    msg = 'Not implemented objective type' #TODO
    logging.critical(msg)
    raise Exception(msg)

  return y


def get_info_objective(OBJECTIVE_NAME, splitted_data_RR, stat_info, GIDN):
  number_of_chunks = len(splitted_data_RR)
  pp = stat_info['selected_pp']
  GIDN_info = pp[pp['GIDN']==GIDN]
  
  obj_value = float(pp[pp['GIDN']==GIDN][OBJECTIVE_NAME])   
  y = [obj_value for i in xrange(number_of_chunks)]
  return y



def get_sleep_interval_objective(splitted_data_RR, stat_info, GIDN):
  """
  Get objective for classification problem related to sleep identification for short pulse intervals
  We suppose that 'sleep' == 1.0, 'awake' == 0.0.
  """
  sleep = stat_info['sleep'] # from .dta file

  #TODO
  
  return y



if __name__ == '__main__':
  pass
  