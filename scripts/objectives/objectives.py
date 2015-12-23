import numpy as np
import logging

import sys
sys.path.append('..')
import project_config as conf
from utils import logg 


def generate_examples(OBJECTIVE_NAME, filtered_data_RR, stat_info):

  if OBJECTIVE_NAME == 'sleep_interval_5_min':
    X, y = get_sleep_interval_objective(filtered_data_RR, stat_info, interval=5.0)

  
  if OBJECTIVE_NAME == 'sex': #TODO
    pass



  return X, y


def get_sleep_interval_objective(filtered_data_RR, stat_info, interval):
  pass



if __name__ == '__main__':
  pass
  