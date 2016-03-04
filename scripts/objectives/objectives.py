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
  objective_classes_names = {}
  objective_classes_names['features code'] = get_features_code()

  if OBJECTIVE_NAME == 'cl_sleep_interval':
    y, objective_classes_names = get_sleep_interval_objective(splitted_data_RR, stat_info, GIDN)

  elif OBJECTIVE_NAME in ['Sex', 'BMIgr', 'tuberculum']: #TODO
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

  elif  OBJECTIVE_NAME == 'some_diseases_ver1':

    objective_classes_names['targets'] = ['stenocardia', 'kidney1', 'kidney2']
    steno_y  = get_info_objective('stenocardia', splitted_data_RR, stat_info, GIDN)
    kid1_y   = get_info_objective('kidney1', splitted_data_RR, stat_info, GIDN)
    kid2_y   = get_info_objective('kidney2', splitted_data_RR, stat_info, GIDN)
    
    y = np.vstack((steno_y, kid1_y, kid2_y)).T

    #print y.shape
    #print y
    
  else:
    msg = 'Not implemented objective type' #TODO
    logging.critical(msg)
    raise Exception(msg)

  if y is None:
    return None, None
  else:
    return y, objective_classes_names




def get_features_code():
  """
  Feature name - Table name - Feature code
  """
  features_code = {'stenocardia':   ['selected', 'M8_4_19'], #!!!
                   'kidney1':       ['selected', 'M8_4_28'], 
                   'kidney2':       ['selected', 'M8_4_29']}

  return features_code

def get_info_objective(OBJECTIVE_NAME, splitted_data_RR, stat_info, GIDN):
  number_of_chunks = len(splitted_data_RR)
  
  if OBJECTIVE_NAME not in ['stenocardia', 'kidney1', 'kidney2']:
    raise Exception('Not implemented')

  else:
    features_code = get_features_code()
    table_name, code = features_code[OBJECTIVE_NAME]

    data = stat_info[table_name]
    
    raw_obj_value = data[data['GIDN']==GIDN][code].values[0] #

    if raw_obj_value == 'No':
      obj_value = 0.0
    elif raw_obj_value == 'Have had':
      obj_value = 1.0
    elif raw_obj_value == 'Have now':
      obj_value = 2.0
    else:
      try:
        if np.isnan(raw_obj_value):
          obj_value = np.nan
        else:
          raise Exception('Unknown status for %s: %s'%(GIDN, raw_obj_value))
      except:
        raise Exception('Unknown status for %s: %s'%(GIDN, raw_obj_value))

  #print obj_value, GIDN_info #!!!
  #zxc #!!

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

# def get_info_objective(OBJECTIVE_NAME, splitted_data_RR, stat_info, GIDN):
#   number_of_chunks = len(splitted_data_RR)

#   objective_in_table = OBJECTIVE_NAME
  
#   if OBJECTIVE_NAME in ['MRT_CVD']:
#     data = stat_info['mortality']
#   elif OBJECTIVE_NAME == 'tuberculum':
#     data = stat_info['selected']
#   else:
#     data = stat_info['selected_pp']
  
#   #GIDN_info = data[data['GIDN']==GIDN]
#   #print 'rfvtf', data[objective_in_table] #!!!
  
#   if OBJECTIVE_NAME != 'tuberculum':
#     obj_value = float(data[data['GIDN']==GIDN][objective_in_table]) 
#   else:
#     objective_in_table = 'M8_4_7'
#     raw_obj_value = data[data['GIDN']==GIDN][objective_in_table]

#     print raw_obj_value

#     zxc

#     if raw_obj_value == 'No':
#       obj_value = 0.0
#     elif raw_obj_value == 'Have had':
#       obj_value = 1.0
#     elif raw_obj_value == 'Have now':
#       obj_value = 2.0
#     else:
#       raise Exception('Unknown tuberculum status for %s: %s'%(GIDN, raw_obj_value))


if __name__ == '__main__':
  pass
  