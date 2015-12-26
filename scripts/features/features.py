from collections import OrderedDict
import numpy as np
import logging


def get_default_pulse_features_params():
  """
  Return dict of default pulse features parameters
  'None' means no actions
  """
  options =  {'time features':      { 'autocorr step': [5, 20] # in beats???
                                    },     

              'frequency features': {'frequency bounds': [0, 0.0033, 0.04, 0.15, 0.5] # Hz
                                    },

              'nonlinear features': 'default' # e.g. 'default' or None
             }

  return options



def generate_pulse_features(splitted_data_RR, features_params):
  """
  Calculate features related to pulse chunks:
    * time-based
    * frequency-based
    * nonlinear

  Args
    splitted_data_RR (list of np.array of np.int64): where np.array has format (time [ms], interval [ms])
    pulse_features_params (dict): see get_default_pulse_features_params()

  Returns
    splitted_features (list of np.array of floats)
    features_names (list of str): features names; the order of features in list relates to their order in splitted_features
  """
  
  splitted_features = []
  for data_RR in splitted_data_RR:

    features_names = []
    data_RR_features = []
    
    if features_params['time features'] is not None:
      time_features, time_features_names = calculate_time_features(data_RR, features_params['time features'])
      data_RR_features += time_features
      features_names += time_features_names

    if features_params['frequency features'] is not None:
      frequency_features, frequency_features_names = calculate_frequency_features(data_RR, features_params['frequency features'])
      data_RR_features += frequency_features
      features_names += frequency_features_names

    if features_params['nonlinear features'] is not None:
      nonlinear_features, nonlinear_features_names = calculate_nonlinear_features(data_RR, features_params['nonlinear features'])
      data_RR_features += nonlinear_features
      features_names += nonlinear_features_names

    splitted_features.append(np.array(data_RR_features)) #!!!

  return splitted_features, features_names



def calculate_time_features(data_RR, time_options):
  """
  Calculate time-based features for given chunk of pulse.
  Only intervals lengths are considered. 

  Args
    data_RR (np.array of np.int64): data in format (time [ms], interval [ms])
    time_options (dict): see get_default_pulse_features_params()

  Returns
    time_features (list of float): calculated featurea for given pulse chunk
    time_features_names (list of str): features names in appropriate order 
  """

  features = []
  intervals = data_RR[:, 1].copy().astype(float) #!!!

  features.append(['meanNN', np.mean(intervals)])
  features.append(['minNN', np.min(intervals)])
  features.append(['maxNN', np.min(intervals)])
  features.append(['medNN', np.median(intervals)])
  features.append(['SDNN', np.std(intervals)])

  SD = intervals[1:] - intervals[:-1] # successive differences
  features.append(['SDSD',  np.std(np.fabs(SD))])
  features.append( [ 'RMSSD', np.sqrt(np.mean(np.power(SD, 2))) ] )

  time_features_names, time_features = zip(*features)
  return list(time_features), list(time_features_names)


def calculate_frequency_features(data_RR, frequency_options):
  """
  TODO
  """

  msg = 'Not implemented.'
  logging.critical(msg)
  raise Exception(msg)

  # Example:
  features = []
  times = data_RR[:, 0].copy().astype(float) #!!!
  intervals = data_RR[:, 1].copy().astype(float) #!!!
  
  # TODO....
  #features.append(['ULF', ULF])


  time_features_names, time_features = zip(*features)
  return list(time_features), list(time_features_names)



def calculate_nonlinear_features(data_RR, nonlinear_options):
  """
  TODO
  """

  msg = 'Not implemented.'
  logging.critical(msg)
  raise Exception(msg)


def get_stat_features(stat_features_names, stat_info, GIDN):
  # TODO

  if stat_features_names:
    msg = 'Not implemented statistical features' #TODO
    logging.critical(msg)
    raise Exception(msg)

  stat_features = None
  return stat_features
  

def get_features_matrix(splitted_pulse_features):
  """
  Args
    splitted_pulse_features (list of np.array of floats)

  Returns
    pulse_features_matrix
  """

  try:
    pulse_features_matrix = np.vstack(splitted_pulse_features)
  except Exception as e:
    msg = "Can\'t transform splitted features to matrix: %s"%e
    logging.critical(msg)
    raise Exception(msg)

  return pulse_features_matrix

  

if __name__ == '__main__':
  pass

