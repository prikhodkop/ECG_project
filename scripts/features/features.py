from collections import OrderedDict
import numpy as np
import logging
import scipy.signal as sg
from scipy.stats import linregress
import pandas as pd

try: 
  import triangulation as tg
except Exception as e:
  logging.warning(e)

def get_default_pulse_features_params():
  """
  Return dict of default pulse features parameters
  'None' means no actions
  """
  options =  {'time features':      { 'step': [60000], #ms
                                      'step_for_hr': [600000], #ms
                                      'bounds': [20, 50], #ms
                                      'triangular': True
                                    },     

              'frequency features': {'frequency bounds': [0, 0.0033, 0.04, 0.15, 0.4] # Hz
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
      frequency_features, features_features_names = calculate_frequency_features(data_RR, features_params['frequency features'])
      data_RR_features += frequency_features
      features_names += features_features_names

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
  See Murukesan14 for feature description.

  Args
    data_RR (np.array of np.int64): data in format (time [ms], interval [ms])
    time_options (dict): see get_default_pulse_features_params()

  Returns
    time_features (list of float): calculated features for given pulse chunk
    time_features_names (list of str): features names in appropriate order 
  """

  times = data_RR[:, 0].copy().astype(float)
  intervals = data_RR[:, 1].copy().astype(float)

  features = pd.Series() 

  features.set_value('maxNN', np.max(intervals))
  features.set_value('minNN', np.min(intervals))
  features.set_value('meanNN', np.mean(intervals))
  features.set_value('medianNN', np.median(intervals))
  features.set_value('SDNN', np.std(intervals))

  if time_options['step']: 
    for step in time_options['step']:
      mean_intervals = []
      std_intervals = []
      for i in xrange(0, int(times[-1]), step):
        mean_intervals.append(np.mean(intervals[(i < times) * (times < i + step)]))
        std_intervals.append(np.std(intervals[(i < times) * (times < i + step)]))
      features.set_value('SDA' + str(step) + 'NN', np.std(mean_intervals))
      features.set_value('SD' + str(step) + 'NNind', np.mean(std_intervals))

  diff = intervals[1:] - intervals[:-1]
  if time_options['bounds']:
    for bound in time_options['bounds']:
      features.set_value('NN' + str(bound), np.sum(np.fabs(diff) > bound))
      features.set_value('pNN' + str(bound), features['NN' + str(bound)] / float(intervals.shape[0] - 1))
  
  features.set_value('RMSSD', np.mean(np.power(diff, 2)) ** 0.5)
  features.set_value('MeanHR', np.mean(60000. / intervals))

  if time_options['step_for_hr']: 
    for step in time_options['step_for_hr']:
      mean_intervals = []
      for i in xrange(0, int(times[-1]), step):
        mean_intervals.append(np.mean(60000. / intervals[(i < times) * (times < i + step)]))
      features.set_value('sd' + str(step) + 'HR', np.std(mean_intervals))
  
  # this part is not ready yet (?)
  if time_options['triangular'] is not None:
    M, N, S = tg.apply_grad_descent(intervals)
    h = 2. / (M + N)
    features.set_value('HRVti', intervals.shape[0] / 2. * (M + N))
    features.set_value('TINN', M - N)

  # extra features, which are not from papers
  features.set_value('meanSD', np.mean(np.fabs(diff)))
  features.set_value('stdSD', np.std(np.fabs(diff)))
  features.set_value('outliers20', np.sum(np.fabs(intervals[1:]/intervals[:-1] - 1) > 0.2) / float(intervals.shape[0] - 1))

  return features


def calculate_frequency_features(data_RR, frequency_options):
  """
  Calculate frequency-based features for given chunk of pulse.

  Args
    data_RR (np.array of np.int64): data in format (time [ms], interval [ms])
    frequency_options (dict): see get_default_pulse_features_params()

  Returns
    features (list of float): calculated features for given pulse chunk
    features_names (list of str): features names in appropriate order 
  """

  times = data_RR[:, 0].copy().astype(float)
  intervals = data_RR[:, 1].copy().astype(float)

  features = []

  f = np.linspace(0.001, 0.5, 100)
  pgram = sg.lombscargle(times/1000, intervals/1000, f)

  bounds_names = ['ULF', 'VLF', 'LF', 'HF']
  frequency_bounds = frequency_options['frequency bounds']
  power = np.mean(pgram) * (frequency_bounds[-1] - frequency_bounds[0])

  for i in xrange(len(frequency_bounds) - 1):
    idx = (f > frequency_bounds[i]) * (f <= frequency_bounds[i+1])
    features.append([bounds_names[i], np.mean(pgram[idx]) * (frequency_bounds[i+1] - frequency_bounds[i])])
    features.append([bounds_names[i] + 'rel', np.sum(pgram[idx]) / power])
    features.append([bounds_names[i] + 'peak', f[idx][np.argmax(pgram[idx])]])
    if i > 1:
      features.append([bounds_names[i] + 'normalized', np.sum(pgram[idx]) / (power - features[3][1])]) #features[3][1] is VLF and is already calculated

  features.append(['LF/HF', pgram[2] / pgram[3]])

  frequency_features_names, frequency_features = zip(*features)
  return list(frequency_features), list(frequency_features_names)



def calculate_nonlinear_features(data_RR, nonlinear_options):
  """
  Calculate nonlinear features for given chunk of pulse.

  Args
    data_RR (np.array of np.int64): data in format (time [ms], interval [ms])
    nonlinear_options (dict): see get_default_pulse_features_params(). Not used # TODO

  Returns
    features (list of float): calculated features for given pulse chunk
    features_names (list of str): features names in appropriate order 
  """

  intervals = data_RR[:, 1].copy().astype(float)

  features = []
  # Poincare plot
  k, b, _, _, _ = linregress(intervals[:-1], intervals[1:])
  def line(x, slope, intercept):
    return slope * x + intercept

  k_ = -1. / k
  b_ = b + np.mean(intervals) * (k + 1. / k)

  projections1 = (intervals[1:] - line(intervals[:-1], k, b)) * k / np.sqrt(1. + k**2)
  projections2 = (intervals[1:] - line(intervals[:-1], k_, b_)) * k_ / np.sqrt(1. + k_**2)
  
  features.append(['poincSD1', np.std(projections1)])
  features.append(['poincSD2', np.std(projections2)])

  w = np.zeros(intervals.shape[0])
  mi = np.mean(intervals)
  w[1:] = np.array([w[i-1] + intervals[i] - mi for i in xrange(1, intervals.shape[0])])
  ls = [2, 4, 6, 8, 11, 16, 20, 25, 32, 35, 46, 58, 64, 72, 80, 96, 128]
  fl = np.zeros(len(ls))
  for li, l in enumerate(ls):
    e = 0
    for j in xrange(l):
      i0 = j * w.shape[0] / l
      i1 = (j + 1) * w.shape[0] / l
      std = linregress(np.arange(i0, i1, 1), w[i0:i1])[-1]
      e += std**2 * w.shape[0] / l
    fl[li] = np.sqrt(e / l)
  l0 = 12
  
  alpha0 = linregress(np.log(ls[:l0])/np.log(10), np.log(fl[:l0])/np.log(10))[0]      
  alpha1 = linregress(np.log(ls[l0:])/np.log(10), np.log(fl[l0:])/np.log(10))[0]

  features.append(['alpha0', alpha0])
  features.append(['alpha1', alpha1])
  features.append(['alphadiff', alpha1 - alpha0])
  
  nonlinear_features_names, nonlinear_features = zip(*features)
  return list(nonlinear_features), list(nonlinear_features_names)


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

