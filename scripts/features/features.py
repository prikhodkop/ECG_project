from collections import OrderedDict
import numpy as np
import logging
import scipy.signal as sg
from scipy.stats import linregress
import pandas as pd
import matplotlib.pyplot as plt

try: 
  import triangulation as tg
except Exception as e:
  logging.warning(e)

def get_default_pulse_features_params():
  """
  Return dict of default pulse features parameters
  'None' means no actions
  """
  options =  {'sampling rate':      1000, #Hz
              'vizualization':      False,
              'time features':      { 'step': [60000], #ms
                                      'step for hr': [600000], #ms
                                      'bounds': [20, 50], #ms
                                      'triangular': True
                                    },     

              'frequency features': { 'frequency bounds': [0., 0.0033, 0.04, 0.15, 0.4], #Hz
                                      'frequency range': [0.001, 0.4, 100]
                                    },

              'nonlinear features': {'box lengths': [4, 6, 8, 10, 12, 14, 16, 20, 30, 60, 120, 180, 240, 300] #beats
                                    } 
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


def calculate_time_features(data_RR, time_options, sampling_rate):
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
  features.set_value('MeanHR', np.mean(60. * sampling_rate / intervals))

  if time_options['step_for_hr']: 
    for step in time_options['step for hr']:
      mean_intervals = []
      for i in xrange(0, int(times[-1]), step):
        mean_intervals.append(np.mean(60. * sampling_rate / intervals[(i < times) * (times < i + step)]))
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


def calculate_frequency_features(data_RR, frequency_options, sampling_rate):
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

  features = pd.Series() 

  frequency_range = frequency_options['frequency range'][1] - frequency_options['frequency range'][0]
  f =  np.linspace(frequency_options['frequency range'][0], frequency_options['frequency range'][1], frequency_options['frequency range'][2])
  pgram = sg.lombscargle(times/sampling_rate, intervals, f)

  bounds_names = ['ULF', 'VLF', 'LF', 'HF']
  frequency_bounds = frequency_options['frequency bounds']
  totalpower = np.mean(pgram) * frequency_range

  idx = {}
  for i in xrange(1, len(frequency_bounds)-1):
    idx[bounds_names[i]] = (f > frequency_bounds[i]) * (f <= frequency_bounds[i+1])
    features.set_value('a' + bounds_names[i], np.mean(pgram[idx[bounds_names[i]]]) * frequency_range)

  for i in xrange(1, len(frequency_bounds)-1):
    features.set_value('peak' + bounds_names[i], f[idx[bounds_names[i]]][np.argmax(pgram[idx[bounds_names[i]]])])

  features.set_value('aTotal', features['aVLF'] + features['aLF'] + features['aHF'])

  for i in xrange(1, len(frequency_bounds)-1):
    features.set_value('p' + bounds_names[i], features['a' + bounds_names[i]] / totalpower)
  
  for i in xrange(2, len(frequency_bounds)-1):
      features.set_value('n' + bounds_names[i], features['a' + bounds_names[i]] / (totalpower - features['aVLF']))

  features.set_value('LF/HF', features['aLF'] / features['aHF'])

  return features


def calculate_nonlinear_features(data_RR, options):
  """
  Calculate nonlinear features for given chunk of pulse.

  Args
    data_RR (np.array of np.int64): data in format (time [ms], interval [ms])
    nonlinear_options (dict): see get_default_pulse_features_params(). Not used # TODO

  Returns
    features (list of float): calculated features for given pulse chunk
    features_names (list of str): features names in appropriate order 
  """
  features = pd.Series()

  box_lengths = np.array(options['nonlinear features']['box lengths'])
  sampling_rate = options['sampling rate']
  vizualization = options['vizualization']
  # TODO add get_boolean

  intervals = data_RR[:, 1].copy().astype(float)
  time = np.sum(intervals) / sampling_rate
  
  # Poincare plot
  k1, b1, _, _, _ = linregress(intervals[:-1], intervals[1:])
  k = k1
  b = b1
  x_cross = np.mean(intervals[:-1])
  y_cross = k * x_cross + b

  projection1x = (intervals[1:] + intervals[:-1] / k - b) / (k + 1.0 / k)
  projection1y = (intervals[1:] * k + intervals[:-1] + b / k) / (k + 1.0 / k)
  distances1 = ((projection1x - x_cross) ** 2 + (projection1y - y_cross) ** 2) ** 0.5
  
  k = -1. / k 
  b = -x_cross * (k + 1. / k) + b

  projection2x = (intervals[1:] + intervals[:-1] / k - b) / (k + 1.0 / k)
  projection2y = (intervals[1:] * k + intervals[:-1] + b / k) / (k + 1.0 / k)
  distances2 = ((projection1x - x_cross) ** 2 + (projection1y - y_cross) ** 2) ** 0.5

  features.set_value('poincSD1', np.std(distances1))
  features.set_value('poincSD2', np.std(distances2))

  if vizualization:
    plt.scatter(intervals[:-1], intervals[1:], c='g', alpha=0.05, s=3)
    plt.plot(np.arange(np.min(intervals[:-1]), np.max(intervals[:-1])), k1 * np.arange(np.min(intervals[:-1]), np.max(intervals[:-1])) + b1, c='b')
    plt.plot(np.arange(np.min(intervals[:-1]), np.max(intervals[:-1])), k * np.arange(np.min(intervals[:-1]), np.max(intervals[:-1])) + b, c='r')
    plt.scatter(projection1x, projection1y, c='b', alpha=0.1, s=10)
    plt.scatter(projection2x, projection2y, c='b', alpha=0.1, s=10)
    plt.show()

  # Non-Linear Domain Analysis
  y = np.zeros(len(intervals))
  B_ave = np.mean(intervals)
  y[0] = intervals[0] - B_ave 
  for k in xrange(1, intervals.shape[0]):
    y[k] = y[k-1] + intervals[k] - B_ave

  i = 0
  F = np.zeros(len(box_lengths))
  for n in box_lengths:

    box_number = intervals.shape[0] / n
    y_n = np.zeros(intervals.shape[0])
    trends = np.zeros((box_number, 2))
    
    for box_idx in xrange(box_number):
      trends[box_idx][0], trends[box_idx][1], _, _, _ = linregress(np.arange(box_idx * n, (box_idx+1) * n), y[box_idx * n : (box_idx+1) * n])
      y_n[np.arange(box_idx * n, (box_idx+1) * n)] = trends[box_idx][0] * np.arange(box_idx * n, (box_idx+1) * n) + trends[box_idx][1]

    F[i] = (np.mean(np.power(y - y_n, 2))) ** 0.5
    i += 1
    
  alpha0, b0, _, _, _ = linregress(np.log(box_lengths), np.log(F))
  alpha1, b1, _, _, _ = linregress(np.log(box_lengths[box_lengths < 17]), np.log(F[box_lengths < 17]))
  alpha2, b2, _, _, _ = linregress(np.log(box_lengths[box_lengths > 16]), np.log(F[box_lengths > 16]))
  
  # plt.plot(np.log(box_lengths[box_lengths < 17]), alpha1 * np.log(box_lengths[box_lengths < 17]) + b1)
  # plt.plot(np.log(box_lengths[box_lengths > 16]), alpha2 * np.log(box_lengths[box_lengths > 16]) + b2)
  # plt.scatter(np.log(box_lengths), np.log(F))
  # plt.show()

  features.set_value('alpha0', alpha0)
  features.set_value('alpha1', alpha1)
  features.set_value('alpha2', alpha2)
  
  return features


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

# #debug
# def read_rr_file(file_name, start, finish):
#   # read rr file and return array of peak coordinates

#   print 'reading rr file'

#   f = open(file_name, 'r')
#   f.readline()
#   f.readline()
#   ans = []
#   temp = 1
  
#   if finish != None:
#     for s in f:
#       if temp != 0:
#         coordinate = [int(s.strip().split()[0]), int(s.strip().split()[1])]
#         if coordinate[0] != 0 and coordinate[0] < finish and coordinate[0] >= start and coordinate[1] != 0:
#           ans.append(coordinate)
#         elif coordinate[0] == 0:
#           temp = 0
#       else:
#         temp = 1
  
#   else:
#     for s in f:
#       if temp != 0:
#         coordinate = [int(s.strip().split()[0]), int(s.strip().split()[1])]
#         if coordinate[0] != 0 and coordinate[0] >= start and coordinate[1] != 0:
#           ans.append(coordinate)
#         elif coordinate[0] == 0:
#           temp = 0
#       else:
#         temp = 1
  
#   f.close()

#   return np.array(ans)

# if __name__ == '__main__':
  
#   # debug
#   # frequency_features, frequency_features_names = calculate_frequency_features(read_rr_file('../../../../520307.rr', 0, None), get_default_pulse_features_params()['frequency features'])
#   # for i in xrange(len(frequency_features_names)):
#     # print frequency_features_names[i], frequency_features[i], get_default_pulse_features_params()['sampling rate']
#   features = calculate_nonlinear_features(read_rr_file('../../../../../520307.rr', 0, None), get_default_pulse_features_params())

#   print features

#   pass