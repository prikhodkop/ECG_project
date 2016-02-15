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

# [1] - L. Murukesan, Machine Learning Approach for Sudden Cardiac Arrest Prediction Based on Optimal Heart Rate Variability Features. 
# [2] - C. K. Peng, Quantification of scaling exponents and crossover phenomena in nonstationary heartbeat time series.
# [3] - M. G. Signorini, Nonlinear analysis of heart rate variability signal for the characterization of cardiac heart failure patients
# [4] - https://en.wikipedia.org/wiki/Sample_entropy 
 
def get_default_pulse_features_params():
  """
  Return dict of default pulse features parameters
  'None' means no actions
  """
  options =  {'sampling rate':      1000, #Hz 
              'vizualization':      False, # demonstration of sample features in graphic manner  
              'save pic':           False, # saving pics of sample features in graphic manner  
              'time features':      { 'step SDANN': [60000], #ms, window size for computation SDANN, see [1]
                                      'step SDNNind': [60000], #ms, window size for computation SDNNind, see [1]
                                      'step sdHR': [600000], #ms, window size for coputation sdHR, see [1]
                                      'threshold NN': [20, 50], #ms, threshold for coputation NNx ans pNNx, see [1]
                                      'threshold outlier': 0.2, #ms, threshold for coputation outlier, see [1]
                                    },     

              'frequency features': { 'frequency bounds': [0., 0.0033, 0.04, 0.15, 0.4], #Hz bounds for ULF, VLF, LF and HF, see[1]
                                      'frequency range': [0.001, 0.4, 200] #Hz lowest and highest frequencies, see[1]
                                    },

              'nonlinear features': {'box lengths': [4, 6, 8, 10, 12, 14, 16, 17, 25, 33, 41, 49, 57, 64], #beats, number of points for each trend, see [2]
                                     'alpha border': 16.5, #beats, border for computation alpha1 and alpha2, see [2]
                                     'embedding dimension': 3, #beats, size of windows for sampen, see [3] and [4], the value of parameter is from [1]
                                     'tolerance': 0.1, #s, threshold for max difference of two groups, see [3] and [4], the value of parameter is from[1]
                                     'limit for sample size': 420 #beats, limit for computation of sampen due to quadratic complexity of algorythm
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
      time_features, time_features_names = calculate_time_features(data_RR, features_params)
      data_RR_features += time_features
      features_names += time_features_names

    if features_params['frequency features'] is not None:
      frequency_features, frequency_features_names = calculate_frequency_features(data_RR, features_params)
      data_RR_features += frequency_features
      features_names += frequency_features_names

    if features_params['nonlinear features'] is not None:
      nonlinear_features, nonlinear_features_names = calculate_nonlinear_features(data_RR, features_params)
      data_RR_features += nonlinear_features
      features_names += nonlinear_features_names

    splitted_features.append(np.array(data_RR_features)) #!!!

  return splitted_features, features_names


def calculate_time_features(data_RR, options):
  """
  Calculate time-based features for given chunk of pulse.
  See [1] for feature description.

  Args
    data_RR (np.array of np.int64): data in format (time [ms], interval [ms])
    options (dict): see get_default_pulse_features_params()

  Returns
    features (pandas.DataFrame): names and calculated features for given pulse chunk 
  """

  times = data_RR[:, 0].copy().astype(float)
  intervals = data_RR[:, 1].copy().astype(float)
  sampling_rate = options['sampling rate']
  
  features = pd.Series() 

  features.set_value('maxNN', np.max(intervals))
  features.set_value('minNN', np.min(intervals))
  features.set_value('meanNN', np.mean(intervals))
  features.set_value('medianNN', np.median(intervals))
  features.set_value('SDNN', np.std(intervals))

  for step in options['time features']['step SDANN']:
    if step < times[-1] / 2:
      mean_intervals = []
      for i in xrange(0, int(times[-1]), step):
        mean_intervals.append(np.mean(intervals[(i < times) * (times < i + step)]))
      features.set_value('SDA' + str(step) + 'NN', np.std(mean_intervals))
    else:
      features.set_value('SDA' + str(step) + 'NN', np.nan)

  for step in options['time features']['step SDNNind']:
    if step < times[-1] / 2:
      std_intervals = []
      for i in xrange(0, int(times[-1]), step):
        std_intervals.append(np.std(intervals[(i < times) * (times < i + step)]))
      features.set_value('SD' + str(step) + 'NNind', np.mean(std_intervals))
    else:
      features.set_value('SD' + str(step) + 'NNind', np.nan)

  diff = intervals[1:] - intervals[:-1]
  for threshold in options['time features']['threshold NN']:
    features.set_value('NN' + str(threshold), np.sum(np.fabs(diff) > threshold))
    features.set_value('pNN' + str(threshold), features['NN' + str(threshold)] / float(intervals.shape[0] - 1))
  
  features.set_value('RMSSD', np.mean(np.power(diff, 2)) ** 0.5)
  features.set_value('MeanHR', np.mean(60. * sampling_rate / intervals))

  for step in options['time features']['step sdHR']:
    if step < times[-1] / 2:
      mean_intervals = []
      for i in xrange(0, int(times[-1]), step):
        mean_intervals.append(np.mean(60. * sampling_rate / intervals[(i < times) * (times < i + step)]))
      features.set_value('sd' + str(step) + 'HR', np.std(mean_intervals))
    else:
      features.set_value('sd' + str(step) + 'HR', np.nan)
  
  # not reviewed
  M, N, S = tg.apply_grad_descent(intervals)
  features.set_value('HRVti', intervals.shape[0] / 2. * (M + N))
  features.set_value('TINN', M - N)

  threshold = options['time features']['threshold outlier']
  features.set_value('outlier', np.sum(np.fabs(intervals[1:]/intervals[:-1] - 1) > threshold) / float(intervals.shape[0] - 1))

  # extra features, which are not from papers
  features.set_value('meanSD', np.mean(np.fabs(diff)))
  features.set_value('stdSD', np.std(np.fabs(diff)))
  
  features_names = list(features.index)
  features_values = list(features)

  return list(features_values), list(features_names)


def calculate_frequency_features(data_RR, options):
  """
  Calculate frequency-based features for given chunk of pulse.
  See [1].

  Args
    data_RR (np.array of np.int64): data in format (time [ms], interval [ms])
    options (dict): see get_default_pulse_features_params()

  Returns
    features (pandas.DataFrame): names and calculated features for given pulse chunk 
  """

  times = data_RR[:, 0].copy().astype(float)
  intervals = data_RR[:, 1].copy().astype(float)
  sampling_rate = options['sampling rate']
  vizualization = options['vizualization']
  save_pic = options['save pic']

  features = pd.Series() 

  frequency_range = options['frequency features']['frequency range'][1] - options['frequency features']['frequency range'][0]
  f =  np.linspace(options['frequency features']['frequency range'][0], options['frequency features']['frequency range'][1], options['frequency features']['frequency range'][2])
  pgram = sg.lombscargle(times/sampling_rate, intervals, f)

  bounds_names = ['ULF', 'VLF', 'LF', 'HF']
  frequency_bounds = options['frequency features']['frequency bounds']
  totalpower = np.mean(pgram) * frequency_range

  if save_pic or vizualization:
    for i in xrange(len(frequency_bounds)-1):
      idx = (f > frequency_bounds[i]) * (f <= frequency_bounds[i+1])
      plt.plot(f[idx], pgram[idx])
      plt.title('Periodogramm')
      plt.xlabel('Frequency, Hz')
      plt.ylabel('Power')
    if vizualization:
      plt.show()
    if save_pic:
      plt.savefig('pgram.png')
    plt.close()

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

  features_names = list(features.index)
  features_values = list(features)

  return list(features_values), list(features_names)


def calculate_nonlinear_features(data_RR, options):
  """
  Calculate nonlinear features for given chunk of pulse.
  See [1] for review

  Args
    data_RR (np.array of np.int64): data in format (time [ms], interval [ms])
    options (dict): see get_default_pulse_features_params().

  Returns
    features (pandas.DataFrame): names and calculated features for given pulse chunk 
  """

  features = pd.Series()

  intervals = data_RR[:, 1].copy().astype(float)
  sampling_rate = options['sampling rate']
  vizualization = options['vizualization']
  save_pic = options['save pic']
  size = intervals.shape[0]

  intervals = data_RR[:, 1].copy().astype(float)
  time = np.sum(intervals) / sampling_rate
  
  # Poincare plot
  # see [1]
  # projections for line of regression and perpendicular line

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
  distances2 = ((projection2x - x_cross) ** 2 + (projection2y - y_cross) ** 2) ** 0.5

  features.set_value('poincSD1', np.std(distances1))
  features.set_value('poincSD2', np.std(distances2))

  if vizualization or save_pic:
    ax1 = plt.subplot2grid((3,2), (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((3,2), (2, 0))
    ax3 = plt.subplot2grid((3,2), (2, 1))
    ax1.scatter(intervals[:-1], intervals[1:], c='g', alpha=0.1, s=6)
    ax1.plot(np.arange(np.min(intervals[:-1]), np.max(intervals[:-1])), k1 * np.arange(np.min(intervals[:-1]), np.max(intervals[:-1])) + b1, c='b')
    ax1.plot(np.arange(np.min(intervals[:-1]), np.max(intervals[:-1])), k * np.arange(np.min(intervals[:-1]), np.max(intervals[:-1])) + b, c='r')
    ax1.set_xlim(np.percentile(intervals, 1), np.percentile(intervals, 99))
    ax1.set_ylim(np.percentile(intervals, 1), np.percentile(intervals, 99))
    sign_distances1 = distances1 * np.sign(projection1y - y_cross)
    sign_distances2 = distances2 * np.sign(y_cross - projection2y)
    ax2.hist(sign_distances1[(sign_distances1 > np.percentile(sign_distances1, 0.1)) * (sign_distances1 < np.percentile(sign_distances1, 99.9))], 50, facecolor='b', alpha=0.75)
    ax3.hist(sign_distances2[(sign_distances2 > np.percentile(sign_distances2, 0.1)) * (sign_distances2 < np.percentile(sign_distances2, 99.9))], 50, facecolor='r', alpha=0.75)
    ax1.set_title('Poincare plot')
    ax1.set_xlabel('Current interval')
    ax1.set_ylabel('Next interval')
    ax2.set_title('Projection for blue line')
    ax2.set_xlabel('Projection')
    ax2.set_ylabel('Frequency')
    ax3.set_title('Projection for red line')
    ax3.set_xlabel('Projection')
    ax3.set_ylabel('Frequency')
    plt.tight_layout()
    if vizualization:
      plt.show()
    if save_pic:
      plt.savefig('poincare.png')
    plt.close()

  # Sample Entropy
  # see [3] and [4] 
  m = options['nonlinear features']['embedding dimension']
  r = options['nonlinear features']['tolerance']
  limit_for_sample_size = options['nonlinear features']['limit for sample size']

  if size > limit_for_sample_size:
    features.set_value('sampen', np.nan)

  else:
    r = 1000*r
    distances = np.zeros((size, size))
    for i in xrange(size):
      for j in xrange(size):
        distances[i, j] = abs(intervals[i] - intervals[j])

    Cheb_distances = np.zeros((size-m+1, size-m+1))
    for i in xrange(size-m+1):
      for j in xrange(size-m+1):
        max_dist = 0
        for k in xrange(m):
          max_dist = max(max_dist, distances[i+k, j+k])
        Cheb_distances[i, j] = max_dist
    B = np.sum(Cheb_distances < r) - (size-m+1)

    Cheb_distances = np.zeros((size-m, size-m))
    for i in xrange(size-m):
      for j in xrange(size-m):
        max_dist = 0
        for k in xrange(m+1):
          max_dist = max(max_dist, distances[i+k, j+k])
        Cheb_distances[i, j] = max_dist
    A = np.sum(Cheb_distances < r) - (size-m)

    features.set_value('sampen', -np.log(float(A)/B))

  # Non-Linear Domain Analysis
  # see [2]
  box_lengths = np.array(options['nonlinear features']['box lengths'])
  alpha_border = np.array(options['nonlinear features']['alpha border'])
  y = np.zeros(len(intervals))
  B_ave = np.mean(intervals)
  y[0] = intervals[0] - B_ave 
  for k in xrange(1, size):
    y[k] = y[k-1] + intervals[k] - B_ave

  i = 0
  F = np.zeros(len(box_lengths))
  for n in box_lengths:

    box_number = size / n
    y_n = np.zeros(size)
    trends = np.zeros((box_number, 2))
    
    for box_idx in xrange(box_number):
      trends[box_idx][0], trends[box_idx][1], _, _, _ = linregress(np.arange(box_idx * n, (box_idx+1) * n), y[box_idx * n : (box_idx+1) * n])
      y_n[np.arange(box_idx * n, (box_idx+1) * n)] = trends[box_idx][0] * np.arange(box_idx * n, (box_idx+1) * n) + trends[box_idx][1]

    F[i] = (np.mean(np.power(y - y_n, 2))) ** 0.5
    i += 1

  if vizualization or save_pic:
    plt.plot(np.arange(y_n.shape[0]), y_n)
    plt.plot(np.arange(y_n.shape[0]), y)
    plt.title('Regression for sum of intervals')
    plt.xlabel('Index of interval')
    plt.ylabel('Sum of intervals')
    if vizualization:
      plt.show()
    if save_pic:
      plt.savefig('regression.png')
    plt.close()
    
  alpha0, b0, _, _, _ = linregress(np.log(box_lengths), np.log(F))
  alpha1, b1, _, _, _ = linregress(np.log(box_lengths[box_lengths < alpha_border]), np.log(F[box_lengths < alpha_border]))
  alpha2, b2, _, _, _ = linregress(np.log(box_lengths[box_lengths > alpha_border]), np.log(F[box_lengths > alpha_border]))

  if vizualization or save_pic:
    plt.plot(np.log(box_lengths), alpha0 * np.log(box_lengths) + b0, linestyle='dashed')
    plt.plot(np.log(box_lengths[box_lengths < alpha_border]), alpha1 * np.log(box_lengths[box_lengths < alpha_border]) + b1, linewidth=2)
    plt.plot(np.log(box_lengths[box_lengths > alpha_border]), alpha2 * np.log(box_lengths[box_lengths > alpha_border]) + b2, linewidth=2)
    plt.scatter(np.log(box_lengths), np.log(F))
    plt.title('Dependence F on n')
    plt.xlabel('Log of box lengths')
    plt.ylabel('Log of F')
    if vizualization:
      plt.show()
    if save_pic:
      plt.savefig('alpha.png')
    plt.close()

  features.set_value('alpha0', alpha0)
  features.set_value('alpha1', alpha1)
  features.set_value('alpha2', alpha2)
  
  features_names = list(features.index)
  features_values = list(features)

  return list(features_values), list(features_names)


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


def get_stat_features(stat_features_names, stat_info, GIDN):
  # TODO

  if stat_features_names:
    msg = 'Not implemented statistical features' #TODO
    logging.critical(msg)
    raise Exception(msg)

  stat_features = None
  return stat_features
  
if __name__ == '__main__':

  # options = get_default_pulse_features_params()
  # options['save pic'] = True
  # print generate_pulse_features([read_rr_file('../../../../../520307.rr', 0, None)[:300]], options)

  pass

