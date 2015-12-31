from collections import OrderedDict
import numpy as np
import logging
import scipy.signal as sg
from scipy.stats import linregress

try: 
  import triangulation as tg
except Exception as e:
  logging.warning(e)

def get_default_pulse_features_params():
  """
  Return dict of default pulse features parameters
  'None' means no actions
  """
  options =  {'time features':      { 'autocorr step': [5, 20], # in beats???
                                      'step': [5, 20],
                                      'bounds': [20, 50],
                                      'triangular': None
                                    },     

              'frequency features': {'frequency bounds': [0, 0.0033, 0.04, 0.15, 0.5] # Hz
                                    },

              'nonlinear features': None # e.g. 'default' or None
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

  Args
    data_RR (np.array of np.int64): data in format (time [ms], interval [ms])
    time_options (dict): see get_default_pulse_features_params()

  Returns
    time_features (list of float): calculated features for given pulse chunk
    time_features_names (list of str): features names in appropriate order 
  """

  times = data_RR[:, 0].copy().astype(float)
  intervals = data_RR[:, 1].copy().astype(float)

  features = []  
  features.append(['meanNN', np.mean(intervals)])
  features.append(['minNN', np.min(intervals)])
  features.append(['maxNN', np.max(intervals)])
  features.append(['medianNN', np.median(intervals)])
  features.append(['stdNN', np.std(intervals)])
  features.append(['meanfabsSD', np.mean(np.fabs(intervals[1:] - intervals[:-1]))])
  features.append(['stdSD', np.std(np.fabs(intervals[1:] - intervals[:-1]))])
  features.append(['meanHR', np.mean(60000. / intervals)])
  features.append(['stdHR', np.std(60000. / intervals)])

  diffs = intervals[1:] / intervals[:-1]
  features.append(['outliers20', diffs[np.fabs(diffs - 1) > 0.2].shape[0] / float(diffs.shape[0])])


  if time_options['bounds']:
    if not (type(time_options['bounds']) == list or type(time_options['bounds']) == np.ndarray):
      time_options['bounds'] = [time_options['bounds']]

    for bound in time_options['bounds']:
      features.append(['pNN' + str(bound), intervals[:-1][np.fabs(intervals[1:] - intervals[:-1]) > bound].shape[0] / float(intervals.shape[0] - 1)])


  # Calculate statistics for averaged data
  if time_options['step']: 
    if not (type(time_options['step']) == list or type(time_options['step']) == np.ndarray):
      time_options['step'] = [time_options['step']]

    for step in time_options['step']:
      mean_intervals = []
      std_intervals = []
      fabs_intervals = []
      hr_intervals = []
      last_step = step
      i = 0
      while i < intervals.shape[0]:
        if i > intervals.shape[0] - last_step:
          last_step = intervals.shape[0] - i
        mean_intervals.append(np.mean(intervals[i:i+last_step]))
        std_intervals.append(np.std(intervals[i:i+last_step]))
        hr_intervals.append(np.mean(60000. / intervals[i:i+last_step]))
        i = i + last_step
      # np.mean(mean_x) = np.mean(x) and just exists in features
      features.append(['meanmeanA' + str(step) + 'NN', np.std(mean_intervals)])
      features.append(['meanstdA' + str(step) + 'NN', np.mean(std_intervals)])
      features.append(['stdstdA' + str(step) + 'NN', np.std(std_intervals)])
      features.append(['meanA' + str(step) + 'HR', np.std(hr_intervals)])
  
  if time_options['triangular'] is not None:
    M, N, S = tg.apply_grad_descent(intervals)
    h = 2. / (M + N)
    features.append(['HRVti', intervals.shape[0] / 2. * (M + N)])
    features.append(['TINN', M - N])


  time_features_names, time_features = zip(*features)
  return list(time_features), list(time_features_names)


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
  power = np.sum(pgram)

  normalized_power = power - pgram[1]
  bounds_names = ['ULF', 'VLF', 'LF', 'HF']
  frequency_bounds = frequency_options['frequency bounds']

  for i in xrange(len(frequency_bounds) - 1):
    idx = (f > frequency_bounds[i]) * (f <= frequency_bounds[i+1])
    features.append([bounds_names[i], np.sum(pgram[idx]) / pgram[idx].shape[0] * (frequency_bounds[i+1] - frequency_bounds[i])])
    features.append([bounds_names[i] + 'rel', np.sum(pgram[idx]) / power])
    features.append([bounds_names[i] + 'peak', f[idx][np.argmax(pgram[idx])]])
    if i > 1:
      features.append([bounds_names[i] + 'normalized', np.sum(pgram[idx]) / normalized_power])

  features.append(['HF/LF', pgram[3] / pgram[2]])

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

