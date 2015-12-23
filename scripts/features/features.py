from collections import OrderedDict


def get_default_features_params():
  """
  Return default features parameters
  """
  options = {'step' : [5, 20], 
             'autocorr_step':[5, 20],
             'bounds':[20, 50],
             'use_sleep_time':True,
             'interval_length':5, # in minutes
             'frequency_interval_length':None, # in minutes
             'include_time':True,
             'frequency_bounds':[0, 0.0033, 0.04, 0.15, 0.5],
             'calculate_time_features':True,
             'calculate_frequency_features':True,
             'calculate_nonlinear_features':False}
  return OrderedDict(options)



if __name__ == '__main__':
  pass