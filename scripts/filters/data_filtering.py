from collections import OrderedDict
import numpy as np
import pandas as pd
import logging 

def get_default_RR_filtering_params():
  """
  Returns OrderedDict of filters parameters. The order is important.
  Value 'None' means no actions.
  """
  filtering_params = OrderedDict(( ('interval type', 'N'),                 # e.g. None 
                                   ('interval range', [200.0, 2000.0]),    # min_interval, max_interval
                                   ('successive intervals ration range', None) 
                                ))


  return filtering_params


def filter_data_RR(data_RR, RR_filtering_params):
  """
  Filter pulse intervals data.

  Args:
    data_RR (np.array): data in format (time since midnight [ms], beat-beat interval [ms], interval type)
    RR_filtering_params (OrderedDict): see example in get_default_RR_filtering_params()

  Returns:
    filtered_data_RR (np.array or None): data in format (time since midnight [ms], intervals [ms]), or
                                         None if no data are available after filtering
    filtration_info (dict): initial and remaining intervals number
  """
  filtration_info = {'initial size': len(data_RR)}

  for filter_name in RR_filtering_params:

    if RR_filtering_params[filter_name] is not None:

      if filter_name == 'interval type':
        permitted_beat_type = RR_filtering_params[filter_name]
        data_RR = data_RR[data_RR[:, -1] == permitted_beat_type]

      elif filter_name == 'interval range':
        min_interval, max_interval = RR_filtering_params[filter_name]

        if min_interval is not None:
          data_RR = data_RR[data_RR[:, 1] >= min_interval]
        if max_interval is not None:
          data_RR = data_RR[data_RR[:, 1] <= max_interval]
      
      elif filter_name == 'successive intervals ration range':
        min_ratio, max_ratio = RR_filtering_params[filter_name]
        
        #TODO
        if min_ratio is not None or max_ratio is not None:  
          msg = 'Filtration based on successive intervals ration range is not implemented.'
          logging.critical(msg)
          raise Exception(msg)

      
      else:
        msg = 'An unknown filter name: %s'%filter_name
        logging.critical(msg)
        raise Exception(msg)
      
    if data_RR.size == 0:
      filtration_info['final size'] = 0
      return None, filtration_info

  
  filtered_data_RR = data_RR[:, :-1] # exclude interval type
  filtration_info['final size'] = len(filtered_data_RR)
  return filtered_data_RR, filtration_info





if __name__ == '__main__':
  
  # 'Simulate' data_RR format
  data_RR = pd.DataFrame([ [43748861,  11, 'N'],
                           [43749368,  507, 'N'],
                           [43749879,  111, 'A'],
                           [122322922, 3523, 'N'],
                           [122323448, 526, 'N'],
                           [122323983, 535, 'A']
                         ]).as_matrix()

  print 'Initial:\n', data_RR

  def_params = get_default_RR_filtering_params()
  print '\ndef_params', def_params
  
  data_RR_filtered, filter_info = filter_data_RR(data_RR, def_params)
  print '\nFiltered for default configuration:\n', data_RR_filtered
  

  params_empty = OrderedDict(( ('interval type', None),
                               ('interval range', [4000, None]), 
                               ('successive intervals ration range', [None, None])
                            ))
  data_RR_filtered, filter_info = filter_data_RR(data_RR, params_empty)
  print '\nFiltered for empty configuration:\n', data_RR_filtered


