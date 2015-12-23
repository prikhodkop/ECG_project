from collections import OrderedDict
import numpy as np
import pandas as pd
import logging 

def get_default_RR_filtering_params():
  """
  Returns OrderedDict of filters parameters. The order is important.
  Value 'None' means no actions.
  """
  filtering_params = OrderedDict(( ('beat type', 'N'),
                                   ('interval range', [200.0, 2000.0]), 
                                   ('successive intervals ration range', None) # e.g. [0.6, 1.4]
                                ))
  return filtering_params


def filter_data_RR(data_RR, RR_filtering_params):
  """
  Filter pulse intervals data.

  Args:
    data_RR (np.array): data in format (time since midnight [ms], interval [ms], beat type)
    RR_filtering_params (OrderedDict): see example in get_default_RR_filtering_params()

  Returns:
    filtered_data_RR (np.array or None): data in format (time since midnight [ms], intervals [ms]), or
                                         None if no data are available after filtering
    filtration_info (???):  #TODO 

  """
  filtration_info = {'initial size': len(data_RR)}

  for filter_name in RR_filtering_params:

    params_filter = RR_filtering_params[filter_name]
    if params_filter is not None:

      if filter_name == 'beat type':
        permitted_beat_type = RR_filtering_params[filter_name]
        data_RR = data_RR[data_RR[:, -1] == permitted_beat_type] #!!!
        zxx

      elif filter_name == 'interval range':
        pass
      elif filter_name == 'successive intervals ration range':
        pass
      else:
        msg = 'An unknown filter name: %s'%filter_name
        logging.critical(msg)
        raise Exception(msg)
  
  filtered_data_RR = data_RR #!!! [:, :-1]
  filtration_info['final size'] = len(filtered_data_RR)
  return filtered_data_RR, filtration_info



def filter_interval_range():
  pass

def filter_successive_intervals_ration():
  pass




if __name__ == '__main__':
  def_params = get_default_RR_filtering_params()

  params = OrderedDict(( ('beat type', 'N'),
                         ('interval range', [None, 2000.0]), 
                         ('successive intervals ration range', [None, None])
                      ))

  params_empty = OrderedDict(( ('beat type', None),
                               ('interval range', [5000.0, 4000.0]), 
                               ('successive intervals ration range', [None, None])
                            ))
  
  # 'Simulate' data_RR format
  data_RR = pd.DataFrame([ [43748861,  11, 'N'],
                           [43749368,  507, 'N'],
                           [43749879,  111, 'A'],
                           [122322922, 3523, 'N'],
                           [122323448, 526, 'N'],
                           [122323983, 535, 'A']
                         ]).as_matrix()

  print filter_data_RR(data_RR, params)


