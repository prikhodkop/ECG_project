import numpy as np
import csv
import os
import logging
import pandas as pd
import h5py

import sys
sys.path.append('..') #!!!
import project_config as conf
from utils import logg 

#TODO
'''
.RR files contain information about pulse: (time [ms], duration [ms], beat type) #!!!
GIDN is the patient identifier, [int.32]
'''


def get_sleep_time(path_to_sleep):
  ''' Read sleep and awake time '''
  sleep = read_dta('sleep', data_folder=conf.path_to_dta)
  GIDNs = sleep['GIDN']
  start = sleep['Hol_Sleep_Start_Trend']
  end = sleep['Hol_Sleep_End_Trend']
  sleep_time = {'start':{}, 'end':{}}

  for i, gidn in enumerate(GIDNs):
    if (start[i] == '') or (end[i] == ''):
      start[i] = 0
      end[i] = 0

    sleep_time['start'][gidn] = float(start[i])
    sleep_time['end'][gidn] = float(end[i])

    # new day
    if sleep_time['start'][gidn] / (60 * 60 * 1000) < 18:
      sleep_time['start'][gidn] += 24 * 60 * 60 * 1000

    if sleep_time['end'][gidn] < sleep_time['start'][gidn]:
      sleep_time['end'][gidn] += 24 * 60 * 60 * 1000
    print sleep_time['start'][gidn], sleep_time['end'][gidn]
  return sleep_time

def get_patients_list_in_folder(path_to_patients):
  ''' Return list that consists of files with patients data '''
  files = os.listdir(path_to_patients)
  files = np.sort(np.array([f[:-3] for f in files if f[-3:] == '.RR'], dtype=np.int))
  return files


def load_RR_data(GIDN, path):
    """
    Get loaded RR data for the given patient.

    Args:
        GIDN (int32): the patient identifier
        path (str): path to data files containing pulse information (.RR)
    Returns:
        - None if .RR file does not exist or
        - np.array: (time since midnight [ms], beat-beat interval [ms], interval type)
          * time of pulse beat [ms], starting from midnight;
          * interval between successive beats [ms];
          * interval type ['N', 'A', etc.]. Note that 'N' is a normal type and preferable.
    
    The output array length varies among patients.    
    """
    
    RR_file_name = path + str(GIDN)+'.RR'
    logging.debug(u'Loading RR data for patient %s'%GIDN)
    try:
        with open(RR_file_name) as csvfile: 
            reader = csv.reader(csvfile, delimiter='\t')
            start_diary = next(reader)[0]  # e.g. 12:08:58
            #print 'start_diary',start_diary
            logging.debug(u'Pulse diary start: %s'%start_diary)

            h,m,s = [int(i) for i in start_diary.split(':')]
            delta = 1000*(h*60*60 + m*60 + s)
            logging.debug(u'Initial time shift is %s ms'%delta)
        
        with open(RR_file_name) as csvfile: 
            data_RR = pd.read_csv(csvfile, sep='\t', skiprows=1, header=None) # skiprows=1 !!!
            data_RR.drop(data_RR.index[:1], inplace=True) # first row data is invalid, e.g. '10350 10350 N'
            data_RR.columns = ['time', 'interval', 'type']

        data_RR = data_RR.values
        data_RR[:, 0] += delta     # add initial time shifting

    except IOError as e:
      data_RR = None
      logging.warning(u'Loading error for RR file: %s'%e)

    except Exception as e:
      raise Exception("Unexpected error: %s"%e)
    
    return data_RR

def read_dta(name, data_folder='../../data/dta/', encoding='cp1252'):
  """
  Read .dta file
  Examples:
    selected_pp
    mortality_SAHR_ver101214
  """
  full_name = data_folder + name +'.dta'
  
  with open(full_name, 'rb') as f:
    data = pd.read_stata(f, encoding=encoding)
  
  return data

def get_GIDNS(path_to_dta):
  """
  Get GIDN identifiers for all patients sorted in ascending order. Based on selected_pp. 
  Output: GIDNS [np.array] of GIDN [int.32]
  """
  selected_pp = read_dta('selected_pp', data_folder=path_to_dta)
  GIDNS = np.array(selected_pp['GIDN'])
  GIDNS = np.array(GIDNS).astype(np.int32)
  return GIDNS


def save_hdf5_sample(sample_name, trainX, trainY, testX, testY):
  hdf5_filename = conf.path_to_sample+sample_name+'.h5'
  h5f = h5py.File(hdf5_filename, 'w')
  h5f.create_dataset('trainX', data=trainX)
  h5f.create_dataset('trainY', data=trainY)
  h5f.create_dataset('testX', data=testX)
  h5f.create_dataset('testY', data=testY)
  h5f.close()
  return hdf5_filename

def load_hdf5_sample(sample_name):
  hdf5_filename = conf.path_to_sample+sample_name+'.h5'
  
  h5f = h5py.File(hdf5_filename,'r')
  trainX = h5f['trainX'][:]
  trainY = h5f['trainY'][:]
  testX = h5f['testX'][:]
  testY = h5f['testY'][:]

  sample_info = {'path': hdf5_filename}
  return trainX, trainY, testX, testY, sample_info


if __name__ == '__main__':
  logg.configure_logging(console_level=logging.DEBUG)

  mortality = read_dta('mortality_SAHR_ver101214', data_folder=conf.path_to_dta)
  selected_pp = read_dta('selected_pp', data_folder=conf.path_to_dta)
  GIDNS = get_GIDNS(path_to_dta=conf.path_to_dta)

  print GIDNS
  print len(GIDNS), 'patients'
  
  GIDN = GIDNS[0]
  data_RR = load_RR_data(GIDN, path=conf.path_to_RR)
  print '\ndata_RR:\n', data_RR, '\n'

  data_RR = load_RR_data(GIDN=1234567890, path=conf.path_to_RR)
  print 'data_RR:', data_RR
  