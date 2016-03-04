import numpy as np
import csv
import os
import logging
import pandas as pd
import h5py
import pickle

import sys
sys.path.append('..') #!!!
import project_config as conf
from utils import logg

#TODO
'''
.RR files contain information about pulse: (time [ms], duration [ms], beat type) #!!!
GIDN is the patient identifier, [int.32]
'''


def get_sleep_time(path_to_sleep, filename='sleep', version=1):
  ''' Read sleep and awake time
      Version 1: returns dict 'start', 'end' with start and end times for all gidns
      Version 2: returns same pandas data frame as in sleep.dta file
      with columns 'sleep_trend_start_corr' and 'sleep_trend_end_corr' added.
  '''
  sleep = read_dta('sleep', data_folder=path_to_sleep)

  GIDNs = sleep['GIDN']

  if version == 1:
      sleep_time = {'start':{}, 'end':{}}
  elif version > 1:
      sleep['sleep_trend_start_corr'] = np.nan
      sleep['sleep_trend_end_corr'] = np.nan
      sleep['sleep_diary_start_corr'] = np.nan
      sleep['sleep_diary_end_corr'] = np.nan
      start_times = []
      end_times = []
      start_diary_times = []
      end_diary_times = []

  def do_correction(start, end):
    if start / (60 * 60 * 1000) < 18:  # new day correction
        start += 24 * 60 * 60 * 1000
    if end < start:
        end += 24 * 60 * 60 * 1000
    return start, end

  for i, gidn in enumerate(GIDNs):
    start = sleep['Hol_Sleep_Start_Trend'][i]
    end = sleep['Hol_Sleep_End_Trend'][i]

    start_diary = sleep['Hol_Sleep_Start_Diary'][i]
    end_diary = sleep['Hol_Sleep_End_Diary'][i]

    start, end = do_correction(start, end)
    start_diary, end_diary = do_correction(start_diary, end_diary)

    if (not np.isnan(start)) and (not np.isnan(end)):
        if version == 1:
          sleep_time['start'][gidn] = float(start)
          sleep_time['end'][gidn] = float(end)
        if version > 1:
            start_times.append(start)
            end_times.append(end)
    else:
        if version > 1:
            start_times.append(np.nan)
            end_times.append(np.nan)

    if version > 1:
        if (not np.isnan(start_diary)) and (not np.isnan(end_diary)):
            start_diary_times.append(start_diary)
            end_diary_times.append(end_diary)
        else:
            start_diary_times.append(np.nan)
            end_diary_times.append(np.nan)

  if version == 1:
      return sleep_time
  elif version > 1:
      sleep['sleep_trend_start_corr'] = start_times
      sleep['sleep_trend_end_corr'] = end_times

      sleep['sleep_diary_start_corr'] = start_diary_times
      sleep['sleep_diary_end_corr'] = end_diary_times
      return sleep

def get_patients_list_in_folder(path_to_patients):
  ''' Return list that consists of files with patients data '''
  files = os.listdir(path_to_patients)
  files = np.sort(np.array([f[:-3] for f in files if f[-3:] == '.RR'], dtype=np.int))
  return files


def load_RR_data(GIDN, path, version=1):
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
            if version == 1:
                data_RR.columns = ['time', 'interval', 'type']
            elif version > 1:
                data_RR.columns = ['time', 'interval', 'beat_type']

        if version > 1:
            data_RR['interval_type'] = [None] + list(data_RR['beat_type'][:-1].as_matrix()+data_RR['beat_type'][1:].as_matrix())

        data_RR['time'] += delta     # add initial time shifting
        data_RR.drop(data_RR.index[:1], inplace=True) # first row data is invalid, e.g. '10350 10350 N'

        if version == 1:
            data_RR = data_RR.values

    except IOError as e:
      data_RR = None
      logging.warning(u'Loading error for RR file: %s'%e)

    # except Exception as e:
    #   raise Exception("Unexpected error: %s"%e)

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


def save_hdf5_sample(sample_name, sample_info, trainX, trainY, testX, testY):
  hdf5_filename = conf.path_to_sample+sample_name+'.h5'
  h5f = h5py.File(hdf5_filename, 'w')
  h5f.create_dataset('trainX', data=trainX)
  h5f.create_dataset('trainY', data=trainY)
  h5f.create_dataset('testX', data=testX)
  h5f.create_dataset('testY', data=testY)
  h5f.close()

  picke_filename = conf.path_to_sample+sample_name+'.pkl'
  with open(picke_filename, 'wb') as f:
    pickle.dump(sample_info, f)

  return hdf5_filename

def load_hdf5_sample(sample_name):
  hdf5_filename = conf.path_to_sample+sample_name+'.h5'

  h5f = h5py.File(hdf5_filename,'r')
  trainX = h5f['trainX'][:]
  trainY = h5f['trainY'][:]
  testX = h5f['testX'][:]
  testY = h5f['testY'][:]
  h5f.close()

  picke_filename = conf.path_to_sample+sample_name+'.pkl'
  with open(picke_filename, 'rb') as f:
    sample_info = pickle.load(f)
  sample_info['path'] = hdf5_filename

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
