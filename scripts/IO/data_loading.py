import numpy as np
import csv
import os
import logging


def get_sleep_time(path_to_sleep):
  ''' Read sleep and awake time '''
  with open(path_to_sleep + 'sleep.csv', 'r') as csvfile:
    r = csv.reader(csvfile, delimiter=',')
    r.next()
    start = {}
    end = {}
    for row in r:
      if not ((row[7] == '') and (row[8] == '')):
        patient = int(row[1])
        start[patient] = float(row[7])
        end[patient] = float(row[8])
        # new day
        if end[patient] < start:
          end[patient] += 24 * 60 * 60 * 1000
  return start, end

def get_patients_list(path_to_patients):
  ''' Return list consist of files with patients data '''
  files = os.listdir(path_to_patients)
  files = np.sort(np.array([f[:-3] for f in files if f[-3:] == '.RR'], dtype=np.int))
  return files

def load_sample(path_to_sample, name):
  ''' Load sample from path_to_sample/name '''
  if os.path.exists(path_to_sample + name):
    with open(path_to_sample + name, 'r') as csvfile:
      r = csv.reader(csvfile, delimiter=',')
      data = []
      patients = []
      for row in r:
        patients.append(row[0])
        data.append(row[1:])
    return np.array(patients, dtype=np.int), np.array(data, dtype=np.float64)
  else:
    logging.error('File ' + path_to_sample + name + ' doesn\'t exist')
    return None, None

def save_sample(path_to_sample, name, patients, data):
  ''' Save sample into path_to_sample/name '''
  with open(path_to_sample + name, 'w') as csvfile:
    w = csv.writer(csvfile, delimiter=',')
    for i in xrange(min(patients.shape[0], data.shape[0])):
      row = [patients[i]]
      for f in data[i, :]:
        row.append(f)
      w.writerow(row)

def read_patient(path_to_patients, patient):
  ''' Read intervals for patient. Choose intervals with types in types list ''' 
  with open(path_to_patients + str(patient) + '.RR', 'r') as csvfile:
    r = csv.reader(csvfile, delimiter='\t')
    start_time_s = r.next()[0]
    start_time = float(start_time_s[:2]) * 60 * 60 * 1000 + float(start_time_s[3:5]) * 60 * 1000 + float(start_time_s[6:8]) * 1000
    intervals = []
    time = []
    types = []
    for row in r:
      time.append(row[0])
      intervals.append(row[1])
      types.append(row[2])
    intervals = np.array(intervals, dtype=np.float64)
    time = start_time + np.array(time, dtype=np.float64)
    types = np.array(types)
  return time, intervals, types


if __name__ == '__main__':
  # choose stream for logging. Should contain 'std' value or a log file name
  log_stream = 'std'
  log_level = logging.INFO
  if log_stream != 'std':
    logging.basicConfig(level=log_level, filename=log_stream)
  else:
    logging.basicConfig(level=log_level)

  options = None#{'step' : [5, 20], 
             #'autocorr_step' : [5, 20]}    

  path_to_sleep = '../../data/dta/'
  path_to_output = '../../data/dta/'
  output_file_name = 'mortality_SAHR_ver101214.dta'
  #output_file_name = 'selected_pp.dta'
  path_to_patients = '../../data/RR/'
  
  patients_list = get_patients_list(path_to_patients)
  print 'Patients list:', patients_list
  start, end = get_sleep_time(path_to_sleep)
  print '\nSleep start:', start
  print 'Sleep end:', end
  
  patient = patients_list[0]
  print '\nPatient', patient
  print read_patient(path_to_patients, patient)

  path_to_sample='../../data/sample/'
  sample_name='temp_sample.csv'
  #print 'Saving...'
  #save_sample(path_to_sample, sample_name, patients, data)
  #print '\nLoading...'
  #print load_sample(path_to_sample, sample_name)
  