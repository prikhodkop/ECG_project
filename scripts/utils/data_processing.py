import numpy as np
import logging


def split_time_chunks(filtered_data_RR, CHUNK_TIME_INTERVAL, MIN_NUMBER_OF_BEATS_IN_CHUNK, 
        MAX_NUMBER_OF_CHUNKS_PER_PATIENT):
  """
  Split data RR into time chunks.

  Args:
    filtered_data_RR (np.array): data in format (time since midnight [ms], intervals [ms])
    CHUNK_TIME_INTERVAL (float): time duration of the chunk [minutes] 
    MIN_NUMBER_OF_BEATS_IN_CHUNK (int or None):
    MAX_NUMBER_OF_CHUNKS_PER_PATIENT (int or None): set number of first chunks to use

  Returns:
    splitted_data_RR (list of np.array or None): np.array of np.int64 in format (time since midnight [ms], intervals [ms]), or
                                                 None if no data are available after processing
  """

  if MAX_NUMBER_OF_CHUNKS_PER_PATIENT is not None:
    msg = 'Not implemented' #TODO
    logging.critical(msg)
    raise Exception(msg)

  if MIN_NUMBER_OF_BEATS_IN_CHUNK is None:
    MIN_NUMBER_OF_BEATS_IN_CHUNK = 1

  fixed_duration = CHUNK_TIME_INTERVAL * 60.0 * 1000 # in milliseconds
  
  start_time = filtered_data_RR[0, 0]   # in milliseconds
  finish_time = filtered_data_RR[-1, 0] # in milliseconds
  chunks_number = int( (finish_time - start_time) / fixed_duration )

  splitted_data_RR = [[] for i in xrange(chunks_number)]
  for cur_time, cur_interval in filtered_data_RR:
    cur_chunk_number = int( (cur_time - start_time) / fixed_duration )
    if cur_chunk_number >= chunks_number:
      break

    splitted_data_RR[cur_chunk_number].append([cur_time, cur_interval])
  
  splitted_data_RR = [np.array(sequence) for sequence in splitted_data_RR 
                        if len(sequence) >= MIN_NUMBER_OF_BEATS_IN_CHUNK]
  
  if len(splitted_data_RR) == 0:
    return None

  return splitted_data_RR


def split_beats_chunks(filtered_data_RR, NUMBER_OF_BEATS_IN_CHUNK, MAX_NUMBER_OF_CHUNKS_PER_PATIENT):
  msg = 'Not implemented' #TODO
  logging.critical(msg)
  raise Exception(msg)


def time_initialization(splitted_data_RR):
  """
  Set start time for each chunk equal zero.
  Data are not copied.

  Args:
    splitted_data_RR (list of np.array): np.array of np.int64 in format (time since midnight [ms], interval [ms])

  Returns:
    splitted_data_RR (list of np.array): np.array of np.int64 in format (time [ms], interval [ms])
    initial_times (list of int): starting time in each chunk before initialization
  """
  initial_times = []
  for data_RR in splitted_data_RR:
    initial_times.append(data_RR[0, 0])
    data_RR[:, 0] -= data_RR[0, 0]

  return splitted_data_RR, initial_times


def split_GIDNS_for_test(GIDNS, TEST_PARTITION, TEST_PORTION):
  """
  Separate train and test GIDNS

  Args:
    GIDNS (list of int): patients for separation
    TEST_PARTITION (str): partition method
    TEST_PORTION (float): portion of GIDNS for test

  Returns:
    train_GIDNS (list of int)
    test_GIDNS (list of int)
    info (dict)
  """

  test_number = int( TEST_PORTION * float(len(GIDNS)) )
  if test_number < 1:
    msg = 'No data for testing'
    logging.error(msg)
    raise Exception(msg)
  elif test_number <= 10:
    msg = 'Only %s patients are used for testing'%test_number
    logging.warning(msg)


  if TEST_PARTITION == 'random':
    test_GIDNS = np.random.choice(GIDNS, size=test_number, replace=False)
    train_GIDNS = [GIDN for GIDN in GIDNS if GIDN not in test_GIDNS]
     
  else:
    msg = 'Not implemented' #TODO
    logging.critical(msg)
    raise Exception(msg)

  info = {'test GIDNS':len(test_GIDNS), 'train GIDNS':len(train_GIDNS)}
  return train_GIDNS, test_GIDNS, info


if __name__ == '__main__':
  pass

