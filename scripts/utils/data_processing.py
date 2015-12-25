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



if __name__ == '__main__':
  pass
