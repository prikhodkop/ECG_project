import logging

try:
  import psutil
  sys_utils = True
except:
  sys_utils = False



def check_memory(memory_threshold=99.0, verbose=False):
  if sys_utils:
    mem_per = psutil.phymem_usage().percent
    msg = "%s%% of memory is used"%mem_per

    if mem_per >= memory_threshold:
      logging.warning(msg)
    else:
      if verbose:
        logging.info(msg)

if __name__ == '__main__':
  pass