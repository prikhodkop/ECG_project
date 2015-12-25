import logging
import os
import time

def configure_logging(console_level=logging.INFO, output_dir='../logs/'):
  # Levels: debug, info, warning, error, critical

  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)
   
  # create console handler and set level to info
  handler = logging.StreamHandler()
  
  '''
  if console_level == 'debug':
    logging_level = logging.DEBUG
  elif console_level == 'info':
    logging_level = logging.INFO
  elif console_level == 'warning':
    logging_level = logging.WARNING
  else:
    raise Exception('An unknown stream logging level')
  handler.setLevel(logging_level)
  '''

  handler.setLevel(console_level)
  #formatter = logging.Formatter("%(levelname)s - %(message)s")
  formatter = logging.Formatter("%(levelname)-8s [%(asctime)s] %(message)s")
  handler.setFormatter(formatter)
  logger.addHandler(handler)

  current_datetime = time.strftime("%Y-%m-%d %H-%M-%S")
  # create error file handler and set level to error

  handler = logging.FileHandler(os.path.join(output_dir, current_datetime+" warning.log"),"w", encoding=None, delay="true")
  handler.setLevel(logging.WARNING)
  formatter = logging.Formatter(u"%(filename)s [%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s")
  handler.setFormatter(formatter)
  logger.addHandler(handler)


  handler = logging.FileHandler(os.path.join(output_dir, current_datetime+" info.log"),"w", encoding=None, delay="true")
  handler.setLevel(logging.INFO)
  formatter = logging.Formatter(u"%(filename)s [%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s")
  handler.setFormatter(formatter)
  logger.addHandler(handler)

  # create debug file handler and set level to debug
  handler = logging.FileHandler(os.path.join(output_dir, current_datetime+" debug.log"),"w")
  handler.setLevel(logging.DEBUG)
  formatter = logging.Formatter(u"%(filename)s [%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s")
  handler.setFormatter(formatter)
  logger.addHandler(handler)


if __name__ == '__main__':
  configure_logging()
