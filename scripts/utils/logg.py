import logging
import os

def configure_logging(output_dir=''):
  # Levels: debug, info, warning, error, critical

  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)
   
  # create console handler and set level to info
  handler = logging.StreamHandler()
  handler.setLevel(logging.DEBUG)
  formatter = logging.Formatter("%(levelname)s - %(message)s")
  handler.setFormatter(formatter)
  logger.addHandler(handler)

  # create error file handler and set level to error
  handler = logging.FileHandler(os.path.join(output_dir, "info.log"),"w", encoding=None, delay="true")
  handler.setLevel(logging.INFO)
  formatter = logging.Formatter(u"%(filename)s [%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s")
  handler.setFormatter(formatter)
  logger.addHandler(handler)

  # create debug file handler and set level to debug
  handler = logging.FileHandler(os.path.join(output_dir, "debug.log"),"w")
  handler.setLevel(logging.DEBUG)
  formatter = logging.Formatter(u"%(filename)s [%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s")
  handler.setFormatter(formatter)
  logger.addHandler(handler)

if __name__ == '__main__':
  pass