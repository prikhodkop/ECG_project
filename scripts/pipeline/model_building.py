import numpy as np
import logging
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

try:
  import user_project_config as conf
except:
  import project_config as conf

from IO import data_loading as dl
from utils import logg 
from utils import data_processing as dp
from models_utils import models_utils as mu


from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier




if __name__ == '__main__':
  
  ###########################################################
  # Settings

  USED_EXAMPLES_NUMBER = None # 'None' means that all examples are used; otherwise randomly selected
  
  EXCLUDE_NANS = True
  OBJECTIVE_NAME = 'cl_sleep_interval' # e.g. 'BMIgr', 'Sex', 'cl_sleep_interval'
  sample_name = OBJECTIVE_NAME + '_3' # train-test filename
  SEED = 0


  classifiers = [
      ("Dummy", DummyClassifier(strategy='stratified')), # see http://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
      # ("Linear SVM", SVC(kernel="linear", C=0.025)),
      # ("RBF SVM", SVC(gamma=2, C=1)),
      ("Decision Tree", DecisionTreeClassifier(max_depth=5)),
      ("Random Forest", RandomForestClassifier(n_estimators=100)),
      ("Nearest Neighbors", KNeighborsClassifier(3)),
      # ("AdaBoost", AdaBoostClassifier()),
      ("Naive Bayes", GaussianNB())
      ] # TODO: xgboost

  ###############################################################
  # Initial configuration
  np.random.seed(SEED)
  logg.configure_logging() # For more details use logg.configure_logging(console_level=logging.DEBUG)

  ################################################################
  # Prepare train and test samples
  trainX, trainY, testX, testY, sample_info = dl.load_hdf5_sample(sample_name)
  sample_info['Features names'] = np.array(sample_info['Features names'])
  logging.info('Training and test samples are loaded from file %s'%sample_info['path'])

  if EXCLUDE_NANS:
    logging.info('Exclude NaNs from train and test features')
    nans_counter = np.sum(np.isnan(trainX[:, :]), axis=0) + np.sum(np.isnan(testX[:, :]), axis=0)
    features_idx = nans_counter == 0
    
    if np.any(~features_idx):
      excluded_features = sample_info['Features names'][~features_idx]
      logging.warning('Some features were excluded because of NaNs: %s'%excluded_features)
    
    trainX = trainX[:, features_idx]
    testX = testX[:, features_idx]

  if USED_EXAMPLES_NUMBER is not None:
    logging.warning('Only %s examples is used for model learning!'%USED_EXAMPLES_NUMBER)
    idx = np.random.choice(range(trainX.shape[0]), size=USED_EXAMPLES_NUMBER, replace=False)
    trainX = trainX[idx]
    trainY = trainY[idx]

  dp.training_stats(trainX, trainY, testX, testY)
  logging.info('Sample info:\n%s'%sample_info)
  logging.info('Standartization of training and test samples.')
  scaler = preprocessing.StandardScaler().fit(trainX)
  trainX = scaler.transform(trainX) 
  testX = scaler.transform(testX) 
                      
  clfs = {}
  if trainY.shape[1] == 1:
    trainY = trainY.ravel()
    testY = testY.ravel()

  for name, clf in classifiers:
    logging.info('Algorithm: ====================== %s =========================='%name)
    logging.info('Features names: %s'%sample_info['Features names'][features_idx])
    clf.fit(trainX, trainY)
    mu.analyse_results(clf, testX, testY)
    logging.info('Objective classes names: %s'%sample_info['Objective classes names']) 
    clfs[name] = clf
  
  models_name = conf.path_to_models + 'clfs_' + sample_name + '.pickle'
  import cPickle as pickle
  with open(models_name, 'wb') as f:
    pickle.dump(clfs, f)

