import numpy as np
import logging
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

try:
  import user_project_config as conf
except:
  import project_config as conf


from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc

def analyse_results(clf, X_test, y_test, show_plots=False):

  y_pred = clf.predict(X_test)
  clf_report = classification_report(y_test, y_pred)
  logging.info('\n'+clf_report)
  acc_score = accuracy_score(y_test, y_pred)
  logging.info('The accuracy is: %s'%acc_score)

  y_pred_probas = clf.predict_proba(X_test[:11])
  #print y_pred_probas.shape
  logging.info('Example of predicted P (11 items):\n%s'%y_pred_probas)
  #zxc
  
  classes = clf.classes_
  logging.info('Objective classes:\n%s'%classes)
  cm = confusion_matrix(y_test, y_pred)
  logging.info('Confusion matrix:\n%s'%cm)
  
  cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  logging.info('Confusion matrix normalized by true classes:\n%s'%cm_normalized)
  
  if 'feature_importances_' in dir(clf):
    importances = clf.feature_importances_
    logging.info('Features importance:\n%s'%importances)

  if show_plots:
    plt.figure(figsize=(25, 25))
    plt.matshow(cm, aspect='auto')
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def analyse_results_precision_recall(clf, X_test, y_test, y_train=None, 
                               prec_classes=[1], figsize=(25, 25)):
    
  y_pred_probas = clf.predict_proba(X_test)

  classes = clf.classes_.tolist()
  sub_size = (len(prec_classes)/3+1, 3)
  figs, axes = plt.subplots(sub_size[0], sub_size[1], figsize=figsize)
  precision = dict()
  recall = dict()
  rand_precision = dict()
  rand_recall = dict()
  average_precision = dict()

  for class_num, prec_class in enumerate(prec_classes):
    ind_class = classes.index(prec_class)

    precision[class_num], recall[class_num], _ = precision_recall_curve(y_test==prec_class, 
                                                y_pred_probas[:, ind_class]) #!!!
    area = auc(recall[class_num], precision[class_num])
    print "\nArea Under Curve for class %s"%prec_class
    print "target classifier: %0.3f" % area  

    rand_precision[class_num], rand_recall[class_num], _ = precision_recall_curve(y_test==prec_class, 
         np.mean(y_train==prec_class) * np.ones((len(y_test),))) #!!!
    rand_area = auc(rand_recall[class_num], rand_precision[class_num])
    print "random classifier: %0.3f" % rand_area

    i = class_num / sub_size[1]
    j = (class_num) % sub_size[1]

    # Plot Precision-Recall curve
    axes[i, j].plot(recall[class_num], precision[class_num], 
        label='Target classifier for class %s'%prec_class)
    axes[i, j].plot(rand_recall[class_num], rand_precision[class_num], 
        label='Random for class %s'%prec_class, linestyle='--')
    axes[i, j].set_xlabel('Recall')
    axes[i, j].set_ylabel('Precision')
    axes[i, j].set_ylim([0.0, 1.05])
    axes[i, j].set_xlim([0.0, 1.0])
    axes[i, j].set_title('Precision-Recall example: AUC={0:0.3f}'.format(area))
    axes[i, j].legend(loc="best")
  plt.show()



if __name__ == '__main__':
  pass