import urllib
import urllib2
import csv
import pandas as pd
import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pylab as pl
from matplotlib import cm



def prepare_dataset(url):
  req = urllib2.Request(url)
  response = urllib2.urlopen(req)
  the_page = response.read()
  string = StringIO(the_page)
  df = pd.read_csv(string, sep=",", header=None)
  train, test = train_test_split(df, test_size = 0.2)
  X = train.iloc[:, :-1]
  print X.head()
  y = train.iloc[:, -1]
  print y
  X_test = test.iloc[:, :-1]
  print X_test.head()
  y_test = test.iloc[:, -1]
  print y_test

  return X, y, X_test, y_test



def makePrediction(model, X, y, X_test, y_test):
  model.fit(X, y)
  expected = y_test
  predicted = model.predict(X_test)
  classification_report = metrics.classification_report(expected, predicted)
  confusion_matrix = metrics.confusion_matrix(expected, predicted)
  confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
  print classification_report
  return confusion_matrix


def init_subplots(ax, set_ticks=True):
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.tick_params(bottom='off', top='off', left='off', right='off')
  if set_ticks:
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])


def plot_matrix(categories, cm1, cm2, cm3, cm4, cm5):
  f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(12,7))

  cax1 = ax1.matshow(cm1, cmap=plt.cm.Blues)
  cax2 = ax2.matshow(cm2, cmap=plt.cm.Blues)
  cax3 = ax3.matshow(cm3, cmap=plt.cm.Blues)
  cax4 = ax4.matshow(cm4, cmap=plt.cm.Blues)
  cax5 = ax5.matshow(cm5, cmap=plt.cm.Blues)

  init_subplots(ax1, False)
  init_subplots(ax2)
  init_subplots(ax3)
  init_subplots(ax4)
  init_subplots(ax5, False)

  labels = categories
  ax1.set_xticklabels([''] + labels, rotation='vertical')
  ax1.set_yticklabels([''] + labels)

  ax5.set_xticklabels([''])
  ax5.set_yticklabels([''])
  ax5.yaxis.set_label_position("right")
  ax5.set_title('Predicted')
  ax5.set_ylabel('True')

  ax1.set_xlabel('DecisionTree')
  ax2.set_xlabel('SVC')
  ax3.set_xlabel('LogisticRegression')
  ax4.set_xlabel('GaussianNB')
  ax5.set_xlabel('KNN')

  f.subplots_adjust(right=0.8)
  # [left, bottom, width, height]
  cbar_ax = f.add_axes([0.20, 0.30, 0.5, 0.025])
  f.colorbar(cax5, cax=cbar_ax, orientation='horizontal')

  pl.title('Confusion matrix for each algorithm')

  pl.show()

def compare_logistic_model(df, categories):
  X, y, X_test, y_test = prepare_dataset(url)

  model_DecisionTree = DecisionTreeClassifier()
  cm1 = makePrediction(model_DecisionTree, X, y, X_test, y_test)

  model_SVC = SVC()
  cm2 = makePrediction(model_SVC, X, y, X_test, y_test)

  model_LogisticRegression = LogisticRegression()
  cm3 = makePrediction(model_LogisticRegression, X, y, X_test, y_test)

  model_GaussianNB = GaussianNB()
  cm4 = makePrediction(model_GaussianNB, X, y, X_test, y_test)

  model_KNN = KNeighborsClassifier()
  cm5 = makePrediction(model_KNN, X, y, X_test, y_test)

  plot_matrix(categories, cm1, cm2, cm3, cm4, cm5)






