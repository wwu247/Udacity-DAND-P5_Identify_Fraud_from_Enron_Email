#!/usr/bin/python

import sys
import pickle
import numpy
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


#########################################################
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'deferred_income', 'long_term_incentive', 'restricted_stock', 'total_payments', 'shared_receipt_with_poi', 'loan_advances']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


#########################################################
### Task 2: Remove outliers
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
data_dict.pop('TOTAL')
data_dict.pop('LOCKHART EUGENE E')


#########################################################
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

min_max_scaler = preprocessing.MinMaxScaler()
features_minmax = min_max_scaler.fit_transform(features)

selector = SelectKBest(score_func=f_classif, k=10)
features_transformed = selector.fit_transform(features_minmax, labels)

from copy import deepcopy
# make a copy of features_list
features_list_new = deepcopy(features_list)
# remove 'poi' from features_list_new
features_list_new.remove('poi')

print "\nSelectKBest Score | Feature"
feat_scores = [] 
for f, s in zip(features_list_new, selector.scores_.tolist()):
    feat_scores.append([float(s), f])
    print float(s), f
feat_scores = sorted(feat_scores, reverse=True)

## Creating new features
## The new features I created, 'poi_email_ratio', 'assets', and 'tsvb_salary_ratio' can be found in the new_features.py file.
## They were not included in the final results since they did not result in meaningful improvements to the precision and recall scores of the machine learning classifier.


#########################################################
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()

# Decision Tree
from sklearn import tree
dt_clf = tree.DecisionTreeClassifier(min_samples_split=30)

### Support Vector Machine Classifier
from sklearn.svm import SVC
s_clf = SVC(kernel='rbf', C=500)

## evaluate SVM with scaled features
from sklearn.cross_validation import train_test_split
ft_train, ft_test, lb_train, lb_test = \
    train_test_split(features_minmax, labels, test_size=0.3, random_state=42)

from sklearn import metrics
from sklearn.metrics import accuracy_score

accuracy = []
precision = []
recall = []
clf = s_clf
clf.fit(ft_train, lb_train)
predictions = clf.predict(ft_test)
print "\nclassifier: SVM"
print "accuracy score:", accuracy_score(lb_test, predictions)
print "precision score:", metrics.precision_score(lb_test, predictions)
print "recall score:", metrics.recall_score(lb_test, predictions)

### Adaboost Classifier
from sklearn.ensemble import AdaBoostClassifier
a_clf = AdaBoostClassifier(algorithm='SAMME')

### Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()

### Selected Classifiers Evaluation
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn import metrics
from sklearn.metrics import accuracy_score

classifiers = [nb_clf, dt_clf, a_clf, rf_clf]

for cf in classifiers:
	accuracy = []
	precision = []
	recall = []
	clf = cf
	clf.fit(features_train, labels_train)
	predictions = clf.predict(features_test)
	print "\nclassifier:", cf
	print "accuracy score:", accuracy_score(labels_test, predictions)
	print "precision score:", metrics.precision_score(labels_test, predictions)
	print "recall score:", metrics.recall_score(labels_test, predictions)

### Final Machine Algorithm Selection
clf = GaussianNB()


#########################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.6, random_state=0)
# Changing the test_size and random_state parameters have no effect on the results.

# Validate algorithm
from sklearn.cross_validation import KFold
kf = KFold(len(labels), 4, shuffle = True)
for train_indices, test_indices in kf:
    kfeatures_train= [features[ii] for ii in train_indices]
    kfeatures_test= [features[ii] for ii in test_indices]
    klabels_train=[labels[ii] for ii in train_indices]
    klabels_test=[labels[ii] for ii in test_indices]

clf = clf.fit(kfeatures_train, klabels_train)
pred = clf.predict(kfeatures_test)

print "\nKFold: k=4 cross validation of algorithm:"
print "accuracy = ", accuracy_score(klabels_test, pred)
print 'precision = ', metrics.precision_score(klabels_test,pred)
print 'recall = ', metrics.recall_score(klabels_test,pred)

#########################################################
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
