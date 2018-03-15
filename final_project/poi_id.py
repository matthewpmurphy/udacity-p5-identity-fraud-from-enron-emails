#!/usr/bin/python
import pickle
from tester import dump_classifier_and_data, test_classifier
import sys
sys.path.append("../tools/")
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from time import time
from feature_format import featureFormat, targetFeatureSplit

# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                 'expenses', 'exercised_stock_options', 'long_term_incentive',
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']


#Define functions

#Find and update NaN values to 0, except for email
def fill_nan_values():
    people = data_dict.keys()
    feature_keys = data_dict[people[0]]
    nan_features = {}
    for feature in feature_keys:
        nan_features[feature] = 0
    for person in people:
        for feature in feature_keys:
            if feature != 'email_address' and \
                data_dict[person][feature] == 'NaN':
                data_dict[person][feature] = 0
                nan_features[feature] += 1

    return nan_features

#Find count and values of POI with missing information
def poi_missing_email():
    poi_count = 0
    poi_keys = []
    for person in data_dict.keys():
        if data_dict[person]["poi"]:
            poi_count += 1
            poi_keys.append(person)

    poi_missing_emails = []
    for poi in poi_keys:
        if (data_dict[poi]['to_messages'] == 'NaN' and data_dict[poi]['from_messages'] == 'NaN') or \
            (data_dict[poi]['to_messages'] == 0 and data_dict[poi]['from_messages'] == 0):
            poi_missing_emails.append(poi)

    return poi_count, poi_missing_emails

#Find salary and bonus outliers
def salary_bonus_outliers():
    bonanza = []
    people = data_dict.keys()
    for person in people:
        if data_dict[person]["bonus"] > 5000000 or data_dict[person]["salary"] > 1000000:
            bonanza.append(person)

    return bonanza

#Create a new feature
def create_new_features():
    people = data_dict.keys()

    for person in people:
        to_poi = float(data_dict[person]['from_this_person_to_poi'])
        from_poi = float(data_dict[person]['from_poi_to_this_person'])
        to_msg_total = float(data_dict[person]['to_messages'])
        from_msg_total = float(data_dict[person]['from_messages'])

        if from_msg_total > 0:
            data_dict[person]['to_poi_fraction'] = to_poi / from_msg_total
        else:
            data_dict[person]['to_poi_fraction'] = 0

        if to_msg_total > 0:
            data_dict[person]['from_poi_fraction'] = from_poi / to_msg_total
        else:
            data_dict[person]['from_poi_fraction'] = 0

    # Add new feature to list
    features_list.extend(['to_poi_fraction', 'from_poi_fraction'])

#Data exploration
def explore_data():
    people = data_dict.keys()
    feature_keys = data_dict[people[0]]
    poi_cnt, poi_missing_emails = poi_missing_email()

    print 'People in dataset: %d' % len(people)
    print 'Features for each person: %d' % len(feature_keys)
    print 'Persons of Interests (POIs) in dataset: %d out of 34 total POIs' % poi_cnt
    print 'Non-POIs in dataset: %d' % (len(people) - poi_cnt)
    print 'POIs with email data in dataset: %d' % len(poi_missing_emails)
    print poi_missing_emails

    # Update nan values
    features_with_nan = fill_nan_values()
    print 'Updating NaN values...'
    print features_with_nan

    # Investigate other high salary and bonuses
    high_salary_bonus = salary_bonus_outliers()
    print 'High Salary/Bonus: \n', high_salary_bonus

    # Remove outlier and odd person value
    print 'Removing two values: Total, The Travel Agency In The Park'
    data_dict.pop('TOTAL')
    data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

    # Create new features
    print '\n'
    print 'Creating features...'
    create_new_features()
    print 'Updated Feature List: \n', features_list

#Build pipeline and tune parameters
def build_classifier_pipeline(classifier_type, kbest, f_list):
    data = featureFormat(my_dataset, f_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)

    # Using stratified shuffle split cross validation because of the small size of the dataset
    sss = StratifiedShuffleSplit(labels, 500, test_size=0.45, random_state=42)

    # Build pipeline
    kbest = SelectKBest(k=kbest)
    scaler = MinMaxScaler()
    classifier = set_classifier(classifier_type)
    pipeline = Pipeline(steps=[('minmax_scaler', scaler), ('feature_selection', kbest), (classifier_type, classifier)])

    # Set parameters for random forest
    parameters = []
    if classifier_type == 'random_forest':
        parameters = dict(random_forest__n_estimators=[25, 50],
                          random_forest__min_samples_split=[2, 3, 4],
                          random_forest__criterion=['gini', 'entropy'])
    if classifier_type == 'logistic_reg':
        parameters = dict(logistic_reg__class_weight=['balanced'],
                          logistic_reg__solver=['liblinear'],
                          logistic_reg__C=range(1, 5),
                          logistic_reg__random_state=42)
    if classifier_type == 'decision_tree':
        parameters = dict(decision_tree__min_samples_leaf=range(1, 5),
                          decision_tree__mix_depth=range(1, 5),
                          decision_tree__class_weight=['balanced'],
                          decision_tree__criterion=['gini', 'entropy'])

    # Get optimized parameters for F1-scoring metrics
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1', cv=sss)
    t0 = time()
    cv.fit(features, labels)
    print 'Classifier tuning: %r' % round(time() - t0, 3)

    return cv

#switch statement
def set_classifier(x):
    return {
        'random_forest': RandomForestClassifier(),
        'decision_tree': DecisionTreeClassifier(),
        'logistic_reg': LogisticRegression(),
        'gaussian_nb': GaussianNB()
    }.get(x)

# Start main program
# Load the dictionary containing the dataset
print '\n'
print 'Loading data...'
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    print 'data_dict of length %d loaded successfully' % len(data_dict)


# Data Exploration and removal of outliers
print '\n'
print 'Exploring data...'
explore_data()
my_dataset = data_dict

# Features List
n_list = ['poi', 'salary', 'total_stock_value', 'expenses', 'bonus',
          'exercised_stock_options', 'deferred_income',
          'to_poi_fraction', 'from_poi_to_this_person', 'from_poi_fraction',
          'shared_receipt_with_poi']

# Feature selection
print '\n'
print 'Selecting features...'
# Feature select is done with SelectKBest where k is selected by GridSearchCV
# Stratify for small and minority POI dataset

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = \
  train_test_split(features, labels, train_size=.45, stratify=labels)

skbest = SelectKBest(k=10)
sk_transform = skbest.fit_transform(features_train, labels_train)
indices = skbest.get_support(True)
print skbest.scores_

n_list = ['poi']
for index in indices:
    print 'features: %s score: %f' % (features_list[index + 1], skbest.scores_[index])
    n_list.append(features_list[index + 1])


# Update features_list with new values
features_list = n_list

clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, min_samples_leaf=2, class_weight='balanced'),
                         n_estimators=50, learning_rate=.8)

# Validate model precision, recall and F1-score
test_classifier(clf, my_dataset, features_list)

# Dump classifier, dataset and features_list
print '\n'
print 'Pickling files....'
# Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)
print 'Files are now pickled.'