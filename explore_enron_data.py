#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


print enron_data["SKILLING JEFFREY K"]

# no. people in dataset
print "# of people:", len(enron_data)

# no. of features
print "# of features:", len(enron_data["SKILLING JEFFREY K"])

# financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)
# email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)
# POI label: [‘poi’] (boolean, represented as integer)

# no. of POIs
def numberpois(data):
	npoi = 0
	for person in data:
		if data[person]["poi"] == 1:
			npoi += 1
	return npoi

print "# of POIs:", numberpois(enron_data)


####################
#print enron_data.keys()

# cursory glance at the names indicate two misleading data points that should be removed: 'THE TRAVEL AGENCY IN THE PARK' and 'TOTAL'
print "THE TRAVEL AGENCY IN THE PARK:", enron_data['THE TRAVEL AGENCY IN THE PARK']
print "TOTAL:", enron_data['TOTAL']

####################
no_info = []

for person in enron_data:
	i = 0
	for info in enron_data[person]:
		if enron_data[person][info] == 'NaN':
			i += 1
	if i == 20:
		no_info.append(enron_data[person])
		print person
		print "person without info:", no_info

# one person has "NaN" values for all features and is not a POI - this data point has no useful information and can be removed

#####################
