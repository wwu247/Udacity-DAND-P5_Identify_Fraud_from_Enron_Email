#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]

data_dict.pop( 'TOTAL', 0 )
data = featureFormat(data_dict, features)


### your code below
#for k,v in data_dict.items():
#    if v['bonus'] == 'NaN' or v['salary'] == 'NaN':
#        pass
#    elif v['bonus'] > 5000000 and v['salary'] > 1000000:
#        print k, "bonus:", v['bonus'], "salary:", v['salary'], '\n'
#    else:
#        pass

#for point in data:
#    salary = point[0]
#    bonus = point[1]

s = []
b = []
for key in data_dict:
	s.append(data_dict[key]["salary"])
	b.append(data_dict[key]["bonus"])

matplotlib.pyplot.scatter(s, b)
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()