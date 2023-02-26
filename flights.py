# importing required dependencies 
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

# loading the dataset 
flights = pd.read_csv("DelayedFlights.csv")

# Check missing values 
# print(flights.isnull().sum())

# Fill missing data with 0
flights_clean=flights.fillna(0)

# Confirm the clean output
# print(flights_clean.isnull().sum())
# print(flights_clean.info())

# Get rid of irrelevant columns
flights_clean=flights_clean.drop(flights_clean.columns[[0,5,6,7,8,9,10,11,12,13,14,17,18,19,20,21,22,23,24]], axis=1)

# Visualize the dataset and explore basic metrics 
# sb.jointplot(data=flights_clean, x="NASDelay", y="ArrDelay")

# sb.jointplot(data=flights_clean, x="DepDelay", y="ArrDelay")

# corr = flights_clean.corr(method='spearman')
# sb.heatmap(corr)
# print(corr)

# stats=flights_clean['ArrDelay'].describe()
# print(stats)

# flights_clean['ArrDelay'].plot(kind='box', title='Arrival Delay')
# plt.show()

# Producing "delay indicator column"
delay_indicator = [] 
for row in flights_clean['ArrDelay']:
  if row > 25:
    delay_indicator.append(1)
  else:
    delay_indicator.append(0)

flights_clean['delay_indicator'] = delay_indicator
flights_clean.value_counts('delay_indicator')

# ArrDelay is categorized by delay_indicator as predictor 
# and the rest of columns are features 
data_drop = flights_clean.drop(['ArrDelay'], axis=1)
data = data_drop.values
X, y = data[:,:-1], data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state = 42)
# Ensuring the same datatype by converting the subsets into dataframes
# column_names = ['Year', 'Month', 'DayofMonth', 'DayofWeek', 'DepDelay', 'CarrierDelay',
#                 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']

# X_train = pd.DataFrame(X_train, columns = column_names)
# X_test = pd.DataFrame(X_test, columns = column_names)
# y_train = pd.DataFrame(y_train, columns = ['delay_indicator'])
# y_test = pd.DataFrame(y_test, columns = ['delay_indicator'])


# Creating train_data and test_data and loading the output into a csv file 
# index = False allows to get rid of an additional index column in each file 
#train_data = pd.concat([X_train, y_train], axis="columns").to_csv('train.csv', index = False)
#test_data= pd.concat([X_test, y_test], axis = 'columns').to_csv('test.csv', index = False)


ss = StandardScaler()
X_train_ss = ss.fit_transform(X_train)
X_test_ss = ss.transform(X_test)


# Decision Tree Classifier 
dtc = DecisionTreeClassifier(max_depth=4, criterion='entropy', max_features=0.6, splitter='best')
dtc = dtc.fit(X_train_ss,y_train)

print("Classification result:")
print(dtc.score(X_test_ss, y_test))
print(dtc.predict(X_test))
result = str(dtc.predict(X_test))
# # Random Forest 
# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
# rf.fit(X_train_ss,y_train)

# #print("The model's accuracy on the training set is: " + str(rf.score(X_train_ss,y_train)*100)+ "%")
# print("The model's accuracy on the test set is: " + str(rf.score(X_test_ss,y_test)*100)+ "%")


