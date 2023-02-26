import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

subway = pd.read_csv("Toronto-Subway-Delay-Jan-2014-Jun-2021.csv")
#print(subway.head())

# Check missing values 
#subway.isnull().sum()

subway['Bound'].fillna('S',inplace=True)
subway['Line'].fillna('YU',inplace=True)
subway['Code'].fillna('MUSC',inplace=True)
subway.dropna(how='any',inplace=True)
#subway.info()

subway=subway.drop(subway.columns[[0,1,2]], axis=1)
#subway.info()

# Convert categorical to indicator variables 
cols1 = ['Bound', 'Station', 'Code', 'Line']
cols2 = ['Min Delay', 'Min Gap', 'Vehicle']

subway_dummies = pd.get_dummies(subway[cols1])
subway = pd.concat([subway[cols2], subway_dummies], axis = 1)
#stats=subway['Min Delay'].describe()
#print(stats)

#subway['Min Delay'].plot(kind='box', title='Min Delay')
#plt.show()

delay_indicator = [] 
for row in subway['Min Delay']:
  if row > 3:
    delay_indicator.append(1)
  else:
    delay_indicator.append(0)
    
subway['delay_indicator'] = delay_indicator

subway.info()
#subway.value_counts('delay_indicator')

# Min Delay is categorized by delay_indicator as predictor and the rest of columns are features 
data_drop = subway.drop(['Min Delay'],axis=1)
data = data_drop.values
X, y = data[:,:-1], data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Ensuring the same datatype by converting the subsets into dataframes

column_names = []
for col in subway.columns:
    column_names.append(col)

column_names.remove('delay_indicator')
column_names.remove('Min Delay')

X_train = pd.DataFrame(X_train, columns = column_names)
X_test = pd.DataFrame(X_test, columns = column_names)
y_train = pd.DataFrame(y_train, columns = ['delay_indicator'])
y_test = pd.DataFrame(y_test, columns = ['delay_indicator'])

# Creating train_data and test_data and loading the output into a csv file 
# index = False allows to get rid of an additional index column in each file 
#train_data = pd.concat([X_train, y_train], axis="columns").to_csv('train.csv', index = False)
#test_data = pd.concat([X_test, y_test], axis = 'columns').to_csv('test.csv', index = False)

ss = StandardScaler()
X_train_ss= ss.fit_transform(X_train)
X_test_ss=ss.transform(X_test)

# Decision Tree Classifier 
dtc = DecisionTreeClassifier(max_depth=4, criterion='entropy', max_features=0.6, splitter='best')
dtc = dtc.fit(X_train_ss,y_train)

pred_train = dtc.predict(X_test_ss)
print("The Decision Tree Classifier model's accuracy on the test set is: " + str(accuracy_score(y_test, pred_train)*100)+ "%")

# Random Forest 
rf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
rf.fit(X_train_ss,y_train)

#print("The model's accuracy on the training set is: " + str(rf.score(X_train_ss,y_train)*100)+ "%")
print("The model's accuracy on the test set is: " + str(rf.score(X_test_ss,y_test)*100)+ "%")

