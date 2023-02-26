# Importing required dependencies 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from joblib import dump

def train():

  train_path = "./train.csv"
  train_data = pd.read_csv(train_path)

  y_train = train_data[train_data.columns[-1]]
  X_train = train_data.drop(train_data.columns[-1], axis = 1) 

  ss = StandardScaler()
  X_train_ss = ss.fit_transform(X_train)

  # Decision Tree Classifier 
  dtc = DecisionTreeClassifier(max_depth = 4, criterion = 'entropy', 
                               max_features = 0.6, splitter = 'best')
  dtc = dtc.fit(X_train_ss, y_train)

  # Random Forest 
  from sklearn.ensemble import RandomForestClassifier
  rf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
  rf.fit(X_train_ss, y_train)

  # saving the model
  dump(dtc, 'decisiontree.joblib')
  dump(rf, 'randomforest.joblib')

if __name__ == '__main__':
    train()
