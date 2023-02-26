# importing required dependencies 
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from joblib import dump

# Define a flask app 
app = Flask(__name__)

@app.route("/")
def home():
  html = "<h2>Decision Tree and Random Forest Model Root</h2>"
  return html

# Run the test 
@app.route('/test', methods = ['POST'])
def test():
  
  filename = request.json['filename'] 
  
  test_data = pd.read_csv(filename)

  y_test = test_data[test_data.columns[-1]]
  X_test = test_data.drop(test_data.columns[-1], axis = 1) 

  ss = StandardScaler()
  X_test_ss = ss.fit_transform(X_test)

  # load the training model and display results to the user 
  decision_tree = joblib.load('decisiontree.joblib')
  random_forest = joblib.load('randomforest.joblib')
  print("Classification result - Decision Tree:")
  print(decision_tree.score(X_test_ss, y_test))
  print(decision_tree.predict(X_test_ss))
  print("Classification result - Random Forest")
  print(random_forest.score(X_test_ss, y_test))
  print(random_forest.predict(X_test_ss))
  result1 = str(decision_tree.score(X_test_ss, y_test))
  result2 = str(random_forest.score(X_test_ss, y_test))
  return jsonify({'prediction_dct': result1, 'prediction_rf':result2})


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000, debug = True)