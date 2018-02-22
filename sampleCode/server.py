import os
from sklearn.externals import joblib
from flask import Flask, jsonify, request
import pandas as pd
import dill as pickle


app = Flask(__name__)

@app.route('/predict', methods = ['POST'])

def apicall():
	
	try:
		test_json = request.get_json()
		test = pd.read_json(test_json, orient = 'records')
		
		test['Dependents'] = [str(x) for x in list(test['Dependents'])]
		
		loan_ids = test['Loan_ID']
		
	except Exception as e:
		raise e
	
	clf = 'model_v1.pk'
	
	if test.empty:
		return(bad_request())
	else:
		print("Loading the model ...")
		loaded_model = None
		
		with open('./model/'+clf, 'rb') as f:
			loaded_model = pickle.load(f)
		print("The model has been loaded...doing predictions now...")
		
		predictions = loaded_model.predict(test)
		
		prediciton_series = list(pd.Series(predictions))
		
		final_predictions = pd.DataFrame(list(zip(loan_ids, prediction_series)))
		responses = jsonify(predictions = final_predictions.to_json(orient="records"))
		
		responses.statuscode = 200
		
		return (responses)
		
		
		
		
		
		
		
		

		
	
