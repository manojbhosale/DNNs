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
		
		#input ='[{"Loan_ID":"LP001015","Gender":1,"Married":1,"Dependents":"0","Education":0,"Self_Employed":0,"ApplicantIncome":5720,"CoapplicantIncome":0,"LoanAmount":110,"Loan_Amount_Term":360,"Credit_History":1,"Property_Area":1}]'
		
		#input = '[{"Loan_ID":"LP001015","Gender":"Male","Married":"Yes","Dependents":"0","Education":"Graduate","Self_Employed":"No","ApplicantIncome":"5720","CoapplicantIncome":0,"LoanAmount":"110","Loan_Amount_Term":"360","Credit_History":"1","Property_Area":"Urban"}]'
		
		print(test_json,type(test_json))
		
		#print(type(jsono), jsono)
		#test = pd.read_json(test_json, orient = 'records')
		test = pd.DataFrame(test_json)
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
		
		with open('./models/'+clf, 'rb') as f:
			loaded_model = pickle.load(f)
		print("The model has been loaded...doing predictions now...")
		testdf = pd.DataFrame(test); #need for succssful run in transformation step
		print(testdf)
		#a = test['Dependents']
		print("Type --- ",type(testdf))
		predictions = loaded_model.predict(testdf)
		
		prediction_series = list(pd.Series(predictions))
		
		final_predictions = pd.DataFrame(list(zip(loan_ids, prediction_series)))
		responses = jsonify(predictions = final_predictions.to_json(orient="records"))
		
		responses.statuscode = 200
		
		return (responses)
		
		
if __name__ == '__main__':
	app.run()
