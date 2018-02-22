from flask import Flask

app = Flask(__name__)


@app:route('user/<string:username>')

def helloWorld(username = None):
	return ("Hello.{}".format(username)) 

	

