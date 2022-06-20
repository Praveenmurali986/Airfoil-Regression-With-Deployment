import pickle
import numpy as np
import pandas as pd
from flask import Flask,request,app,jsonify,url_for,render_template
from flask import Response

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    for direct api calls through request
    '''
    data=request.json['data']
    print(data)
    new_data=[list(data.values())]
    output=model.predict(new_data)[0]
    return jsonify(output)

@app.route('/predict',methods=['POST'])
def predict():
    '''
    for direct api calls through request
    '''
    data=[float(x) for x in request.form.values()]
    final_data=[np.array(data)]
    print(data)
    #new_data=[list(data.values())]
    output=model.predict(final_data)[0]
    print(output)
    return render_template('home.html',prediction_text=f'Airfoil pressure is  {output}')


if __name__=="__main__":
    app.run(host='127.0.0.1',debug=True)