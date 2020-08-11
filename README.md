# Liver Disease Prediction System

### Files
* app.py : REST API
* main_logic.py : Load the pickled scaler, encoder, mean values, and model. Use scaler, encoder and mean values for preprocessing prediction. The use the model to predict
* encoder.pkl : Pickled fitted OneHotEncoder
* scaler.pkl : Pickled fitted MinMaxScaler
* mean_values.pkl : Pickled column-vise mean values
* model_rfc : Pickled RandomForestClassifier
* requirements.txt : Libraries with relavent versions
* Req1.png, Req2.png : Request template
* Procfile : For Heroku deployment


### Setup - Local

`conda create -n lmp python=3.6 pip`

`conda activate lmp`

`pip install -r requirements.txt`

### Setup - Heroku

`heroku login`

`heroku create <appname>`

`heroku git:remote -a <appname>`# ML_Liver_pred
