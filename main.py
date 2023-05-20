from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model
import joblib
from sklearn.model_selection import train_test_split
from datetime import datetime
import json

app = Flask(__name__)

# Load the trained model and necessary preprocessing objects
# model = load_model('cnn_model.h5')
# u_rf_model = joblib.load('models/update_RF_model.pkl')
u_rf_model = joblib.load('models/update_RF_model_eng.pkl')
rf_model = joblib.load('models/RandomForest_model.pkl')
lstm_model = load_model('models/algo3_final.h5')
algo1 = load_model('models/algo1_ann.h5')
algo2 = load_model('models/algo2_lstm.h5')
algo5 = joblib.load('models/KNN_model.pkl')
# cnn_model = load_model('models/LSTM_model.h5')

scaler = StandardScaler()
label_encoder = LabelEncoder()

# load data
# df = pd.read_csv('data/cat_data.csv')
df = pd.read_csv('data/cat_data_eng.csv')

# handale categrical variables
# cat_vars = ['Local Name','Fish Temp In Category']
cat_vars = ['English Name','Fish Temp In Category']
label_encoders = {}
for var in cat_vars:
    label_encoders[var] = LabelEncoder()
    df[var] = label_encoders[var].fit_transform(df[var])
# label_encoders['Local Name'].classes_ = np.append(label_encoders['Local Name'].classes_, 'Unknown')

    
# split the data into training and testing sets
# X = df[['season', 'chl_df_a', 'sst', 'Local Name', 'Fish Temp In Category']]
# X = df[['season', 'Local Name', 'Fish Temp In Category']]
X = df[['season', 'English Name', 'Fish Temp In Category']]
y = df[['lat', 'lon']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

current_month = datetime.now().month

if current_month in range(1, 6):
    current_season = 1   #"Season 1 (Spring/Summer)"
elif current_month in range(9, 12):
    current_season = 2  #"Season 2 (Fall/Winter)"
else:
    current_season = 1

# new_data_encoded = np.zeros((new_data.shape[0], num_features))
# for i, var in enumerate(cat_vars[:3]):
#     new_data_encoded[:, i] = label_encoders[var].transform(new_data[:, i])

# Define API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Preprocess the input data
    data = pd.DataFrame(data)
    # print(data['Local Name'])
    
    # for var in cat_vars:
    # data['Local Name'] = label_encoders['Local Name'].transform(data['Local Name'])
    data['English Name'] = label_encoders['English Name'].transform(data['English Name'])
    # print(data) 
    data['season'] = current_season
    # Fish_Temp = df[(df['Local Name'] == data['Local Name'][0])]['Fish Temp In Category'].unique()[0]
    l = []
    for val in data['English Name']:
        Fish_Temp = df[(df['English Name'] == val)]['Fish Temp In Category'].unique()[0]
        l.append(Fish_Temp)
    data['Fish Temp In Category'] = l
    
    data = data.loc[:,['season','English Name','Fish Temp In Category']]
    print(data)
    
    
    # input_data['Local Name'] = label_encoder.transform(input_data['Local Name'])
    # input_features = scaler.transform(input_data[['season', 'chl_df_a', 'sst']])
    # input_features = input_features.reshape(input_features.shape[0], input_features.shape[1], 1)
    
    # for var in cat_vars:
    #     df[var] = label_encoders[var].transform(df[var])
    
   
    input_data = scaler.transform(data)
    
    ## algo 1
    # predictions = algo1.predict(input_data)
    # print(f"Predicted latitude: {predictions[0][0]}")
    # print(f"Predicted longitude: {predictions[0][1]}")
    # predictions = np.array(predictions, dtype=np.float32)
    # predictions_list = predictions.tolist()
    # json_predictions = json.dumps(predictions_list)
    # return json_predictions
    ## algo 1 end
    
    # # algo 2
    # X_new_3d = np.reshape(input_data, (input_data.shape[0], 1, input_data.shape[1]))
    # # make predictions
    # predictions = algo2.predict(X_new_3d)
    # print(f"Predicted latitude: {predictions[0][0]}")
    # print(f"Predicted longitude: {predictions[0][1]}")
    # predictions = np.array(predictions, dtype=np.float32)
    # predictions_list = predictions.tolist()
    # json_predictions = json.dumps(predictions_list)
    # return json_predictions
    # # algo 2 end
    
    # # algo 3
    input_data = np.reshape(input_data, (1, 1, input_data.shape[1]))
    # make predictions
    predictions = lstm_model.predict(input_data)
    # print the predicted latitude and longitude values
    print(f"Predicted latitude: {predictions[0][0]}")
    print(f"Predicted longitude: {predictions[0][1]}")
    predictions = np.array(predictions, dtype=np.float32)
    predictions_list = predictions.tolist()
    json_predictions = json.dumps(predictions_list)
    return json_predictions
    # # algo 3 end
    
    # # # algo 4
    # predictions = u_rf_model.predict(input_data)
    # # Prepare the response
    # response = {}
    # for i,val in enumerate(predictions):
    #     response[str(data['English Name'][i])] = list(val)
   
    # return jsonify(response)
    # # # algo 4 end
    
    # # # algo 5
    # predictions = algo5.predict(input_data)
    # # Prepare the response
    # response = {}
    # for i,val in enumerate(predictions):
    #     response[str(data['English Name'][i])] = list(val)
    # return jsonify(response)
    # # algo 5 end
    
    

if __name__ == '__main__':
    app.run(debug=True)
