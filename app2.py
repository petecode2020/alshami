
import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd

#Create an app object using the Flask class. 
app = Flask(__name__)

#Load the trained model. (Pickle file)
xg_model = pickle.load(open('C:/Users/arash.tayyebi/Desktop/scale/data/done/last updated 0206/XG9y.pkl', 'rb'))
rf_model = pickle.load(open('C:/Users/arash.tayyebi/Desktop/scale/data/done/last updated 0206/RF9y2.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

import matplotlib.pyplot as plt
from io import BytesIO
import base64


def create_bar_chart(result_dict):
    values = result_dict
    chemicals = ['CaSO4-Anhydrite', 'CaCO3-Aragonite', 'BaSO4-Barite', 'CaCO3-Calcite', 'SrSO4-Celestite', 'CaSO4:2H2O-Gypsum', 'NaCl-Halite', 'SiO2-Quartz', 'FeCO3-Siderite']

    # Create a subplot grid with 1 row and 1 column
    fig, ax = plt.subplots(figsize=(10, 8))

    # Assigning colors based on positive and negative values
    colors = ['green' if value >= 0 else 'red' for value in values]

    # Creating a horizontal bar chart
    ax.barh(chemicals, values, color=colors)

    # Adding labels and title
    ax.set_xlabel('Values')
    ax.set_ylabel('Chemicals')
    ax.set_title('SI values')

    # Adding value labels next to the bars
    for chemical, value, color in zip(chemicals, values, colors):
        ax.text(value, chemical, f'{value:.2f}', ha='left' if value >= 0 else 'right', va='center', color='black' if color == 'green' else 'black')

    # Adjust the position of the subplot
    ax.set_position([0.25, 0.2, 0.7, 0.7])  # Adjust the values as needed

    # Saving the plot to a BytesIO object
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)

    # Closing the plot to avoid memory issues
    plt.close()

    # Converting the BytesIO object to a base64-encoded string
    image_base64 = base64.b64encode(image_stream.read()).decode('utf-8')

    return image_base64

feature_order = ['pH', 'ICP_Al', 'ICP_B', 'ICP_Ba', 'ICP_HCO3', 'ICP_Ca', 'ICP_Cl', 'ICP_Cu', 'ICP_Fe',
                  'ICP_K', 'ICP_Mg', 'ICP_Mn', 'ICP_Na', 'ICP_Pb', 'ICP_S', 'ICP_Sulfate', 'ICP_Si',
                  'ICP_Sr', 'ICP_Zn']

@app.route('/predict',methods=['POST'])
def predict():
    selected_model = request.form['model']
    # Convert form data to a dictionary
  #  form_data = {key: float(value) for key, value in request.form.items()}
    form_data = {key: float(value) if value.replace('.', '', 1).isdigit() else value for key, value in request.form.items()}


    # Create a DataFrame with the form data in the expected order
    
    if selected_model == 'RF':
        model = rf_model
    elif selected_model == 'XG':
        model = xg_model
    else:
        return render_template('index.html', prediction_text='Invalid model selection', bar_chart_image=None)
    
    input_data = pd.DataFrame([form_data])[feature_order]

    # Perform prediction using the XGBoost model
    prediction = model.predict(input_data)

    # Assuming your result_dict structure is similar to the previous code
    result_dict = {'CaSO4-Anhydrite': round(prediction[0][0],2),'CaCO3-Aragonite': round(prediction[0][1],2)\
                      , 'BaSO4-Barite': round(prediction[0][2],2), 'CaCO3-Calcite': round(prediction[0][3],2)\
                          , 'SrSO4-Celestite': round(prediction[0][4],2), 'CaSO4:2H2O-Gypsum': round(prediction[0][5],2)\
                              ,'NaCl-Halite': round(prediction[0][6],2)\
                              , 'SiO2-Quartz': round(prediction[0][7],2), 'FeCO3-Siderite': round(prediction[0][8],2)}

    bar_chart_image = create_bar_chart(list(result_dict.values()))

    return render_template('index.html', prediction_text=result_dict, bar_chart_image=bar_chart_image)

#@app.route('/predict',methods=['POST'])
#def predict():
 #   data = [float(x) for x in request.form.values()] 
 #   prediction = model.predict([data])
   #ok result_dict = {'A': round(prediction[0][0],2), 'B': round(prediction[0][1],2)}
 #   result_dict = {'CaSO4-Anhydrite': round(prediction[0][0],2),'CaCO3-Aragonite': round(prediction[0][1],2)\
   #                , 'BaSO4-Barite': round(prediction[0][2],2), 'CaCO3-Calcite': round(prediction[0][3],2)\
   #                    , 'SrSO4-Celestite': round(prediction[0][4],2), 'CaSO4:2H2O-Gypsum': round(prediction[0][5],2)\
   #                        ,'NaCl-Halite': round(prediction[0][6],2)\
  #                         , 'SiO2-Quartz': round(prediction[0][7],2), 'FeCO3-Siderite': round(prediction[0][8],2)}
#    bar_chart_image = create_bar_chart(list(result_dict.values()))

#    return render_template('index.html', prediction_text=result_dict, bar_chart_image=bar_chart_image)
										


  



if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)