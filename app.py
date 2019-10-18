import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import pandas as pd
import pickle



########### Initiate the app
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server

file = open('resources/AAPL_final_model.pkl', 'rb')
model= pickle.load(file)
file.close()
# define the new observation from the chosen values
new_obs =[[2, 1, 0, 2, 1, 2]]
my_prediction = model.predict(new_obs)

app.layout = html.Div(children=[f'the prediction is {my_prediction}'])



############ Execute the app
if __name__ == '__main__':
    app.run_server(debug=True)# automatically reload
