import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

@st.cache
def get_data(filename):
    taxi_data = pd.read_csv(filename)
    return taxi_data

def create_website(header, datasets, features, model_training):

    with header:
        st.title('Sample Project')
        st.write('[Just insert some description about your project]')


    with datasets:
        st.header('New York City Taxi Datasets')
        st.write('[Explain how you get the datasets. You can include a website to this one]')
        taxi_data = get_data('./yellow_tripdata_2021-01.csv')  
        st.write(taxi_data.head()) # display the data
        pulocation_distribution = pd.DataFrame(taxi_data['PULocationID'].value_counts())
        st.subheader('Pick Up Location Distribution')
        st.bar_chart(pulocation_distribution.head(20)) # show the bar chart
        


    with features:
        st.header('Features')
        st.markdown('* **First Feature: ** I Created this feature because of this... I calculated it using this logic')
        st.markdown('* **Second Feature: ** I Created this feature because of this... I calculated it using this logic')


    with model_training:
        st.header('Train the model')
        st.write('Explain and choose the hyperparameters of the model and see how the performance changes')

        selection_col, display_col = st.columns(2)
        minimum_value = 10
        maximum_value = 100
        default_value = 20
        difference = 10
        max_depth = selection_col.slider('Max depth', minimum_value, maximum_value, default_value, difference)
        number_estimator = selection_col.selectbox('amount of trees',options = [100, 200, 300, 'No Limit'])

        # List the input feature
        selection_col.write('Here is the list of features in my data:')
        selection_col.write(taxi_data.columns)

        input_feature = selection_col.text_input('Select feature to be used', 'PULocationID')

        if(number_estimator  == 'No Limit'):
            regr = RandomForestRegressor(max_depth=max_depth)
        else :
            regr = RandomForestRegressor(max_depth=max_depth, n_estimators=number_estimator)

        # st.write(max_depth, number_estimator)
        x = taxi_data[[input_feature]]
        y = taxi_data[['trip_distance']]
        regr.fit(x, y)
        prediction = regr.predict(y)

        display_col.subheader('Mean absolute error: ')
        display_col.write(mean_absolute_error(y, prediction))

        display_col.subheader('Mean squared error: ')
        display_col.write(mean_squared_error(y, prediction))

        display_col.subheader('R squared error: ')
        display_col.write(r2_score(y, prediction))


def main():
    header = st.container()
    datasets = st.container()
    features = st.container()
    model_training = st.container()

    create_website(header, datasets, features, model_training)



# main driver
main()