import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(df):
    # Define numerical and categorical attributes
    numerical_attribs = ['Accommodates', 'Bedrooms', 'Beds', 'Number of Reviews', 'Host Since Days', 
        'Review Scores Rating', 'Review Scores Accuracy', 'Review Scores Cleanliness', 'Review Scores Checkin',
        'Review Scores Communication', 'Review Scores Location', 'Review Scores Value']

    categorical_attribs = ['Host Response Time']
    all_categorical_attribs = ['Host Response Time', 'Neighbourhood Cleansed', 'City', 'Country', 'Property Type', 'Room Type', 'Bed Type', 'Cancellation Policy']

    # Define custom order for Host Response Time
    custom_order = ['within an hour', 'within a few hours', 'within a day', 'a few days or more']

    other_features = ['Neighbourhood Cleansed', 'City', 'Country', 'Property Type', 'Room Type', 'Bed Type', 'Cancellation Policy', 'Host Identity Verified', 'Host Is Superhost', 'TV', 'Wireless Internet', 'Kitchen', 'Heating', 'Family/kid friendly', 'Washer', 'Smoke detector', 'Fire extinguisher', 'Essentials', 'Cable TV', 'Internet', 'Dryer', 'First aid kit', 'Safety card', 'Shampoo', 'Hangers', 'Laptop friendly workspace', 'Air conditioning', 'Breakfast', 'Free parking on premises', 'Elevator in building', 'Buzzer/wireless intercom', 'Hair dryer', 'Private living room', 'Iron', 'Wheelchair accessible', 'Hot tub', 'Carbon monoxide detector', '24-hour check-in', 'Pets live on this property', 'Dog(s)', 'Gym', 'Lock on bedroom door', 'Private entrance', 'Indoor fireplace', 'Smoking allowed', 'Pets allowed', 'Cat(s)', 'Self Check-In', 'Doorman Entry', 'Suitable for events', 'Pool', 'Lockbox', 'Bathtub', 'Room-darkening shades', 'Game console', 'Doorman', 'High chair', 'Pack ’n Play/travel crib', 'Keypad', 'Other pet(s)', 'Smartlock', 'Price']

    # Define pipelines for preprocessing
    numerical_pipeline = Pipeline([
            ('std_scaler', StandardScaler()),
        ])

    categorical_ordinal_pipeline = Pipeline([
        ('ordinal_encoder', OrdinalEncoder(categories=[custom_order]))
    ])

    # Combine preprocessing steps using ColumnTransformer
    preprocessing_pipeline = ColumnTransformer([
        ("categorical_ordinal", categorical_ordinal_pipeline, ['Host Response Time']),
        ("numerical", numerical_pipeline, numerical_attribs),
    ], remainder='passthrough')  # Pass through other features

    # Fit and transform data using the combined pipeline
    preprocessed_data = preprocessing_pipeline.fit_transform(df)

    # Get column names after preprocessing
    preprocessed_columns = (['Host Response Time'] +
                            numerical_attribs +
                            [col for col in df.columns if col not in ['Host Response Time'] + numerical_attribs])

    # Create a DataFrame using the preprocessed data and column names
    preprocessed_df = pd.DataFrame(data=preprocessed_data, columns=preprocessed_columns)

    return preprocessed_df