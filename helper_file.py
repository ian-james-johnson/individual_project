import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from pandas.plotting import register_matplotlib_converters

from datetime import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt

from scipy.stats import pearsonr, spearmanr, ttest_ind

import statsmodels.api as sm
from statsmodels.tsa.api import Holt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score

import warnings
warnings.filterwarnings("ignore")

# Sets precision of 2 and suppresses scientific notation
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Seperates thousands, millions, etc with commas
pd.options.display.float_format = '{:,}'.format

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def aquire_data():
    '''
    This function reads the required local csv's to create datasets.
    '''

    df = pd.read_csv('covid_19_india.csv')
    vaccine_df = pd.read_csv('covid_vaccine_statewise.csv')
    return df, vaccine_df

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def prepare_data(df, vaccine_df):
    '''
    Cleans data for further analysis. 
    Gives states the same names in both source dataframes.
    Gives better names to features.
    Drops unwanted features.
    '''

    # Correct issues with state names
    # Asterixes were removed from names
    # Some states had several spellings, and were combined into one form
    # Some states were not state names and were changed to null
    df = df.replace('Bihar****', 'Bihar')
    df = df.replace('Cases being reassigned to states', 'Other')
    df = df.replace('Dadra and Nagar Haveli', 'Dadra and Nagar Haveli and Daman and Diu')
    df = df.replace('Daman & Diu', 'Dadra and Nagar Haveli and Daman and Diu')
    df = df.replace('Himanchal Pradesh', 'Himachal Pradesh')
    df = df.replace('Karanataka', 'Karnataka')
    df = df.replace('Madhya Pradesh***', 'Madhya Pradesh')
    df = df.replace('Maharashtra***', 'Maharashtra')
    df = df.replace('Telengana', 'Telangana')
    df = df.replace('Unassigned', 'Other')
    vaccine_df = vaccine_df.replace('India', 'Other')
    
    # Change column names for ease-of-use
    df = df.rename(columns={'Date':'date', 
                            'State/UnionTerritory':'state',
                            'Cured':'cured', 
                            'Deaths':'deaths'})
    
    vaccine_df = vaccine_df.rename(columns={'Updated On':'date', 
                                            'State':'state', 
                                            'Total Doses Administered':'total_doses', 
                                            ' Covaxin (Doses Administered)':'covaxin', 
                                            'CoviShield (Doses Administered)':'covishield', 
                                            'Sputnik V (Doses Administered)':'sputnik', 
                                            '18-44 Years(Individuals Vaccinated)':'young_adults_vaccinated', 
                                            '45-60 Years(Individuals Vaccinated)':'midaged_vaccinated', 
                                            '60+ Years(Individuals Vaccinated)':'elderly_vaccinated', 
                                            'Male(Individuals Vaccinated)':'males_vaccinated', 
                                            'Female(Individuals Vaccinated)':'females_vaccinated'}) 
    
    # Drop unwanted columns
    df = df.drop(columns=['Sno', 'Time', 'ConfirmedIndianNational', 
                          'ConfirmedForeignNational', 'Confirmed'])
    
    vaccine_df = vaccine_df.drop(columns=['18-44 Years (Doses Administered)', 
                                          '45-60 Years (Doses Administered)', 
                                          '60+ Years (Doses Administered)',
                                          'Male (Doses Administered)', 
                                          'Female (Doses Administered)',
                                          'Transgender(Individuals Vaccinated)', 
                                          'Total Individuals Vaccinated', 
                                          'First Dose Administered', 
                                          'Second Dose Administered',
                                          'Transgender (Doses Administered)', 
                                          'Sessions', 
                                          ' Sites '])

    # Drop nulls from vaccine dataframe, date and state had no nulls
    # Assume that other nulls occured because healthcare workers do not
    # always fill out a list of things that they don't give a patient.
    # Thus, the real value is probably 0
    # The other dataframe has no nulls
    vaccine_df = vaccine_df.fillna(0)
    
    # Convert 'date' to datetime type for both dataframes
    df.date = pd.to_datetime(df.date)
    vaccine_df.date = pd.to_datetime(vaccine_df.date)
    # Set date as index for both dataframes
    df = df.set_index('date').sort_index()
    vaccine_df = vaccine_df.set_index('date').sort_index()
      
    return df, vaccine_df

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def combine_data(df, vaccine_df):
    '''
    Combines and reorganizes the survival and vaccination datasets.
    '''

    df['month'] = df.index.month
    vaccine_df['month'] = vaccine_df.index.month
    
    df = df.reset_index()
    vaccine_df = vaccine_df.reset_index()
    
    df = df.drop(columns='date')
    vaccine_df = vaccine_df.drop(columns='date')
    
    # The other df only goes to month 8, so we will cap this df at 8
    # The year does not change in this dataset
    vaccine_df = vaccine_df[vaccine_df.month <= 8]
    
    # Sorting by state and then merging the two dataframes side-by-side
    # Drop one state or we will have two state columns after merging
    delhi_d = df[df.state == 'Delhi']
    delhi_v = vaccine_df[vaccine_df.state == 'Delhi'].drop(columns='state')
    delhi_s = pd.merge(delhi_d, delhi_v, on='month')
    
    tamil_d = df[df.state == 'Tamil Nadu']
    tamil_v = vaccine_df[vaccine_df.state == 'Tamil Nadu'].drop(columns='state')
    tamil_s = pd.merge(tamil_d, tamil_v, on='month')
    
    madhya_d = df[df.state == 'Madhya Pradesh']
    madhya_v = vaccine_df[vaccine_df.state == 'Madhya Pradesh'].drop(columns='state')
    madhya_s = pd.merge(madhya_d, madhya_v, on='month')
    
    
    # List of dataframes to be concatenated
    frames = [delhi_s, tamil_s, madhya_s]
    
    # Concatenate the dataframes
    # Dataframes are stacked below each other, ignoring the index
    sum_df = pd.concat(frames, ignore_index=True, axis=0)
    
    # Replace state names with numbers so that they may be used for calculations
    sum_df = sum_df.replace('Delhi', 1)
    sum_df = sum_df.replace('Tamil Nadu', 2)
    sum_df = sum_df.replace('Madhya Pradesh', 3)
    
    return sum_df

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def univariate_exploration(df):
    '''
    Makes histograms for each of the features.
    '''

    for col in df.columns:
        plt.figure(figsize=(12,4))
        df[col].hist()
        plt.ylabel(col)
        plt.title(col)
        plt.show()

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def time_v_cured_daily(df):
    plt.scatter(df.index, df.cured)
    plt.xlabel('Time')
    plt.ylabel('Cured')

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def vaccine_v_doses_daily(df):
    plt.scatter(df.index, df.total_doses)
    plt.xlabel('Time')
    plt.ylabel('Total Doses Administered')

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def month_state_v_cured(df):
    grid = sns.FacetGrid(df, col='state')
    grid.map(sns.lineplot, 'month', 'cured')
    grid.add_legend()
    plt.show()

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def state_month_v_deaths(df):
    grid = sns.FacetGrid(df, col='state')
    grid.map(sns.lineplot, 'month', 'deaths')
    grid.add_legend()
    plt.show()

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def state_month_v_doses(df):
    grid = sns.FacetGrid(df, col='state')
    grid.map(sns.lineplot, 'month', 'total_doses')
    grid.add_legend()
    plt.show()

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def state_month_v_AEFI(df):
    grid = sns.FacetGrid(df, col='state')
    grid.map(sns.lineplot, 'month', 'AEFI')
    grid.add_legend()
    plt.show()

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def proportion_vaccines(df):
    grid = sns.FacetGrid(df, col='state')
    grid.map(sns.lineplot, 'month', 'covaxin', label='covaxin', color='red')
    grid.map(sns.lineplot, 'month', 'covishield', label='covishield', color='blue')
    grid.map(sns.lineplot, 'month', 'sputnik', label='sputnik', color='yellow')
    grid.add_legend()
    plt.show()

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def month_v_vaccination_age(df):
    grid = sns.FacetGrid(df, col='state')
    grid.map(sns.lineplot, 'month', 'young_adults_vaccinated', label='young_adults_vaccinated', color='red')
    grid.map(sns.lineplot, 'month', 'midaged_vaccinated', label='midaged_vaccinated', color='blue')
    grid.map(sns.lineplot, 'month', 'elderly_vaccinated', label='elderly_vaccinated', color='yellow')
    grid.add_legend()
    plt.show()

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def month_v_gender(df):
    grid = sns.FacetGrid(df, col='state')
    grid.map(sns.lineplot, 'month', 'males_vaccinated', label='males_vaccinated', color='red')
    grid.map(sns.lineplot, 'month', 'females_vaccinated', label='females_vaccinated', color='blue')
    grid.add_legend()
    plt.show()

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def data_split(df):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=123, 
                                            stratify = df.month)
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=123,
                                       stratify=train_validate.month)
    return train, validate, test
# I am stratifying the split across time, because time influences most features

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def state_v_col(train):
    for col in train.columns:
        plt.figure(figsize=(12,4))
        sns.boxplot( x = train.state, y = train[col])
        plt.ylabel('State')
        plt.title(col)
        plt.show()

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def doses_v_col(train):
    for col in train.columns:
        grid = sns.FacetGrid(train, col='state')
        grid.map(sns.lineplot, 'total_doses', col)
        plt.ylabel('Total Doses')
        plt.title(col)
        plt.show()

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def doses_v_vaccine_type(train):
    grid = sns.FacetGrid(train, col='state')
    grid.map(sns.lineplot, 'total_doses', 'covaxin', label='covaxin', color='red')
    grid.map(sns.lineplot, 'total_doses', 'covishield', label='covishield', color='blue')
    grid.map(sns.lineplot, 'total_doses', 'sputnik', label='sputnik', color='yellow')
    grid.add_legend()
    plt.show()

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def vaccine_type_v_AEFI(train):
    grid = sns.FacetGrid(train, col='state')
    grid.map(sns.lineplot, 'covaxin', 'AEFI', label='covaxin', color='red')
    grid.map(sns.lineplot, 'covishield', 'AEFI', label='covishield', color='blue')
    grid.map(sns.lineplot, 'sputnik', 'AEFI', label='sputnik', color='yellow')
    grid.add_legend()
    plt.show()

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def age_v_AEFI(train):
    grid = sns.FacetGrid(train, col='state')
    grid.map(sns.lineplot, 'AEFI', 'young_adults_vaccinated', label='young_adults_vaccinated', color='red')
    grid.map(sns.lineplot, 'AEFI',  'midaged_vaccinated', label='midaged_vaccinated', color='blue')
    grid.map(sns.lineplot, 'AEFI', 'elderly_vaccinated', label='elderly_vaccinated', color='yellow')
    grid.add_legend()
    plt.show()

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def deaths_v_age(train):
    grid = sns.FacetGrid(train, col='state')
    grid.map(sns.lineplot, 'deaths', 'young_adults_vaccinated', label='young_adults_vaccinated', color='red')
    grid.map(sns.lineplot, 'deaths',  'midaged_vaccinated', label='midaged_vaccinated', color='blue')
    grid.map(sns.lineplot, 'deaths', 'elderly_vaccinated', label='elderly_vaccinated', color='yellow')
    grid.add_legend()
    plt.show()

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def prep_X_y_train(train, validate, test):
    '''
    Seperates target features from predictive features in split data.
    '''
    # Dropping the target from the other features
    X_train = train.drop(columns='AEFI') 
    X_validate = validate.drop(columns='AEFI')
    X_test = test.drop(columns='AEFI')
    
    # Creating a set containing the target
    y_train = train.AEFI
    y_validate = validate.AEFI
    y_test = test.AEFI
    
    return X_train, X_validate, X_test, y_train, y_validate, y_test

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def scale_data(X_train, X_validate, X_test):
    '''
    Scales the data.
    '''

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_validate = scaler.transform(X_validate)
    X_test = scaler.transform(X_test)
    
    return X_train, X_validate, X_test

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def back_to_df(X_train, X_validate, X_test, y_train, y_validate, y_test):
    '''
    Turns scaled data back into pandas dataframes.
    '''

    X_train = pd.DataFrame(X_train, columns = ['state', 'cured', 'deaths', 'month', 'total_doses', 'covaxin', 'covishield', 'sputnik', 'young_adults_vaccinated', 'midaged_vaccinated', 'elderly_vaccinated', 'males_vaccinated', 'females_vaccinated'])
    X_validate = pd.DataFrame(X_validate, columns = ['state', 'cured', 'deaths', 'month', 'total_doses', 'covaxin', 'covishield', 'sputnik', 'young_adults_vaccinated', 'midaged_vaccinated', 'elderly_vaccinated', 'males_vaccinated', 'females_vaccinated'])
    X_test = pd.DataFrame(X_test, columns = ['state', 'cured', 'deaths', 'month', 'total_doses', 'covaxin', 'covishield', 'sputnik', 'young_adults_vaccinated', 'midaged_vaccinated', 'elderly_vaccinated', 'males_vaccinated', 'females_vaccinated'])
    
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)
    
    return X_train, X_validate, X_test, y_train, y_validate, y_test

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def mean_baseline(X_train, X_validate, X_test, y_train, y_validate, y_test):
    '''
    Runs the mean baseline model.
    '''

    # Create a mean baseline
    AEFI_mean = y_train.AEFI.mean()
    y_train['AEFI_pred_mean'] = AEFI_mean
    y_validate['AEFI_pred_mean'] = AEFI_mean
    y_test['AEFI_pred_mean'] = AEFI_mean
    
    # RMSE of AEFI_pred_mean
    rmse_train = mean_squared_error(y_train.AEFI,
                                    y_train.AEFI_pred_mean) ** .5
    rmse_validate = mean_squared_error(y_validate.AEFI, 
                                       y_validate.AEFI_pred_mean) ** (1/2)

    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
          "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))
    
    return X_train, X_validate, X_test, y_train, y_validate, y_test, rmse_train, rmse_validate

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Make a table comparing metrics from the different models
def make_metric_df(y, y_pred, model_name, metric_df):
    '''
    Creates a table of metrics to compare models.
    '''
    if metric_df.size ==0:
        metric_df = pd.DataFrame(data=[
            {
                'model': model_name, 
                'RMSE_validate': mean_squared_error(
                    y,
                    y_pred) ** .5,
                'r^2_validate': explained_variance_score(
                    y,
                    y_pred)
            }])
        return metric_df
    else:
        return metric_df.append(
            {
                'model': model_name, 
                'RMSE_validate': mean_squared_error(
                    y,
                    y_pred) ** .5,
                'r^2_validate': explained_variance_score(
                    y,
                    y_pred)
            }, ignore_index=True)

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def lm_model(X_train, X_validate, X_test, y_train, y_validate, y_test, metric_df):
    '''
    Runs the linear model (OLS).
    '''
    
    # Linear regression, OLS
    
    # Create the object
    lm = LinearRegression(normalize=True)
    
    # Fit the object
    lm.fit(X_train, y_train.AEFI)
    
    # Use the object
    # Store predicted target values from linear model in new column
    y_train['AEFI_pred_lm'] = lm.predict(X_train)
    
    # Calculate RMSE for train
    rmse_train = mean_squared_error(y_train.AEFI, y_train.AEFI_pred_lm) ** (1/2)
    
    # Store predicted target values from linear model in new column
    y_validate['AEFI_pred_lm'] = lm.predict(X_validate)
    
    # Calculate RMSE for validate
    rmse_validate = mean_squared_error(y_validate.AEFI, y_validate.AEFI_pred_lm) ** (1/2)
    
    print("RMSE for LinearRegression(OLS)\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)
    
    # Add the linear model RMSE and R^2 values to metric_df
    metric_df = metric_df.append({
    'model': 'Linear Regression (OLS)', 
    'RMSE_validate': rmse_validate,
    'r^2_validate': explained_variance_score(y_validate.AEFI, y_validate.AEFI_pred_lm)}, ignore_index=True)

    return X_train, X_validate, X_test, y_train, y_validate, y_test, rmse_train, rmse_validate, metric_df

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def lasslars_model(X_train, X_validate, X_test, y_train, y_validate, y_test, metric_df):    
    '''
    Runs the LassoLars model.
    '''
    
    # Create the object
    lars = LassoLars(alpha=1)
    
    # Fit the object 
    # We must specify the column in y_train, 
    # because we have converted it to a dataframe from a series
    lars.fit(X_train, y_train.AEFI)
    
    # Use the object
    # Store predicted target values from LassoLars in new column
    y_train['AEFI_pred_lars'] = lars.predict(X_train)
    
    # Calculate RMSE for train
    rmse_train = mean_squared_error(y_train.AEFI, y_train.AEFI_pred_lars) ** (1/2)
    
    # Store predicted target values from LassoLars in new column
    y_validate['AEFI_pred_lars'] = lars.predict(X_validate)
    
    # Calculate RMSE for validate
    rmse_validate = mean_squared_error(y_validate.AEFI, y_validate.AEFI_pred_lars) ** (1/2)
    
    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)
    
    # Add the LassoLars model RMSE and R^2 values to metric_df
    metric_df = metric_df.append({
    'model': 'LassoLars', 
    'RMSE_validate': rmse_validate,
    'r^2_validate': explained_variance_score(y_validate.AEFI, y_validate.AEFI_pred_lars)}, ignore_index=True)

    return X_train, X_validate, X_test, y_train, y_validate, y_test, rmse_train, rmse_validate, metric_df

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def glm_model(X_train, X_validate, X_test, y_train, y_validate, y_test, metric_df):    
    '''
    Runs the generalized linear model.
    '''
    
    # Create the object
    glm = TweedieRegressor(power=1, alpha=0)
    
    
    # Fit the object 
    # We must specify the column in y_train, 
    # becuase we  converted it to a dataframe from a series
    glm.fit(X_train, y_train.AEFI)
    
    # Use the object
    # Store predicted target values from GLM in new column
    y_train['AEFI_pred_glm'] = glm.predict(X_train)
    
    # Calculate RMSE for train
    rmse_train = mean_squared_error(y_train.AEFI, y_train.AEFI_pred_glm) ** (1/2)
    
    # Store predicted target values from GLM in new column
    y_validate['AEFI_pred_glm'] = glm.predict(X_validate)
    
    # Calculate RMSE for validate
    rmse_validate = mean_squared_error(y_validate.AEFI, y_validate.AEFI_pred_glm) ** (1/2)
    
    print("RMSE for GLM using Tweedie\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)
    
    # Add the LassoLars model RMSE and R^2 values to metric_df
    metric_df = metric_df.append({
    'model': 'Generalized Linear Model', 
    'RMSE_validate': rmse_validate,
    'r^2_validate': explained_variance_score(y_validate.AEFI, y_validate.AEFI_pred_glm)}, ignore_index=True)

    return X_train, X_validate, X_test, y_train, y_validate, y_test, rmse_train, rmse_validate, metric_df



#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def best_model(X_train, X_validate, X_test, y_train, y_validate, y_test, metric_df):    
    '''
    Runs the best model on test data.
    '''

    # Bring back the best model
    lm = LinearRegression(normalize=True)
    lm.fit(X_train, y_train.AEFI)
    
    # Store predicted target values from LM in new column
    y_test['AEFI_pred_lm'] = lm.predict(X_test)
    
    # Calculate RMSE for validate
    rmse_test = mean_squared_error(y_test.AEFI, y_test.AEFI_pred_lm) ** (1/2)
    
    # Add the LassoLars model RMSE and R^2 values to metric_df
    metric_df = metric_df.append({
    'model': 'Linear Model (OLS) on Test Data', 
    'RMSE_validate': rmse_test,
    'r^2_validate': explained_variance_score(y_test.AEFI, y_test.AEFI_pred_lm)}, ignore_index=True)

    return X_train, X_validate, X_test, y_train, y_validate, y_test, rmse_test, metric_df

