#!/usr/bin/env python3
"""
Statistical analysis functions for crime data.
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

def crime_rate_by_area(df, area_column, population_column=None):
    """
    Calculate crime rates by area.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing crime data
    area_column : str
        Column name for geographical areas
    population_column : str, optional
        Column name for population data (to calculate per capita rates)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with crime counts and rates by area
    """
    # Count crimes by area
    crime_counts = df[area_column].value_counts().reset_index()
    crime_counts.columns = [area_column, 'crime_count']
    
    # Calculate rates if population data is available
    if population_column and population_column in df.columns:
        # Get unique area-population pairs
        population_data = df[[area_column, population_column]].drop_duplicates()
        
        # Merge with crime counts
        result = pd.merge(crime_counts, population_data, on=area_column)
        
        # Calculate per capita rate (per 100,000 people)
        result['crime_rate_per_100k'] = (result['crime_count'] / result[population_column]) * 100000
        
        return result
    else:
        return crime_counts

def predict_theft_crimes(df, date_column, crime_column=None, theft_type=None, prediction_periods=36):
    """
    Predict theft crimes for the next three years (36 months) using time series forecasting.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing crime data
    date_column : str
        Column name for dates
    crime_column : str, optional
        Column name for crime types
    theft_type : str, optional
        Specific theft type to predict (e.g., 'THEFT', 'BURGLARY', 'ROBBERY')
        If None, all theft-related crimes will be considered
    prediction_periods : int, default 36
        Number of periods (months) to predict
    
    Returns:
    --------
    dict
        Dictionary containing prediction results and model evaluation
    """
    # Import required libraries for prediction
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # Ensure date column is datetime type
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Filter for theft crimes if crime_column is provided
    if crime_column and crime_column in df.columns:
        if theft_type:
            # Filter for specific theft type
            theft_data = df[df[crime_column] == theft_type]
        else:
            # Filter for all theft-related crimes (customize based on your data)
            theft_keywords = ['THEFT', 'BURGLARY', 'ROBBERY', 'LARCENY', 'STOLEN']
            theft_mask = df[crime_column].str.contains('|'.join(theft_keywords), case=False)
            theft_data = df[theft_mask]
    else:
        # If no crime column specified, use all data (assuming it's all theft-related)
        theft_data = df
    
    # Create monthly time series
    monthly_series = pd.Series(
        index=theft_data[date_column],
        data=np.ones(len(theft_data))
    ).resample('M').sum().fillna(0)
    
    # Ensure we have enough data (at least 24 months)
    if len(monthly_series) < 24:
        return {
            'error': 'Not enough data for prediction. Need at least 24 months of data.',
            'data_points_available': len(monthly_series)
        }
    
    # Split data into train and test sets (80% train, 20% test)
    train_size = int(len(monthly_series) * 0.8)
    train, test = monthly_series[:train_size], monthly_series[train_size:]
    
    # If test set is empty, use last 20% of train for testing
    if len(test) == 0:
        train_size = int(len(monthly_series) * 0.8)
        train, test = monthly_series[:train_size], monthly_series[train_size:]
    
    # Try different models and select the best one
    models = {
        'ARIMA(1,1,1)': ARIMA(train, order=(1,1,1)),
        'ARIMA(2,1,2)': ARIMA(train, order=(2,1,2)),
        'SARIMAX(1,1,1)(1,1,1,12)': SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
    }
    
    best_model = None
    best_aic = float('inf')
    best_model_name = None
    model_results = {}
    
    for name, model in models.items():
        try:
            fitted_model = model.fit()
            aic = fitted_model.aic
            
            # Forecast on test set
            forecast = fitted_model.forecast(steps=len(test))
            
            # Calculate error metrics
            mse = mean_squared_error(test, forecast)
            mae = mean_absolute_error(test, forecast)
            
            model_results[name] = {
                'aic': aic,
                'mse': mse,
                'mae': mae,
                'model': fitted_model
            }
            
            # Update best model if this one is better
            if aic < best_aic:
                best_aic = aic
                best_model = fitted_model
                best_model_name = name
        except:
            model_results[name] = {
                'error': 'Model fitting failed'
            }
    
    # If no model could be fitted
    if best_model is None:
        return {
            'error': 'All models failed to fit the data',
            'model_results': model_results
        }
    
    # Refit the best model on the entire dataset
    if best_model_name.startswith('ARIMA'):
        order = best_model.model.order
        final_model = ARIMA(monthly_series, order=order).fit()
    else:  # SARIMAX
        order = best_model.model.order
        seasonal_order = best_model.model.seasonal_order
        final_model = SARIMAX(monthly_series, order=order, seasonal_order=seasonal_order).fit()
    
    # Generate future dates for prediction
    last_date = monthly_series.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=prediction_periods, freq='M')
    
    # Make predictions for the next three years
    forecast = final_model.forecast(steps=prediction_periods)
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'predicted_crimes': forecast
    })
    
    # Ensure predictions are non-negative
    forecast_df['predicted_crimes'] = forecast_df['predicted_crimes'].apply(lambda x: max(0, x))
    
    # Calculate confidence intervals
    if hasattr(final_model, 'get_forecast'):
        forecast_obj = final_model.get_forecast(steps=prediction_periods)
        conf_int = forecast_obj.conf_int()
        forecast_df['lower_ci'] = conf_int.iloc[:, 0].apply(lambda x: max(0, x))
        forecast_df['upper_ci'] = conf_int.iloc[:, 1].apply(lambda x: max(0, x))
    
    return {
        'historical_data': monthly_series,
        'forecast': forecast_df,
        'best_model': best_model_name,
        'model_summary': str(final_model.summary()),
        'model_results': model_results,
        'evaluation': {
            'aic': final_model.aic,
            'bic': final_model.bic
        }
    }

def time_series_analysis(df, date_column, crime_column=None, freq='M'):
    """
    Perform time series analysis on crime data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing crime data
    date_column : str
        Column name for dates
    crime_column : str, optional
        Column name for crime types (to analyze specific crime types)
    freq : str, default 'M'
        Frequency for time series aggregation ('D' for daily, 'M' for monthly, etc.)
    
    Returns:
    --------
    dict
        Dictionary containing time series analysis results
    """
    # Ensure date column is datetime type
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Filter by crime type if specified
    if crime_column and crime_column in df.columns:
        crime_types = df[crime_column].unique()
        results = {}
        
        for crime_type in crime_types:
            crime_data = df[df[crime_column] == crime_type]
            
            # Create time series
            time_series = pd.Series(
                index=crime_data[date_column],
                data=np.ones(len(crime_data))
            ).resample(freq).sum()
            
            # Fill missing periods with zeros
            time_series = time_series.fillna(0)
            
            # Perform analysis if we have enough data points
            if len(time_series) > 12:  # Need at least a year of data for seasonal analysis
                # Test for stationarity
                adf_result = adfuller(time_series)
                
                # Decompose time series
                try:
                    decomposition = seasonal_decompose(time_series, model='additive')
                    
                    results[crime_type] = {
                        'time_series': time_series,
                        'adf_test': {
                            'statistic': adf_result[0],
                            'p_value': adf_result[1],
                            'is_stationary': adf_result[1] < 0.05
                        },
                        'decomposition': {
                            'trend': decomposition.trend,
                            'seasonal': decomposition.seasonal,
                            'residual': decomposition.resid
                        }
                    }
                except:
                    # If decomposition fails (e.g., not enough data)
                    results[crime_type] = {
                        'time_series': time_series,
                        'adf_test': {
                            'statistic': adf_result[0],
                            'p_value': adf_result[1],
                            'is_stationary': adf_result[1] < 0.05
                        }
                    }
            else:
                results[crime_type] = {
                    'time_series': time_series,
                    'error': 'Not enough data points for seasonal analysis'
                }
                
        return results
    else:
        # Analyze all crimes together
        time_series = pd.Series(
            index=df[date_column],
            data=np.ones(len(df))
        ).resample(freq).sum()
        
        # Fill missing periods with zeros
        time_series = time_series.fillna(0)
        
        # Perform analysis if we have enough data points
        if len(time_series) > 12:
            # Test for stationarity
            adf_result = adfuller(time_series)
            
            # Decompose time series
            try:
                decomposition = seasonal_decompose(time_series, model='additive')
                
                return {
                    'all_crimes': {
                        'time_series': time_series,
                        'adf_test': {
                            'statistic': adf_result[0],
                            'p_value': adf_result[1],
                            'is_stationary': adf_result[1] < 0.05
                        },
                        'decomposition': {
                            'trend': decomposition.trend,
                            'seasonal': decomposition.seasonal,
                            'residual': decomposition.resid
                        }
                    }
                }
            except:
                return {
                    'all_crimes': {
                        'time_series': time_series,
                        'adf_test': {
                            'statistic': adf_result[0],
                            'p_value': adf_result[1],
                            'is_stationary': adf_result[1] < 0.05
                        }
                    }
                }
        else:
            return {
                'all_crimes': {
                    'time_series': time_series,
                    'error': 'Not enough data points for seasonal analysis'
                }
            }