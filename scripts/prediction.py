import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

def crime_rate_by_area(df, area_column, population_column=None):
    "Calculate crime rates by area."

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

def time_series_analysis(df, date_column, crime_column=None, freq='M'):
    "Perform time series analysis on crime data. "
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