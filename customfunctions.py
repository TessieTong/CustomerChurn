#!/usr/bin/env python
# coding: utf-8

# Custom function for transforming data in data pipeline.

import pandas as pd
import numpy as np

## combine non-frequent categories into one
def combine_cat(s, cutoff=6, replace=6):
    """ Replace the categories that are greater than or equal to the cutoff with replace
        s: a Pandas series.
        cutoff: a scalar.
        cat: scalar.
        
        return: a series with replaced values.
    """
    s_ = s.copy()
    mask = s > cutoff
    s_[mask==True] = replace
    return s_

# balance, debit, credit, days_since_last_transaction
def log_transform(df):
    """ Log transform the values in the df.
            for values < 0, log tranform absolute value, and then reverse to negative.
        df: a panda dataframe or array like.
        return: a pandas dataframe with log transformed values.
    """
    df_ = pd.DataFrame(df.copy())
    s_= []
    for i in range(df.shape[1]):
        s_.append([np.log1p(x) if x>=0 else -np.log1p(-x) for x in df_.iloc[:,i]])
    s_ = pd.DataFrame.from_records(s_).transpose()
    return s_

## balance features
def impute_balance(df):
    """ Fill in missing values in each column with the average value of other columns in the same row
        df: a panda dataframe. To be imputed.
        
        return: a pandas dataframe without missing values
    """
    s_= []
    for i in range(df.shape[1]):
        s = df.iloc[:,i] 
        s_.append(s.fillna(df.mean(axis=1,skipna=True)))
    s_ = pd.DataFrame.from_records(s_).transpose()
    return s_

# debit and credit features
def impute_credit_debit(df):
    """ Fill in missing values in each column with the average value of other columns in the same row
            then add some randomness to the replacement value.
        df: a panda dataframe. To be imputed.
        
        return: a pandas dataframe without missing values
    """
    s_= []
    for i in range(df.shape[1]):
        s = df.iloc[:,i] 
        s_.append(s.fillna(df.mean(axis=1)*(1+np.random.randn())))
    s_ = pd.DataFrame.from_records(s_).transpose()
    return s_

## Engineer new features 
# percentage changes
def calculate_pct_change(df):
    """ calculate percent changes in balance between consecutive periods
        df: Pandas dataframe or array. Balance columns of two consecutive peroids
        
        return: a dataframe containing percent changes with one less number of columns.
    """
    df_ = pd.DataFrame(df.copy())
    s_ = []
    for i in range(df_.shape[1]-1):
        s1 = df_.iloc[:,i]
        s2 = df_.iloc[:,i+1]
        s_.append((s1-s2)/(s2+1)*100) # s2+1 to avoid dividing-by-zero
    df_ = pd.DataFrame.from_records(s_).transpose()
    return df_ 

# vintage/(day_since_last_transaction) & per person values in a household
def calculate_ratio(df):
    """ calculate the ratio of two features. First column is denominator
        df: Pandas dataframe or numpy array.
        
        return: a dataframe containing ratio with one less number of columns.
    """
    s_ = []
    df_ = pd.DataFrame(df.copy())
    s1 = df_.iloc[:,0]
    for i in range(1,df_.shape[1]):
        s2 = df_.iloc[:,i]
        s_.append(s2/(s1+1)) #to avoid dividing by zero
    df_ = pd.DataFrame.from_records(s_).transpose()
    return df_ 

# vintage_age score 1 - equal distance
def calculate_vintage_age_score_eqdist(df):
    """ Calculate vintage-age combined score with equal-distance bins (pd.cut)
        df: an array of shape (*,2) or a dataframe
            df.shape[0]: for vintage column
            df.shape[1]: for age column
        
        return: a 2D array (shape (*,1)) withe the scores
    """
    # df = df[['vintage','age']]
    # Vintage
    df_ = pd.DataFrame(df.copy())
    mask = df_.iloc[:,0] >= 7000
    df_['vintage_score'] = df_.iloc[:,0]//1000 + 1
    df_['vintage_score'][mask] = 8
    
    # age
    cut_score = [1, 2, 3, 4, 5, 6, 7, 8]
    cut_bins = [0, 10, 19, 29, 39, 49, 59, 69, 100]
    df_['age_score'] = pd.cut(df_.iloc[:,1], bins=cut_bins, labels=cut_score).astype(int)
    df_['vintage_age_score'] = df_['vintage_score'] * df_['age_score']
    # return 2D arrage required
    return df_['vintage_age_score'].values.reshape(-1,1)

# vintage_age score 2 - equal population
def calculate_vintage_age_score_eqdens(df):
    """ Calculate vintage-age combined score with equal-population bins (pd.qcut)
        df: an array of shape (*,2) or a dataframe
            df.shape[0]: for vintage column
            df.shape[1]: for age column
        
        return: a 2D array (shape (*,1)) withe the scores
    """
    # df = df[['vintage','age']]
    df_ = pd.DataFrame(df.copy())
    cut_score = [1, 2, 3, 4, 5, 6, 7, 8]
    df_['vintage_score'] = pd.qcut(df_.iloc[:,0], q=8, labels=cut_score).astype(int)
    df_['age_score'] = pd.qcut(df_.iloc[:,1], q=8, labels=cut_score).astype(int)
    df_['vintage_age_score'] = df_['vintage_score'] * df_['age_score']
    # return 2D arrage required
    return df_['vintage_age_score'].values.reshape(-1,1)

