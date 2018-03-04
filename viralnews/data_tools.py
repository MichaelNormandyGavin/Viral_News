import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd

def add_sent_max(df,normalize=False,drop_columns=False):

	#This function is to select the most volatile sentiment from either the Title or Headline columns

    SentimentMax = []

    for index, row in df.iterrows():
        max_sentiment = row['SentimentTitle'] if abs(row['SentimentTitle']) > abs(row['SentimentHeadline']) else row['SentimentHeadline']
        SentimentMax.append(max_sentiment)
    
    df['SentimentMax'] = SentimentMax
    
    if normalize:
        df['SentimentMax'] = df['SentimentMax'].div(df['SentimentMax'].mean())
        
    if drop_columns:
        df = df.drop(['SentimentTitle','SentimentHeadline'],axis=1)
        
    return df
	
def trendify_mean(df,normalize=False,periods=7,drop_all=False,drop_alpha=False,drop_nums=False):

	#add trend columns to prepare for visualization
	
	assert isinstance(df,pd.DataFrame), "Please use a dataframe!"
	assert isinstance(df.index,pd.DatetimeIndex), "Make sure to use time series!"
	
	baseline = df.head(1).index

	periods = int(periods)
	
	if drop_all:
		drop_alpha, drop_nums = True, True
		
	numeric_columns = list(df.select_dtypes(include=['int64','float64']).columns)
	alpha_columns = list(df.select_dtypes(exclude=['int64','float64']).columns)
	
	for c in numeric_columns:
		trend_c = 'trend_{}'.format(c)

		df[trend_c] = df[c].rolling(periods).mean()

		if normalize:
			norm_c = 'norm_'+trend_c

			df[norm_c] = df[trend_c].div(df[trend_c].values[periods-1]).sub(1).mul(100)
			df = df.drop(trend_c,axis=1)
	
	if drop_alpha:
		df = df.drop(alpha_columns,index=1)
		
	if drop_nums:
		df = df.drop(numeric_columns,index=1)
	
	return df
		
	
def trendify_median(df,normalize=False,periods=7,drop_alpha=False,drop_nums=False):
	#add trend columns to prepare for visualization
	
	assert isinstance(df,pd.DataFrame), "Please use a dataframe!"
	assert isinstance(df.index,pd.DatetimeIndex), "Make sure to use time series!"
	
	baseline = df.head(1).index

	periods = int(periods)
	
	if drop_all:
		drop_alpha, drop_nums = True, True
		
	numeric_columns = list(df.select_dtypes(include=['int64','float64']).columns)
	alpha_columns = list(df.select_dtypes(exclude=['int64','float64']).columns)
	
	for c in numeric_columns:
		trend_c = 'trend_{}'.format(c)

		df[trend_c] = df[c].rolling(periods).median()

		if normalize:
			norm_c = 'norm_'+trend_c

			df[norm_c] = df[trend_c].div(df[trend_c].values[periods-1]).sub(1).mul(100)
			df = df.drop(trend_c,axis=1)
	
	if drop_alpha:
		df = df.drop(alpha_columns,index=1)
		
	if drop_nums:
		df = df.drop(numeric_columns,index=1)
	
	return df
	
def trendify_std(df,normalize=False,periods=7,drop_alpha=False,drop_nums=False):
	#add trend columns to prepare for visualization
	
	assert isinstance(df,pd.DataFrame), "Please use a dataframe!"
	assert isinstance(df.index,pd.DatetimeIndex), "Make sure to use time series!"
	
	baseline = df.head(1).index

	periods = int(periods)
	
	if drop_all:
		drop_alpha, drop_nums = True, True
		
	numeric_columns = list(df.select_dtypes(include=['int64','float64']).columns)
	alpha_columns = list(df.select_dtypes(exclude=['int64','float64']).columns)
	
	for c in numeric_columns:
		trend_c = 'trend_{}'.format(c)

		df[trend_c] = df[c].rolling(periods).std()

		if normalize:
			norm_c = 'norm_'+trend_c

			df[norm_c] = df[trend_c].div(df[trend_c].values[periods-1]).sub(1).mul(100)
			df = df.drop(trend_c,axis=1)
	
	if drop_alpha:
		df = df.drop(alpha_columns,index=1)
		
	if drop_nums:
		df = df.drop(numeric_columns,index=1)
	
	return df