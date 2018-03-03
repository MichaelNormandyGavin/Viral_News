import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd

def add_sent_max(df,normalize=False,drop_columns=False,normalize=False):

	#This function is to select the most volatile sentiment from either the Title or Headline columns

    SentimentMax = []

    for index, row in news_final.iterrows():
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
	assert isinstance(df.index,pd.DateTimeIndex), "Make sure to use time series!"
	
	baseline = df.head(1).index
	
	if drop_all:
		drop_alpha, drop_nums = True, True
		
	numeric_columns = list(df.select_dtypes(include=['int64','float64']).columns)
	alpha_columns = list(df.select_dtypes(include=['int64','float64']).columns)
	
	for c in numeric_columns:
		df['trend_{}'.format(c)] = df[c].rolling(int(periods)).mean()
		if normalize:
			df['trend_{}'.format(c)] = df['trend_{}'.format(c)].div.loc[baseline,c].sub(1).mul(100)
	
	if drop_alpha:
		df = df.drop(alpha_columns,index=1)
		
	if drop_nums:
		df = df.drop(numeric_columns,index=1)
	
	return df
		
	
def trendify_median(df,normalize=False,periods=7,drop_alpha=False,drop_nums=False):
	#add trend columns to prepare for visualization
	
	assert isinstance(df,pd.DataFrame), "Please use a dataframe!"
	assert isinstance(df.index,pd.DateTimeIndex), "Make sure to use time series!"
	
	baseline = df.head(1).index
	
	if drop_all:
		drop_alpha, drop_nums = True, True
		
	numeric_columns = list(df.select_dtypes(include=['int64','float64']).columns)
	alpha_columns = list(df.select_dtypes(include=['int64','float64']).columns)
	
	for c in numeric_columns:
		df['trend_{}'.format(c)] = df[c].rolling(int(periods)).median()
		if normalize:
			df['trend_{}'.format(c)] = df['trend_{}'.format(c)].div.loc[baseline,c].sub(1).mul(100)
	
	if drop_alpha:
		df = df.drop(alpha_columns,index=1)
		
	if drop_nums:
		df = df.drop(numeric_columns,index=1)
	
	return df
	
def trendify_std(df,normalize=False,periods=7,drop_alpha=False,drop_nums=False):
	#add trend columns to prepare for visualization
	
	assert isinstance(df,pd.DataFrame), "Please use a dataframe!"
	assert isinstance(df.index,pd.DateTimeIndex), "Make sure to use time series!"
	
	baseline = df.head(1).index
	
	if drop_all:
		drop_alpha, drop_nums = True, True
		
	numeric_columns = list(df.select_dtypes(include=['int64','float64']).columns)
	alpha_columns = list(df.select_dtypes(include=['int64','float64']).columns)
	
	for c in numeric_columns:
		df['trend_{}'.format(c)] = df[c].rolling(int(periods)).std()
		if normalize:
			df['trend_{}'.format(c)] = df['trend_{}'.format(c)].div.loc[baseline,c].sub(1).mul(100)
	
	if drop_alpha:
		df = df.drop(alpha_columns,index=1)
		
	if drop_nums:
		df = df.drop(numeric_columns,index=1)
	
	return df