import re
import requests

import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd

from itertools import chain
from collections import Counter

pattern = r'[F|L|G]\w+_\w+'

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00432/Data/'

def url_find(url=url,pattern=pattern):
	u = requests.get(url)
	url_list = sorted(list(set(re.findall(pattern,u.text))))
	return url_list

def df_read(file,url=url):
	full_path = url+file+'.csv'
	platform, topic = file.split('_')
	df = pd.read_csv(full_path,index_col='IDLink')
	df.index = df.index.astype('int64').astype('object')
	df['Platform'] = platform
	df['Topic'] = topic
	df['Platform_Topic']= file
	return df