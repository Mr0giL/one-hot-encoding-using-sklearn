import numpy as np
import pandas as pd
data = pd.read_csv('5th data.csv')
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop='first')
x = ohe.fit_transform(data['Car Model'].values.reshape(-1,1)).toarray()
data_int = data.drop('Car Model',axis =1)
print(data['Car Model'].value_counts())
data_int[['Car Model_BMW X5 ','Car Model_Mercedez Benz C class']] = x.tolist()
data_int.to_csv('ohe-concancated-data.csv')
print(data_int)