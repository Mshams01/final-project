## main
import numpy as np
import pandas as pd
import os

## skelarn -- preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector

import warnings
warnings.filterwarnings('ignore')

## using pandas
TRAIN_DATA_PATH = os.path.join(os.getcwd(),'train.csv')
df = pd.read_csv(TRAIN_DATA_PATH)

## drop un important columns
df.drop(columns=df[['Unnamed: 0','id']],axis=1 ,inplace=True)

X=df.drop(columns=['satisfaction'],axis=1)
y = df['satisfaction']

# We use .map() on a specific column and provide a dictionary to perform the mapping
df['satisfaction'] = df['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied':1})

## split to train and valid
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, shuffle=True, random_state=35)

## Slicing cols according to their datatypes
num_cols = X_train.select_dtypes(include='number').columns.tolist()
categ_cols = X_train.select_dtypes(exclude='number').columns.tolist()


print (X_train.columns)
print ('**'*20)


print (num_cols)
print ('**'*20)
print (categ_cols)
print ('**'*20)

print(X_train.iloc[0:1, :])


## My Pipline:
## Numerical: imputing using (median) and  then standardize with  StandardScaler()
## categorical:imputing using (mode) and label encoding


## Numerical pipeline
num_pipeline = Pipeline(steps=[
                    ('selector', DataFrameSelector(num_cols)),
                    ('imputer', SimpleImputer(strategy='median')),
                    ('standardize', StandardScaler())
                    ])

## Categorical Pipeline
categ_pipeline = Pipeline(steps=[
                        ('selector', DataFrameSelector(categ_cols)),
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('encder', OrdinalEncoder())
                    ])

## Get all together
all_pipeline = FeatureUnion(transformer_list=[
                                ('numerical', num_pipeline),
                                ('category', categ_pipeline)
                                ])

## applY
_=all_pipeline.fit(X_train)


def process_new(X_new):
    df_new=pd.DataFrame([X_new])
    df_new.columns = X_train.columns
    
    df_new['Gender']=df_new['Gender'].astype('str')
    df_new['Customer Type']=df_new['Customer Type'].astype('str')
    df_new['Age']=df_new['Age'].astype('float')
    df_new['Type of Travel']=df_new['Type of Travel'].astype('str')
    df_new['Class']=df_new['Class'].astype('str') 
    df_new['Flight Distance']=df_new['Flight Distance'].astype('str') 
    df_new['Inflight wifi service']=df_new['Inflight wifi service'].astype('int') 
    df_new['Departure/Arrival time convenient']=df_new['Departure/Arrival time convenient'].astype('int')
    df_new['Ease of Online booking']=df_new['Ease of Online booking'].astype('int')
    df_new['Gate location']=df_new['Gate location'].astype('int')
    df_new['Food and drink']=df_new['Food and drink'].astype('int')
    df_new['Online boarding']=df_new['Online boarding'].astype('int')
    df_new['Seat comfort']=df_new['Seat comfort'].astype('int')
    df_new['Inflight entertainment']=df_new['Inflight entertainment'].astype('int')
    df_new['On-board service']=df_new['On-board service'].astype('int')
    df_new['Leg room service']=df_new['Leg room service'].astype('int')
    df_new['Baggage handling']=df_new['Baggage handling'].astype('int')
    df_new['Checkin service']=df_new['Checkin service'].astype('int')
    df_new['Inflight service']=df_new['Inflight service'].astype('int')
    df_new['Cleanliness']=df_new['Cleanliness'].astype('int')
    df_new['Departure Delay in Minutes']=df_new['Departure Delay in Minutes'].astype('float')
    df_new['Arrival Delay in Minutes']=df_new['Arrival Delay in Minutes'].astype('float')
  
    
    
    
    
    Tr_new=all_pipeline.transform(df_new)
     
    return Tr_new