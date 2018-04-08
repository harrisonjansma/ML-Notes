import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
#Remove irrelevent features, fill NaN, reset index, set dtypes
titanic = pd.read_csv('titanic_train.csv')

#Set index
titanic = titanic.set_index('PassengerId')

#Split name into Surname and Title for family tracking
titanic['Name_Mod'] = titanic.Name.apply(lambda x: x.split())
titanic['Surname'] = titanic.Name_Mod.apply(lambda x: x[0][:-1])
titanic['Title'] = titanic.Name_Mod.apply(lambda x: x[1])
titanic = titanic.drop(['Ticket', 'Name', 'Name_Mod', 'Cabin'], axis = 1)

titanic.info()
# drop the NaN Values. (Could find a way to impute, but that is an exercise for another day)
titanic  = titanic.dropna()
titanic.info()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
titanic.Sex = le.fit_transform(titanic.Sex)
titanic.Embarked = le.fit_transform(titanic.Embarked)
titanic.head()


from scipy import stats
OutTitanic = titanic[(np.abs(stats.zscore(titanic[['Age','Fare']])) < 3).all(axis=1)]
