import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

milkData = pd.read_csv(r"C:\\Users\\Lenovo\\Downloads\\milknew.csv")

milkData.describe()
milkData.shape

milkData.dtypes

milkData.dropna()
milkData.shape
milkData.duplicated().sum()

milkData.loc[milkData.duplicated(),:]

milkData.isnull().sum()

milkData.nunique()

milkData.info

milkData1= pd.DataFrame(milkData)

milkData2=milkData.drop(['Grade'], axis=1)

milkData2
milkData['Grade'].replace({'high':2,'medium':1,'low':0},inplace=True)
milkData.head()

cols = milkData.columns[:6]
densityplot = milkData[cols].plot(kind='density')

milkData.Grade.value_counts()

import seaborn as sns
import matplotlib.pyplot as plt

milkData.corr()

f, axes = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(milkData.corr(), vmin = -1, vmax = 1,  linewidths = 1,
           annot = True, fmt = ".2f", annot_kws = {"size": 14}, cmap = "bwr")

x=milkData.drop(['Grade'],axis=1)
y=milkData['Grade']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

X_train

from sklearn.metrics import confusion_matrix  
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,max_depth=3,random_state=0)
model.fit(X_train,y_train)
prediction = model.predict(X_test)
confusionmatrix = confusion_matrix(y_test,prediction)
print(confusionmatrix)

print(accuracy_score(y_test,prediction))


from tensorflow.keras.models import save_model

# Save the model to a folder
save_model(model, 'C:\\Users\\Lenovo\\Downloads\\milk\\milk\\random_forest_model.h5')
