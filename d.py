# pip install pandas
# pip install Keras
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

Diabetic_df=pd.read_csv(r'C:\Users\farza\OneDrive\Documents\Desktop\diabetes\diabetes_prediction_dataset.csv')

Diabetic_df=Diabetic_df.select_dtypes(include=(['int64','float64']))
Q1= Diabetic_df.quantile(0.25)
Q3=Diabetic_df.quantile(0.75)
IQR=Q3-Q1

Lowerbound=(Q1-15*IQR)

Upperbound=(Q3+1.5*IQR)
outliers= (Diabetic_df < Lowerbound) | (Diabetic_df > Upperbound)

Diabetic_df=pd.DataFrame(Diabetic_df)
Lowerlimit=5
Upperlimit=95
columns_to_winsorize = ['age','hypertension','heart_disease','HbA1c_level','bmi','blood_glucose_level']
for column in columns_to_winsorize:
    Lowerbound = np.percentile(Diabetic_df[column], Lowerlimit)
    Upperbound = np.percentile(Diabetic_df[column], Upperlimit)
    Diabetic_df[column] = np.where(Diabetic_df[column] < Lowerbound , Lowerbound ,Diabetic_df[column])
    Diabetic_df[column] = np.where(Diabetic_df[column] > Upperbound , Upperbound ,Diabetic_df[column])


#standard scale split values in between 0 to 9
from sklearn.preprocessing import StandardScaler
x=Diabetic_df.drop('diabetes', axis = 1)
y= Diabetic_df['diabetes']
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(x)
X_test_scaled = scaler.transform(x)

#After standard scaling, spliting the dataset into test and train go give it to model and test and train variable should be used as given in scalling code.
from sklearn.model_selection import train_test_split
x_train_scaled, x_test_scaled, y_train, y_test =train_test_split(x,y, test_size=.20, random_state = 101)


# from imblearn.over_sampling import SMOTE

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

smote=SMOTE(random_state=27)
smote_x_train, smote_y_train= smote.fit_resample(x_train_scaled, y_train)
print("Before Sampling class distribution: ", (y_train.count))
print("After sampling class distribution: ",(smote_y_train.count))



X_train_scaled = scaler.fit_transform(x)
X_test_scaled = scaler.transform(x)

#After standard scaling, spliting the dataset into test and train go give it to model and test and train variable should be used as given in scalling code.
from sklearn.model_selection import train_test_split
x_train_scaled, x_test_scaled, y_train, y_test =train_test_split(smote_x_train, smote_y_train, test_size=.20, random_state = 101)



#Import RandomForest model
from sklearn.ensemble import RandomForestClassifier
Rf =  RandomForestClassifier()

Rf.fit(smote_x_train, smote_y_train)

# ACCURACY SCORE OF THE MODEL

from sklearn.metrics import accuracy_score
y_pred = Rf.predict(x_test_scaled)
y_binary_prediction = np.round(y_pred)
test_accuracy = accuracy_score(y_test, y_binary_prediction)
print("Accuracy score is: ", test_accuracy)

# PRECISION OF THE MODEL
from sklearn.metrics import precision_score
y_pred = Rf.predict(x_test_scaled)
y_binary_prediction = np.round(y_pred)
test_Precision = precision_score(y_test, y_binary_prediction)
print("Precision score is: ", test_Precision)

# RECALL SCORE OF THE MODEL

from sklearn.metrics import recall_score
y_pred = Rf.predict(x_test_scaled)
y_binary_prediction = np.round(y_pred)
test_accuracy = recall_score(y_test, y_binary_prediction)
print("recall score is: ", test_accuracy)

from sklearn.metrics import confusion_matrix
y_pred = Rf.predict(x_test_scaled)
y_binary_prediction = np.round(y_pred)
test_accuracy = confusion_matrix(y_test, y_binary_prediction)
print("Confusion_matrix score is: ", test_accuracy)

plt.figure(figsize=(5,3))
sns.heatmap(test_accuracy , cmap= 'plasma', annot=True)
plt.show()

print(Diabetic_df.head())


#joblib.dump to save a trained model to a file after training it
import joblib
joblib.dump(Rf, 'RandomForest_Model.pkl')

# %%
# # %%
# #Fit
# #Import LOGISTIC REGRESSION model
# from sklearn.linear_model import LogisticRegression
# Ireg = LogisticRegression()

# Ireg.fit(smote_x_train, smote_y_train)
# # %%
# # PRECISION OF THE MODEL
# from sklearn.metrics import precision_score
# y_pred = Ireg.predict(x_test_scaled)
# y_binary_prediction = np.round(y_pred)
# test_Precision = precision_score(y_test, y_binary_prediction)
# print("Precision score is: ", test_Precision)
# # %%
# # ACCURACY SCORE OF THE MODEL

# from sklearn.metrics import accuracy_score
# y_pred = Ireg.predict(x_test_scaled)
# y_binary_prediction = np.round(y_pred)
# test_accuracy = accuracy_score(y_test, y_binary_prediction)
# print("Accuracy score is: ", test_accuracy)
# # %%
# # RECALL SCORE OF THE MODEL

# from sklearn.metrics import recall_score
# y_pred = Ireg.predict(x_test_scaled)
# y_binary_prediction = np.round(y_pred)
# test_accuracy = recall_score(y_test, y_binary_prediction)
# print("recall score is: ", test_accuracy)
# # %%
# # F1 SCORE OF THE MODEL

# from sklearn.metrics import f1_score

# y_pred = Ireg.predict(x_test_scaled)
# y_binary_prediction = np.round(y_pred)
# test_f1_score = f1_score(y_test, y_binary_prediction)
# print("F1 score is:", test_f1_score)

# # %%
# from sklearn.metrics import confusion_matrix
# y_pred = Ireg.predict(x_test_scaled)
# y_binary_prediction = np.round(y_pred)
# test_accuracy = confusion_matrix(y_test, y_binary_prediction)
# print("Confusion_matrix score is: ", test_accuracy)

# plt.figure(figsize=(5,3))
# sns.heatmap(test_accuracy , cmap= 'plasma', annot=True)
# plt.show()