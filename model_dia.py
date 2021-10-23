import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# %matplotlib inline
# dia_data = pd.read_csv("diabetes.csv")
# dia_data.shape
# dia_data.head()
# dia_data.info()
# dia_data.shape
# dia_data.isnull().sum()
# dia_data.duplicated().sum()
# # droping duplicate data
# dia_data.drop_duplicates(inplace = True)
# dia_data.shape
# print("total no of rows :: {} ".format(len(dia_data)))
# print("total no of rows missing Pregnancies :: {} ".format(len(dia_data.loc[dia_data['Pregnancies'] == 0])))
# print("total no of rows missing glucose :: {} ".format(len(dia_data.loc[dia_data['Glucose'] == 0])))
# print("total no of rows missing bp :: {} ".format(len(dia_data.loc[dia_data['BloodPressure'] == 0])))
# print("total no of rows missing insulin :: {} ".format(len(dia_data.loc[dia_data['Insulin'] == 0])))
# print("total no of rows missing SkinThickness :: {} ".format(len(dia_data.loc[dia_data['SkinThickness'] == 0])))
# print("total no of rows missing DiabetesPedigreeFunction :: {} ".format(len(dia_data.loc[dia_data['DiabetesPedigreeFunction'] == 0])))
# print("total no of rows missing bmi :: {} ".format(len(dia_data.loc[dia_data['BMI'] == 0])))
# print("total no of rows missing age :: {} ".format(len(dia_data.loc[dia_data['Age'] == 0])))
# print("total no of rows missing Pregnancies :: {} ".format(len(dia_data.loc[dia_data['Pregnancies'] == 0])))

# fig , s= plt.subplots(3,2, figsize = (15,10))
# s[0][0].set_title("Histogram of pregnancies column")
# s[1][0].set_title("Histogram of Glucose column")
# s[2][0].set_title("Histogram of BloodPressure column")
# s[0][1].set_title("Histogram of Insulin column")
# s[1][1].set_title("Histogram of SkinThickness column")
# s[2][1].set_title("Histogram of BMI column")

# s[0][0].hist(dia_data['Pregnancies'], rwidth = 0.8)
# s[1][0].hist(dia_data['Glucose'], rwidth = 0.8)
# s[2][0].hist(dia_data['BloodPressure'], rwidth = 0.8)
# s[0][1].hist(dia_data['Insulin'] ,rwidth = 0.8)
# s[1][1].hist(dia_data['SkinThickness'],rwidth = 0.8)
# s[2][1].hist(dia_data['BMI'], rwidth = 0.8)
# plt.show()

# plt.figure(figsize=(15,5))
# sns.scatterplot(x= 'Age',y= 'Pregnancies', hue = 'Outcome', data = dia_data)
# plt.show()

# plt.figure(figsize=(15,5))
# sns.scatterplot(x= 'Age',y= 'Glucose', hue = 'Outcome', data = dia_data)
# plt.show()

# plt.figure(figsize=(15,5))
# sns.scatterplot(x= 'Age',y= 'BloodPressure', hue = 'Outcome', data = dia_data)
# plt.show()

# plt.figure(figsize=(15,5))
# sns.scatterplot(x= 'Age',y= 'SkinThickness', hue = 'Outcome', data = dia_data)
# plt.show()

# plt.figure(figsize=(15,5))
# sns.scatterplot(x= 'Age',y= 'Insulin', hue = 'Outcome', data = dia_data)
# plt.show()

# plt.figure(figsize=(15,5))
# sns.scatterplot(x= 'Age',y= 'BMI', hue = 'Outcome', data = dia_data)
# plt.show()

# plt.figure(figsize=(15,5))
# sns.scatterplot(x= 'Age',y= 'DiabetesPedigreeFunction', hue = 'Outcome', data = dia_data)
# plt.show()

# corr_data = dia_data.corr() # correlated metrics
# top_corr_features = corr_data.index

# corr_data

# top_corr_features

# plt.figure(figsize = (8,5))
# # annot is used to show each values
# # cmap is used for color map on the graph
# sns.heatmap(corr_data, annot = True, cmap = 'RdYlGn')

from sklearn.model_selection import train_test_split
# features set (independent data)
features_column = list(dia_data.iloc[:,:-1].columns)
predicted_column = ['Outcome']
print("features columns :: {} \n predicted columns :: {}".format(features_column,predicted_column))

X = dia_data[features_column].values
y = dia_data[predicted_column].values

# print("features columns :: {} \n predicted columns :: {}".format(X,y))

X.shape  , y.shape

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 10)

x_train.shape, y_train.shape  , x_test.shape, y_test.shape

# dia_data.head()

from sklearn.impute import SimpleImputer
fill_null_value = SimpleImputer(missing_values = 0, strategy = 'mean')

x_train = fill_null_value.fit_transform(x_train)
x_test = fill_null_value.fit_transform(x_test)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 10, random_state = 10)

# train the model 
rf.fit( x_train , y_train.ravel())

# Predicting values from the model 
y_pred = rf.predict(x_test)

y_pred = np.array([0 if i < 0.5 else 1 for i in y_pred])

y_pred.shape, y_test.ravel().shape

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

from sklearn.metrics import classification_report
class_report = classification_report(y_test, y_pred)

print("confusion matrix :: {} \n\n Accuracy = {} \n\n classification report :: \n{}".format(cm,acc,class_report))

import pickle 
pickle_out = open("classifier.pkl", mode = "wb") 
pickle.dump(rf, pickle_out) 
pickle_out.close()

# %%writefile app.py
 
import pickle
import streamlit as st
 
# loading the trained model
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)
 
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):   
 
    # Pre-processing user input    

    
 
    # Making predictions 
    prediction = classifier.predict( 
        [[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
     
    if prediction == 0:
        pred = 'not diabeteic'
    else:
        pred = 'diabeteic'
    return pred
      
      # this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">diabetes_detection App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    Pregnancies=st.number_input("number of pregnancies")
    Glucose=st.number_input("glucose amount")
    BloodPressure=st.number_input("blood pressure amount")
    SkinThickness=st.number_input("skin thickness")
    Insulin=st.number_input("insulin amount")
    BMI=st.number_input("BMI")
    DiabetesPedigreeFunction=st.number_input("genitic history")
    Age=st.number_input("age")
    Outcome =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age) 
        st.success('The patient {}'.format(Outcome))
     
if __name__=='__main__': 
    main()

from pyngrok import ngrok
 
public_url = ngrok.connect('8501')
public_url
# !streamlit run app.py &>/dev/null&
