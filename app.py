%%writefile app.py
 
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
        pred = 'not diabetic'
    else:
        pred = 'diabetic'
    return pred
      
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> 
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
