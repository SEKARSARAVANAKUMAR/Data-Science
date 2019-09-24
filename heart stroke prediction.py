# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 16:16:35 2018

@author: vbcsaran
"""

#import data set 
import pandas as pd 


#Before_pre_processing_dataset
df=pd.read_csv("C:\\Users\\vbcsaran\\Documents\\heart stroke dataset\\train_2v.csv")
df.info()
df.describe(include="all")
df["age"].median()



#Remove_the_null_values
df.isnull().sum()
df=df.dropna()
df.isnull().sum()

#Over_view_about_dataset
df.head()
df.describe(include="all")



#Exploratory_data_analysis
import seaborn as sb

sb.countplot(x="stroke",hue="gender",data=df)

sb.countplot(x="stroke" ,hue="ever_married",data=df)

sb.countplot(x="stroke" ,hue="work_type",data=df)

sb.countplot(x="stroke" ,hue="Residence_type",data=df)

sb.countplot(x="stroke" ,hue="smoking_status",data=df)


sb.boxplot(x="stroke",y="avg_glucose_level",hue="gender",data=df)

sb.boxplot(x="stroke",y="avg_glucose_level",hue="ever_married",data=df)

sb.boxplot(x="stroke",y="avg_glucose_level",hue="Residence_type",data=df)

sb.boxplot(x="stroke",y="avg_glucose_level",hue="smoking_status",data=df)

sb.boxplot(x="stroke",y="avg_glucose_level",hue="work_type",data=df)
df_wt=df[df.work_type == "Govt_job"]  
df_wt_p=df_wt[df_wt.stroke == 1]
df_wt_p["avg_glucose_level"].median()




#Data_wrangling
df["gender"]=df["gender"].replace(["Male","Female","Other"],(1,0,2))

df["ever_married"]=df["ever_married"].replace(["Yes","No"],(1,0))

df["work_type"]=df["work_type"].replace(["Private","Self-employed","Govt_job","children","Never_worked"],(1,2,3,4,5))

df["Residence_type"]=df["Residence_type"].replace(["Urban","Rural"],(1,0))

df["smoking_status"]=df["smoking_status"].replace(["smokes","never smoked","formerly smoked"],(0,1,2))


df.head()


#model_Selection


#model1
x=df[["gender","age","hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level","bmi","smoking_status"]]
y=df[["stroke"]]


#normalize
#from sklearn import preprocessing
#x=preprocessing.StandardScaler().fit(x).transform(x)


#train _tes 
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=4)


#model
from sklearn.linear_model import LogisticRegression 


#declare_object_and_fit_the_model
LR = LogisticRegression().fit(x,y) 

#Prediction
LR_Prediction = LR.predict([[0,70,0,0,1,1,0,69,35,2]])
LR_Prediction


#Prediction_with_Probability
Prediction_Probability=LR.predict_proba([[0,70,0,0,1,1,0,69,35,2]])
Prediction_Probability



#Behind_of_Statistics
import pandas
import numpy as np

lm_co=LR.coef_
lm_coef=pandas.DataFrame(l)

lm_intercept=LR.intercept_

#formula=1/1+e^-(b0+b1*x1+b2*x2..n)
#here we are applying sigmoid/logistic formula using mathematical calculation
sigmoid_process_1=(lm_intercept+(lm_coef[0]*0)+(lm_coef[1]*70)+(lm_coef[2]*0)+(lm_coef[3]*0)+(lm_coef[4]*1)+(lm_coef[5]*1)+(lm_coef[6]*0)+(lm_coef[7]*69)+(lm_coef[8]*35)+(lm_coef[9]*2))
sigmoid_process_2=2.71838**(-sigmoid_process_1) #2.71838 is e value
sigmoid_process_3=sigmoid_process_2+1
prediction=1/sigmoid_process_3
prediction_output=np.array([[prediction]])



if prediction_output > 0.5:
    print("Positive")
else:
    print("Negative")
print("chance of getting heart stroke probability is ",prediction_output)



#Model_evolution
LR.score(x,y)

from sklearn.metrics import confusion_matrix,mean_squared_error
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
 
LR.fit(xtrain,ytrain)

y_pred=LR.predict(xtest)
mean_squared_error(ytest,y_pred)

y_pred1=LR.predict(xtrain)
mean_squared_error(ytrain,y_pred1)


#but the right metric is 
confusion_matrix(ytest,y_pred)


#If model performance is good means, we can integrate our model into our website/user_ineractive_tools 

#for_User_ application
gender=int(input("enter your gender male=1 female=0"))

age=int(input("enter your age"))

hbp=int(input("enter if you have hbp is yes=1  no=0"))

heartdisease=int(input("enter if you have heart disease is yes=1 no=0"))

married=int(input("enter if you married yes=1 no=0"))

work=int(input("work_type if private=1,govt_job=2,self_emp=3,child=4,no_work=5"))

residence=int(input("enter your residence if urban=1 , rural=0"))

avg_glucose_level=int(input("enter your glucos level"))

bmi=int(input("enter your body mass index measure"))

smoke = int(input("enter your smoking status if yes=1 ,no=0,sometime=2" ))



LR_Prediction=LR.predict(gender,age,hbp,heartdisease,married,work,residence,avg_glucose_level,bmi,smoke)
Prediction_Probability=LR.predict_proba(gender,age,hbp,heartdisease,married,work,residence,avg_glucose_level,bmi,smoke)




















