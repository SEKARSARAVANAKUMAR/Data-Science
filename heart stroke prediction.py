# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 16:16:35 2018

@author: vbcsaran
"""

#import data set 
import pandas as pd 


#Before_pre_processing_dataset
b=pd.read_csv("C:\\Users\\vbcsaran\\Documents\\heart stroke dataset\\train_2v.csv")
b.info()
b.describe(include="all")
b["age"].median()



#Remove_the_null_values
b.isnull().sum()
b=b.dropna()
b.isnull().sum()

#Over_view_about_dataset
b.head()
b.describe(include="all")



#Exploratory_data_analysis
import seaborn as sb

sb.countplot(x="stroke",hue="gender",data=b)

sb.countplot(x="stroke" ,hue="ever_married",data=b)

sb.countplot(x="stroke" ,hue="work_type",data=b)

sb.countplot(x="stroke" ,hue="Residence_type",data=b)

sb.countplot(x="stroke" ,hue="smoking_status",data=b)


sb.boxplot(x="stroke",y="avg_glucose_level",hue="gender",data=b)

sb.boxplot(x="stroke",y="avg_glucose_level",hue="ever_married",data=b)

sb.boxplot(x="stroke",y="avg_glucose_level",hue="Residence_type",data=b)

sb.boxplot(x="stroke",y="avg_glucose_level",hue="smoking_status",data=b)

sb.boxplot(x="stroke",y="avg_glucose_level",hue="work_type",data=b)
gj=b[b.work_type == "Govt_job"]  
gj=gj[gj.stroke == 1]
gj["avg_glucose_level"].median()




#Data_wrangling
b["gender"]=b["gender"].replace(["Male","Female","Other"],(1,0,2))

b["ever_married"]=b["ever_married"].replace(["Yes","No"],(1,0))

b["work_type"]=b["work_type"].replace(["Private","Self-employed","Govt_job","children","Never_worked"],(1,2,3,4,5))

b["Residence_type"]=b["Residence_type"].replace(["Urban","Rural"],(1,0))

b["smoking_status"]=b["smoking_status"].replace(["smokes","never smoked","formerly smoked"],(0,1,2))


b.head()


#model_Selection


#model1
x=b[["gender","age","hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level","bmi","smoking_status"]]
y=b[["stroke"]]


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
LR_P = LR.predict([[0,70,0,0,1,1,0,69,35,2]])
LR_P


#Prediction_with_Probability
LR_PP=LR.predict_proba([[0,70,0,0,1,1,0,69,35,2]])
LR_PP



#Behind_of_Statistics
import pandas
import numpy as np

l=LR.coef_
l1=pandas.DataFrame(l)

t=LR.intercept_

#formula=1/1+e^-(b0+b1*x1+b2*x2..n)
j=(t+(l1[0]*0)+(l1[1]*70)+(l1[2]*0)+(l1[3]*0)+(l1[4]*1)+(l1[5]*1)+(l1[6]*0)+(l1[7]*69)+(l1[8]*35)+(l1[9]*2))
j1=2.71838**(-j)
j2=j1+1
lr=1/j2
lr=np.array([[lr]])



if lr > 0.05:
    print("Positive")
else:
    print("Negative")
print("chance of getting heart stroke probability is ",lr)



#for_application
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



LR=LR.predict(gender,age,hbp,heartdisease,married,work,residence,avg_glucose_level,bmi,smoke)
LR_PP=LR.predict_proba(gender,age,hbp,heartdisease,married,work,residence,avg_glucose_level,bmi,smoke)






#model2_without_avg_glucose_level
x=b[["gender","age","hypertension","heart_disease","ever_married","work_type","Residence_type","bmi","smoking_status"]]
y=b[["stroke"]]

#fit
LR.fit(x,y)
LR=LR.predict(gender,age,hbp,heartdisease,married,work,residence,bmi,smoke)
LR_PP=LR.predict_proba(gender,age,hbp,heartdisease,married,work,residence,bmi,smoke)


#Behind_of_Statistics
import pandas
import numpy as np

l=LR.coef_
l1=pandas.DataFrame(l)

t=LR.intercept_


#formula=1/1+e^-(b0+b1*x1+b2*x2..n)
j=(t+(l1[0]*gender)+(l1[1]*age)+(l1[2]*hbp)+(l1[3]*heartdisease)+(l1[4]*married)+(l1[5]*work)+(l1[6]*residence)+(l1[7]*avg_glucose_level)+(l1[8]*bmi)+(l1[9]*smoke))
j1=2.71838**(-j)
j2=j1+1
lr=1/j2
lr=np.array([[lr]])



if lr > 0.05:
    print("Positive")
else:
    print("Negative")
print("chance of getting heart stroke probability is ",lr)    







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









