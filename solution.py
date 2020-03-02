def years(s):
    y=int(s[0])
    r=s[::-1]
    for i in r:
        m=0
        for j in r[3:5]:
            if j!=' ':
                m=m*10+int(j)
            else:
                break
    return(12*y+m)

def dateconvert(d,m,y): #Function for calculating days between dates
    month=[0,31,28,31,30,31,30,31,31,30,31,30,31]
    if y%4:
        month[2]=29
    d1=17
    m1=4
    y1=19
    s=month[m]-d
    m+=1
    while 1:
        s+=month[m]
        m+=1
        if m>12:
            y+=1
            m=1
        if m>=m1 and y>=y1:
            s+=d1
            break
    return(s)


import numpy as np
import pandas as pd

train=pd.read_csv("C:/Users/hp/Desktop/lt/train.csv")
test=pd.read_csv("C:/Users/hp/Desktop/lt/test_bqCt9Pv.csv")
#to avoid memory error
train_11 = train.iloc[:,:]
train = pd.DataFrame(train_11)
train_1=pd.DataFrame(train_11)
train_1=train_1.drop(["loan_default"],axis=1)

train_1=pd.concat([train_1, test], axis = 0, ignore_index=True)

print(train_1.head())
train_1.info()
train_1.describe()
from sklearn.neighbors import KNeighborsClassifier
#since out of all 42 variables we have 6 variables as factor type
#challange2 , create dummy variables for "PERFORM_CNS.SCORE.DESCRIPTION"
dum_var=pd.get_dummies(train_1['Employment.Type'])
dum_var1=pd.get_dummies(train_1['PERFORM_CNS.SCORE.DESCRIPTION'])
print(len(dum_var.columns))
print(len(dum_var1.columns))
print(dum_var)
print(dum_var1)
data=train_1[["disbursed_amount","asset_cost","ltv","branch_id","supplier_id","DisbursalDate","manufacturer_id","Current_pincode_ID","State_ID","Employee_code_ID","MobileNo_Avl_Flag","Aadhar_flag","PAN_flag","VoterID_flag","Driving_flag","Passport_flag","PERFORM_CNS.SCORE","PRI.NO.OF.ACCTS","PRI.ACTIVE.ACCTS","PRI.OVERDUE.ACCTS","PRI.CURRENT.BALANCE","PRI.SANCTIONED.AMOUNT","PRI.DISBURSED.AMOUNT","SEC.NO.OF.ACCTS","SEC.ACTIVE.ACCTS","SEC.OVERDUE.ACCTS","SEC.CURRENT.BALANCE","SEC.SANCTIONED.AMOUNT","SEC.DISBURSED.AMOUNT","PRIMARY.INSTAL.AMT","SEC.INSTAL.AMT","NEW.ACCTS.IN.LAST.SIX.MONTHS","DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS","AVERAGE.ACCT.AGE","CREDIT.HISTORY.LENGTH","NO.OF_INQUIRIES"]]


print(data.head())
##NOW ADD THOSE COLUMNS FOR WHICH DUMMY VARIABLES ARE MADE
print(dum_var.dtypes)
data=pd.concat([data,dum_var,dum_var1],axis=1)
print(data.head())
###MOST UPDATED DATAFRAME WE HAVE IS DATA, also PERFORM_CNS SCOREis BE DONE##
print(data.isnull())
print(data.columns)
##CHALLANGE 3 : HANDLING MISSING VALUES
print(data.isnull())
#from sklearn.preprocessing import Imputer
#from sklearn.svm import SVC

# Setup the Imputation transformer: imp
#imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
# Instantiate the SVC classifier: clf
#clf = SVC()

# Setup the pipeline with the required steps: steps
#steps = [('imputation', imp),
        #('SVM', clf)]

#Converting the Date.of.Birth to age of each account holder
age=[]
for i in train_1["Date.of.Birth"]:
    age.append(abs(100-int(i[len(i)-2:])+19))
age={'age':age}
age=pd.DataFrame(age)
print(age)
print(age.shape)
#Adsdimg the new 'age' attribute in our dataset
data_new=pd.concat([data,age], sort=False ,axis=1)
#print(data.describe())
print(data.head())
print(data_new.columns)
##NEXT STEP IS DOING FEATURE ENGINEERING, DIMENSION REDUCTION
import seaborn as sns
import matplotlib.pyplot as plt


#plt.figure(figsize=(40,40))
# play with the figsize until the plot is big enough to plot all the columns
# of your dataset, or the way you desire it to look like otherwise

#sns.heatmap(data_new.corr())
#plt.show()
#print(data_new.corr(method='pearson'))

#data_new1=data_new['DisbursalDate'].fillna('missing')
#print(data_new1)
print(data_new['DisbursalDate'])

#print(data_new.tail())
#print(data_new['DisbursalDate'].head())
#### Counting missing values
print(data_new['DisbursalDate'].isna().sum())
print(data_new.shape)

####Converting the 'DisbursalDate' attribute to no. of days since loan was disbursed
days=[]
for i in data_new['DisbursalDate']:

        d=int(i[0:2])
        m=int(i[3:5])
        y=18
        days.append(dateconvert(d,m,y))

disbDate={'DisbursalDays':days}
disbDate=pd.DataFrame(disbDate)
print(disbDate)

####Coverting account age and credit history length
accage=[]
credhist=[]
for i in data_new["AVERAGE.ACCT.AGE"]:
    accage.append(years(i))
for i in data_new["CREDIT.HISTORY.LENGTH"]:
    credhist.append(years(i))
accage={"Acc_Age" : accage}
credhist={"Credit_History": credhist}
accage=pd.DataFrame(accage)
credhist=pd.DataFrame(credhist)
##NUMBER OF DAYS , SINCE DISBURSAL OF LOAN IS STORED IN disbDate
df=pd.concat([data_new,disbDate, accage, credhist],axis=1)
print(df.shape)
print(df.columns)
################TO APPLY MODEL############
df2 = df.drop(['DisbursalDate','supplier_id','CREDIT.HISTORY.LENGTH','AVERAGE.ACCT.AGE','manufacturer_id','Passport_flag','Employee_code_ID','State_ID'],axis=1)

print(df2.columns)

##BEFORE DOING PCA WE MUST NORMALIZE OUR DATA
# Perform the necessary imports
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Create a PCA instance: pca




# Create the pipeline: pipeline
#pipeline = Pipeline(steps)
#print(df2.columns)
#X_train=df2.drop(['loan_default'],axis=1)
#y_train=df2['loan_default']
#parameters = {'SVM__C':[1, 10, 100],
             #'SVM__gamma':[0.1, 0.01]}
# Instantiate the GridSearchCV object: cv
#cv = GridSearchCV(pipeline,parameters,cv=3)
print(data_new.isnull().any())

# Fit to the training set


#x=preprocessing.scale(X)
# Separating out the target
#y = df2.loc[:,['loan_default']].values
# Standardizing the features
#count=0
#for i in X:
 #   if i == '':
  #     count=count+1
#print(count)
#print(x)
##now seperating test and train data again
###FINAL DATASET WE HAVE IS DF2
print(df2.shape)


print(train.shape)
print(test.shape)
training_data=df2.iloc[0:233154,:]
test_data=df2.iloc[233154:,:]
print(training_data.shape)
print(test_data.shape)
print(training_data.columns)
print(train.columns)
loan_default1=train['loan_default']
df_training=pd.concat([training_data,loan_default1],axis=1)
print(df_training.columns)
# ########################################NOR FOR TRAINING :- df_training , FOR TESTING test_data##############################33
x_train= df_training.drop(['loan_default'],axis=1)
y_train=df_training['loan_default']
print(df_training.dtypes)

print(x_train.shape)
#y_prediction=pd.DataFrame(y_prediction, columns=["loan_default"])
#print(y_prediction.tail())
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
#scaler.fit(y_train)
xscale=scaler.transform(x_train)
#yscale=scaler.transform(y)
scaler.fit(test_data)
test_scaled=scaler.transform(test_data)




#y_prediction.to_csv("C:/Users/hp/Desktop/lt/submitl22.csv")
####################MAKING TEST SET TO SAME TYPE###########
#test= pd.read_csv("C:/Users/hp/Desktop/lt/test_bqCt9Pv.csv")
################################################USING RANDOM FOREST ###########################
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

from  keras.callbacks import EarlyStopping
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

# Convert the target to categorical: target
#target = to_categorical(df_training.loan_default)

# Set up the model
model = Sequential()

# Add the first layer
model.add(Dense(40, kernel_initializer='normal',activation='relu' , input_shape=(54,)))
model.add(Dense(18, activation='relu'))
# Add the output layer
model.add(Dense(1, activation='linear'))

# Compile the model
# Compile the model
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
history = model.fit(xscale, y_train, epochs=150, batch_size=50,  verbose=1)
#print(history.history.keys())
# "Loss"
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Fit the model


y_prediction = model.predict(test_scaled)
print(y_prediction)
m=y_prediction["loan_default"].median()
a=y_prediction["loan_default"].values
for i in range(0, len(a)):
    if a[i]>m:
        a[i]=1
    else:
        a[i]=0
y_prediction=pd.DataFrame(a, columns=["loan_default"])
print(y_prediction.tail())



#y_prediction.rename(index=str, columns=['loan_default'])
y_prediction=pd.concat([test['UniqueID'],y_prediction], axis=1, sort=False)

y_prediction.to_csv("C:/Users/hp/Desktop/lt/ssubmit final dl solution.csv")









#y_prediction=pd.DataFrame(y_prediction, columns=["loan_default"])

#y_prediction=pd.concat([test['UniqueID'],y_prediction], axis=1, sort=False)

#y_prediction.to_csv("C:/Users/hp/Desktop/lt/submitdeeplearning.csv")


















#y_pred.to_csv("C:/Users/hp/Desktop/lt/submitRF.csv")
