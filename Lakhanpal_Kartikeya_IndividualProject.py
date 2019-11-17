# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 19:45:40 2019

@author: kartikeya
"""
#Code to run Classification and Regression models in one click for kickstarter-grading dataset
#Please refer to appendix section for the code developed on kickstarter data

#Importing relevant libraries
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier  
from sklearn.ensemble import RandomForestRegressor  
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


#-------------Please upload Kickstarter dataset  Required for dummification sets---------
kick = pd.read_excel(r"C:\Users\DELL\OneDrive - McGill University\Warut\Individual assignment\Kickstarter.xlsx")
#-------------Please upload Grading dataset ----------------
grading = pd.read_excel(r"C:\Users\DELL\OneDrive - McGill University\Warut\Individual assignment\Kickstarter-Grading.xlsx")
#-------------merging kickstarer with grading
df = pd.concat([kick, grading], axis=0) 
#-------------Data pre processing-----------------------------                        
df = df[(df.state=="successful") | (df.state=="failed")] #Selecting only fail or select

df['category'] = np.where(df['category'].isnull(),'Other',df['category']) #Inserting others for blank categories
df.isnull() #Testing for nulls
df.columns[df.isnull().any()]
null_columns1 = df.columns[df.isnull().any()]
df[null_columns1].isnull().sum() #Getting a understanding of null columns
df["goal"] = df["goal"]*df["static_usd_rate"] #converting goal to usd
df.drop(['launch_to_state_change_days'],axis = 1) #Dropping this column due to massive NAs
bins_day = [0,7,14,21,32] #Creating bins to reduce dummies in days
bins_hour = [0,8,16,25] #Creating bins to reduce dummies in hours
#Creating bins
df['deadline_day'] = pd.cut(df['deadline_day'], bins_day)
df['deadline_hr'] = pd.cut(df['deadline_hr'], bins_hour)
df['created_at_day'] = pd.cut(df['created_at_day'], bins_day)
df['created_at_hr'] = pd.cut(df['created_at_hr'], bins_hour)
df['launched_at_day'] = pd.cut(df['launched_at_day'], bins_day)
df['launched_at_hr'] = pd.cut(df['launched_at_hr'], bins_hour)
#Select relevant columns as per prelimnary logic
X=df[['goal','disable_communication','country','category','name_len_clean','blurb_len_clean','deadline_weekday','created_at_weekday','launched_at_weekday','deadline_month','deadline_day','deadline_yr','deadline_hr','created_at_month','created_at_day','created_at_yr','created_at_hr','launched_at_month','launched_at_day','launched_at_yr','launched_at_hr','create_to_launch_days','launch_to_deadline_days']]
#removed currency (perfectly correlated with country), backers_count (comes after), statis_usd_rate(not needed)
y=df['state']

#Dummyfication
X = pd.get_dummies(X, columns = ['disable_communication','country','category','deadline_weekday','created_at_weekday','launched_at_weekday','deadline_month','deadline_day','deadline_yr','deadline_hr','created_at_month','created_at_day','created_at_yr','created_at_hr','launched_at_month','launched_at_day','launched_at_yr','launched_at_hr'])


#------------Feature list for classification-----------------
fl1 = ['goal', 'name_len_clean', 'blurb_len_clean', 'create_to_launch_days', 'launch_to_deadline_days', 'country_GB', 'country_US', 'category_Apps', 'category_Experimental', 'category_Festivals', 'category_Gadgets', 'category_Hardware', 'category_Musical', 'category_Other', 'category_Plays', 'category_Software', 'category_Sound', 'category_Wearables', 'category_Web', 'deadline_weekday_Friday', 'deadline_weekday_Monday', 'deadline_weekday_Saturday', 'deadline_weekday_Sunday', 'deadline_weekday_Thursday', 'deadline_weekday_Tuesday', 'deadline_weekday_Wednesday', 'created_at_weekday_Friday', 'created_at_weekday_Monday', 'created_at_weekday_Saturday', 'created_at_weekday_Sunday', 'created_at_weekday_Thursday', 'created_at_weekday_Tuesday', 'created_at_weekday_Wednesday', 'launched_at_weekday_Friday', 'launched_at_weekday_Monday', 'launched_at_weekday_Saturday', 'launched_at_weekday_Sunday', 'launched_at_weekday_Thursday', 'launched_at_weekday_Tuesday', 'launched_at_weekday_Wednesday', 'deadline_month_2', 'deadline_month_3', 'deadline_month_4', 'deadline_month_5', 'deadline_month_6', 'deadline_month_7', 'deadline_month_8', 'deadline_month_9', 'deadline_month_10', 'deadline_month_11', 'deadline_month_12', 'deadline_day_(0, 7]', 'deadline_day_(7, 14]', 'deadline_day_(14, 21]', 'deadline_day_(21, 32]', 'deadline_yr_2013', 'deadline_yr_2014', 'deadline_yr_2015', 'deadline_yr_2016', 'deadline_hr_(0, 8]', 'deadline_hr_(8, 16]', 'deadline_hr_(16, 25]', 'created_at_month_1', 'created_at_month_2', 'created_at_month_3', 'created_at_month_4', 'created_at_month_5', 'created_at_month_6', 'created_at_month_7', 'created_at_month_8', 'created_at_month_9', 'created_at_month_10', 'created_at_month_11', 'created_at_month_12', 'created_at_day_(0, 7]', 'created_at_day_(7, 14]', 'created_at_day_(14, 21]', 'created_at_day_(21, 32]', 'created_at_yr_2013', 'created_at_yr_2014', 'created_at_yr_2015', 'created_at_yr_2016', 'created_at_hr_(0, 8]', 'created_at_hr_(8, 16]', 'created_at_hr_(16, 25]', 'launched_at_month_1', 'launched_at_month_2', 'launched_at_month_3', 'launched_at_month_4', 'launched_at_month_5', 'launched_at_month_6', 'launched_at_month_7', 'launched_at_month_8', 'launched_at_month_9', 'launched_at_month_10', 'launched_at_month_11', 'launched_at_month_12', 'launched_at_day_(0, 7]', 'launched_at_day_(7, 14]', 'launched_at_day_(14, 21]', 'launched_at_day_(21, 32]', 'launched_at_yr_2014', 'launched_at_yr_2015', 'launched_at_yr_2016', 'launched_at_hr_(0, 8]', 'launched_at_hr_(8, 16]', 'launched_at_hr_(16, 25]']
#------------Feature list for Regression---------------------

fl2 = ['name_len_clean', 'create_to_launch_days', 'country_GB', 'country_US', 'category_Apps', 'category_Blues', 'category_Experimental', 'category_Festivals', 'category_Flight', 'category_Gadgets', 'category_Hardware', 'category_Immersive', 'category_Makerspaces', 'category_Musical', 'category_Other', 'category_Plays', 'category_Robots', 'category_Shorts', 'category_Sound', 'category_Spaces', 'category_Wearables', 'deadline_weekday_Tuesday', 'created_at_weekday_Monday', 'created_at_weekday_Tuesday', 'launched_at_weekday_Tuesday', 'deadline_yr_2011', 'deadline_yr_2016', 'deadline_hr_(16, 25]', 'created_at_month_6', 'created_at_yr_2013', 'created_at_hr_(8, 16]', 'launched_at_month_5', 'launched_at_month_11', 'launched_at_yr_2012', 'launched_at_yr_2013', 'launched_at_hr_(0, 8]']

#Features selected
fc = X[fl1]
fr = X[fl2]

#-----------Splitting data back into kickstarter and Grading datasets for classification
X_trainC = fc.iloc[:15685][:]
X_testC = fc.iloc[15685:][:]

y_trainC = y.iloc[:15685]
y_testC = y.iloc[15685:]

#-----------Splitting data back into kickstarter and Grading datasets for Regression
X_trainR = fr.iloc[:15685][:]
X_testR = fr.iloc[15685:][:]

target=df['usd_pledged']

reg_train = target.iloc[:15685]
reg_test = target.iloc[15685:]

#-------------------------Classification Model-------------------------#
rfc= RandomForestClassifier(random_state=0, max_features=70, max_depth =10, min_samples_split=79, min_samples_leaf=2, bootstrap=0, n_estimators=100)

model= rfc.fit(X_trainC, y_trainC)
y_test_pred= rfc.predict(X_testC)
score = accuracy_score(y_testC, y_test_pred)



#-------------------------Regression Model-------------------------#


rf = RandomForestRegressor(random_state = 0, max_features = 24,max_depth = 6, min_samples_split = 40, min_samples_leaf = 6, bootstrap = 1,n_estimators = 100)
model = rf.fit(X_trainR,reg_train)
reg_test_pred = rf.predict(X_testR)

mse_reg = mean_squared_error(reg_test, reg_test_pred)


#---------------------Final printing of results------------------#
print("For classification problem the accuracy is",score )
#print(confusion_matrix(y_testC, y_test_pred))
print("For Regression problem the MSE is",mse_reg)

#--------------------------------------------------------------------------------#
'''
                            APPENDIX
'''                            
#--------------------------------------------------------------------------------#

#------Appendix i Random Forest Feature Selection classification-----------------#
'''
randomforest = RandomForestClassifier(random_state=0)
model = randomforest.fit(X,y)

from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(model, threshold=0.0035)
sfm.fit(X,y)

#random_forest_output.to_csv("Random_Forest_Feature_Select3.csv")
random_forest_output = pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','Gini coefficient'])
random_forest_output = random_forest_output[random_forest_output['Gini coefficient']>0.0035]
random_forest_output = random_forest_output['predictor']
feature_list1 = list(random_forest_output)
print(feature_list1)
feature_list = X[feature_list1]

'''

#------Appendix i Lasso Feature Selection Regression-----------------#

'''
from sklearn.linear_model import Lasso
model = Lasso(alpha=50, positive=True)
model.fit(X_std,y_dummy)
lasso_alpha_0 = pd.DataFrame(list(zip(X.columns,model.coef_)), columns = ['predictor','coefficient'])

lasso_alpha_0 = lasso_alpha_0[lasso_alpha_0['coefficient']>0]
lasso_alpha_0 = lasso_alpha_0['predictor']
feature_list1 = list(lasso_alpha_0)
'''
#------Appendix i Model Building Classification-----------------#
'''
num_pred = [25,75,100,106]
max_f = 0
max_depth = 0
min_sample_split = 0
min_samples_leaf = 11
bootstrap = 0
max_score = 0
best_pred = [0,0,0,0,0,0]
for max_f in num_pred:
    for max_depth in range(2,20):
        for min_sample_split in range(2,50,10):
            for min_samples_leaf in range(80,107):
                for bootstrap in range(0,2):
                    rf= RandomForestRegressor(random_state=5, max_features= max_f, max_depth = max_depth, min_samples_split=min_sample_split, min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, n_estimators=100)
                     scores = cross_val_score(estimator = rfc, X=feature_list.iloc[:,:max_f], y=y, cv=5)
                    if np.average(scores)>max_score:
                        max_score = np.average(scores)
                        best_pred = [max_f,max_depth,min_sample_split,min_samples_leaf,bootstrap]
             
'''
#------Appendix Model Building Regression-----------------------#
'''
max_f = 0
max_depth = 0
min_sample_split = 0
min_samples_leaf = 11
bootstrap = 0
min_mse = 99000000000
best_pred = [0,0,0,0,0,0]
for max_f in range (2,152,20):
    for max_depth in range(2,20):
        for min_sample_split in range(2,25):
            for min_samples_leaf in range(2,25):
                for bootstrap in range(0,2):
                    rf= RandomForestRegressor(random_state=5, max_features= max_f, max_depth = max_depth, min_samples_split=min_sample_split, min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, n_estimators=100)
                    model= rf.fit(X_train, y_train)
                    y_test_pred= rf.predict(X_test)
                    mse = mean_squared_error(y_test, y_test_pred)
                    if(mse<min_mse):
                        min_mse = mse
                        best_pred = [max_f,max_depth,min_sample_split,min_samples_leaf,bootstrap]
'''

#----------- Appendix Feature Selection -------------------------#

#---Recursive Function Selection---

#Logistic Regression
'''
from sklearn.linear_model import LogisticRegression 
from sklearn.feature_selection import RFE 
lr = LogisticRegression()

rfe = RFE(lr,1)             #Used feature = 1 to get ranking
y.unique()
model= rfe.fit(X,y)
log_rank_outout = pd.DataFrame(list(zip(X.columns,model.ranking_)), columns = ['predictor','ranking'])
'''
'''
from sklearn.ensemble import RandomForestClassifier rf= RandomForestClassifier(random_state=5)
rfe = RFE(rf,1)             #Used feature = 1 to get ranking
model= rfe.fit(X,y)
rf_rank_outout = pd.DataFrame(list(zip(X.columns,model.ranking_)), columns = ['predictor','ranking'])
log_rank_outout.to_csv("rf_rank_outout.csv")
'''



