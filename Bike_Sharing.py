#!/usr/bin/env python
# coding: utf-8

# # Bike Sharing Assignment

# ---
# ## Environment Setup
# ---

# In[1]:


# To get multiple outputs in the same cell

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[2]:


# Supress Warnings

import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Import the EDA required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import plotly.express as px

# Import the machine learning libraries

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

# Importing VIF from statsmodels 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Import the generic utility libraries

import os
import random
import datetime as dt

#Importing the function
from pandas_profiling import ProfileReport


# In[4]:


# Set the required global options

# To display all the columns in dataframe
pd.set_option( "display.max_columns", None)
pd.set_option( "display.max_rows", None)

# Setting the display fromat
# pd.set_option('display.float_format', lambda x: '%.2f' % x)

#pd.reset_option('display.float_format')

sns.set(style='whitegrid')

get_ipython().run_line_magic('matplotlib', 'inline')


# ---
# ## Data Smthing
# ---

# - **_Reading the Bike Sharing dataset csv file_**

# In[5]:


# Read the raw csv file 'day.csv' - containing the basic data of the loans
# encoding - The type of encoding format needs to be used for data reading

day = pd.read_csv('day.csv', low_memory=False)


# In[6]:


day.head()


# In[7]:


day.shape


# In[8]:


#day['dteday'] = pd.to_datetime(day['dteday'], format='%d-%m-%Y')


# In[9]:


# df = pd.DataFrame()
# df['dteday'] = day['dteday']
# df['year'] = day['dteday'].dt.year
# df['month'] = day['dteday'].dt.month
# df['day'] = day['dteday'].dt.day
# df['weekday'] = day['dteday'].dt.weekday
# df['dayname'] = day[['dteday']].apply(lambda x: dt.datetime.strftime(x['dteday'], '%A'), axis=1)
# df


# #### Dataset characteristics
# 
# - day.csv have the following fields:
# 	
# 	- instant: record index
# 	- dteday : date
# 	- season : season (1:spring, 2:summer, 3:fall, 4:winter)
# 	- yr : year (0: 2018, 1:2019)
# 	- mnth : month ( 1 to 12)
# 	- holiday : weather day is a holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
# 	- weekday : day of the week
# 	- workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
# 	+ weathersit : 
# 		- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
# 		- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# 		- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# 		- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# 	- temp : temperature in Celsius
# 	- atemp: feeling temperature in Celsius
# 	- hum: humidity
# 	- windspeed: wind speed
# 	- casual: count of casual users
# 	- registered: count of registered users
# 	- cnt: count of total rental bikes including both casual and registered

# - **_Checking the missing values_**

# In[10]:


miss = day.isna().sum()
len(miss[miss > 0])


# _There are no missing values in the data as evident from **.isna()**_

# In[11]:


day.describe()


# ---
# ## Data Cleaning
# ---

# In[12]:


# Checking the values of 'instant'

day['instant'].nunique()
print(" % 3.1f%% unique values in variable instant" %(day['instant'].nunique()/len(day)*100))


# In[13]:


# Checking the values of 'dteday' 

day['dteday'].nunique()
print(" % 3.2f%% unique values in variable instant" %(day['dteday'].nunique()/len(day)*100))


# _Also year, month, weekday, workingday, holiday are the important metrics already derived from 'dteday' variable._ <br>
# _So, the variables **'instant'** and **'dteday'** can be dropped as it contains 100% distinct values, not useful for prediction._

# In[14]:


# Dropping the variables 'instant' and 'dteday'

day.drop(['instant','dteday'], axis=1, inplace=True)


# In[15]:


# Dropping the variables 'casual' and 'registered' as the target variable will be 'cnt' - sum of 'casual' and 'registered'

day.drop(['casual','registered'], axis=1, inplace=True)


# In[16]:


day.head()


# In[17]:


day['season_label'] = day['season'].map({1:'spring',2:'summer',3:'fall',4:'winter'})


# In[18]:


day['yr_label'] = day['yr'].map({0:'2018',1:'2019'})


# In[19]:


day['mnth_label'] = day['mnth'].map({1:'january',2:'feburary',3:'march',4:'april',5:'may',6:'june',7:'july',8:'august',9:'september',10:'october',11:'november',12:'december'})


# In[20]:


day['weekday_label'] = day['weekday'].map({0:'tuesday',1:'wednesday',2:'thursday',3:'friday',4:'saturday',5:'sunday',6:'monday'})


# In[21]:


day['weathersit_label'] = day['weathersit'].map({1:'clear',2:'mist',3:'rain',4:'heavy rain'})


# In[22]:


day.drop(['season','yr','mnth','weekday','weathersit'], axis=1,inplace=True)


# In[23]:


day.sample(5)


# - **_Checking the datatypes_**

# In[24]:


day.info()


# In[25]:


# Converting the categorical variables to type 'category'

category = day.columns[7:]

day['holiday'] = day['holiday'].astype('category')
day['workingday'] = day['workingday'].astype('category')
day[category] = day[category].astype('category')


# In[26]:


day.info()


# ---
# ## Data Analysis
# ---

# In[27]:


day.corr()


# In[28]:


plt.figure(figsize=(12,7))
sns.heatmap(day.corr(), annot=True);


# In[29]:


# num_dtype_ser = day.dtypes
# num_list = num_dtype_ser[num_dtype_ser == 'float64'].index
# num_list

# sns.pairplot(day[num_list])


# In[30]:


# Create scatterplots to visulaize the relationship between quantitave/numerical variables 

sns.pairplot(day.select_dtypes(include='number'))


# **_Correlation coefficient for temp and atemp is 0.99, which is quite high and is also visible through scatter-plot_** <br>
# _NOT dropping the variable temp although there is a strong correlataion between temp and atemp._ <br>
# _Will be handled in Feature Selection process._

# In[31]:


# Dropping the variable temp as there is a strong correlataion between temp and atemp

# day.drop(['temp'], axis=1, inplace=True)


# - **_atemp variable is somewhat having a linear relationship with cnt, the target variable_** <br>
# - **_cnt against hum and windspeed does not seem to have good correlation and the points are scattered all around in the plot._** <br>
#     - _Also evident from heatmap, the Correlation coefficient is **-0.099 for hum vs cnt** and **-0.24 for windspeed vs cnt**, which is on a very low side._ <br>
# 
# - _**NOT Dropping the variables hum and windspeed** although, there is a **very weak correlataion between them and cnt**, the target variable._ <br>
# - _Will be handled in Feature Selection process._

# In[32]:


# Dropping the variable temp as there is a strong correlataion between temp and atemp

# day.drop(['hum','windspeed'], axis=1, inplace=True)


# In[33]:


cat_dtype_ser = day.dtypes
category_list = cat_dtype_ser[cat_dtype_ser == 'category'].index
category_list


# In[34]:


plt.figure(figsize=(18, 23))
for i,var in enumerate(category_list):
    plt.subplot(4,2,i+1)
    sns.boxplot(x=var, y='cnt', data=day);
plt.show();


# In[35]:


# sns.catplot(x="yr_label", y="cnt", data=day);


# In[36]:


# sns.catplot(x="weathersit_label", y="cnt",col="yr_label", data=day);


# In[37]:


# sns.catplot(x="weathersit_label", y="cnt", hue="yr_label",col = 'holiday', data=day);
# sns.catplot(x="weathersit_label", y="cnt", hue="yr_label",col = 'workingday', data=day);
# sns.catplot(x="weekday_label", y="cnt", hue="yr_label",col = 'weathersit_label', data=day);


# In[38]:


category_list


# In[39]:


# def bivariate(x_var, y_var = 'cnt', hue = 'yr_label', data = day):
#     ax = sns.catplot(x=x_var, y= y_var, hue= hue, data=data)


# In[40]:


# plt.figure(figsize=(18, 23));
# for i,var in enumerate(category_list):
#     bivariate(x_var=var);
# plt.show()


# In[41]:


# bivariate(x_var='season_label');
# bivariate(x_var='mnth_label');


# In[42]:


# bivariate(x_var='holiday');
# bivariate(x_var='weekday_label');
# bivariate(x_var='workingday');


# ---
# ## Data Preparation for modelling
# ---

# ### Dummy Coding

# In[43]:


day.head()


# In[44]:


# Create the dummy variables for the categorical features

# cat_var = ['season_label','yr_label','mnth_label','weekday_label','holiday','workingday','weathersit_label']

dummy = pd.get_dummies(day[category_list], drop_first = True)
dummy.sample(4)


# In[45]:


# df.dtypes


# In[46]:


# df.describe()


# In[47]:


# Dropping the original categorical features

day.drop(category_list,axis=1,inplace=True)


# In[48]:


# Adding the dummy features to the original day dataframe

day = pd.concat([day,dummy], axis=1)


# In[49]:


day.sample(4)


# In[50]:


# Checking the shape of day dataframe

day.shape


# ### Splitting the data into training and testing sets

# In[51]:


# Specify random_state so that the train and test data set always have the same rows, respectively

day_train, day_test = train_test_split(day, train_size = 0.7, random_state = 100)


# In[52]:


day_train.shape


# In[53]:


day_test.shape


# _The train and test data shape looks good._

# ### Feature Scaling

# In[54]:


plt.figure(figsize=(15,6))
plt.subplot(1,2,1);
sns.boxplot(x = day.atemp);
plt.subplot(1,2,2);
sns.histplot(x = day.atemp, kde=True);


# **_There are no outliers in atemp feature and so we can apply Standardized Scaling, instead of MinMax Scaling._**

# In[55]:


scaler = StandardScaler()


# - **_Feature scaling should be done only for numerical features and not for categorical or dummy variables_**

# In[56]:


# Apply scaler() to all the variables except the 'yes-no' and 'dummy' variables.

num_vars = ['atemp', 'cnt']

day_train[num_vars] = scaler.fit_transform(day_train[num_vars])


# _1. Training set should be fit as well transformed._ <br>
# _2. But the testing set should never be used to fit as this dataset is not available at the time of model building (in real world scenario)._ <br>
# _3. The testing set should only be transformed with the fit of training set._

# In[57]:


day_train.sample(5)


# ### Splitting the targets into X and y sets for the model building

# In[58]:


y_train = day_train.pop('cnt')
X_train = day_train


# In[59]:


# y_train[10:15]
# X_train.sample(4)


# ---
# ## Building a Linear Regession Model
# ---

# - **Algorithm Introduction:**
# 
#     - **_Linear Regression or Ordinary Least Squares Regression (OLS)_** is one of the simplest machine learning algorithms and produces both accurate and interpretable results on most types of continuous data.<br>
#     - While more sophisticated algorithms like random forest will produce more accurate results, they are know as “black box” models because it’s tough for analysts to interpret the model.<br>
#     - In contrast, **_OLS regression results are clearly interpretable because each predictor value (beta) is assigned a numeric value (coefficient) and a measure of significance for that variable (p-value)_**. This allows us to interpret the effect of difference predictors on the model and tune it easily.
# 
#     - Equation of linear regression<br>
#         - $y = c + m_1x_1 + m_2x_2 + ... + m_nx_n$
# 
#             -  $y$ is the response
#             -  $c$ is the intercept
#             -  $m_1$ is the coefficient for the first feature
#             -  $m_n$ is the coefficient for the nth feature<br>
# 
#     - The $m$ values are called the model **coefficients** or **model parameters**.

# In[60]:


# Building a Linear Model

# By default, the statsmodels library fits a line on the dataset which passes through the origin.
# But in order to have an intercept, we need to manually use the add_constant attribute of statsmodels. 
# Add a constant to get an intercept
X_train_cn = sm.add_constant(X_train)

# Fit the resgression line using 'OLS'
model = sm.OLS(y_train, X_train_cn)
res = model.fit()

# Print the parameters, i.e. the intercept and the slope of the regression line fitted
res.params


# In[61]:


# Performing a summary operation lists out all the different parameters of the regression line fitted

print(res.summary())


# - **_Looking at the p-values, it looks like some of the variables aren't really significant (in the presence of other variables)._**
# 
#     - _We could simply drop the variable with the highest, non-significant p value. **A better way would be to cross-verify this with the VIF information.**_

# - **_Colinearity is the state where two variables are highly correlated and contain similiar information about the variance within a given dataset. To detect colinearity among variables, simply create a correlation matrix and find variables with large absolute values._**
# 
# - **_Multicolinearity on the other hand is more troublesome to detect because it emerges when three or more variables, which are highly correlated, are included within a model. To make matters worst multicolinearity can emerge even when isolated pairs of variables are not colinear._**
# 
#     - _Multicollinearity will **not affect the model's output or prediction strength.**_
#     - _Multicollinearity will **only affect the coefficient values** for the predictor variables by inflating their importance._

# ### Feature Selection with scikit learn RFE

# In[62]:


# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[63]:


# Running RFE with the output number of the variable equal to 10

# Create linear regression object
sk_model = LinearRegression()

# Train the model using the training sets
sk_model.fit(X_train, y_train)


# In[64]:


# Running RFE

# Create the RFE object
rfe = RFE(sk_model, n_features_to_select = 14)

rfe = rfe.fit(X_train, y_train)


# In[65]:


# Features with rfe.support_ values

list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[66]:


# Creating a list of rfe supported features
feats = X_train.columns[rfe.support_]
feats

# Creating a list of non-supported rfe features
drop_feats = X_train.columns[~rfe.support_]
drop_feats


# ### Dropping the features and updating the Model

# In[67]:


X_train.shape


# In[68]:


# Creating X_train dataframe with RFE selected variables

X_train.drop(drop_feats,axis=1,inplace=True)


# In[69]:


X_train.shape


# In[70]:


# Building a Linear Model

# By default, the statsmodels library fits a line on the dataset which passes through the origin.
# But in order to have an intercept, we need to manually use the add_constant attribute of statsmodels. 
# Add a constant to get an intercept
X_train_cn = sm.add_constant(X_train)

# Fit the resgression line using 'OLS'
model = sm.OLS(y_train, X_train_cn)
res = model.fit()

# Print the parameters, i.e. the intercept and the slope of the regression line fitted
res.params


# In[71]:


# Performing a summary operation lists out all the different parameters of the regression line fitted

print(res.summary())


# ### Checking VIF
# 
# - Variance Inflation Factor or VIF, gives a basic quantitative idea about **how much the feature variables (independent/predictor) are correlated with each other.**<br>
# - It is an extremely important parameter to test our linear model. The formula for calculating `VIF` is:
# 
# - $ VIF_i = \frac{1}{1 - {R_i}^2} $

# In[72]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[73]:


# Custom function to create a dataframe that will contain the names of all the feature variables and their respective VIFs

def vif():
    vif = pd.DataFrame()
    vif['Features'] = X_train.columns
    vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return vif


# In[74]:


# Calling the Custom function to create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif()


# **_We generally want a VIF that is less than 5. So there are clearly no variables with VIF more than 5._**

# ### Dropping the feature and updating the model
# 
# As you can see from the summary, mnth_label_feburary is insignificant with p-value of `0.065`. We ll drop this variable.

# In[75]:


X_train.shape


# In[76]:


# Dropping highly insignificant variable

X_train.drop(['mnth_label_feburary'], axis=1, inplace=True)


# In[77]:


X_train.shape


# - Looks good.

# In[78]:


# Building a Linear Model

# By default, the statsmodels library fits a line on the dataset which passes through the origin.
# But in order to have an intercept, we need to manually use the add_constant attribute of statsmodels. 
# Add a constant to get an intercept
X_train_cn = sm.add_constant(X_train)

# Fit the resgression line using 'OLS'
model = sm.OLS(y_train, X_train_cn)
res = model.fit()

# Print the parameters, i.e. the intercept and the slope of the regression line fitted
res.params


# In[79]:


# Performing a summary operation lists out all the different parameters of the regression line fitted

print(res.summary())


# In[80]:


# Calling the Custom function to create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif()


# ### Dropping the feature and updating the model
# 
# As you can see from the summary, mnth_label_may is less significant with p-value of `0.026`. We ll drop this variable.

# In[81]:


X_train.shape


# In[82]:


# Dropping highly insignificant variable

X_train.drop(['mnth_label_may'], axis=1, inplace=True)


# In[83]:


X_train.shape


# - Looks good.

# In[84]:


# Building a Linear Model

# By default, the statsmodels library fits a line on the dataset which passes through the origin.
# But in order to have an intercept, we need to manually use the add_constant attribute of statsmodels. 
# Add a constant to get an intercept
X_train_cn = sm.add_constant(X_train)

# Fit the resgression line using 'OLS'
model = sm.OLS(y_train, X_train_cn)
res = model.fit()

# Print the parameters, i.e. the intercept and the slope of the regression line fitted
res.params


# In[85]:


# Performing a summary operation lists out all the different parameters of the regression line fitted

print(res.summary())


# In[86]:


# Calling the Custom function to create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif()


# ### Dropping the feature and updating the model
# 
# As you can see from the summary, mnth_label_january is less significant with p-value of `0.012`. We ll drop this variable.

# In[87]:


X_train.shape


# In[88]:


# Dropping highly insignificant variable

X_train.drop(['mnth_label_january'], axis=1, inplace=True)


# In[89]:


X_train.shape


# - Looks good.

# In[90]:


# Building a Linear Model

# By default, the statsmodels library fits a line on the dataset which passes through the origin.
# But in order to have an intercept, we need to manually use the add_constant attribute of statsmodels. 
# Add a constant to get an intercept
X_train_cn = sm.add_constant(X_train)

# Fit the resgression line using 'OLS'
model = sm.OLS(y_train, X_train_cn)
res = model.fit()

# Print the parameters, i.e. the intercept and the slope of the regression line fitted
res.params


# In[91]:


# Performing a summary operation lists out all the different parameters of the regression line fitted

print(res.summary())


# In[92]:


# Calling the Custom function to create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif()


# ### Dropping the feature and updating the model
# 
# As you can see from the summary, mnth_label_december is less significant with p-value of `0.027`. We ll drop this variable.

# In[93]:


X_train.shape


# In[94]:


# Dropping highly insignificant variable

X_train.drop(['mnth_label_december'], axis=1, inplace=True)


# In[95]:


X_train.shape


# - Looks good.

# In[96]:


# Building a Linear Model

# By default, the statsmodels library fits a line on the dataset which passes through the origin.
# But in order to have an intercept, we need to manually use the add_constant attribute of statsmodels. 
# Add a constant to get an intercept
X_train_cn = sm.add_constant(X_train)

# Fit the resgression line using 'OLS'
model = sm.OLS(y_train, X_train_cn)
res = model.fit()

# Print the parameters, i.e. the intercept and the slope of the regression line fitted
res.params


# In[97]:


# Performing a summary operation lists out all the different parameters of the regression line fitted

print(res.summary())


# In[98]:


# Calling the Custom function to create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif()


# ### Dropping the feature and updating the model
# 
# As you can see from the summary, mnth_label_november is less significant with p-value of `0.052`. We ll drop this variable.

# In[99]:


X_train.shape


# In[100]:


# Dropping highly insignificant variable

X_train.drop(['mnth_label_november'], axis=1, inplace=True)


# In[101]:


X_train.shape


# - Looks good.

# In[102]:


# Building a Linear Model

# By default, the statsmodels library fits a line on the dataset which passes through the origin.
# But in order to have an intercept, we need to manually use the add_constant attribute of statsmodels. 
# Add a constant to get an intercept
X_train_cn = sm.add_constant(X_train)

# Fit the resgression line using 'OLS'
model = sm.OLS(y_train, X_train_cn)
res = model.fit()

# Print the parameters, i.e. the intercept and the slope of the regression line fitted
res.params


# In[103]:


# Performing a summary operation lists out all the different parameters of the regression line fitted

print(res.summary())


# In[104]:


# Calling the Custom function to create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif()


# ### Visualising the data with a scatter plot and the fitted regression line

# ---
# ## Residual Analysis
# ---

# - **_Residual Analysis needs to be done to validate assumptions of the model, and hence the reliability for inference._**
# 
# ### Distribution of the Error terms
# - We need to check if the error terms are also normally distributed (which is one of the major assumptions of linear regression).
# - Plotting a histogram of the error terms and see what it looks like.

# In[105]:


y_train_pred = res.predict(X_train_cn)
# y_train_pred.head()


# In[106]:


# Calculating the residuals

residuals = (y_train - y_train_pred)


# In[107]:


# Plot the histogram of the error terms/residuals

# plt.figure(figsize=(11,7))
# sns.distplot(residuals, hist=True)
# plt.title('Residuals Analysis', fontsize = 24)                 # Plot heading 
# plt.xlabel('Errors / Residuals', fontsize = 12);                    # X-label


# In[108]:


# Plot the histogram of the error terms/residuals

plt.figure(figsize=(10,6))
sns.histplot(residuals, stat="density", kde=True, color='#d62728')
plt.title('Residuals Analysis', fontsize = 24)                 # Plot heading 
plt.xlabel('Errors / Residuals', fontsize = 12);                    # X-label


# **_We can conclude that the Error terms/Residuals follow a Normal-Distribution curve._**

# - **_Normal distribution of the residuals can be validated by plotting a q-q plot._**
# 
# - **_Using the q-q plot we can infer if the data comes from a normal distribution._**
# - **_If yes, the plot would show fairly straight line. Absence of normality in the errors can be seen with deviation in the straight line._**

# In[109]:


residuals_fit = res.resid
fig = sm.qqplot(residuals_fit, fit=True, line='45')
plt.show()


# **_The q-q plot of the bike sharing data set shows that the errors(residuals) are fairly normally distributed._**

# ### Homoscedasticity Assumption

# - _**Homoscedasticity** describes a situation in which the **error term** (that is, the “noise” or random disturbance in the relationship between the features and the target) **is the same across all values of the independent variables.**_
# - A scatter plot of residual values vs predicted values is a goodway to check for homoscedasticity.There should be **no clear pattern in the distribution** and **if there is a specific pattern,the data is heteroscedastic.**
# 
# ![image.png](attachment:image.png)

# In[110]:


# Predicting the y_train
y_train_pred = res.predict(X_train_cn)

# Calculating the residuals
residuals = (y_train - y_train_pred)

# Visualizing the residuals and predicted value on train set
# plt.figure(figsize=(25,12))
sns.jointplot(x = y_train_pred, y = residuals, kind='reg', color='#d62728')
plt.title('Residuals of Linear Regression Model', fontsize = 20, pad = 100) # Plot heading 
plt.xlabel('Predicted Value', fontsize = 12)                     # X-label
plt.ylabel('Residuals', fontsize = 12);                          # Y-label


# **_Homoscedasticity Assumption holds true, as there is no clear pattern in the distribution_**

# ### Little or No autocorrelation in the residuals
# 
# - Autocorrelation occurs when the residual errors are dependent on each other.The presence of correlation in error terms drastically reduces model’s accuracy.
# 
# - **Autocorrelation** can be tested with the help of **Durbin-Watson test**.The null hypothesis of the test is that there is no serial correlation.
# 
# - The test statistic is approximately equal to **2*(1-r)** where **r is the sample autocorrelation of the residuals**. Thus, **for r == 0, indicating no serial correlation, the test statistic equals 2**. This statistic will always be between 0 and 4. The closer to 0 the statistic, the more evidence for positive serial correlation. The closer to 4, the more evidence for negative serial correlation.
# 
# - In our summary results, **Durbin-Watson is 2.065**, which tells us that the **residuals are not correlated**.

# In[ ]:





# ---
# ## Making Predictions using the Final Model
# ---

# ### Applying the scaling on the test sets

# In[111]:


# Apply scaler() to all the variables except the 'yes-no' and 'dummy' variables.
# scaler = StandardScaler() - scaler object is already instantiated while scaling train set

num_vars = ['atemp', 'cnt']

day_test[num_vars] = scaler.transform(day_test[num_vars])


# In[112]:


# day_test.head()


# ### Splitting into X_test and y_test

# In[113]:


y_test = day_test.pop('cnt')
X_test = day_test


# ### Predict the target

# In[114]:


# Now let's use our model to make predictions.

# Creating X_test dataframe by dropping variables from X_test
X_test = X_test[X_train.columns]

# Adding a constant variable 
X_test_cn = sm.add_constant(X_test)

# Making predictions
y_pred = res.predict(X_test_cn)


# ---
# ## Model Evaluation
# ---

# In[115]:


# Plotting y_test and y_pred to understand the spread.
plt.figure(figsize=(12,7))
sns.scatterplot(x = y_test, y = y_pred, color='#d62728')
plt.title('y_test vs y_pred', fontsize=25, pad = 25)              # Plot heading 
plt.xlabel('y_test', fontsize=16)                          # X-label
plt.ylabel('y_pred', fontsize=16);                          # Y-label


# ### The scatterplot is almost a straight line which also depicts that predicted y resemble the y test.

# In[116]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ### Looking at the RMSE

# In[117]:


#Returns the mean squared error; we'll take a square root
np.sqrt(mean_squared_error(y_test, y_pred))


# **_The closer the value of RMSE is to zero , the better is the Regression Model._**<br>
# **_In reality, we will not have RMSE equal to zero, in that case we will be checking how close the RMSE is to zero._**

# ### Checking the R-squared on the test set

# In[118]:


# R2 scroe on test data

r_squared = r2_score(y_test, y_pred)
r_squared


# In[119]:


# R2 scroe on train data

r_squared = r2_score(y_train, y_train_pred)
r_squared


# ### **_R2 score of train and test data is very close. Hence we can say that the model has performed well on the test data._**

# # Summary
# 
# - _Model Selected with Mix Approach_ - **RFE Technique and Manual selection guided by VIF**
# - _R-Squared_ : **0.822**
# - _Adjusted R-Squared_ : **0.818**
# - _R2_Score_ : **0.81**
# - _Root Mean Squared Error_ : **0.42**

# In[ ]:




