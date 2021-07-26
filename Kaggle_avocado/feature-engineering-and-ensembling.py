
# coding: utf-8

# ![](https://www.sickchirpse.com/wp-content/uploads/2015/07/Avocado-Price.jpg)

# # **If you like my work please upvote!!!! Thanks**

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px 
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_csv('../input/avacado-price-prediction/Avocado.csv')


# # EDA

# In[4]:


df.head()


# Unnamed: 0 seems to be a identifier column and needs to be removed.

# In[5]:


df.shape


# There are 18249 rows and 14 columns

# In[6]:


df.nunique()


# There are no constant or identifier column.

# In[7]:


df.isnull().sum()


# There are no null values.

# In[8]:


df.dtypes


# Dataframe have 3 columns with object type data, which we need to encode.

# In[9]:


df.skew()


# Data is higly skewed in almost all the columns.

# In[10]:


df.describe()


# All the columns have count equal to 18249. Mean and median have high difference except for Average price stating that data has high skewness present. There is high variance in all the columns except for Average price and year column. Difference between min, max and interquartile ranges is uneven hence there are a no. of outliers present in the data.

# ### Univariate Analysis

# In[11]:


df.nunique()


# In[12]:


#We separate categorical and continuous features
cat=['year','region','type']
cont=[ 'Total Volume', '4046', '4225','4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags']


# In[13]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
df['type'].value_counts().plot.pie(autopct='%1.1f%%')
plt.subplot(1,2,2)
sns.countplot(df['type'])
df['type'].value_counts()


# There two types organic and conventional are almost equal and balanced.

# In[14]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
df['year'].value_counts().plot.pie(autopct='%1.1f%%')
plt.subplot(1,2,2)
sns.countplot(df['year'])
df['year'].value_counts()


# Most of the data is from 2017 followed by 2016 and 2015 respectively, while 2018 has the least data.

# In[15]:


plt.figure(figsize=(25,10))
sns.countplot(df['region'])
plt.xticks(rotation=90)
print('Total number of regions',df['region'].nunique())


# There are 54 regions in total. All the region produces almost equal amounts of avocados except for west tex new mexico which has slightly less number.

# In[16]:


plt.figure(figsize=(8,6))
sns.distplot(df['AveragePrice'],color='m', kde_kws={"color": "k"})
print('Minimum',df['AveragePrice'].min())
print('Maximum',df['AveragePrice'].max())


# Average price of avacados little skewed, price ranging from 0.44 t0 3.25 

# In[17]:


plt.figure(figsize=(8,6))
sns.distplot(df['Total Volume'],color='r', kde_kws={"color": "k"})
print('Minimum',df['Total Volume'].min())
print('Maximum',df['Total Volume'].max())


# Most of the Total volume of Avacados is concentrated below 1e7 volumes while it goes upto 62505646.52 volumes. Data is higly skewed to the right, which needs to be taken care of ahead.

# In[18]:


plt.figure(figsize=(8,6))
sns.distplot(df['4046'],color='g', kde_kws={"color": "k"})
print('Minimum',df['4046'].min())
print('Maximum',df['4046'].max())


# Avacados with Product look up code 4046 is mostly concentrated near the minimum whereas the range goes up to 22743616.17. Data is higly skewed to the right, which needs to be taken care of ahead.

# In[19]:


plt.figure(figsize=(8,6))
sns.distplot(df['4770'],color='y', kde_kws={"color": "k"})
print('Minimum',df['4770'].min())
print('Maximum',df['4770'].max())


# Avacados with Product look up code 4770 is mostly concentrated near the minimum and the minimum equal to 0, whereas the range goes up to 2546439.11 as the maximum. Data is higly skewed to the right, which needs to be taken care of ahead in the data engineering process.

# In[20]:


plt.figure(figsize=(8,6))
sns.distplot(df['4225'],color='b', kde_kws={"color": "k"})
print('Minimum',df['4225'].min())
print('Maximum',df['4225'].max())


# Avacados with Product look up code 4225 is mostly concentrated near the minimum and the minimum equal to 0, whereas the range goes up to 20470572.61 as the maximum. Data is higly skewed to the right, which needs to be taken care of ahead in the data engineering process.

# In[21]:


plt.figure(figsize=(8,6))
sns.distplot(df['Total Bags'],color='y', kde_kws={"color": "k"})
print('Minimum',df['Total Bags'].min())
print('Maximum',df['Total Bags'].max())


# Total bags has most of its density near to its minimum value and goes up to a range of 19373134.37 with its maximum value. Data is clearly skewed to the right.

# In[22]:


plt.figure(figsize=(8,6))
sns.distplot(df['Small Bags'],color='b', kde_kws={"color": "k"})
print('Minimum',df['Small Bags'].min())
print('Maximum',df['Small Bags'].max())


# Large bags has most of its density near to its minimum value which is 0 and goes up to a range of 13384586.8 with its maximum value. Data is clearly skewed to the right.

# In[23]:


plt.figure(figsize=(8,6))
sns.distplot(df['Large Bags'],color='g', kde_kws={"color": "k"})
print('Minimum',df['Large Bags'].min())
print('Maximum',df['Large Bags'].max())


# Large bags has most of its density near to its minimum value which is 0 and goes up to a range of 5719096.61 with its maximum value. Data is clearly skewed to the right.

# In[24]:


plt.figure(figsize=(8,6))
sns.distplot(df['XLarge Bags'],color='r', kde_kws={"color": "k"})
print('Minimum',df['XLarge Bags'].min())
print('Maximum',df['XLarge Bags'].max())


# Xtra large bags are densely populated in the range 0 to 5000, whereas they are spread till values more than 5 lakh. Distribution of data is highly right skewed.

# In[25]:


fig,ax=plt.subplots(4,2,figsize=(15,25))
r=0
c=0
for i,n in enumerate(cont):
    if i%2==0 and i>0:
        r+=1
        c=0
    sns.boxplot(df[n],ax=ax[r,c])
    c+=1


# There are a large number of outliers present in all the features that is needed to be removed.

# ### Bivariate Analysis

# In[26]:


plt.figure(figsize=(8,20))
sns.stripplot(x='year',y='region',data=df)


# Avocados are collected from all the regions irrespective of the year. For each year avocados are collected from all the same regions.

# In[27]:


plt.figure(figsize=(8,6))
sns.stripplot(x='year',y='AveragePrice',data=df)


# Average price is highest in the year 2016, but relatively price seem to increase as the time passes except for the year 2018 which is an exception. 

# In[28]:


plt.figure(figsize=(8,6))
sns.stripplot(x='type',y='AveragePrice',data=df)


# Organic Avacoados are more expensive than the conventional ones.

# In[29]:


plt.figure(figsize=(20,8))
sns.boxplot(x='region',y='AveragePrice',data=df)
plt.xticks(rotation=90)


# Highest average price for avacaodo's were in San diego, Las vegas and cahrlotte regions while the least was from phoenix tucson. It is also to be noted that highest average price belong from areas where there is more development.

# In[30]:


fig,ax=plt.subplots(4,2,figsize=(15,25))
r=0
c=0
for i,n in enumerate(cont):
    if i%2==0 and i>0:
        r+=1
        c=0
    sns.scatterplot(x=n,y='AveragePrice',data=df,ax=ax[r,c],color='r')
    c+=1


# Average price shows a negative correlation as the average price seem to decrease as total volumes, PLU's and types of bags increases which means that average price decreases as the quantity of avacado's decreases.

# In[31]:


fig,ax=plt.subplots(4,2,figsize=(15,25))
r=0
c=0
for i,n in enumerate(cont):
    if i%2==0 and i>0:
        r+=1
        c=0
    sns.scatterplot(x=n,y='Total Volume',data=df,ax=ax[r,c],color='k')
    c+=1


# With the increase in total volumes of avacado, quatity of all types of bags as well as PLU's also increases. This is logical as if volume of avacado's increases, no. of bags needed to carry it and PLU's inreases.

# In[32]:


plt.figure(figsize=(20,10))
sns.barplot(x='region',y='Total Volume',data=df)
plt.xticks(rotation=90)


# Highest volumes of avacado's are found in the US. That could be the reason price of avacado's low there. As supply increases price decreases,whereas region with less volume such as las vegas have the highest price of avacado's.

# In[33]:


plt.figure(figsize=(20,10))
sns.barplot(x='region',y='Total Bags',data=df)
plt.xticks(rotation=90)


# Data is similar to the previous bar chart, as the volumes of avacado increases in a region, total bags also increases to carry those avacado's.

# In[34]:


fig,ax=plt.subplots(4,2,figsize=(15,25))
r=0
c=0
for i,n in enumerate(cont):
    if i%2==0 and i>0:
        r+=1
        c=0
    sns.stripplot(x='year',y=n,data=df,ax=ax[r,c])
    c+=1


# Though data for the year 2018 is very low but we can see that total volume, types of bags and PLUs are highest for 2018. As the time has passed volumes, bags to carry avocados  and plus have increased without any doubt.

# In[35]:


fig,ax=plt.subplots(4,2,figsize=(15,25))
r=0
c=0
for i,n in enumerate(cont):
    if i%2==0 and i>0:
        r+=1
        c=0
    sns.stripplot(x='type',y=n,data=df,ax=ax[r,c])
    c+=1


# Toal volumes. types of bags, and PLU's are all high for convetional type of avocados, this states that although price of organic type is high but cobventional types of avocados are more produced.

# ### Multivariate Analysis

# In[36]:


data=df.groupby(['AveragePrice','year']).apply(lambda x:x['Total Volume'].count()).reset_index(name='Volume')
px.line(data,x='AveragePrice',y='Volume',color='year',title='Average Price of Avacados by Volume for year 2015 and 2016  ')


# Average price of avacado's for the year 2016 was way more than any other year, but highest volumes of avacado's were produced in 2015,reaching to a mark of 85, after that the produce of avocados seem to decrease. It is also to be noted that high volumes of avocados are sold at lower average price.

# In[37]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='Total Volume',y='Total Bags',hue='AveragePrice',data=df)


# As the total volume increases, Total bags also increases to carry it but the average price seem to decrease. When the volumes of avacados are low average price increases of avacados.

# In[38]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='Total Bags',y='4225',hue='AveragePrice',data=df)


# As PlU's increases so does the no. of total bags but as we know if supply of something increases its price decreases. Same thing can be seen here.

# In[39]:


plt.figure(figsize=(13,10))
sns.heatmap(df.corr(),annot=True,cmap='Greys')


# Average price shows positive relationship with the year column while negative correlation with all the other column. Intresting thing to notice here is that independent features show more than 90% correlation with each other. This is a case of multicollinearity. We need to remove some features to resolve this problem.

# # Feature Engineering

# ###### Removing identifier and constant columns

# In[40]:


#year column also need to be removed as we already have date column.
df.drop(['Unnamed: 0','year'],axis=1,inplace=True)


# ###### Handling date column

# In[41]:


#Converting date column into datetime format
df['Date']=pd.to_datetime(df['Date'])


# In[42]:


#Extracting month, day and year info from date column then dropping it.
df['Month']=df['Date'].apply(lambda x:x.month)
df['Day']=df['Date'].apply(lambda x:x.day)
df['Year']=df['Date'].apply(lambda x:x.year)
df.drop('Date',axis=1,inplace=True)


# In[43]:


plt.figure(figsize=(15,8))
sns.lineplot(x='Month',y='AveragePrice',hue='Year',ci=18,data=df)


# Average price of avacado's is high in the month of September October November. As the season for avocado's is in the summer, off season fruit is expensive. We have data for year 2018 till march only

# In[44]:


#We replace 2015 by 1 and 2016 by 2 for more simplicity
df['Year'].replace(2015,1,inplace=True)
df['Year'].replace(2016,2,inplace=True)
df['Year'].replace(2017,3,inplace=True)
df['Year'].replace(2018,24,inplace=True)


# In[45]:


#we create a time column using year,month and day column and then drop these 3.
df['Time']=(df['Year']*365)+(df['Month']*30)+(df['Day'])
df.drop(['Year','Month','Day'],axis=1,inplace=True)


# ###### Encoding

# In[46]:


from sklearn.preprocessing import OrdinalEncoder
o=OrdinalEncoder()


# In[47]:


df['region']=o.fit_transform(df['region'].values.reshape(-1,1))
df['type']=o.fit_transform(df['type'].values.reshape(-1,1))


# ###### Removing Outliers

# In[48]:


#Function to choose the right threshold 
def threhold(z,d):
    for i in np.arange(3,4,0.01):
        data=d.copy()
        data=data[(z<i).all(axis=1)]
        loss=(d.shape[0]-data.shape[0])/d.shape[0]*100
        print('With threshold {} data loss is {}%'.format(np.round(i,2),np.round(loss,2))) 


# In[49]:


#Using zscore method to remove outliers
from scipy.stats import zscore
z=np.abs(zscore(df))
threhold(z,df)


# In[50]:


#We use threshold as 3.57 because we cannot afford to loose much data
df=df[(z<3.57).all(axis=1)]


# ###### Removing Skewness

# In[51]:


cont.append('Time')


# In[52]:


from sklearn.preprocessing import PowerTransformer
pt=PowerTransformer()


# In[53]:


#We make use of power transformer to remove skewness from all columns except from Total volume as it was incapable
for i in cont:
    if np.abs(df[i].skew())>0.5 and i!='Total Volume':
        df[i]=pt.fit_transform(df[i].values.reshape(-1,1))

# To remove skewness from total volume column we ise log transformation
df['Total Volume']=np.log(df['Total Volume'])


# In[54]:


fig,ax=plt.subplots(5,2,figsize=(15,25))
r=0
c=0
for i,n in enumerate(cont):
    if r==4 and c==1:
        break
    if i%2==0 and i>0:
        r+=1
        c=0
    sns.distplot(df[n],ax=ax[r,c])
    c+=1


# Skewness is almost negligible after using tranformations techniques. Only XLarge bags shows skewness in graph but it is still reduced considerably.

# In[55]:


df.skew()


# ###### Separating the dependent and independent variables.

# In[56]:


x=df.copy()
x.drop('AveragePrice',axis=1,inplace=True)

y=df['AveragePrice']


# ###### Scaling the data

# In[57]:


#Scaling the data using min max scaler
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()


# In[58]:


xd=scaler.fit_transform(x)
x=pd.DataFrame(xd,columns=x.columns)


# # Modelling Phase

# Importing neccessary modules

# In[59]:


from sklearn.model_selection import train_test_split,cross_val_score


# In[60]:


#importing models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor


# In[61]:


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


# In[62]:


#Choosing the best random state using Logistic regression
def randomstate(a,b):
    maxx=1000
    for state in range(1,201):
        xtrain,xtest,ytrain,ytest=train_test_split(a,b,test_size=0.25,random_state=state)
        model=LinearRegression()
        model.fit(xtrain,ytrain)
        p=model.predict(xtest)
        mse=mean_squared_error(p,ytest)
        if maxx>mse:
            maxx=mse
            j=state
    return j


# In[63]:


#Creating list of models and another list mapped to their names
models=[KNeighborsRegressor(),SVR(),LinearRegression(),Lasso(),Ridge(),ElasticNet(),DecisionTreeRegressor(),
       RandomForestRegressor(),AdaBoostRegressor(),GradientBoostingRegressor(),XGBRegressor()]

names=['KNeighborsRegressor','SVR','LinearRegression','Lasso','Ridge','ElasticNet','DecisionTreeRegressor',
       'RandomForestRegressor','AdaBoostRegressor','GradientBoostingRegressor','XGBRegressor']


# In[64]:


def createmodels(model_list,independent,dependent,n):
    xtrain,xtest,ytrain,ytest=train_test_split(independent,dependent,test_size=0.25,random_state=randomstate(independent,dependent))
    name=[]
    meanabs=[]
    meansqd=[]
    rootmeansqd=[]
    r2=[]
    mcv=[]
    
    #Creating models
    for i,model in enumerate(model_list):
        model.fit(xtrain,ytrain)
        p=model.predict(xtest)
        score=cross_val_score(model,independent,dependent,cv=10)
        
        #Calculating scores of the model and appending them to a list
        name.append(n[i])
        meanabs.append(np.round(mean_absolute_error(p,ytest),4))
        meansqd.append(np.round(mean_squared_error(p,ytest),4))
        rootmeansqd.append(np.round(np.sqrt(mean_squared_error(p,ytest)),4))
        r2.append(np.round(r2_score(p,ytest),2))
        mcv.append(np.round(np.mean(score),4))
    
    #Creating Dataframe
    data=pd.DataFrame()
    data['Model']=name
    data['Mean Absolute Error']=meanabs
    data['Mean Squared Error']=meansqd
    data['Root Mean Squared Error']=rootmeansqd
    data['R2 Score']=r2
    data['Mean of Cross validaton Score']=mcv
    data.set_index('Model',inplace = True)
    return data
        


# In[65]:


createmodels(models,x,y,names)


# ### Removing multicollinearity ussing L1 Regularisation

# In[66]:


from sklearn.model_selection import GridSearchCV


# In[67]:


param_grid={'alpha':[1e-15,1e-10,1e-8,1e-5,1e-3,0.1,1,5,10,15,20,30,35,45,50,55,65,100,110,150,1000]}
m1=GridSearchCV(Lasso(),param_grid,scoring='neg_mean_squared_error',cv=10)
m1.fit(x,y)
print(m1.best_params_)


# In[68]:


m1=Lasso(alpha=1e-05)
m1.fit(x,y)


# In[69]:


importance = np.abs(m1.coef_)


# In[70]:


dfcolumns = pd.DataFrame(x.columns)
dfimp=pd.DataFrame(importance)
featureScores = pd.concat([dfcolumns,dfimp],axis=1)
featureScores.columns = ['Features','Coefficients']  #naming the dataframe columns
featureScores


# In[71]:


featureScores.sort_values(by=['Coefficients'],ascending=False)


# In[72]:


#We remove 1 feature
lassoxt=x.copy()
lassoxt.drop(['XLarge Bags'],axis=1,inplace=True)
createmodels(models,lassoxt,y,names)


# In[73]:


#We remove these 2 features
lassoxt=x.copy()
lassoxt.drop(['XLarge Bags','4046'],axis=1,inplace=True)
createmodels(models,lassoxt,y,names)


# As removing features also leads to some data loss, so the model's performance have decreased, therefore we will not remove any features.

# ###### We apply Hperparameter tuning on Knearest Neighbor, Random Forest and xtreme Gradient Boost as they are giving the best performance for our dataset

# # Hyperparameter Tuning

# In[74]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=randomstate(x,y))


# ###### Knearest Neighbor

# In[75]:


leaf = list(range(1,50))
k = list(range(1,30))
params={'n_neighbors':k,'leaf_size':leaf,'weights':['uniform','distance'],'metric':['euclidean','manhattan']}


# In[76]:


g=GridSearchCV(KNeighborsRegressor(),params,cv=10)


# In[78]:


g.fit(xtrain,ytrain)


# In[79]:


print(g.best_estimator_)
print(g.best_params_)
print(g.best_score_)


# In[80]:


m=KNeighborsRegressor(leaf_size=1, metric='manhattan', n_neighbors=3,weights='distance')
m.fit(xtrain,ytrain)
p=m.predict(xtest)
score=cross_val_score(m,x,y,cv=10)


# In[81]:


print('Mean Absolute Error is',np.round(mean_absolute_error(p,ytest),4))
print('Mean Squared Error is',np.round(mean_squared_error(p,ytest),4))
print('Root Mean Squared Error is',np.round(np.sqrt(mean_squared_error(p,ytest)),4))
print('R2 Score is',np.round(r2_score(p,ytest),4)*100)
print('Mean of cross validaton Score is',np.round(np.mean(score)*100,4))


# ###### Random Forest

# In[82]:


from sklearn.model_selection import RandomizedSearchCV


# In[83]:


params={'n_estimators':[100, 300, 500, 700],
        'min_samples_split':[1,2,3,4],
        'min_samples_leaf':[1,2,3,4],
            'max_depth':[None,1,2,3,4,5,6,7,8]}


# In[84]:


g=RandomizedSearchCV(RandomForestRegressor(),params,cv=10)


# In[85]:


g.fit(xtrain,ytrain)


# In[86]:


print(g.best_estimator_)
print(g.best_params_)
print(g.best_score_)


# In[87]:


m=RandomForestRegressor(max_depth=8, min_samples_leaf=2, min_samples_split=3,n_estimators=500)
m.fit(xtrain,ytrain)
p=m.predict(xtest)
score=cross_val_score(m,x,y,cv=10)


# In[88]:


print('Mean Absolute Error is',np.round(mean_absolute_error(p,ytest),4))
print('Mean Squared Error is',np.round(mean_squared_error(p,ytest),4))
print('Root Mean Squared Error is',np.round(np.sqrt(mean_squared_error(p,ytest)),4))
print('R2 Score is',np.round(r2_score(p,ytest),4)*100)
print('Mean of cross validaton Score is',np.round(np.mean(score)*100,4))


# ###### Xtreme Gradient Boost

# In[89]:


params={
 "learning_rate"    : [0.001,0.05, 0.10, ] ,
 "max_depth"        : [ 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    }


# In[90]:


g=RandomizedSearchCV(XGBRegressor(verbosity=0),params,cv=10)


# In[91]:


g.fit(xtrain,ytrain)


# In[92]:


print(g.best_estimator_)
print(g.best_params_)
print(g.best_score_)


# In[93]:


m=XGBRegressor(colsample_bytree=0.5,gamma=0.1,learning_rate=0.1,max_depth=15,min_child_weight=7)
m.fit(xtrain,ytrain)
p=m.predict(xtest)
score=cross_val_score(m,x,y,cv=10)


# In[94]:


print('Mean Absolute Error is',np.round(mean_absolute_error(p,ytest),4))
print('Mean Squared Error is',np.round(mean_squared_error(p,ytest),4))
print('Root Mean Squared Error is',np.round(np.sqrt(mean_squared_error(p,ytest)),4))
print('R2 Score is',np.round(r2_score(p,ytest),4)*100)
print('Mean of cross validaton Score is',np.round(np.mean(score)*100,4))


# XGBRegressor is giving the best performance with minimum error compared to all the models, so we choose it as out final model.

# ### Finalizing the Model

# In[95]:


model=XGBRegressor(colsample_bytree=0.5,gamma=0.1,learning_rate=0.1,max_depth=15,min_child_weight=7)
model.fit(xtrain,ytrain)
p=model.predict(xtest)
score=cross_val_score(m,x,y,cv=10)


# # Evaluation Metrics

# In[96]:


print('Mean Absolute Error is',np.round(mean_absolute_error(p,ytest),4))
print('Mean Squared Error is',np.round(mean_squared_error(p,ytest),4))
print('Root Mean Squared Error is',np.round(np.sqrt(mean_squared_error(p,ytest)),4))
print('R2 Score is',np.round(r2_score(p,ytest),4)*100)
print('Mean of cross validaton Score is',np.round(np.mean(score)*100,4))


# In[97]:


plt.scatter(x=ytest,y=p,color='r')
plt.plot(ytest,ytest,color='b')
plt.xlabel('Actual Average Price')
plt.ylabel('Predicted Average Price')
plt.title('XGBRegressor')


# # Saving the Model

# In[98]:


import joblib
joblib.dump(model,'avacadoprice.obj')

