
# coding: utf-8

# Approx Hypothesis:
# 1. Chances of approval decresease with increase in no of dependents.
# 2. Chances are more for graduate people.
# 3. Chances increase with increase in applicant and coapplicant income.
# 4. Chances decrease with increase in loan amount.
# 5. Chances increase for those who have repayed previous loans.
# 6. Chances are more for urban people.
# 

# In[249]:


import pandas as pd
import numpy as np                     # For mathematical calculations
import seaborn as sns                  # For data visualization
import matplotlib.pyplot as plt        # For plotting graphs


# In[250]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[251]:


train_df=pd.read_csv('train_df.csv')
test_df=pd.read_csv('test_df.csv')
combine=[train_df,test_df]


# In[252]:


test=pd.read_csv('test_df.csv')
print(test.head())


# In[253]:


train_df.head()


# In[254]:


train_df.describe()


# Now, we will analyse different variables.
# Starting with categorical variables.

# In[255]:


plt.subplot(3,2,1)
train_df['Gender'].value_counts(normalize=False).plot.bar(figsize=(20,20),title='Gender freq')
plt.subplot(3,2,2)
train_df['Married'].value_counts(normalize=False).plot.bar(figsize=(20,20),title='Married freq')
plt.subplot(3,2,3)
train_df['Education'].value_counts(normalize=False).plot.bar(figsize=(20,20),title='Graduation freq')
plt.subplot(3,2,4)
train_df['Self_Employed'].value_counts(normalize=False).plot.bar(figsize=(20,20),title='Self Employed freq')
plt.subplot(3,2,5)
train_df['Credit_History'].value_counts(normalize=False).plot.bar(figsize=(20,20),title='Credit History')
plt.show()


# We get some info about these variables:
# 1. No of males>no of females
# 2. No of married applicants>unmarried
# 3. No of graduated applicants>ungraduated
# 4. There sre less no of self employed applicants
# 5. Credit history of max applicants is positive.

# Now, taking into consideration ordinal variables.

# In[256]:


plt.subplot(1,2,1)
plt.hist(train_df['Dependents'],bins='auto',histtype='bar',rwidth=1)
plt.title('Frequency of no of dependents')
plt.xlabel('No of dependents')
plt.ylabel('frequency')
plt.subplot(1,2,2)
train_df['Property_Area'].value_counts(normalize=False).plot.bar(title='Property area freq')
plt.show()


# Now, looking at independent or continous numeric variables.
# Let us the see the variation of no of people with different incomes.

# In[257]:


plt.subplot(1,2,1)
plt.hist(train_df['ApplicantIncome'],bins='auto',histtype='bar',rwidth=0.9)
plt.title('Applicant income')
plt.xlabel('Applicant Income')
plt.ylabel('No of applicants')
plt.subplot(1,2,2)
plt.hist(train_df['CoapplicantIncome'],bins='auto',histtype='bar',rwidth=0.9)
plt.title('CoApplicant income')
plt.xlabel('CoApplicant Income')
plt.ylabel('No of applicants')
plt.show()


# We can see that applicant income after 20000 has very less frequency. Similarly, coaaplicant income after about 10000 has very less frequency

# 
# Now, let us do bivariate analysis, that is, find loan status relation to other variables. We will do it graphically.
# The main aim to do bivariate analysis is to find out seperate relations of each individual variable with the target variable. We can then later discard some variable if it is found to be potentially unimportantor club to variables to create a new variable if they have the same releationship with target variable. Also, we can try to find out some completely new variable thata may affect the target variable.
# 

# Firstly lets start with the categorical variables-
# Gender
# Married
# Education
# Self_Employed
# Credit_History

# In[258]:


Gender=pd.crosstab(train_df['Gender'],train_df['Loan_Status'])
Gender.div(Gender.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)


# In[259]:


Married=pd.crosstab(train_df['Married'],train_df['Loan_Status'])
Married.div(Married.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)

#crosstab will create a data frame with rows as Maried and unmarried and columns as loan status accepted or not.
#.div will divide the dataframe according to the condition given


# In[260]:


Education=pd.crosstab(train_df['Education'],train_df['Loan_Status'])
Education.div(Education.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)


# In[261]:


Employment=pd.crosstab(train_df['Self_Employed'],train_df['Loan_Status'])
Employment.div(Employment.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)


# In[262]:


Credit_History=pd.crosstab(train_df['Credit_History'],train_df['Loan_Status'])
Credit_History.div(Credit_History.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)


# What we observe is that the important variables that affect the loan status include 
# 1. Credit History
# 2. Education 
# 3. Marital Status
# 
# The other 2 categorical variables , i.e ; gender and self employment status do not affect the loan status.
# 
# 

# Now, taking into consideration ordinal variables.
# Dependents
# Property_Area

# In[263]:


Dependents=pd.crosstab(train_df['Dependents'],train_df['Loan_Status'])
Dependents.div(Dependents.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)


# In[264]:


Property_Area=pd.crosstab(train_df['Property_Area'],train_df['Loan_Status'])
Property_Area.div(Property_Area.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)


# We observe that one and three no of dependents have the least of all and almost the same loan status acceptance ratio.
# The most acceptance ratio is for 2 no of dependents.
# 
# 

# Loans are least accepted for people in rural areas and most for people in semi urban areas or developing areas.

# Now, moving on to the continous variables.

# In[265]:


sns.stripplot(x='Loan_Status',y='ApplicantIncome',data=train_df,jitter=True)

##stripplot doesn't give a clear picture.


# In[266]:


sns.violinplot(x='Loan_Status',y='ApplicantIncome',data=train_df)


# Here, we can not clearly find any significant difference between the applicant incomes of accepted and not accepted loan statuses. It may be related to some third variable. For eg, credit history plays a major role in determining the loan status as we know so let us segregate using credit history as the third variable.

# In[267]:


sns.swarmplot(x='Loan_Status',y='ApplicantIncome',hue='Credit_History',data=train_df)


# Here we can more cleearly see that almost all people will negative credit history don't recieve the loan, also one person will excellent income doesn't recieve the loan because of negative credit history, whereas many people with very low incomes and a positive credit history do recieve the loan.
# 

# One more problem is that we still can't find the exact variation of income with loan status because even though the total income range is from 0 to about 80000, max poeple lie in the range of 0 to 20000. So we need to analye that data seperately.

# In[268]:


print(train_df.iloc[0,6])


# In[269]:


d=[]
for i in range(0,614):
    if train_df['ApplicantIncome'][i]<=20000:
        d.append(train_df['ApplicantIncome'][i])
print(len(d))
        


# In[270]:


train_df1=pd.read_csv('train_df.csv')


# In[271]:


##data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
##        'year': [2012, 2012, 2013, 2014, 2014], 
##        'reports': [4, 24, 31, 2, 3]}
##df = pd.DataFrame(data, index = ['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'])
##df


# In[272]:


##df.drop(df.index[2])
##THIS IS HOW TO DROP A ROW BY ITS INDEX.


# In[273]:


s=[]
for i in range(0,614):
    if train_df1['ApplicantIncome'][i]>=15000:
        s.append(i)
        i=i+1
train_df1=train_df1.drop(train_df1.index[s])


# In[274]:


print(train_df1)


# In[275]:


sns.swarmplot(x='Loan_Status',y='ApplicantIncome',hue='Credit_History',data=train_df1)


# Here, we see that our initial approx hypothesis that poeple with higher applicant income have more laon approval chances does not hold really good. But again this does not make sense that people with lower income have higher approval chances. Thus we will first analyze coaaplicant income with Loan Status and then deduce what to do further.

# In[276]:


sns.swarmplot(x='Loan_Status',y='CoapplicantIncome',hue='Credit_History',data=train_df)


# Even this graph does not give a very good idea about what is happening.Infact what we can see is many of the applicants do not have any coapplicant and thus the coapplicant income is zero, but they have still recieved the loan. Therefore we need to analyze both the applicant and coapplicant income together deducing a good relationship between total income and loan status.
# So let's create a new variable TotalIncome and add in the training data.

# In[277]:


train_df1['TotalIncome']=train_df1['ApplicantIncome']+train_df1['CoapplicantIncome']
train_df['TotalIncome']=train_df['ApplicantIncome']+train_df['CoapplicantIncome']
sns.swarmplot(x='Loan_Status',y='TotalIncome',data=train_df1)


# Here, we can see total no of approvals and disapprovals but we cannot deduce the percentage of approved loans in various income segments which would actually give us an idea as to how variation in income varies loan status.
# To visualize fraction of yes and no out of total applicants in different income ranges in, lets plot a bar graph with bins defined by us.

# In[278]:


bins=[0,2000,4000,6000,10000,85000]
segments=['Very Low','Low','Average','High','Very High']
train_df['TotalIncomeBins']=pd.cut(train_df['TotalIncome'],bins,labels=segments)


# In[279]:


TotalIncomeBins=pd.crosstab(train_df['TotalIncomeBins'],train_df['Loan_Status'])
TotalIncomeBins.div(TotalIncomeBins.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
plt.xlabel('TotalApplicantIncome')
plt.ylabel('Percentage')
plt.show()


# This graph gives us a better picture. We can see that very low income people have very less loan approval compared to those of others.

# Let's now find relation between loan approval and loan amount. Also, loan amount term.

# In[280]:


sns.swarmplot(x='Loan_Status',y='LoanAmount',data=train_df)


# In[281]:


sns.swarmplot(x='Loan_Status',y='LoanAmount',hue='Loan_Amount_Term',data=train_df)


# The range is from about 0 to 750

# In[282]:


bins2=[0,75,150,225,300,700]
train_df['LoanAmountBins']=pd.cut(train_df['LoanAmount'],bins2)
LoanAmountBins=pd.crosstab(train_df['LoanAmountBins'],train_df['Loan_Status'])
LoanAmountBins.div(LoanAmountBins.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)


# We can see that loan approval for smaller amount is more which is in cordinance with our original approximate hypothesis.

# In[283]:


train_df=train_df.drop(['LoanAmountBins','TotalIncomeBins'],axis=1)


# Let's replace the yes no variables with 1 and 0 so that we can use them for calculations with other numeric variables.
# Also, let's replace dependents from 3+ to 3.

# In[284]:


train_df['Dependents'].replace('3+',3,inplace=True)
test_df['Dependents'].replace('3+',3,inplace=True)
train_df['Loan_Status'].replace('N',0,inplace=True)
train_df['Loan_Status'].replace('Y',1,inplace=True)


# Let's look at the correlation between all numeric variables

# In[285]:


correlations=train_df.corr()
f,ax=plt.subplots()
sns.heatmap(correlations,vmax=1,square=True,cmap="BuPu"); ##heatmap gives correlation of all numeric variables with each other.


# We can see that credit history is closely correlated to loan status.
# We can also see that Total Income,Applicant income is correlated to loan amount.

# Now, let us check no of missing values in all fields.

# In[286]:


train_df.isnull().sum()


# In[287]:


test_df.isnull().sum()


# In[288]:


##replacing categorical missing values with mode
train_df['Gender'].fillna(train_df['Gender'].mode()[0],inplace=True)
train_df['Married'].fillna(train_df['Married'].mode()[0],inplace=True)
train_df['Self_Employed'].fillna(train_df['Self_Employed'].mode()[0],inplace=True)
train_df['Credit_History'].fillna(train_df['Credit_History'].mode()[0],inplace=True)
train_df['Dependents'].fillna(train_df['Dependents'].mode()[0],inplace=True)


# In[289]:


##Loan amount terms also have to be fixed(for eg:half year or 1 year). So, printing the loan amount term data:
train_df['Loan_Amount_Term'].value_counts()


# In[290]:


##replacing by mode as 360 appears much more than others
train_df['Loan_Amount_Term'].fillna(train_df['Loan_Amount_Term'].mode()[0],inplace=True)


# In[291]:


train_df['LoanAmount'].value_counts()


# In[292]:


##there are various values. There are outliers so it would be better to replace by median than mean.
train_df['LoanAmount'].fillna(train_df['LoanAmount'].median(),inplace=True)


# In[293]:


##recheck if any missing value left
train_df.isnull().sum()


# In[294]:


##similarly for test dataset
test_df['Gender'].fillna(train_df['Gender'].mode()[0], inplace=True)
test_df['Dependents'].fillna(train_df['Dependents'].mode()[0], inplace=True)
test_df['Self_Employed'].fillna(train_df['Self_Employed'].mode()[0], inplace=True)
test_df['Credit_History'].fillna(train_df['Credit_History'].mode()[0], inplace=True)
test_df['Loan_Amount_Term'].fillna(train_df['Loan_Amount_Term'].mode()[0], inplace=True)
test_df['LoanAmount'].fillna(train_df['LoanAmount'].median(), inplace=True)


# Now, outliers need to be treated as they affect the overall quantities of data like mean, standard deviation, etc.
# By the previous plotting and observations we can see that income has some outliers. Also loan amount may have some outliers.
# Income outliers have chances of being natural outliers and so we can not ignore them. Thus one method to treat the outliers can be taking log of all the observations.

# In[295]:


train_df['LoanAmount_Log']=np.log(train_df['LoanAmount'])
##Checking if this works
plt.hist(train_df['LoanAmount'],bins='auto')

plt.show()


# Here, we can see many outliers.

# In[296]:


plt.hist(train_df['LoanAmount_Log'])


# In[297]:


##Let's create new variables which will be more relevant using the existing ones. Total income is already created
train_df['EMI']=train_df['LoanAmount']/train_df['Loan_Amount_Term']
##We should relate total income to loan amount. For eg, if a person with high income has high loan amount it is more 
##favorable than a person with low income asking for high loan amount.
train_df['Dif_In_Sal_And_LoanAmount']=train_df['TotalIncome']-train_df['EMI']*1000 ##one months payable loan(that is EMI)-one months salary. 
##Can also be done for whole tenure by taking loan amount instead of EMI and total income multiplied by tenure instead of total income.


# In[298]:


##Now that we have made new variables, let us remove the older not of much use variables.
train_df=train_df.drop(['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term'],axis=1)


# In[299]:



print(train_df.head())


# In[300]:


plt.hist(train_df['EMI'],bins='auto')


# In[301]:



##let's drop loan id as it does not have any effect on loan status
train_df=train_df.drop(['Loan_ID'],axis=1)


# In[302]:


test_df=test_df.drop(['Loan_ID'],axis=1)


# In[303]:


##Also we need to perform same actions on test set as we did on training set.
test_df['TotalIncome']=test_df['ApplicantIncome']+test_df['CoapplicantIncome']
test_df['EMI']=test_df['LoanAmount']/test_df['Loan_Amount_Term']
test_df['Dif_In_Sal_And_LoanAmount']=test_df['TotalIncome']-test_df['EMI']*1000
test_df['LoanAmount_log']=np.log(test_df['LoanAmount'])


# In[304]:


test_df=test_df.drop(['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term'],axis=1)


# In[305]:


##scikit learn requires target variable to be in different dataframe
x=train_df.drop(['Loan_Status'],axis=1)
y=train_df['Loan_Status']


# In[306]:


##creating dummy variables for all categorical variables, basically converting them to numerical form.
x=pd.get_dummies(x)
train_df=pd.get_dummies(train_df)
test_df=pd.get_dummies(test_df)


# In[307]:


from sklearn.model_selection import train_test_split
x_train,x_cv,y_train,y_cv=train_test_split(x,y,test_size=0.3)


# In[308]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[309]:


log_model=LogisticRegression()
log_model.fit(x_train,y_train)


# In[310]:


predicted_cv=log_model.predict(x_cv)


# In[311]:


print(predicted_cv)


# In[312]:


accuracy_score(y_cv,predicted_cv)


# In[313]:


print(test_df.head())
print(train_df.head())


# In[314]:


pred_test = log_model.predict(test_df)


# In[315]:


print(pred_test)


# In[316]:


submission=pd.read_csv("Sample_Submission_ZAuTl8O_FK3zQHh.csv")


# In[317]:


submission['Loan_Status']=pred_test


# In[318]:


submission['Loan_Status'].value_counts()


# In[319]:


test_df1=pd.read_csv('test_df.csv')
submission['Loan_ID']=test_df1['Loan_ID']
print(submission)


# In[320]:


submission['Loan_Status'].replace(0, 'N',inplace=True)
submission['Loan_Status'].replace(1, 'Y',inplace=True)


# In[321]:


print(submission)


# In[322]:


pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('submission.csv',index=False)

