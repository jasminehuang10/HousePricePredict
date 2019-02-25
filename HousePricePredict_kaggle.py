#用户：jasmineHuang

#日期：2019-02-24   

#时间：09:29   

#文件名称：PyCharm

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

pd.set_option('display.max_columns',None)
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

#查看数据概况
print(train.describe())
print(train.info())

#缺失值处理
print(train.isnull().sum().sort_values(ascending=False).head(30))
total_null=train.isnull().sum()
total=train.isnull().count()
percent=total_null/total
null_data=pd.concat([total_null,percent],axis=1,keys=['total_null','percent'])
null_data.sort_values(by='total_null',ascending = False,inplace=True)
print(null_data)

#删除含高比例缺失值的变量
train = train.drop(null_data[null_data['total_null']>258].index,axis=1)
test=test.drop(null_data[null_data['total_null']>258].index,axis=1)

#删除少量比例缺失值的记录
cols=['Electrical','GarageYrBlt','GarageCond','GarageType','GarageFinish',
      'GarageQual','BsmtFinType2','BsmtExposure','BsmtQual','BsmtCond',
      'BsmtFinType1','MasVnrArea','MasVnrType']
print(null_data[null_data.percent>0].index)
for col in cols:
    train= train.drop(train.loc[train[col].isnull()].index)
print('train_shape',train.shape)

#探索性分析
sns.distplot(train.SalePrice,bins=30)
plt.show()
print(train.SalePrice.skew())
print(train.SalePrice.kurt())
y_train=train.SalePrice
y_train=np.log(y_train)
sns.distplot(y_train,bins=30)
plt.show()

#相关性分析
corr=train.corr()
plt.figure(figsize=(12,9))
k=10
cols=corr.nlargest(k,'SalePrice')['SalePrice'].index
cm=np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm=sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={
    'size':10},yticklabels=cols.values,xticklabels=cols.values)
plt.show()

#删除相关性高的类似变量
train.drop(['1stFlrSF','GarageArea','TotRmsAbvGrd'],axis=1,inplace=True)
test.drop(['1stFlrSF','GarageArea','TotRmsAbvGrd'],axis=1,inplace=True)
print('corr>0.5')
print(corr[corr['SalePrice']>0.5])
cols=['SalePrice','OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
      'FullBath', 'YearBuilt']
sns.pairplot(train[cols],size=2.5)
plt.show()

#离散型-哑变量处理
data=pd.concat([train,test])
data_new=data.drop('Id',axis=1)
non_numeric_cols=data_new.columns[data_new.dtypes=='object']
df=pd.get_dummies(data_new[non_numeric_cols])
data=pd.concat([data,df],axis=1)
data.drop(non_numeric_cols,axis=1,inplace=True)
print(data.describe())
train=data[data.SalePrice>0]
test=data[data.SalePrice.isnull()]

#建模与分析-选用随机森林
from sklearn.ensemble import RandomForestRegressor
RF=RandomForestRegressor()
x_train=train.drop('Id',axis=1)
x_train=x_train.drop('SalePrice',axis=1)
RF.fit(x_train,y_train)
print('test_describe',test.isnull().sum().sort_values(ascending=False).head(20))
cols=['GarageYrBlt','MasVnrArea','BsmtFullBath','BsmtHalfBath','BsmtFinSF1',
      'BsmtFinSF2',
      'BsmtUnfSF','TotalBsmtSF','GarageCars']
for col in cols:
    test[col]=test[col].fillna(value=0)
print('test_describe',test.isnull().sum().sort_values(ascending=False).head(20))
print(test.shape)
submit=pd.DataFrame()
submit['Id']=test.Id
test=test.drop('Id',axis=1)
test=test.drop('SalePrice',axis=1)
RF_pred=RF.predict(test)
submit['SalePrice']=np.exp(RF_pred)
submit.to_csv('submission_2.csv',index=False,encoding='utf-8')

#Evaluation score:0.11429  292/4097  2019/02/25
