#'讀取缺乏忽略金額的資料'
def DropStd(df,col,std=3):
    tempmean = df[col].mean()
    tempstd = df[col].std()
    return df[np.abs(df[col]-tempmean)<=std*tempstd]

data = pd.read_csv('data2017(notfill).csv')

#需處理:被保人ID, 身體健康狀況, 被保人殘障原因, 認識幾年, 薪資所得_萬, 其他所得_萬, 要保人累積保額_萬, 
#主要經濟者, 主要經濟者_累計保額_萬, 本人職業, 本人工作內容, 配偶職業, 配偶工作內容, 被保人薪資所得_萬, 被保人其他所得_萬
data = data.drop(['MAX_of_主約投保日期','被保人殘障原因','本人工作內容','配偶職業','配偶工作內容','本人職業','主要經濟者',
                  '身體健康狀況','認識幾年','薪資所得_萬','其他所得_萬','要保人累積保額_萬','主要經濟者_累計保額_萬','被保人薪資所得_萬'
                  ,'被保人其他所得_萬'],axis=1,errors='ignore')



data = data[data['BMI']>=10]
data = DropStd(data,'BMI')
data = data[data['BMI']<=55]

data = data[data['體重FIN']>2]
data = data[data['體重FIN']<150]

data = data[data['身高FIN']>50]
data = data[data['身高FIN']<240]

#data = data[data['被保人累積保額_萬']>=0.05]
#data = data[data['被保人累積保額_萬']<100000]

data.columns.values

# One-Hot encoding

def SparseCol(df,col):
    if len(df[col].value_counts())>=10:
        return df
    else:
        for i in set(df[col].value_counts().keys()):
            df[col+'_'+str(int(i))] = (df[col]==i).astype(int)
        return df.drop([col],axis=1)
    

data = SparseCol(data,'婚姻狀況')
data = SparseCol(data,'工作車輛')
data = SparseCol(data,'職業')
data = SparseCol(data,'服役')
data = SparseCol(data,'被保人學歷')

#'建立字串置換對應表，把所有年度的理賠金額，增加理賠指數、單位理賠金額對應欄位'
cols = []
for term in [s for s in data.columns.values if ("理賠金額" in s) and not ('年理賠金額' in s)]:
    cols.append((term,term.replace('理賠金額','單位理賠金額')))

#'理賠金額轉換為單位理賠金額'
for col in cols:
    data[col[0]] = data[col[0]] / data['B5總保額']


#進model
######################################################################################
import numpy as pd
import pandas as pd

data2016 = pd.read_csv('data2016(notfill).csv') 
data2017 = pd.read_csv('data2017(notfill).csv')   
data2018 = pd.read_csv('data2018(notfill).csv') 
data2018 = data2018.drop(['Unnamed: 0','RANK', 'MIN_of_RANK', 'MIN_of_MIN_of_RANK',
                          '婚姻居住狀況','婚姻狀況_5','役別','被保人聾啞','被保人肢體殘缺畸形_部位及程度','被保人肥胖瘦弱'],axis=1,errors='ignore')
#data2017.to_csv('data2017(notfill).csv',index=False)
ID_2018 = data2018['被保人ID']    
data2016 = data2016.drop(['理賠金額_住院2016','理賠金額_住院2017', '理賠金額_住院2018','理賠金額_手術2017', 
                          '理賠金額_手術2018','被保人ID','近三年理賠金額_住院_13_15',
                          '近五年理賠金額_住院_11_15', '近14年理賠金額_住院_02_15',
                          '近三年理賠金額_手術_13_15', '近五年理賠金額_手術_11_15', '近14年理賠金額_手術_02_15'],axis=1,errors='ignore')
data2017 = data2017.drop(['理賠金額_手術2002','理賠金額_住院2002','理賠金額_住院2017', '理賠金額_住院2018', 
                          '理賠金額_手術2018','被保人ID','近三年理賠金額_住院_13_15','近五年理賠金額_住院_11_15', 
                          '近14年理賠金額_住院_02_15','近三年理賠金額_手術_13_15', '近五年理賠金額_手術_11_15', '近14年理賠金額_手術_02_15'],axis=1,errors='ignore')
data2018 = data2018.drop(['理賠金額_住院2002', '理賠金額_住院2003','理賠金額_手術2002','理賠金額_手術2003',
                          '近三年理賠金額_住院_13_15','近五年理賠金額_住院_11_15','近14年理賠金額_住院_02_15',
                          '近三年理賠金額_手術_13_15', '近五年理賠金額_手術_11_15', '近14年理賠金額_手術_02_15','理賠金額_住院2018','被保人ID'],axis=1,errors='ignore')
# 切性別做model  ##################################
    
data2016 = data2016[data2016['被保人性別'] == 1]  
data2017 = data2017[data2017['被保人性別'] == 1]
data2018 = data2018[data2018['被保人性別'] == 1]  
    
        
    
    
##################################################    
train_y = data2016['理賠金額_手術2016']
train_x = data2016.drop(['理賠金額_手術2016'],axis=1,errors='ignore')
test_y = data2017['理賠金額_手術2017']
test_x = data2017.drop(['理賠金額_手術2017'],axis=1,errors='ignore')
test2018_y = data2018['理賠金額_手術2018']
test2018_x = data2018.drop(['理賠金額_手術2018'],axis=1,errors='ignore')


for i in (3,5,7,10,14):
    #data['近'+str(i)+'年理賠金額_手術'] 
    train_x['近'+str(i)+'年理賠金額手術'] = train_x.filter(a for a in train_x.columns.values if '理賠金額_手術' in a and int(a[7:])>=(2016-i)).sum(axis=1)
    train_x['近'+str(i)+'年理賠金額住院'] = train_x.filter(a for a in train_x.columns.values if '理賠金額_住院' in a and int(a[7:])>=(2016-i)).sum(axis=1)
for i in (3,5,7,10,14):
    #data['近'+str(i)+'年理賠金額_手術'] 
    test_x['近'+str(i)+'年理賠金額手術'] = test_x.filter(a for a in test_x.columns.values if '理賠金額_手術' in a and int(a[7:])>=(2017-i) and int(a[8:])!=0).sum(axis=1)
    test_x['近'+str(i)+'年理賠金額住院'] = test_x.filter(a for a in test_x.columns.values if '理賠金額_住院' in a and int(a[7:])>=(2017-i) and int(a[8:])!=0).sum(axis=1)
for i in (3,5,7,10,14):
    #data['近'+str(i)+'年理賠金額_手術'] 
    test2018_x['近'+str(i)+'年理賠金額手術'] = test2018_x.filter(a for a in test2018_x.columns.values if '理賠金額_手術' in a and int(a[7:])>=(2018-i) and int(a[8:])!=0).sum(axis=1)
    test2018_x['近'+str(i)+'年理賠金額住院'] = test2018_x.filter(a for a in test2018_x.columns.values if '理賠金額_住院' in a and int(a[7:])>=(2018-i) and int(a[8:])!=0).sum(axis=1)

# Use lightgbm to predict

from lightgbm import LGBMClassifier

#Check whether columns are inconsistent
#def checkcolumn(X,Y):
    #for i in range(X.shape[1]):
       # if(X.columns.values[i] != Y.columns.values[i]):
            #print(X.columns.values[i],Y.columns.values[i])

# PCA
#time_trainx住院 = train_x[['理賠金額_住院2002','理賠金額_住院2003',
       #'理賠金額_住院2004', '理賠金額_住院2005', '理賠金額_住院2006', '理賠金額_住院2007',
       #'理賠金額_住院2008', '理賠金額_住院2009', '理賠金額_住院2010', '理賠金額_住院2011',
       #'理賠金額_住院2012', '理賠金額_住院2013', '理賠金額_住院2014', '理賠金額_住院2015']]

#time_trainx手術 = train_x[['理賠金額_手術2002','理賠金額_手術2003', '理賠金額_手術2004', '理賠金額_手術2005',
       #'理賠金額_手術2006', '理賠金額_手術2007', '理賠金額_手術2008', '理賠金額_手術2009',
       #'理賠金額_手術2010', '理賠金額_手術2011', '理賠金額_手術2012', '理賠金額_手術2013',
       #'理賠金額_手術2014','理賠金額_手術2015']]    
    
#nontime_trainx = train_x.drop(['理賠金額_住院2002','理賠金額_住院2003',
       #'理賠金額_住院2004', '理賠金額_住院2005', '理賠金額_住院2006', '理賠金額_住院2007',
       #'理賠金額_住院2008', '理賠金額_住院2009', '理賠金額_住院2010', '理賠金額_住院2011',
       #'理賠金額_住院2012', '理賠金額_住院2013', '理賠金額_住院2014', '理賠金額_住院2015',
       #'理賠金額_手術2002','理賠金額_手術2003', '理賠金額_手術2004', '理賠金額_手術2005',
       #'理賠金額_手術2006', '理賠金額_手術2007', '理賠金額_手術2008', '理賠金額_手術2009',
       #'理賠金額_手術2010', '理賠金額_手術2011', '理賠金額_手術2012', '理賠金額_手術2013',
       #'理賠金額_手術2014','理賠金額_手術2015'],axis=1)    
    
#time_trainx住院.rename(columns = lambda x: 'y'+str(int(x.replace('理賠金額_住院', ''))-2001), inplace=True) 
#time_trainx手術.rename(columns = lambda x: 'y'+str(int(x.replace('理賠金額_手術', ''))-2001), inplace=True)  
    
#time_testx住院 = test_x[['理賠金額_住院2003',
       #'理賠金額_住院2004', '理賠金額_住院2005', '理賠金額_住院2006', '理賠金額_住院2007',
       #'理賠金額_住院2008', '理賠金額_住院2009', '理賠金額_住院2010', '理賠金額_住院2011',
       #'理賠金額_住院2012', '理賠金額_住院2013', '理賠金額_住院2014', '理賠金額_住院2015','理賠金額_住院2016']]

#time_testx手術 = test_x[['理賠金額_手術2003', '理賠金額_手術2004', '理賠金額_手術2005',
       #'理賠金額_手術2006', '理賠金額_手術2007', '理賠金額_手術2008', '理賠金額_手術2009',
       #'理賠金額_手術2010', '理賠金額_手術2011', '理賠金額_手術2012', '理賠金額_手術2013',
       #'理賠金額_手術2014', '理賠金額_手術2015','理賠金額_手術2016']]
#nontime_testx = test_x.drop(['理賠金額_住院2003',
       #'理賠金額_住院2004', '理賠金額_住院2005', '理賠金額_住院2006', '理賠金額_住院2007',
       #'理賠金額_住院2008', '理賠金額_住院2009', '理賠金額_住院2010', '理賠金額_住院2011',
       #'理賠金額_住院2012', '理賠金額_住院2013', '理賠金額_住院2014', '理賠金額_住院2015','理賠金額_住院2016',
       #'理賠金額_手術2003', '理賠金額_手術2004', '理賠金額_手術2005',
       #'理賠金額_手術2006', '理賠金額_手術2007', '理賠金額_手術2008', '理賠金額_手術2009',
       #'理賠金額_手術2010', '理賠金額_手術2011', '理賠金額_手術2012', '理賠金額_手術2013',
       #'理賠金額_手術2014', '理賠金額_手術2015','理賠金額_手術2016'],axis=1)

#time_testx住院.rename(columns = lambda x: 'y'+str(int(x.replace('理賠金額_住院', ''))-2002), inplace=True) 
#time_testx手術.rename(columns = lambda x: 'y'+str(int(x.replace('理賠金額_手術', ''))-2002), inplace=True) 


from sklearn.decomposition import PCA, TruncatedSVD, NMF,MiniBatchSparsePCA,DictionaryLearning,FastICA,FactorAnalysis
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, Normalizer,QuantileTransformer
from sklearn.pipeline import make_pipeline

#Scaler1 = QuantileTransformer()
#reducedPCA1 = PCA(n_components=10)
#Scaler2 = QuantileTransformer()
#reducedPCA2 = PCA(n_components=10)
#reduced_Scaler住院 = make_pipeline(Scaler1, reducedPCA1)
#reduced_Scaler手術 = make_pipeline(Scaler2, reducedPCA2)

#time_trainx住院_pca = reduced_Scaler住院.fit_transform(time_trainx住院)
#time_testx住院_pca = reduced_Scaler住院.transform(time_testx住院)
#time_trainx手術_pca = reduced_Scaler手術.fit_transform(time_trainx手術)
#time_testx手術_pca = reduced_Scaler手術.transform(time_testx手術)

#time_trainx住院_pca = pd.DataFrame(time_trainx住院_pca,columns=['pca_1','pca_2','pca_3','pca_4','pca_5','pca_6','pca_7','pca_8','pca_9','pca_10'])
#time_trainx手術_pca = pd.DataFrame(time_trainx手術_pca,columns=['pca_11','pca_12','pca_13','pca_14','pca_15','pca_16','pca_17','pca_18','pca_19','pca_20'])
#time_testx住院_pca = pd.DataFrame(time_testx住院_pca,columns=['pca_1','pca_2','pca_3','pca_4','pca_5','pca_6','pca_7','pca_8','pca_9','pca_10'])
#time_testx手術_pca = pd.DataFrame(time_testx手術_pca,columns=['pca_11','pca_12','pca_13','pca_14','pca_15','pca_16','pca_17','pca_18','pca_19','pca_20'])

#train_x_pca = nontime_trainx.join(time_trainx住院_pca)
#train_x_pca = train_x_pca.join(time_trainx手術_pca)
#test_x_pca = nontime_testx.join(time_testx住院_pca)
#test_x_pca = test_x_pca.join(time_testx手術_pca)


train_y_binary = train_y.astype('bool').astype('int')
test_y_binary = test_y.astype('bool').astype('int')

# No-PCA


# Use Bayesian to tune hyperparameter
from bayes_opt import BayesianOptimization

def target(train_x, train_y_binary,test_x, test_y_binary,
           n_estimators,learning_rate,num_leaves,colsample_bytree,
           subsample,reg_alpha,reg_lambda,min_split_gain,min_child_weight):
    clf = LGBMClassifier(
            n_estimators=int(n_estimators),
            learning_rate=learning_rate,
            num_leaves=int(num_leaves),
            colsample_bytree=colsample_bytree,
            subsample=subsample,
            max_depth=50,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            silent=-1,
            verbose=-1,
            )
    clf.fit(train_x_pca, train_y_binary, eval_set=[(train_x_pca, train_y_binary), (test_x_pca, test_y_binary)], 
            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 500)
    return clf.best_score_['valid_1']['auc']   


myobject = BayesianOptimization(lambda n_estimators,learning_rate,num_leaves,colsample_bytree,
           subsample,reg_alpha,reg_lambda,min_split_gain,min_child_weight :
           target(train_x, train_y_binary,test_x, test_y_binary,n_estimators,learning_rate,num_leaves,colsample_bytree,
           subsample,reg_alpha,reg_lambda,min_split_gain,min_child_weight),
           {'n_estimators':(100,1000000),'learning_rate':(0.0001,0.5),'num_leaves':(2,150),'colsample_bytree':(0.0001,1),
           'subsample':(0.0001,1),'reg_alpha':(0.0001,1),'reg_lambda':(0.0001,1),'min_split_gain':(0.0001,1),'min_child_weight':(1,1000)})


myobject.explore({'n_estimators':(100,1000000),'learning_rate':(0.0001,0.5),'num_leaves':(2,150),'colsample_bytree':(0.0001,1),
           'subsample':(0.0001,1),'reg_alpha':(0.0001,1),'reg_lambda':(0.0001,1),'min_split_gain':(0.0001,1),'min_child_weight':(1,1000)})

myobject.maximize()
print(myobject.res['max'])

# The best hyperparameters
clf = LGBMClassifier(
            nthread=7,
            is_unbalance=True,
            n_estimators=10000,
            learning_rate=0.01,
             num_leaves=32,
             colsample_bytree=0.9497036,
             subsample=0.8715623,
             max_depth=8,
             reg_alpha=0.04,
             reg_lambda=0.0735294,
             min_split_gain=0.0222415,
             min_child_weight=39.3259775,
            silent=-1,
            #'scale_pos_weight' : 15,
#            'device' : 'gpu',
            tree_learner='voting',
#             'objective':'xentropy',
#             'boosting' : 'dart',
#             'drop_rate':0.2,
# #            'bagging_freq' : 5,
#             'min_gain_to_split':0.2,
#             'histogram_pool_size': 10*1024
            )

#feats = [f for f in train_x.columns if f not in ['被保人ID','ans']]
#sorted(zip(clf.feature_importances_,train_x[feats].columns),reverse=True)

########################################################################################




time_2018x住院 = test2018_x[[
       '理賠金額_住院2004', '理賠金額_住院2005', '理賠金額_住院2006', '理賠金額_住院2007',
       '理賠金額_住院2008', '理賠金額_住院2009', '理賠金額_住院2010', '理賠金額_住院2011',
       '理賠金額_住院2012', '理賠金額_住院2013', '理賠金額_住院2014', '理賠金額_住院2015','理賠金額_住院2016','理賠金額_住院2017']]

time_2018x手術 = test2018_x[['理賠金額_手術2004', '理賠金額_手術2005',
       '理賠金額_手術2006', '理賠金額_手術2007', '理賠金額_手術2008', '理賠金額_手術2009',
       '理賠金額_手術2010', '理賠金額_手術2011', '理賠金額_手術2012', '理賠金額_手術2013',
       '理賠金額_手術2014', '理賠金額_手術2015','理賠金額_手術2016','理賠金額_手術2017']]
nontime_2018x = test2018_x.drop(['理賠金額_住院2004', '理賠金額_住院2005', '理賠金額_住院2006', '理賠金額_住院2007',
       '理賠金額_住院2008', '理賠金額_住院2009', '理賠金額_住院2010', '理賠金額_住院2011',
       '理賠金額_住院2012', '理賠金額_住院2013', '理賠金額_住院2014', '理賠金額_住院2015','理賠金額_住院2016','理賠金額_住院2017',
       '理賠金額_手術2004', '理賠金額_手術2005',
       '理賠金額_手術2006', '理賠金額_手術2007', '理賠金額_手術2008', '理賠金額_手術2009',
       '理賠金額_手術2010', '理賠金額_手術2011', '理賠金額_手術2012', '理賠金額_手術2013',
       '理賠金額_手術2014', '理賠金額_手術2015','理賠金額_手術2016','理賠金額_手術2017'],axis=1)

time_2018x住院.rename(columns = lambda x: 'y'+str(int(x.replace('理賠金額_住院', ''))-2003), inplace=True) 
time_2018x手術.rename(columns = lambda x: 'y'+str(int(x.replace('理賠金額_手術', ''))-2003), inplace=True) 

time_trainx住院_pca = reduced_Scaler住院.fit_transform(time_trainx住院)
time_2018x住院_pca = reduced_Scaler住院.transform(time_2018x住院)
time_trainx手術_pca = reduced_Scaler手術.fit_transform(time_trainx手術)
time_2018x手術_pca = reduced_Scaler手術.transform(time_2018x手術)


time_2018x住院_pca = pd.DataFrame(time_testx住院_pca,columns=['pca_1','pca_2','pca_3','pca_4','pca_5','pca_6','pca_7','pca_8','pca_9','pca_10'])
time_2018x手術_pca = pd.DataFrame(time_testx手術_pca,columns=['pca_11','pca_12','pca_13','pca_14','pca_15','pca_16','pca_17','pca_18','pca_19','pca_20'])

test_2018x_pca = nontime_2018x.join(time_2018x住院_pca)
test_2018x_pca = test_2018x_pca.join(time_2018x手術_pca)


######################################################################
test2018_y_binary = test2018_y.astype('bool').astype('int')

#2016當training
clf.fit(train_x, train_y_binary, eval_set=[(train_x, train_y_binary), (test_x, test_y_binary)], 
            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 500)
ans_train_2016 = clf.predict_proba(test2018_x)
ans_train_2016 = pd.DataFrame(ans_train_2016[:,1],columns=['理賠發生機率'])
ID_2018 = pd.DataFrame(ID_2018,columns=['被保人ID'])
ans_2016 = ID_2018.join(ans_train_2016)

from sklearn.metrics import roc_auc_score
roc_auc_score(test2018_y_binary,ans_train_2017)

#2017當training
clf.fit(test_x, test_y_binary, eval_set=[(test_x, test_y_binary), (train_x, train_y_binary)], 
            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 500)
ans_train_2017 = clf.predict_proba(test2018_x)
ans_train_2017 = pd.DataFrame(ans_train_2017[:,1],columns=['理賠發生機率'])
ans_2017 = ID_2018.join(ans_train_2017)

ans_2016.to_csv('predict_data2016_as_training.csv',index=False)
ans_2017.to_csv('predict_data2017_as_training.csv',index=False)