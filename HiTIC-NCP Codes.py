# -*- coding: utf-8 -*-

# In[Import python libs]
import numpy as np
import pandas as pd
import time
import lightgbm as lgb
from sklearn import metrics
from osgeo import gdal
import os
from sklearn.model_selection import GridSearchCV

# In[Functions: seek filename in specific folder]
def bseek(bootdir,Ftype,Alist):
    '''
    Parameters
    ----------
    bootdir : Home folder to retrieve
    Ftype : The data suffix to be found
    Alist : list to store filename
    '''
    import os
    sfds=os.listdir(bootdir) #search under the specific folder
    sfds.sort()
    for sfd in sfds:
      s="/".join((bootdir,sfd))
      if os.path.isdir(s):
        bseek(s,Ftype,Alist)
      elif os.path.isfile(s):
        Alist.append(s)


#%% [Step 1: Input data]
# In[Load data]

###------ Load Training and Testing Data ------ 
Xn_train_path = r"/home0/lix/Research/Data/2 Split Data/Xn_train.csv"
Xn_train = pd.read_csv(Xn_train_path)
ss = Xn_train.filter(regex = "Unname")
Xn_train = Xn_train.drop(ss, axis = 1) 
del ss,Xn_train_path

Yn_train_path = r"/home0/lix/Research/Data/2 Split Data/Yn_train.csv"
Yn_train = pd.read_csv(Yn_train_path)
ss = Yn_train.filter(regex = "Unname")
Yn_train = Yn_train.drop(ss, axis = 1)
del ss,Yn_train_path

Xn_test_path = r"/home0/lix/Research/Data/2 Split Data/Xn_test.csv"
Xn_test = pd.read_csv(Xn_test_path)
ss = Xn_test.filter(regex = "Unname")
Xn_test = Xn_test.drop(ss, axis = 1) 
del ss,Xn_test_path

Yn_test_path = r"/home0/lix/Research/Data/2 Split Data/Yn_test.csv"
Yn_test = pd.read_csv(Yn_test_path)
ss = Yn_test.filter(regex = "Unname")
Yn_test = Yn_test.drop(ss, axis = 1) 
del ss,Yn_test_path

###------ load point data including lon/lat ------
Point_Lonlat_path = r"/home0/lix/Research/Data/0 huabei_Shpfile/point.csv"
Point_Lonlat = pd.read_csv(Point_Lonlat_path)
ss = Point_Lonlat.filter(regex = "Unname")
Point_Lonlat = Point_Lonlat.drop(ss, axis = 1) 
Point_Lonlat = Point_Lonlat.sort_values(by=['id'] ,ascending=[True])
del ss,Point_Lonlat_path
# In[Chose data ]
# Note:Modeling each index year by year

###------ Choose the Index waiting for training ------ 
year = 2020 #from 2003 to 2020
Xn_train_Ayear = Xn_train.loc[Xn_train['year'].isin([year])]
Xn_test_Ayear = Xn_test.loc[Xn_test['year'].isin([year])]
Yn_train_Ayear = Yn_train.loc[Yn_train['year'].isin([year])]
Yn_test_Ayear = Yn_test.loc[Yn_test['year'].isin([year])]
print("id's shape:",np.unique(Xn_train["id"]).shape[0])

Y_col = "SAT" 
# indices including ['SAT', 'WBT', 'ATin', 'ATout', 'ET', 'DI', 'MDI', 'HMI','NET', 'sWBGT', 'WCT', 'HI']
x_train = Xn_train_Ayear.drop(columns=['id', 'year', 'month', 'day'],axis=1)
x_test = Xn_test_Ayear.drop(columns=['id', 'year', 'month', 'day'],axis=1)
y_train = Yn_train_Ayear[Y_col]
y_test = Yn_test_Ayear[Y_col]


#%% [Step 2: Hyperparameter Optimization]
# Note: The value and its testing range in this parts can be set by yourself
 

##---------- setp 1:  n_estimators ---------- 
start_time1 = time.time()
params = {    
    'boosting_type': 'gbdt', 
    'objective': 'regression', 
    'learning_rate': 0.1, 
    'num_leaves': 50, 
    'max_depth': 6,
    'subsample': 0.8, 
    'colsample_bytree': 0.8, 
    }

data_train = lgb.Dataset(x_train, y_train, silent=True)

cv_results = lgb.cv(
        params, 
        data_train, 
        num_boost_round=100000, 
        nfold=5, 
        stratified=False, 
        shuffle=True, 
        metrics='rmse',
        early_stopping_rounds=1000, 
        verbose_eval=100, 
        show_stdv=True, 
        seed=0)

best_estimator = len(cv_results['rmse-mean'])
print('best n_estimators:',best_estimator )
print('best cv score:', cv_results['rmse-mean'][-1])
end_time1 = time.time()
print("cose times:",end_time1-start_time1)


##---------- Step 2: max_dept and num_leaves ---------- 
#from sklearn.model_selection import GridSearchCV
start_time = time.time()
model_lgb = lgb.LGBMRegressor(objective='regression',
                boosting_type= 'gbdt', 
                num_leaves=50,
                learning_rate=0.1, 
                n_estimators = best_estimator, 
                # max_depth = 6,
                metric='rmse', 
                bagging_fraction = 0.8,
                feature_fraction = 0.8)

 
params_test1={    
'max_depth': range(2, 11, 3),    
'num_leaves':range(10, 150, 30)
}

gsearch1 = GridSearchCV(estimator=model_lgb, 
             param_grid=params_test1, 
             scoring='neg_mean_squared_error', 
             cv=5, 
             verbose=1, 
             n_jobs=-1)

gsearch1.fit(x_train, y_train)
best_depth = gsearch1.best_params_["max_depth"]
best_leaves = gsearch1.best_params_["num_leaves"]
print("CV:", gsearch1.cv_results_) 
print("best params:",gsearch1.best_params_)
print("best score:", gsearch1.best_score_)
end_time2 = time.time()
print("cose times:",end_time2-start_time)

##--- Refinement Testing for max_depth and num_leaves --- 
# Note: this can be added repeatedly
start_time = time.time()
model_lgb = lgb.LGBMRegressor(objective='regression',
                boosting_type= 'gbdt',
                learning_rate=0.1, 
                n_estimators = best_estimator, 
                metric='rmse', 
                bagging_fraction = 0.8,
                feature_fraction = 0.8)
params_test2={
    'max_depth': [best_depth-1,best_depth,best_depth+1], 
    'num_leaves':[best_leaves-5,best_leaves,best_leaves+5], 
}
 
gsearch2 = GridSearchCV(estimator=model_lgb, 
             param_grid=params_test2,
             scoring='neg_mean_squared_error', 
             cv=5, 
             verbose=1,
             n_jobs=-1)

gsearch2.fit(x_train, y_train)
best_depth = gsearch2.best_params_["max_depth"]
best_leaves = gsearch2.best_params_["num_leaves"]
print("CV:", gsearch2.cv_results_) 
print("best params:",gsearch2.best_params_)
print("best score:", gsearch2.best_score_)
end_time3 = time.time()
print("cose times:",end_time3-start_time)


##---------- setp 3: min_data_in_leaf and min_sum_hessian_in_leaf ---------- 
start_time = time.time()
params_test3={
    'min_child_samples': [6,9,12,16],
    'min_child_weight':[0.001,0.002]
}

model_lgb = lgb.LGBMRegressor(objective='regression',
                boosting_type= 'gbdt',
                n_estimators = best_estimator,
                num_leaves = best_leaves,
                max_depth = best_depth, 
                metric='rmse', 
                bagging_fraction = 0.8, 
                feature_fraction = 0.8,
                learning_rate= 0.1 )

gsearch3 = GridSearchCV(estimator=model_lgb, 
             param_grid=params_test3, 
             scoring='neg_mean_squared_error', 
             cv=5, 
             verbose=1, 
             n_jobs=-1)


gsearch3.fit(x_train, y_train)

best_child_samples = gsearch3.best_params_["min_child_samples"]
best_child_weight = gsearch3.best_params_["min_child_weight"]
print("CV:", gsearch3.cv_results_) 
print("best params:",gsearch3.best_params_)
print("best score:", gsearch3.best_score_)
end_time4 = time.time()
print("cose times:",end_time4-start_time)

##--- Refinement Testing for min_data_in_leaf and min_sum_hessian_in_leaf --- 
# Note: this can be added repeatedly
start_time = time.time()
params_test3={
    'min_child_samples': [best_child_samples-2,best_child_samples-1,best_child_samples,best_child_samples+1,best_child_samples+2],
    'min_child_weight':[best_child_weight]
}

model_lgb = lgb.LGBMRegressor(objective='regression',
                boosting_type= 'gbdt',
                n_estimators = best_estimator,  
                num_leaves = best_leaves,
                max_depth = best_depth, 
                metric='rmse', 
                learning_rate= 0.1, 
                bagging_fraction = 0.8, 
                feature_fraction = 0.8,)

gsearch3 = GridSearchCV(estimator=model_lgb, 
             param_grid=params_test3, 
             scoring='neg_mean_squared_error', 
             cv=5, 
             verbose=1, 
             n_jobs=-1)

gsearch3.fit(x_train, y_train)

best_child_samples = gsearch3.best_params_["min_child_samples"]
print("CV:", gsearch3.cv_results_) 
print("best params:",gsearch3.best_params_)
print("best score:", gsearch3.best_score_)
end_time5 = time.time()
print("cose times:",end_time5-start_time)


##---------- Step 4: feature_fraction and bagging_fraction ---------- 
start_time = time.time()
params_test4={
    'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9],
    'bagging_fraction': [0.6, 0.8, 1.0]
}

model_lgb = lgb.LGBMRegressor(objective='regression',
                boosting_type= 'gbdt',
                n_estimators = best_estimator,
                num_leaves = best_leaves,
                max_depth = best_depth, 
                
                min_child_samples = best_child_samples, 
                min_child_weight = best_child_weight,
                metric='rmse', 
                bagging_freq = 5,  
                learning_rate= 0.1,
                )

gsearch4 = GridSearchCV(estimator=model_lgb, 
             param_grid=params_test4,
             scoring='neg_mean_squared_error', 
             cv=5, 
             verbose=1, 
             n_jobs=4)

gsearch4.fit(x_train, y_train)

best_feature_fraction = gsearch4.best_params_["feature_fraction"]
best_bagging_fraction = gsearch4.best_params_["bagging_fraction"]
print("CV:", gsearch4.cv_results_) 
print("best params:",gsearch4.best_params_)
print("best score:", gsearch4.best_score_)
end_time6 = time.time()
print("cose times:",end_time6-start_time)

##--- Refinement Testing for feature_fraction --- 
# Note: this can be added repeatedly
start_time = time.time()
params_test5={
    'feature_fraction': [best_feature_fraction-0.1,best_feature_fraction-0.05,best_feature_fraction-0.02,best_feature_fraction,best_feature_fraction+0.02,best_feature_fraction+0.05,best_feature_fraction+0.1]
}

model_lgb = lgb.LGBMRegressor(objective='regression',
                boosting_type= 'gbdt',
                n_estimators = best_estimator, 
                num_leaves = best_leaves,
                max_depth = best_depth, 
                
                min_child_samples = best_child_samples,
                min_child_weight = best_child_weight,
                
                bagging_fraction= best_bagging_fraction,
                metric='rmse', 
                bagging_freq = 5,  
                learning_rate= 0.1,
                )

gsearch5 = GridSearchCV(estimator=model_lgb, 
             param_grid=params_test5,                       
             scoring='neg_mean_squared_error', 
             cv=5, 
             verbose=1, 
             n_jobs=4)

gsearch5.fit(x_train, y_train)

best_feature_fraction = gsearch5.best_params_["feature_fraction"]
print("CV:", gsearch5.cv_results_) 
print("best params:",gsearch5.best_params_)
print("best score:", gsearch5.best_score_)
end_time7 = time.time()
print("cose times:",end_time7-start_time)


##---------- step 5: learning_rate ---------- 
start_time = time.time()
params_test5={
    'learning_rate': [0.1,0.001,0.002,0.005,0.008]
}

model_lgb = lgb.LGBMRegressor(objective='regression',
                boosting_type= 'gbdt',
                n_estimators = best_estimator, 
                num_leaves = best_leaves,
                max_depth = best_depth, 
                
                min_child_samples = best_child_samples,
                min_child_weight = best_child_weight,
                
                bagging_fraction= best_bagging_fraction,
                feature_fraction = best_feature_fraction,
    
                metric='rmse', 
                bagging_freq = 5,  
                )

gsearch5 = GridSearchCV(estimator=model_lgb, 
             param_grid=params_test5,                       
             scoring='neg_mean_squared_error', 
             cv=5, 
             verbose=1, 
             n_jobs=4)

gsearch5.fit(x_train, y_train)


best_learning_rate = gsearch5.best_params_["learning_rate"]
print("CV:", gsearch5.cv_results_) 
print("best params:",gsearch5.best_params_)
print("best score:", gsearch5.best_score_)
end_time8 = time.time()
print("cose times:",end_time8-start_time)

print("best_estimator:",best_estimator)
print("best_leaves:",best_leaves)
print("best_depth:",best_depth)
print("best_child_samples:",best_child_samples)
print("best_child_weight:",best_child_weight)
print("best_bagging_fraction:",best_bagging_fraction)
print("best_feature_fraction:",best_feature_fraction)
print("best_learning_rate:",best_learning_rate)
end_time9 = time.time()
print("cose times:",end_time9-start_time1)

#%% [Step 3: Train LGBM]
# In[By lightgbm (lgb)]

###------ Build estimator ------ 
start_time = time.time()
LightGBM_model = lgb.LGBMRegressor(
    n_estimators = best_estimator,
    num_leaves = best_leaves,
    max_depth = best_depth, 
    
    min_child_samples = best_child_samples, 
    min_child_weight = best_child_weight,
    
    bagging_fraction= best_bagging_fraction,
    feature_fraction = best_feature_fraction,
    
    learning_rate = best_learning_rate,
    
    verbosity = 30,
    boosting_type='gbdt', #default is gbdt
    objective = 'regression',
    importance_type = "gain",
    n_jobs=-1
)
end_time1 = time.time()

###------ Train ------ 
LightGBM_model.fit(x_train,y_train)
end_time2 = time.time()

###------ Predict ------ 
LGBM_y_pred = LightGBM_model.predict(x_test) 
end_time3 = time.time()

print('LGBM-Building estimator uses {} s'.format(end_time1-start_time))
print('LGBM-Training model uses {} s'.format(end_time2-end_time1))
print('LGBM-Predicting model uses {} s'.format(end_time3-end_time2))

###------ Accuracy Assessment ------ 
print('R²： ',metrics.r2_score(y_test, LGBM_y_pred,multioutput='uniform_average'))
print('Mean Absolute Error:  ', metrics.mean_absolute_error(y_test, LGBM_y_pred))
print('Root Mean Squared Error:  ',np.sqrt(metrics.mean_squared_error(y_test, LGBM_y_pred)))

del start_time, end_time1, end_time2, end_time3

#%% [Step 4: Prodicte HPT - LGBM]

###------ seek original image and check saving path ------ 
#--- Retrieve the whole year original image---
Alist = [] #saving path
bootpath = r"/media/lix/Saving/Original/2 Huabei_img/{}/".format(year) 
bseek(bootpath,"tif",Alist)

#--- add the corresponding folde ---
OutputPath_root = "/home0/lix/Research/Data/3 Predicting_Image/predicted/{}/".format(year) 
if not os.path.exists(OutputPath_root):
        os.mkdir(OutputPath_root)
OutputPath_root = OutputPath_root + "{}/".format(Y_col)
if not os.path.exists(OutputPath_root):
        os.mkdir(OutputPath_root)


regsor = "_LGBM"  #name suffix

print('{} start:'.format(regsor))
startTime11 = time.time()
for num in range(len(Alist)):
    img_meta = gdal.Open(Alist[num])
    img = img_meta.ReadAsArray()
    # print(img)
    
    new_shape = (img.shape[0],-1)
    img_as_array = img[:,:,:].reshape(new_shape).T
    
    #Mask
    mask = pd.DataFrame(img_as_array)!= -9999.0
    final_mask = mask[0]
    for col in mask.columns:
        final_mask = np.bitwise_and(final_mask,mask[col])
    
    #Fill the null with 0
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values = -9999.0, strategy = 'constant', fill_value = 0)
    mask_all = imputer.fit_transform(img_as_array)
    mask_all = pd.DataFrame(mask_all) 
    mask_all.columns = x_test.columns
    
    #predict
    img_pred = LightGBM_model.predict(mask_all) 
    
    img_pred = img_pred.reshape(img[1,:,:].shape)
    final_mask = final_mask.values.reshape(img[1,:,:].shape)
      
    #adjust mask
    import numpy.ma as ma
    img_pred = img_pred * final_mask
    masked_img_pred = ma.masked_array(data=img_pred, mask=~final_mask)
    filled_masked_img_pred = masked_img_pred.filled(-9999.0)
    
    #output image
    #--- set name rule ---
    OutputPath_1 = OutputPath_root + Alist[num].split("/")[8] 
    if not os.path.exists(OutputPath_1):
        os.mkdir(OutputPath_1)
    
    output_filename = OutputPath_1 + "/" +Alist[num].split("/")[9].split(".")[0][:-6] + regsor + ".tif"
    
    x_pixels = img_pred.shape[1]
    # print(x_pixels)
    y_pixels = img_pred.shape[0]
    # print(y_pixels)
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_filename, x_pixels, y_pixels, 1, gdal.GDT_Float32)
      
    #Add geographic coordinates and projections
    geotrans = img_meta.GetGeoTransform()
    proj = img_meta.GetProjection()
    dataset.SetGeoTransform(geotrans)
    dataset.SetProjection(proj)
    dataset.GetRasterBand(1).WriteArray(filled_masked_img_pred)
    dataset.GetRasterBand(1).SetNoDataValue(-9999.0)
    dataset.FlushCache()
    dataset = None
    print('--- {} has Done! ---'.format(Alist[num].split("/")[9].split(".")[0][:-6]))

startTime12 = time.time()
print("%f"%(startTime12-startTime11))

print('====== {} have Done! ======'.format(regsor[1:]))

