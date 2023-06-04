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

# In[Functions]
#seek filename in specific folder
def bseek(bootdir,Ftype,Alist):
    import os
    sfds=os.listdir(bootdir) #search under the specific folder
    sfds.sort()
    for sfd in sfds:
      s="/".join((bootdir,sfd))
      if os.path.isdir(s):
        bseek(s,Ftype,Alist)
      elif os.path.isfile(s):
        Alist.append(s)


def check_leap(input_year):
    if ((input_year%400 == 0) | (input_year%4 == 0)):
        return 1
    else:
        return 0

#%% [Step 1: Input data]
# In[Load data]

###------ Load Training and Testing Data ------
Train_path = r"C:\Users\lixiang\Desktop\Data Samples\Sampledata\Trainsample1000.csv"
Train = pd.read_csv(Train_path)
ss = Train.filter(regex = "Unname")
Train = Train.drop(ss, axis = 1) 
del ss,Train_path

Test_path = r"C:\Users\lixiang\Desktop\Data Samples\Sampledata\Testsample1000.csv"
Test = pd.read_csv(Test_path)
ss = Test.filter(regex = "Unname")
Test = Test.drop(ss, axis = 1) 
del ss,Test_path

###------ load point data including lon/lat ------
Point_Lonlat_path = r"C:\Users\lixiang\Desktop\Data Samples\Sampledata\point.csv"
Point_Lonlat = pd.read_csv(Point_Lonlat_path)
ss = Point_Lonlat.filter(regex = "Unname")
Point_Lonlat = Point_Lonlat.drop(ss, axis = 1) 
Point_Lonlat = Point_Lonlat.sort_values(by=['id'] ,ascending=[True])
del ss,Point_Lonlat_path

# In[Chose data ]
# Note:Modeling each index year by year

###------ Choose the Index waiting for training ------ 
year = 2020 #from 2003 to 2020
Train_Ayear = Train.loc[Train['year'].isin([year])]
Test_Ayear = Test.loc[Test['year'].isin([year])]
print("id's shape:",np.unique(Train_Ayear["id"]).shape[0])

Y_col = "SAT" 
# indices including ['SAT', 'WBT', 'ATin', 'ATout', 'ET', 'DI', 'MDI', 'HMI','NET', 'sWBGT', 'WCT', 'HI']
x_train = Train_Ayear.iloc[:,1:8]
y_train = Train_Ayear[Y_col]
x_test = Test_Ayear.iloc[:,1:8]
y_test = Test_Ayear[Y_col]

###------ Dataframe for saving results ------ 
Result_data = pd.DataFrame()
Result_data["id"] = Test_Ayear["id"]
Result_data['lon'] = np.nan
Result_data['lat'] = np.nan
Result_data["year"] = Test_Ayear["year"]
Result_data["month"] = Test_Ayear["month"]
Result_data["day"] = Test_Ayear["day"]
Result_data["y_test"] = y_test

for num in range(Result_data.shape[0]):
    # num = 0
    Aid = Result_data[num:num+1]["id"].values[0] 
    Arowdata = Point_Lonlat.loc[Point_Lonlat['id'].isin([Aid])]
    Result_data.iloc[num,1] = Arowdata["lon"].values[0]
    Result_data.iloc[num,2] = Arowdata["lat"].values[0]
del num,Aid,Arowdata,Point_Lonlat
del Train_Ayear,Test_Ayear

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

#%% [Step 3-4: Train LGBM and Accuracy evaluation]
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
    boosting_type='gbdt',
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

# Note:
# Output the prediction results of a single year, and then combine all 18-year results.
# Accuracy evaluation of index is in the "Fig 9" sections of "Figure code.py"
Result_data["LGBM_y_pred"] = LGBM_y_pred
Result_data["difference"] = Result_data["LGBM_y_pred"]-Result_data["y_test"]
output_path = r"C:/Users/lixiang/Research/Data/TrainModel_PredictedData/{}/{}y_test_data.csv".format(Y_col,year)
Result_data.to_csv(output_path, encoding='utf-8')
del output_path,Result_data

index_list =['SAT', 'WBT', 'ATin', 'ATout', 'ET', 'DI', 'MDI', 'HMI','NET', 'sWBGT', 'WCT', 'HI']
for index_name in index_list:
    Result_data_All = pd.DataFrame() #dataframef for saving predicted results during 2003-2020

    for Ayear in range(2003,2021):
        Ayear_path = r"C:/Users/lixiang/Research/Data/TrainModel_PredictedData/{}/{}y_test_data.csv".format(index_name,Ayear)
        Ayear_index = pd.read_csv(Ayear_path)
        ss = Ayear_index.filter(regex = "Unname")
        Ayear_index = Ayear_index.drop(ss, axis = 1) 
        Ayear_index = Ayear_index.sort_values(by=['id'] ,ascending=[True])
        Result_data_All = pd.concat([Result_data_All,Ayear_index],axis=0,ignore_index=True)
        del Ayear_path,ss,Ayear_index
    
    Result_data_All_exportPath = r"C:\Users\lixiang\Desktop\Data Samples\Fig9_data\{}_AllYears_y_test.csv".format(index_name)
    Result_data_All.to_csv(Result_data_All_exportPath, encoding='utf-8')
    del Result_data_All_exportPath,Result_data_All,Ayear

del Train,Test,x_train,y_train,x_test,y_test

#%% [Step 5: Prodicte HPT - LGBM]
# Note: Using tunned model to predict HPTs when the model had desirable performance

###------ Build estimator ------ 
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
    boosting_type='gbdt',
    objective = 'regression',
    importance_type = "gain",
    n_jobs=-1
)

###------ seek original image and check saving path ------ 
#--- Retrieve the whole year original image ---
Alist = [] #saving path
bootpath = r"/home0/lix/Research/Data/2 Huabei_img/{}/".format(year) 
bseek(bootpath,"tif",Alist)

#preparation
OutputPath_root = "/home0/lix/Research/Data/3 predicting/predicted/{}/".format(year) 
if not os.path.exists(OutputPath_root):
        os.mkdir(OutputPath_root)
OutputPath_root = OutputPath_root + "{}/".format(Y_col)
if not os.path.exists(OutputPath_root):
        os.mkdir(OutputPath_root)

Test_path = r"C:\Users\lixiang\Desktop\Data Samples\Sampledata\Testsample1000.csv"
Test = pd.read_csv(Test_path)
ss = Test.filter(regex = "Unname")
Test = Test.drop(ss, axis = 1) 
Test_Ayear = Test.loc[Test['year'].isin([year])]
x_test = Test_Ayear.iloc[:,1:8]
del ss,Test_path,Test_Ayear

#Run
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

#%% [Step 6: Transform TIF into NetCDF]
import pandas as pd
import numpy as np
from osgeo import gdal
import xarray as xr
import os
import zipfile
import time


###------ TIF to NC ------ 
index_name = "SAT" # ['SAT', 'WBT', 'ATin', 'ATout', 'ET', 'DI', 'MDI', 'HMI','NET', 'sWBGT', 'WCT', 'HI']

#Obtain the latitude and longitude reference
lon_=[]
lat_=[]

tiff_path = "/home0/lix/Research0 - HPT_Huabei/Data/3 Predicting_Image/predicted/2003/Atin/2003-1/2003-1-1_LGBM.tif"
gdal.UseExceptions()
ds = gdal.Open(tiff_path)
im_width = ds.RasterXSize
im_height = ds.RasterYSize
im_geotrans = ds.GetGeoTransform()

for col in range(im_width):
    px_single = im_geotrans[0] + col*im_geotrans[1]
    lon_.append(px_single)
for row in range(im_height):
    py_single = im_geotrans[3] + row*im_geotrans[5]
    lat_.append(py_single)

del ds,im_width,im_height,im_geotrans,col,px_single,row,py_single

      
leap_year = [31,29,31,30,31,30,31,31,30,31,30,30] #only 365
Not_leapyear = [31,28,31,30,31,30,31,31,30,31,30,31]

start_time = time.time()
for NC_year in range(2003,2021):
    tiff = gdal.Open(tiff_path)
    band = tiff.GetRasterBand(1)
    image_= band.ReadAsArray()
    WholeYear_image = np.full([365, image_.shape[0], image_.shape[1]], np.nan)
    del tiff,band,image_
    
    if check_leap(NC_year):
        mon_list = leap_year;
    else:
        mon_list = Not_leapyear;
    
    #merge whole year images
    i=0
    for mon in range(1,13):
        for day in range(1,mon_list[mon-1]+1):
            # print(day)
            path = "/home0/lix/Research/Data/3 predicting/predicted/{}/{}/{}-{}/{}-{}-{}_LGBM.tif".format(NC_year,index_name,NC_year, mon, NC_year, mon, day)
            tiff = gdal.Open(path)
            band = tiff.GetRasterBand(1)
            image= band.ReadAsArray()
            image[image < -9000] = np.nan#去除无效值
            image[image > 9000] = np.nan
            WholeYear_image[i] = image
            i=i+1
            print("--- number of %d has done ---"%(i))
    
    del i,mon,day,path,tiff,band,image
    
    #build time_index
    start = "%d0101"%(NC_year)
    end = "%d12%d"%(NC_year,mon_list[11])
    time_= pd.date_range(start=start, end=end, freq="D")
    del start,end
    
    #Build NC data
    WholeYear_image = xr.DataArray(WholeYear_image,
                      coords={
                              "time":time_,
                              "lon":lon_,
                              "lat":lat_
                          },
                       dims=["time","lat","lon"],
                      )#dims对应的是array的shape
    
    #output folder
    export_RootPath = "/home0/lix/Research/Data/4 NC_Data/1 Yearly_NCData/{}".format(index_name)
    if not os.path.exists(export_RootPath):
            os.mkdir(export_RootPath)
    
    OutputPath = export_RootPath + "/{}_{}_DailyData.nc".format(index_name,NC_year)
    WholeYear_image.to_netcdf(OutputPath)
    print("======== Year: %d has done ========"%(NC_year))
    
end_time = time.time()
print('Costs {} s'.format(end_time-start_time)) 
print("========== {} All have done ==========".format(index_name))
del NC_year,mon_list
del WholeYear_image,time_
del export_RootPath,OutputPath
del start_time,end_time
del leap_year,Not_leapyear


###------ Compress NC (To int type) ------ 
for year in range(2003,2021):
    output_path = "/home0/lix/Research/Data/4 NC_Data/2 Compress_yearly"
    Yearly_data = xr.open_dataset(r"/home0/lix/Research/Data/4 NC_Data/1 Yearly_NCData/{}/{}_{}_DailyData.nc".format(index_name,index_name,year))
    Yearly_data = Yearly_data['__xarray_dataarray_variable__'].to_dataset(name='%s'%index_name)
    
    #to int 
    Yearly_data = Yearly_data*100
    Yearly_data['%s'%index_name] = Yearly_data['%s'%index_name].astype(np.int16)
    
    #re-output
    OutputPath_root = output_path + "/{}/".format(index_name)
    del output_path
    if not os.path.exists(OutputPath_root):#如果没有文件夹则建立一个
        os.mkdir(OutputPath_root)
    
    #save as ".nc"
    output_path = OutputPath_root + "HiTIC_NCP_Daily_{}_{}.nc".format(index_name,year)
    Yearly_data.to_netcdf(output_path) #导出
    print("===== {}-{} has done! =====".format(index_name,year))


###------ Conver to RAR ------ 
for year in range(2003,2021):
    input_path_root = "/home0/lix/Research/Data/4 NC_Data/"
    input_path = input_path_root + "2 Compress_yearly/{}/HiTIC_NCP_Daily_{}_{}.nc".format(index_name,index_name,year)
    
    #output folder
    outputPath_root = "/home0/lix/Research/Data/4 NC_Data/3 RAR_files/{}/".format(index_name) #自动建立文件夹
    if not os.path.exists(outputPath_root):#如果没有文件夹则建立一个
        os.mkdir(outputPath_root)
    
    #save as ".rar"
    outputPath = outputPath_root + "HiTIC_NCP_Daily_{}_{}.rar".format(index_name,year)
    zipped_file = zipfile.ZipFile(outputPath,"w")
    zipped_file.write(input_path)
    print("===== Zipped: {}-{} =====".format(index_name,year))
    zipped_file.close()

