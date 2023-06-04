# %%[1 Obtain data - by GEE]
# Note:this section code includes two parts: (1) obtaining daily multi-band images, 
# and (2) extracting daily gridded datasets using meteorological stations.
# The results obtained in Part 2 (".csv ") will be cleaned and divided into training sets and validation sets

//Function → check leap year
function Check_year(years){
  if(years % 4 === 0 && years % 100 !== 0 || years % 400 === 0)
    return 1;
  else
    return 0;
}

//Function → obtain daily LST collection of specific year
// var year=2004; //for test
function Allyearcol(year){
  //January
  var first_path ='users/LeeXiang/China_lst'+year+'/month1';
  var lst_img = ee.Image(first_path);
  var singleImg = lst_img.slice(0,1);
  var list_img = ee.List([singleImg]);
  // print(list_img);
  
  for(var i=1;i<31;i++){
      singleImg = lst_img.slice(i,i+1);
      list_img = list_img.add(singleImg);
      // print(list_img);
    }
  // print(list_img);
  // print("part1:Done!");
  
  //other months
  
  var leap_month = [31,29,31,30,31,30,31,31,30,31,30,30];
  var Nonleap_month = [31,28,31,30,31,30,31,31,30,31,30,31];

  if(Check_year(year)==1){
    for(var num_y=2;num_y<13;num_y++){
      // var num_y=8; //for test
      var img_path = 'users/LeeXiang/China_lst'+year+'/month'+num_y;
      // print(img_path);
      var mul_bands = ee.Image(img_path);
      // print(mul_bands);
      var endBand_num_y = leap_month[num_y-1];
      // print(endBand_num_y);
      for(i=0;i<endBand_num_y;i++){
        singleImg = mul_bands.slice(i,i+1);
        list_img = list_img.add(singleImg);
      }
      // print("list_img",list_img)
    }
  }
  else{
    for(var num_n=2;num_n<13;num_n++){
      var img_path_n = 'users/LeeXiang/China_lst'+year+'/month'+num_n;
      // print(img_path_n);
      var mul_bands_n = ee.Image(img_path_n);
      // print(mul_bands_n);
      var endBand_num_n = Nonleap_month[num_n-1];
      // print(endBand_num_n);
      for(i=0;i<endBand_num_n;i++){
        singleImg = mul_bands_n.slice(i,i+1);
        list_img = list_img.add(singleImg);
      }
    }
  }
  // print("part2:Done!");

  
  var lst = ee.ImageCollection.fromImages(list_img);
  // print("LST_imagecollection",lst);
  return ee.ImageCollection(lst);
}

//Function → build daily multiple bands image 
function CombineImge(t1,t2,lst_collection){ 
  // t1:starttime
  // t2:endtime
  // lst_collection:ImageCollection for LST
  
  /* doy/dem/aspect/slope */
  var dem_clipped = ee.Image("MERIT/DEM/v1_0_3")
                      .select("dem")
                      .reproject('EPSG:4326', null, 1000);
  
  var slope = ee.Terrain.slope(dem_clipped)
                        .reproject('EPSG:4326', null, 1000)
                        .clip(area.geometry())
                        .add(ee.Image(3000))
                        .unmask(-6999)
                        .subtract(ee.Image(3000));
                        
  var aspect = ee.Terrain.aspect(dem_clipped)
                        .reproject('EPSG:4326', null, 1000)
                        .clip(area.geometry())
                        .add(ee.Image(3000))
                        .unmask(-6999)
                        .subtract(ee.Image(3000));
                         
  var dem = dem_clipped.clip(area.geometry())
                       .add(ee.Image(3000))
                       .unmask(-6999)
                       .subtract(ee.Image(3000));
                       
  var doy = t1.getRelative("day", "year");
  
  var dayofyear = ee.Image(doy.toInt())
                    .setDefaultProjection('EPSG:4326', null, 1000)
                    .clip(area.geometry());
                    
  var lon = dayofyear.addBands(dem)
                    .addBands(aspect)
                    .addBands(slope)
                    .rename(['doy','dem','aspect','slope']);//rename
                     
  // print(dem_clipped);
  // print(lon);
  // print(slope);
  // print(aspect);
  
  //***Visualization***
  // var visualization = {
  //   bands: ['dem'],
  //   min: -3,
  //   max: 18,
  //   palette: ['000000', '478FCD', '86C58E', 'AFC35E', '8F7131',
  //           'B78D4F', 'E2B8A6', 'FFFFFF']
  // };
  // Map.addLayer(dem,visualization,"dem")
  // Map.addLayer(aspect,{},"aspect")
  // Map.addLayer(slope,{},"slope")
  // Map.addLayer(dayofyear,{},"dayofyear")
  // Map.addLayer(lon,{},"lon")
  
  /* lst */
  var imgcolList = lst_collection.toList(365);
  var doy1 = t1.getRelative("day", "year");
  // print("doy",doy);
  
  function getImg(num){
    var lst = ee.Image(imgcolList.get(doy1))
                 .rename("lst")
                 .add(ee.Image(3000))
                 .unmask(-6999,false) 
                 .subtract(ee.Image(3000))
                 .clip(area.geometry());
    return lst;
  }
  function getImg366(){
    return ee.Image(-9999).reproject('EPSG:4326', null, 1000)
                            .clip(area.geometry())
                            .rename("lst");
  }
  
  var lst = ee.Image(ee.Algorithms.If(doy1.gte(365),getImg366(),getImg(doy1)));
  // print("lst",lst);
  
  //Visualization
  // var visParams = {
  //     min: -20.0,  // Degrees C
  //     max: 12.0,
  //     palette: ['000000', '005aff', '43c8c8', 'fff700', 'ff0000'],
  //     };

  // Map.addLayer(lst,visParams,"lst")
  
  /* pop */
  var t5 = ee.Date.fromYMD({
        day:1,
        month:1,
        year:y,
  });
  var t6 = ee.Date.fromYMD({
        day:31,
        month:12,
        year:y,
  });
  var pop_dataset = ee.ImageCollection("WorldPop/GP/100m/pop")
                      .select('population')
                      .filterDate(t5,t6);
  // print("pop_dataset",pop_dataset);
  
  var pop_img = pop_dataset.median()
                  .reproject('EPSG:4326', null, 1000)
                  .clip(area.geometry())
                  .rename('pop')
                  .add(ee.Image(3000))
                  .unmask(-6999)
                  .subtract(ee.Image(3000));             
  // print("pop_img",pop_img);
  
  var pop_substt =ee.Image(-9999) //Create an empty set(-9999) to avoid having no data
                    .setDefaultProjection('EPSG:4326', null, 1000)
                    .clip(area.geometry())
                    .rename('pop');
  // print("pop_substt",pop_substt);
  
  var pop = ee.Image(ee.Algorithms.If(pop_dataset.size(), pop_img, pop_substt));
  // print('pop',pop);
  
  /* Precipitable water vapor */
  var wv_dataset = ee.ImageCollection('MODIS/006/MCD19A2_GRANULES')
                  .select('Column_WV')
                  .filterDate(t1, t2);
  // print('wv_dataset',wv_dataset);
  
  var wv_img = wv_dataset.mean()
                         .multiply(0.001)
                         .reproject('EPSG:4326', null, 1000)
                         .clip(area.geometry())
                         .rename('wv')
                         .add(ee.Image(3000))
                         .unmask(-6999)
                         .subtract(ee.Image(3000));
  // print('wv_img',wv_img);
  
  var wv_substt =ee.Image(-9999)
                   .setDefaultProjection('EPSG:4326', null, 1000)
                   .clip(area.geometry())
                   .rename('wv');
  // print("ndvi_substt",ndvi_substt);
  
  var wv = ee.Image(ee.Algorithms.If(wv_dataset.size(), wv_img, wv_substt));
  // print('wv',wv);
  
  //Visualization
  // var visParams = {
  //   min: -1,
  //   max: 1,
  //   palette: ['000000', '005aff', '43c8c8', 'fff700', 'ff0000'],
  // };
  
  // Map.addLayer(wv,visParams,"wv")

  /* combined as a multi-band image */
  var final = lon.addBands(lst)
                 .addBands(pop)
                 .addBands(wv)
                 .toFloat()
                 .set("system:time_start", t1);//add a timestamp
  
  return ee.Image(final);
}

//Function → download daily multiple bands image 
function ExportImge(mul_img,y,m,d){
  Export.image.toDrive({
          image: mul_img,
          description: y+'-'+m+'-'+d+'image',
          folder: "Image",
          region: area.geometry(),
          scale: 1000,
          crs: 'EPSG:4326'
        });
}

//Function → Extracted daily data by meteorological stations 
function ExportData(t1,multibandsImg,Point,foldername,columns_name){
  
  //Using point to extract the image value
  var final_point = multibandsImg.reduceRegions({
    collection:Point,
    reducer:ee.Reducer.mean(),
    scale:1000
  });
  
  //rename filename
  var year =ee.String(t1.get("year"));
  var month =ee.String(t1.get("month"));
  var day =ee.String(t1.get("day"));
  var filename_js = year.cat("-").cat(month).cat("-").cat(day).cat("final");
  var filename = filename_js.getInfo();
  
  //export point value
  Export.table.toDrive({
    collection:final_point,
    description:filename,
    folder:foldername,
    fileNamePrefix:filename,
    fileFormat: 'CSV',
    selectors: columns_name
  });
}

/*=== extracted daily gridded datasets by weather stations ===*/
//Load data
var China2419point = ee.FeatureCollection("users/LeeXiang/HPT/point");
var area = ee.FeatureCollection("users/LeeXiang/HPT/huabei_area");

//Extracted by point
var y =2020;
var starttime = ee.Date.fromYMD(y, 1, 1); //Need to adjust
var endtime = starttime.getRange('year').end();
var doyAll = endtime.advance(-1,"day").getRelative("day", "year");
// print(starttime);
// print(endtime);
// print(doyAll);
if(Check_year(y)){
  var wholeyear = ee.List.sequence(0,doyAll.subtract(1));
  // print("wholeyear1",wholeyear);
}
else{
  var wholeyear = ee.List.sequence(0,doyAll);
  // print("wholeyear2",wholeyear);
}
var lst_collection = Allyearcol(y);
// print("lst_collection",lst_collection);
var foldername = "Huabei_HPT";
var columns_name = ["id","doy","dem","aspect","slope","lst","pop","wv"];

wholeyear.evaluate(function(numbers){
  
  numbers.map(function(number){
    var startDay = starttime.advance(number,"day");
    // print(startDay);
    var nextDay = startDay.advance(1,"day");
    // print(nextDay)
    var image = CombineImge(startDay,nextDay,lst_collection);
    ExportData(startDay,image,China2419point,foldername,columns_name);
  });
});

/*=== export multi-band image ===*/
var y=2006; 
for(var m=1;m<13;m++)
{
if( m===1 || m===3 || m===5 || m===7 || m===8 || m===10 || m===12)
{
  for(var d=1;d<31;d++)
  {
	var t1 = ee.Date.fromYMD({
		  day:d,
		  month:m,
		  year:y,
		  //timeZone:'Asia/Harbin',//黑龙江和吉林省时区
	});
	var t2 = ee.Date.fromYMD({
		  day:d+1,
		  month:m,
		  year:y,
		  //timeZone:'Asia/Harbin',
	});
	//Obtain lst_collection
	var lst_collection = Allyearcol(y);
	//Obtain multiple bands
	var final = CombineImge(t1,t2,lst_collection);
	//Export image with all bands
	ExportImge(final,y,m,d);
  }
  d=31;
  if(m===12)
  {
	var t1 = ee.Date.fromYMD({
		day:d,
		month:m,
		year:y,
	});
	var t2 = ee.Date.fromYMD({
		  day:1,
		  month:1,
		  year:y+1,
	});
  }
  else
  {
	var t1 = ee.Date.fromYMD({
		day:d,
		month:m,
		year:y,
	});
	var t2 = ee.Date.fromYMD({
		  day:1,
		  month:m+1,
		  year:y,
	});
  }
  //Obtain lst_collection
  var lst_collection = Allyearcol(y);
  //Obtain multiple bands
  var final = CombineImge(t1,t2,lst_collection);
  //Export image with all bands
  ExportImge(final,y,m,d);
}

else if(m===4 || m===6 || m===9 || m===11)
  {
	for(var d=1;d<30;d++)
	{
	  var t1 = ee.Date.fromYMD({
			day:d,
			month:m,
			year:y,
	  });
	  var t2 = ee.Date.fromYMD({
			day:d+1,
			month:m,
			year:y,
	  });
	  //Obtain lst_collection
	  var lst_collection = Allyearcol(y);
	  //Obtain multiple bands
	  var final = CombineImge(t1,t2,lst_collection);
	  //Export image with all bands
	  ExportImge(final,y,m,d);
	}
	d=30;
	var t1 = ee.Date.fromYMD({
		day:d,
		month:m,
		year:y,
	});
	var t2 = ee.Date.fromYMD({
		  day:1,
		  month:m+1,
		  year:y,
	});
	//Obtain lst_collection
	var lst_collection = Allyearcol(y);
	//Obtain multiple bands
	var final = CombineImge(t1,t2,lst_collection);
	//Export image with all bands
	ExportImge(final,y,m,d);
  }

else //February
{
  if(Check_year(y)===1)
  {
	for(var d=1;d<29;d++)
	{
	  var t1 = ee.Date.fromYMD({
			day:d,
			month:m,
			year:y,
	  });
	  var t2 = ee.Date.fromYMD({
			day:d+1,
			month:m,
			year:y,
	  });
	  //Obtain lst_collection
	  var lst_collection = Allyearcol(y);
	  //Obtain multiple bands
	  var final = CombineImge(t1,t2,lst_collection);
	  //Export image with all bands
	  ExportImge(final,y,m,d);
	}
	d=29;
	var t1 = ee.Date.fromYMD({
		day:d,
		month:m,
		year:y,
	});
	var t2 = ee.Date.fromYMD({
		  day:1,
		  month:m+1,
		  year:y,
	});
	//Obtain lst_collection
	var lst_collection = Allyearcol(y);
	//Obtain multiple bands
	var final = CombineImge(t1,t2,lst_collection);
	//Export image with all bands
	ExportImge(final,y,m,d);
  }
  else 
  {
	for(var d=1;d<28;d++)
	{
	  var t1 = ee.Date.fromYMD({
			day:d,
			month:m,
			year:y,
	  });
	  var t2 = ee.Date.fromYMD({
			day:d+1,
			month:m,
			year:y,
	  });
	  //Obtain lst_collection
	  var lst_collection = Allyearcol(y);
	  //Obtain multiple bands
	  var final = CombineImge(t1,t2,lst_collection);
	  //Export image with all bands
	  ExportImge(final,y,m,d);
	}
	d=28;
	var t1 = ee.Date.fromYMD({
		day:d,
		month:m,
		year:y,
	});
	var t2 = ee.Date.fromYMD({
		  day:1,
		  month:m+1,
		  year:y,
	});
	//Obtain lst_collection
	var lst_collection = Allyearcol(y);
	//Obtain multiple bands
	var final = CombineImge(t1,t2,lst_collection);
	//Export image with all bands
	ExportImge(final,y,m,d);
  }
}
}

# %%[2 Data cleaning]
import pandas as pd

###------ load data ------
data2003_2020_path = "/home0/lix/Research/Data/1 Allcombined_cleanedData/CombinedALLData2003_2020.csv"
data = pd.read_csv(data2003_2020_path)
unname_col = data.filter(regex = "Unname")
data = data.drop(unname_col, axis=1)
columnsList = data.columns[1:]
print(data.shape)

###------ cleaning ------
for i in columnsList:
  data_cln = data.loc[~data[i].isin([-9999.0])]
  print("-9999: {} left-{}".format(i,data_cln.shape))

for m in columnsList:
    data_cln = data_cln.dropna(axis='index', how='any',subset=[m])
print("Nan: {} left-{}".format(m,data_cln.shape))

###------ export ------
out_path = r"/home0/lix/Research/Data/1 Allcombined_cleanedData/Cleaned_ALLData2003_2020.csv"
data_cln.to_csv(out_path, encoding='utf-8')

del data2003_2020_path,unname_col
del columnsList,i,m
del data,data_cln,out_path

# %%[3 Splitting data into training sets and validation sets]
import pandas as pd
from sklearn.model_selection import train_test_split

###------ load data ------
cleaningData_path = r"/home0/lix/Research/Data/1 Allcombined_cleanedData/Cleaned_ALLData2003_2020.csv"
cleaningData = pd.read_csv(cleaningData_path)
ss = cleaningData.filter(regex = "Unname")
cleaningData = cleaningData.drop(ss, axis = 1)
del ss,cleaningData_path

###------ split training/validation ------
percent = 0.2
id_list = np.unique(cleaningData['id'])
TrainData = pd.DataFrame()
TestData = pd.DataFrame()
num=0

for i in id_list:
  id_data = cleaningData.loc[cleaningData["id"].isin([i])] #split by individual stations
  id_traindata, id_testdata = train_test_split(id_data, test_size=percent, random_state=42)
  TrainData = pd.concat([TrainData,id_traindata],axis=0,ignore_index=False)
  TestData = pd.concat([TestData,id_testdata],axis=0,ignore_index=False)
  print("No.{} has done, which id is {}".format(num,i))
  num+=1

print("original:",cleaningData.shape,
      "training:",TrainData.shape,
      "validation:",TestData.shape)
TrainData.to_csv(r"/home0/lix/Research/Data/1 Allcombined_cleanedData/Test_all.csv", encoding='utf-8')
TestData.to_csv(r"/home0/lix/Research/Data/1 Allcombined_cleanedData/Train_all.csv", encoding='utf-8')
del percent, id_list, num, i, id_data, id_traindata, id_testdata


# %%[additionally: Split Xn/Yn for saving]
# Note: the training and validation sets were stored separately as “Xn_train/Yn_train/ Xn_test/Yn_test” 
#       just for storage and to facilitate the recall at different configurations of computer devices.


#data columns → ['id', 'lon', 'lat', 'doy', 'dem', 'aspect', 'slope', 'wbt', 'spre',
                # 'lst', 'pop', 'wv', 'year', 'month', 'day', 'SAT', 'WBT', 'ATin',
                # 'ATout', 'ET', 'DI', 'MDI', 'HMI', 'NET', 'sWBGT', 'WCT', 'HI',
                # 'year', 'month', 'day']

#Xn_train and Yn_train
cols = ['id', 'doy', 'dem', 'aspect', 'slope', 'lst', 'pop', 'wv', 'year', 'month', 'day']
Xn_train = pd.DataFrame()
for col in cols:
    Xn_train[col] = TrainData[col]
Xn_train.to_csv(r"/home0/lix/Research/Data/2 Split Data/Xn_train.csv", encoding='utf-8')
del cols,col

cols = ['id', 'SAT', 'WBT', 'ATin', 'ATout', 'ET', 'DI',\
        'MDI', 'HMI', 'NET', 'sWBGT', 'WCT', 'HI', 'year', 'month', 'day']
Yn_train = pd.DataFrame()
for col in cols:
    Yn_train[col] = TrainData[col]
Yn_train.to_csv(r"/home0/lix/Research/Data/2 Split Data/Yn_train.csv", encoding='utf-8')
del cols,col


#Xn_test and Yn_test
cols = ['id', 'doy', 'dem', 'aspect', 'slope', 'lst', 'pop', 'wv', 'year', 'month', 'day']
Xn_test = pd.DataFrame()
for col in cols:
    Xn_test[col] = TestData[col]
Xn_test.to_csv(r"/home0/lix/Research/Data/2 Split Data/Xn_test.csv", encoding='utf-8')
del cols,col

cols = ['id', 'SAT', 'WBT', 'ATin', 'ATout', 'ET', 'DI',\
        'MDI', 'HMI', 'NET', 'sWBGT', 'WCT', 'HI', 'year', 'month', 'day']
Yn_test = pd.DataFrame()
for col in cols:
    Yn_test[col] = TestData[col]
Yn_test.to_csv(r"/home0/lix/Research/Data/2 Split Data/Yn_test.csv", encoding='utf-8')
del cols,col


del TrainData,TestData
del Yn_test,Xn_test,Xn_train,Yn_train
del cleaningData