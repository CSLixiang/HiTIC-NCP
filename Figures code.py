# -*- coding: utf-8 -*-

# %%[Fig 3-4:Spatial distribution of one-day human thermal indices]
import numpy as np
import xarray as xr
import geopandas as gpd
from osgeo import gdal
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker 

index_name= "ATin" #"ATin","NET"
# indices including ['SAT', 'WBT', 'ATin', 'ATout', 'ET', 'DI', 'MDI', 'HMI','NET', 'sWBGT', 'WCT', 'HI']

 
###------ load data ------ 
MapShp = gpd.read_file(r'C:\Users\lixiang\Desktop\Data Samples\Fig3and4_data\area_shp.shp')

date = "2013-8-13" #"2013-8-13","2008-1-3"
path = r"C:\Users\lixiang\Desktop\Data Samples\Fig3and4_data\{}_{}_LGBM.tif".format(index_name,date)
tifdata01 = gdal.Open(path)
bands = tifdata01.RasterCount
if bands<1:
  print("without bands！")
else:
  band = tifdata01.GetRasterBand(1)#选择需要展示的波段
  A_band = band.ReadAsArray()
del path,tifdata01,bands,band
export_path = r'C:\Users\lixiang\Desktop\Data Samples\Fig3and4_data\Results\{}_{}.pdf'.format(index_name,date)
if date=="2008-1-3":
    corbarValues = [-16,0,16] #[min,center,max]
elif "2013-8-13":
    corbarValues = [0,15,45] #[min,center,max]

A_band[A_band < -9000] = np.nan
A_band[A_band > 9000] = np.nan

###------ plot ------ 
fig, ax = plt.subplots(figsize=(12,13), dpi=200)
plt.grid(False) 

ax.spines['right'].set_visible(True)
ax.spines['top'].set_visible(True)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylim(34,41.01) 
ax.set_yticks(np.arange(34,41.01, 2))
ax.set_xlim(113,121.1)
ax.set_xticks(np.arange(113,121.1, 2))
plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f E'))
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f N'))
ax.tick_params(axis="x",labelsize=10)
ax.tick_params(axis="y",labelsize=10)
plt.xlabel('Longitude',fontsize=12, color='black')
plt.ylabel('Latitude',fontsize=12, color='black')

#colorbar range
vmax = np.nanmax(A_band)
vmin = np.nanmin(A_band)

if vmin>=0:
    cmap_slope = "OrRd"
    norm = mpl.colors.Normalize(vmin=corbarValues[1],vmax=corbarValues[2]) 
elif vmin<0:
    cmap_slope = "RdYlBu_r"
    norm = mpl.colors.TwoSlopeNorm(vmin=corbarValues[0], vcenter=0, vmax=corbarValues[2])

#raster
extent = (113,121.0,34,41.01)
im1 = ax.imshow(A_band,
                extent = extent,
                norm = norm,
                cmap = cmap_slope,)
#area_shp
MapShp.geometry.plot(ax=ax,
                    facecolor='none',
                    edgecolor='dimgrey',
                    alpha=1,
                    linewidth=0.5,
                    )

#corlorbar
fig.subplots_adjust(right=0.95,bottom=0.07,top=0.97)
position = fig.add_axes([0.10, 0.02, 0.80, 0.030 ])

cb = fig.colorbar(im1,
              orientation="horizontal",
              cax=position)

colorbarfontdict = {"size":12,"color":"k",'family':'ubuntu'}
cb.ax.set_title('Temperature',fontdict=colorbarfontdict,pad=6)
cb.ax.set_xlabel('Values',fontdict=colorbarfontdict)
cb.ax.tick_params(labelsize=13,direction='out') 

plt.savefig(export_path)

del fig,ax,im1,vmin,vmax,extent,cmap_slope,norm
del cb,colorbarfontdict,position
del index_name,MapShp,A_band,corbarValues,export_path


# %%[Fig 5: Scatterplots of predicted versus observed daily human thermal indices]
import numpy as np
import pandas as pd
from sklearn import metrics
from scipy import optimize
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker 

# ###------ Concat yearly data ------ 
# index_list =['SAT', 'WBT', 'ATin', 'ATout', 'ET', 'DI', 'MDI', 'HMI','NET', 'sWBGT', 'WCT', 'HI']
# for index_name in index_list:
    
#     ###------ random sampling ------
#     Result_data_All_path = r"C:\Users\lixiang\Desktop\Data Samples\Fig5_data\{}_AllYears_y_test.csv".format(index_name)
#     Result_data_All = pd.read_csv(Result_data_All_path)
#     ss = Result_data_All.filter(regex = "Unname")
#     Result_data_All = Result_data_All.drop(ss, axis = 1) 
#     del Result_data_All_path,ss
    
#     Result_data_sample = Result_data_All.sample(n=1000)
    
#     Result_data_samPath = r"C:\Users\lixiang\Desktop\Data Samples\Fig5_data\Sample_{}.csv".format(index_name)
#     Result_data_sample.to_csv(Result_data_samPath, encoding='utf-8')
#     del Result_data_All,Result_data_samPath,Result_data_sample
# del index_list,index_name


index_name = "NET"
# indices including ['SAT', 'WBT', 'ATin', 'ATout', 'ET', 'DI', 'MDI', 'HMI','NET', 'sWBGT', 'WCT', 'HI']

###------ load sample data ------
Result_data_samPath = r"C:\Users\lixiang\Desktop\Data Samples\Fig5_data\Sample_{}.csv".format(index_name)
Result_data_sample = pd.read_csv(Result_data_samPath)
ss = Result_data_sample.filter(regex = "Unname")
Result_data_sample = Result_data_sample.drop(ss, axis = 1) 
del Result_data_samPath,ss

x = Result_data_sample["y_test"]
y = Result_data_sample["LGBM_y_pred"]

###------ Accuracy Assessment ------ 
print('R²：',metrics.r2_score(x, y,multioutput='uniform_average'))
print('Mean Absolute Error:', metrics.mean_absolute_error(x, y))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(x, y)))

###------ scatter plot ------ 
x = x.values.ravel()
y = y.values.ravel()

R_square = metrics.r2_score(x, y,multioutput='uniform_average')
RMSE = np.sqrt(metrics.mean_squared_error(x, y))
xlabel = "Observation"
ylabel = "Prediction"

plt.rc('font',family='Arial') 
plt.rc('axes',linewidth="1",grid='False')

x2 = np.linspace(-40,50)
y2 = x2
  
#fitted line
def f_1(x,A,B):
  return A*x + B
A1,B1 = optimize.curve_fit(f_1,x,y)[0]
y3 = A1*x + B1

fig,ax=plt.subplots(figsize=(22.44, 20.9),dpi=300)
    
#Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

scatter = ax.scatter(x=x,
                    y=y,
                    marker='.',
                    c=z*100,
                    edgecolors='none',
                    s=12,
                    label='LST',
                    cmap='Spectral_r',
                    zorder=2
                    )

cbar=plt.colorbar(scatter,
                  shrink=1,
                  orientation='vertical',
                  extend='both',
                  pad=0.015,
                  aspect=30,
                  )
cbar.ax.locator_params(nbins=8)
cbar.ax.tick_params(which='both',
                    direction='in',                        
                    labelsize=35,
                    left=False)
ax.plot(x2,
        y2,
        color='k',
        linewidth=1,
        linestyle='--',
        alpha = 0.7,
        zorder=1
        )
ax.plot(x,
        y3,
        color='r',
        linewidth=2,
        linestyle='-',
        alpha=0)

plt.legend(["R²=%.2f,RMSE=%.2f°C"%(R_square,RMSE)], loc=2, fontsize=25)
plt.grid(False, alpha=0.01)

#corlorbar range of each index
list_a = ["DI","MDI","SAT","WBT"]
list_b = ["NET","WCT","ATout"]
list_c = ["HI","HMI"]
if index_name in list_a:
    start_t = -30
    end_t = 35
elif index_name in list_b:
    start_t = -40
    end_t = 40
elif index_name in list_c:
    start_t = -40
    end_t = 50
elif index_name=="ATin":
    start_t = -30
    end_t = 40
elif index_name=="ET":
    start_t = -15
    end_t = 25
elif index_name=="sWBGT":
    start_t = -15
    end_t = 40
ax.set_xlim((start_t,end_t))
ax.set_ylim((start_t,end_t))
ax.set_xticks(np.arange(start_t,end_t,step=15))
ax.set_yticks(np.arange(start_t,end_t,step=15))

plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%d °C'))
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%d °C'))
ax.tick_params(axis="x",labelsize=35)
ax.tick_params(axis="y",labelsize=35)

#save 
plt.savefig(r'C:\Users\lixiang\Desktop\Data Samples\Fig5_data\Results\{}.pdf'.format(index_name))

del A1,B1,x2,y2,y3,xy,z
del xlabel,ylabel,RMSE,R_square,scatter
del fig,ax,cbar,list_a,list_b,list_c,start_t,end_t
del index_name,Result_data_sample,x,y

# %%[Fig 6: R2, MAE, and RMSE values of 12 predicted human thermal indices]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###------ load data ------ 
Indices_Accuracies_path = r"C:\Users\lixiang\Desktop\Data Samples\Fig6_data\Fig6_data.csv"
Indices_Accuracies = pd.read_csv(Indices_Accuracies_path, encoding="utf-8")
ss = Indices_Accuracies.filter(regex="Unname")
Indices_Accuracies = Indices_Accuracies.drop(ss, axis=1)
del Indices_Accuracies_path,ss

# Indices_Accuracies.set_index(["Index"], inplace=True) 

labels = list(Indices_Accuracies["Index"])
y1 = Indices_Accuracies["R2"]
y2 = Indices_Accuracies["MAE"]
y3 = Indices_Accuracies["RMSE"]
y_right = "RMSE" #"MAE"

###------ plot ------ 
fig,ax1 = plt.subplots(figsize=(18, 12), dpi=200)

#set xaxis
gap = np.linspace(0.5,9.5,12)
width = 0.2 #width of bar
x_gap = gap-width/2

#yaxis_left:R_square
ax1.spines['right'].set_visible(True)
ax1.spines['top'].set_visible(True)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')
ax1.set_ylim(0.970,1.001) 
ax1.set_yticks(np.arange(0.970,1.001, 0.005))
ax1.tick_params(axis="y",labelsize=20, labelcolor = 'cornflowerblue')
ax1.set_ylabel('R Square',fontsize=18, color='cornflowerblue')#label

#bar-R_square
pic1 = ax1.bar(x_gap, #对轴的位置进行设置
                y1, 
                width,
                # color="tomato",
                color="cornflowerblue",
                # color="mediumseagreen",
                alpha=0.8,
                label="R²"
                )

#yaxis_right:MAE and RMSE
ax2 = ax1.twinx() 

ax2.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.yaxis.set_ticks_position('right') #将y轴放在右边

if y_right == "MAE":
    #MAE
    ax2.set_ylim(0.4,1.5) 
    ax2.set_yticks(np.arange(0.4,1.5,0.15)) #建立刻度
    ax2.set_ylabel('MAE',fontsize=18, color='darkorange')#设置轴的label
    ax2.tick_params(axis="y",labelsize=20,labelcolor='darkorange')#调整y轴的数字大小
elif y_right == "RMSE":
    #RMSE
    ax2.set_ylim(0.6,2.01) 
    ax2.set_yticks(np.arange(0.6,2.01,0.2)) #建立刻度
    ax2.set_ylabel('RMSE',fontsize=18, color='mediumseagreen')#设置轴的label
    ax2.tick_params(axis="y",labelsize=20,labelcolor='mediumseagreen')#调整y轴的数字大小


for i in range(len(x_gap)):
    x_gap[i] = x_gap[i] + width
#bar-MAE
pic2 = ax2.bar(x_gap,
                y2, 
                width,
                color="darkorange",
                # color="cornflowerblue",
                alpha=0.8,
                label="MAE"
                )

for i in range(len(x_gap)):
    x_gap[i] = x_gap[i] + width
#bar-RMSE   
pic3 = ax2.bar(x_gap,
                y3, 
                width,
                # color="darkorange",
                color="mediumseagreen",
                alpha=0.6,
                label="RMSE"
                )

for i in range(len(x_gap)):
    x_gap[i] = x_gap[i] - width

ax1.set_xticks(x_gap)
ax1.set_xticklabels(labels)
ax1.tick_params(axis="x",labelsize=23)

plt.legend(handles = [pic1, pic2, pic3],
           loc = 'upper left',
           prop={'size': 23})

plt.savefig(r'C:\Users\lixiang\Desktop\Data Samples\Fig6_data\Results\R2_%s.pdf'%y_right)

plt.show()

del fig,ax1,ax2,gap,i,labels,width,x_gap
del pic1,pic2,pic3,y1,y2,y3
del y_right,Indices_Accuracies

# %%[Fig 7-9: Spatial distribution of R2 of 12 predicted human thermal indices at individual meteorological stations ]
import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker 

index_name= "ATin"
# indices including ['SAT', 'WBT', 'ATin', 'ATout', 'ET', 'DI', 'MDI', 'HMI','NET', 'sWBGT', 'WCT', 'HI']

###------ load data ------ 
Index_By_Id_path = r"C:\Users\lixiang\Desktop\Data Samples\Fig7to9_data\{}_AllYears_y_test_ById.csv".format(index_name)
Index_By_Id = pd.read_csv(Index_By_Id_path)
ss = Index_By_Id.filter(regex = "Unname")
Index_By_Id = Index_By_Id.drop(ss, axis = 1) 
del Index_By_Id_path, ss

##point
pointshp = gpd.read_file(r'C:\Users\lixiang\Desktop\Data Samples\Fig7to9_data\point_shp.shp')

##area_shp
MapShp = gpd.read_file(r'C:\Users\lixiang\Desktop\Data Samples\Fig7to9_data\area_shp.shp')

num=0
pointshp["R_2"]=np.nan
pointshp["MAE"]=np.nan
pointshp["RMSE"]=np.nan
for Aid in pointshp["id"]:
    if(Aid in Index_By_Id["id"].values):
        AStation = Index_By_Id.loc[Index_By_Id["id"].isin([Aid])]
        pointshp.iloc[num,-3] = AStation["R_2"].values[0]
        pointshp.iloc[num,-2] = AStation["MAE"].values[0]
        pointshp.iloc[num,-1] = AStation["RMSE"].values[0]
    num+=1
print("{} has computed Done!".format(num))
del num,Aid,AStation


###------ plot ------ 
indicator = "RMSE" #"R_2","MAE","RMSE"
title = 'Spatial distributions of {}({})'.format(indicator,index_name)

plt.rc('axes',unicode_minus='False') 
plt.rc('axes',linewidth='2.0',grid='True') 
plt.rc('font',family='Microsoft YaHei',size='15')

fig, ax = plt.subplots(figsize=(15, 12))
ax.set_alpha(0.8)
plt.grid(False, alpha=0.01)

#set corlorbar
if indicator=="R_2":
    colors = ['mistyrose','salmon','maroon']
    cmap1 = mcolors.LinearSegmentedColormap.from_list('cmap1',colors)
    norm = mcolors.Normalize(vmin=0.975,vmax=1.0)
elif indicator=="MAE":
    colors = ['maroon','salmon','mistyrose']
    cmap1 = mcolors.LinearSegmentedColormap.from_list('cmap1',colors)
    norm = mcolors.Normalize(vmin=0.5,vmax=1.2)
elif indicator=="RMSE":
    colors = ['maroon','salmon','mistyrose']
    cmap1 = mcolors.LinearSegmentedColormap.from_list('cmap1',colors)
    norm = mcolors.Normalize(vmin=0.7,vmax=1.6)
    
#area_shp
MapShp.geometry.plot(ax=ax,
                        facecolor='cornsilk',
                        edgecolor='dimgrey',
                        alpha=1,
                        label = 'Mainland China'
                        )
#point_shp
if indicator==None:
    pts = gpd.GeoSeries(pointshp['geometry'])  # 创建点要素数据集
    pts.plot(ax=ax,
            facecolor='black',
            edgecolor='black',
            marker='.',
            markersize=30,
            label='Meteorological station',
            alpha=1)
else:
    pointdf_ = gpd.GeoDataFrame(pointshp,
                                geometry=gpd.points_from_xy(pointshp["lon"],pointshp["lat"]))
    
    pointdf_.plot(column=indicator,
                    ax=ax,
                    edgecolor='gray',
                    legend=True,
                    label='Meteorological station',
                    cmap =cmap1,
                    norm=norm
                    )


ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim(33.5,41.5)
ax.set_yticks(np.arange(33.5, 41.5, 1))
ax.set_xlim(112.5,121.5)
ax.set_xticks(np.arange(112.5,121.5, 1))
plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f °E'))
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f °N'))
plt.xlabel('Longitude',fontsize=18)
plt.ylabel('Latitude',fontsize=18)
plt.title(title, fontdict={'weight': 'normal', 'size': 25}) # 设置图名&改变图标题字体
plt.rcParams['savefig.dpi'] = 300 #图片像素

#save
plt.savefig(r'C:\Users\lixiang\Desktop\Data Samples\Fig7to9_data\Results\{}-{}.pdf'.format(index_name,indicator))
plt.show()

del fig, ax,title,indicator,colors,cmap1,norm
del Index_By_Id,index_name,pointshp,MapShp


# %%[Fig 10: Prediction accuracies of 12 human thermal indices over NCP in individual years ]
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

###------ load data ------ 
indicator = "R2" #including "R2","MAE","RMSE"
yearly_accurary_path = r"C:\Users\lixiang\Desktop\Data Samples\Fig10_data\{}_yearly_accurary.csv".format(indicator)
yearly_accurary = pd.read_csv(yearly_accurary_path)
ss = yearly_accurary.filter(regex = "Unname")
yearly_accurary = yearly_accurary.drop(ss, axis = 1) 
yearly_accurary.set_index(["index"], inplace=True)
del yearly_accurary_path,ss

if indicator=="R2":
    colors = ['mistyrose','salmon','tomato','maroon']
    cmap1 = mcolors.LinearSegmentedColormap.from_list('cmap1',colors)
    vmin=0.96
    vmax=1.0
    center = None
elif indicator=="MAE":
    colors = ['mistyrose','tomato','maroon']
    cmap1 = mcolors.LinearSegmentedColormap.from_list('cmap1',colors)
    center = None
    vmin=0.2
    vmax=2
elif indicator=="RMSE":
    colors = ['mistyrose','salmon','tomato','maroon']
    cmap1 = mcolors.LinearSegmentedColormap.from_list('cmap1',colors)
    center = None
    vmin=0.5
    vmax=2

###------ plot ------ 
plt.subplots(figsize=(20, 8))
ax = sns.heatmap(yearly_accurary,
                 vmin=vmin,
                 vmax=vmax,
                 cmap=cmap1,
                 center = center,
                 xticklabels=yearly_accurary.columns,
                 yticklabels=yearly_accurary.index,
                 )
# ax.invert_yaxis()
title = "{} Yearly Accuracy ".format(indicator)
plt.title(title, fontdict={'weight': 'normal', 'size': 20})
plt.rcParams['savefig.dpi'] = 300
plt.savefig(r'C:\Users\lixiang\Desktop\Data Samples\Fig10_data\Results\{}.pdf'.format(indicator))

plt.show()

del ax,title
del colors,cmap1,center,vmin,vmax
del indicator,yearly_accurary


