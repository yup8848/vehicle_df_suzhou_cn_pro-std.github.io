#引入包和加载数据
from pandas import Series,DataFrame
from sklearn import preprocessing
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt                                                  #NumPy的可视化操作界面
import seaborn as sns                                                            #seaborn提供了一种高度交互式界面，便于用户能够做出各种有吸引力的统计图表
import plotly.graph_objs as go                                                   #导入plotly中所有图形对象,实现数据可视化
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot    #从离线Plot包加载所需要的函数

#加载python可视化库
import pandas_profiling as pp 
import plotly.express as px 

#一、加载数据
data = pd.read_csv(r'C:\Users\Administrator\Desktop\苏州_data20180918n.csv',header=None,skiprows=12,names=['RTICIDNEW','LEN','FUNCLASS','UPDATE_TIME','V_SPEED'])
data = pd.DataFrame(data)

#查看数据维度及类型
print(data.head(5))#查看数据前5行
print(data.info())#查看每列数据类型及空值情况
print(data.describe().T)#查看连续数值属性基本统计情况
print(data.describe(include=['O']).T)#查看object属性数据统计情况
'''
查看数据类型发现变量并不都是数值型，'V_SPEED'与'UPDATE_TIME'为objet，
需要转换为float64才能进行后续计算，转换过程如下
'''
data['V_SPEED'] = pd.to_numeric(data['V_SPEED'],errors='coerce')
data['UPDATE_TIME'] = pd.to_numeric(data['UPDATE_TIME'],errors='coerce')
print(data.info())#检查'V_SPEED'与'UPDATE_TIME'变量数值是否转化成float64型
data = data.drop('RTICIDNEW',axis=1)# 无关属性列删除

#二、分析缺失数据
print(data.isnull())#元素级别的判断，把对应的所有元素的位置都列出来，元素为空或者NA就显示True，否则就是False
print(data.isnull().any()) #列级别的判断，只要该列有为空或者NA的元素，就为True，否则False
missing = data.columns[data.isnull().any()].tolist()#将为空或者NA的列找出来
missing_count = data[missing].isnull().sum()#将列中为空或者NA的个数统计出来
print('为空或者NA的列:',missing,'\n列中为空或者NA的个数统计:\n',missing_count)

# 将某一列中缺失元素的值，用value值进行填充。处理缺失数据时，比如该列都是字符串，不是数值，可以将出现次数最多的字符串填充缺失值。
def cat_imputation(column, value):
    data.loc[data[column].isnull(),column] = value

#三、统计分析
print(data['V_SPEED'].value_counts())#统计某一列中各个元素值出现的次数
'''
偏度的衡量是相对于正态分布来说，正态分布的偏度为0。因此我们说，若数据分布是对称的，偏度为0。
若偏度>0，则可认为高峰在左，分布为右偏，即分布有一条长尾在右；Skewness: 0.530372
若偏度<0，则可认为高峰在右，分布为左偏，即分布有一条长尾在左。
若偏度 = 0 - mean = median, the distribution is symmetrical around the mean.
同时偏度的绝对值越大，说明分布的偏移程度越严重。
'''
print("Skewness: %f" % data['V_SPEED'].skew())#列出数据的偏斜度
'''
峰度其实是一个相对于正态分布的对比量，正态分布的峰度系数为3，但是为了比较起来方便，
很多软件（spss，python中的pandas工具）将峰度系数减去3，此时正态分布峰度值定为0。
而均匀分布的峰度为-1.2，指数分布的峰度为6。本数据的Kurtosis: 0.027380接近0，说明
速度数据接近正太分布
'''
print("Kurtosis: %f" % data['V_SPEED'].kurt())#列出数据的峰度
print('V_SPEED与LEN相关度：',data['V_SPEED'].corr(data['LEN']))#计算'V_SPEED'与'LEN'相关性
data['SqrtV_SPEED']=np.sqrt(data['V_SPEED'])  #将列的数值求根，并赋予一个新列
print(data[['FUNCLASS', 'V_SPEED']].groupby(['FUNCLASS'], as_index=False).mean())#跟FUNCLASS进行分组，并求分组后的平均值

#四、数据处理　
#1)删除相关

data['FUNCLASS'].dropna()    #去掉为空值或者NA的元素
data.dropna(axis=0)         #删除带有空值的行
data.dropna(axis=1)         #删除带有空值的列

#2）缺失值填充处理

data['UPDATE_TIME']=data['UPDATE_TIME'].fillna(0) #将该列中的空值或者NA填充为0
data['UPDATE_TIME'].fillna(method='pad')#使用前一个数值替代空值或者NA，就是NA前面最近的非空数值替换     
data['UPDATE_TIME'].fillna(method='bfill',limit=1)#使用后一个数值替代空值或者NA，limit=1就是限制如果几个连续的空值，只能最近的一个空值可以被填充。
data['UPDATE_TIME'].fillna(data['UPDATE_TIME'].mean())#使用平均值进行填充
data['UPDATE_TIME'].interpolate() # 使用插值来估计NaN 如果index是数字，可以设置参数method='value' ，如果是时间，可以设置method='time'
data= data.fillna(data.mean())#将缺失值全部用该列的平均值代替，这个时候一般已经提前将字符串特征转换成了数值。
'''
注：如果数据的缺失达到15%，且并没有发现该变量有多大作用，就删除该变量！
'''
#3）数据保存
#index=False，写入的时候不写入列的索引序号
data.to_csv('merge_data.csv',index=False)    
#4）数据转换
#采用log(1+x)方式对原数据进行处理，改变原数据的偏斜度，使数据更加符合正态曲线分布。
data["V_SPEED"] = np.log1p(data["V_SPEED"])  
#把内容为数值的特征列找出来   
numeric_feats =data.dtypes[data.dtypes != "object"].index 
 #另外一种形式数据转换，将字符串特征列中的内容分别提出来作为新的特征出现，这样就省去了将字符串内容转化为数值特征内容的步骤了。    
data= pd.get_dummies(data)  
print(data)


#5）数据标准化
'''　
大多数的梯度方法（几乎所有的机器学习算法都基于此）对于数据的缩放很敏感。
因此，在运行算法之前，我们应该进行标准化，或所谓的规格化。标准化包括替
换所有特征的名义值，让它们每一个的值在0和1之间。而对于规格化，它包括数
据的预处理，使得每个特征的值有0和1的离差。Scikit-Learn库已经为其提供
了相应的函数。
'''
# normalize the data attributes
normalized_data = preprocessing.normalize(data['V_SPEED'].values.reshape(-1, 1))
print(normalized_data)
# standardize the data attributes
standardized_data = preprocessing.scale(data['V_SPEED'].values.reshape(-1, 1))
print(standardized_data)

#subplots设计显示图像大小
#Heatmap将数据绘制为颜色方格（编码矩阵），热力图可以看数据表里多个特征两两的相似度
f,ax = plt.subplots(figsize=(10,8))
sns.heatmap(data.iloc[:,:].corr(),annot=True, linewidths=.5, fmt='.2f', ax=ax)
#设置坐标字体方向
label_y = ax.get_yticklabels()
plt.setp(label_y, rotation=0, horizontalalignment='right')
label_x = ax.get_xticklabels()
plt.setp(label_x, rotation=45, horizontalalignment='right')
plt.title('Heatmap-explor data')#标题
plt.show()

