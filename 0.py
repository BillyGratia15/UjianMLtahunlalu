import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#df semua provinsi (Indonesia not included)
df = pd.read_excel(
    'Indo_12_1.xls',
    header = 3,
    skipfooter = 3,                 #3 footer di skip (supaya Indonesia gaikutan)
    index_col = 0,
    na_values = ['-']
)

#df prov 2010 jumlah penduduk max
# print(df)
# print(df[2010].max())
dfMax2010 = df[df[2010]== df[2010].max()]
namaMax2010 = dfMax2010.index[0]
print(dfMax2010)                                    #show semua data yang maksimal di tahun 2010
# print(dfMax2010.index[0])                           #buat index nya jadi 'Jawa Barat'
# print(df.loc[dfMax2010.index[0]])                   #munculin dimana data maksimal tahun 2010
print(dfMax2010.columns.values)
print(dfMax2010.iloc[0])                            #munculin dimana data maksimal tahun 2010

#df prov 1971 jumlah penduduk min
df.dropna(subset = [1971])                          #hapus data di kolom 1971 itu NaN
dfMin1971 = df[df[1971] == df[1971].min()]
namaMin1971 = dfMin1971.index[0]
print(dfMin1971)
print(dfMin1971.columns.values)
print(dfMin1971.iloc[0])

# cari Indonesia
df = pd.read_excel(
    'Indo_12_1.xls',
    header = 3,
    skipfooter = 2,                 #3 footer di skip (supaya Indonesia gaikutan)
    index_col = 0,                  #supaya angka xls ga kebaca
    na_values = ['-']
)

dfIndo = df[df[2010]==df[2010].max()]
namaIndo = dfIndo.index[0]
print(dfIndo)
print(dfIndo.columns.values)
print(dfIndo.iloc[0])

#linear regression
from sklearn.linear_model import LinearRegression
modelMax2010 =  LinearRegression()
modelMin1971 = LinearRegression()
modelIndo = LinearRegression()

#training
x = dfMax2010.columns.values.reshape(-1,1)
y = dfMax2010.values[0]
modelMax2010.fit(x,y)

x = dfMin1971.columns.values.reshape(-1,1)
y = dfMin1971.values[0]
modelMin1971.fit(x,y)

x = dfIndo.columns.values.reshape(-1,1)
y = dfIndo.values[0]
modelIndo.fit(x,y)

#prediksi 2050
max1050 = int(round(modelMax2010.predict([[2050]])[0]))
min7150 = int(round(modelMin1971.predict([[2050]])[0]))
indo50 = int(round(modelIndo.predict([[2050]])[0]))
print('Prediksi jumlah penduduk {} di th 2050='.format(namaMax2010), max1050)
print('Prediksi jumlah penduduk {} di th 2050='.format(namaMin1971), min7150)
print('Prediksi jumlah penduduk {} di th 2050='.format(namaIndo), indo50)

plt.plot(
    dfMax2010.columns.values, dfMax2010.iloc[0],'g-',
    dfMin1971.columns.values, dfMin1971.iloc[0],'m-',
    dfIndo.columns.values, dfIndo.iloc[0],'r-'
)

# best fit line
plt.plot(
    dfMax2010.columns.values, 
    modelMax2010.coef_ * dfMax2010.columns.values + modelMax2010.intercept_,
    'y-'
)
plt.plot(
    dfMin1971.columns.values, 
    modelMin1971.coef_ * dfMin1971.columns.values + modelMin1971.intercept_,
    'y-'
)
plt.plot(
    dfIndo.columns.values, 
    modelIndo.coef_ * dfIndo.columns.values + modelIndo.intercept_,
    'y-'
)

plt.scatter(dfMax2010.columns.values,dfMax2010.iloc[0],color='g',s=80)

plt.scatter(dfMin1971.columns.values,dfMin1971.iloc[0],color='m',s=80)

plt.scatter(dfIndo.columns.values,dfIndo.iloc[0],color='r',s=80)

plt.title('Jumlah Penduduk {} (1971-2010)'.format(namaIndo)) #tidak boleh tulis IndoNESIA
plt.legend([namaMax2010,namaMin1971,namaIndo])
plt.grid(True)
plt.show()
