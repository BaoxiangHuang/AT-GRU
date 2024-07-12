import datetime
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from openpyxl import Workbook
import os

# Data paths
BGC_PATH = 'path/to/your/bgc'
Chla_NC_PATH = 'path/to/your/Chla/'
Ouput_PATH = 'path/to/your/finish/'
NC_FILE = 'path/to/your/Chla/Chla.nc'

# Obtain spatiotemporal information from BGC
df = pd.read_excel(BGC_PATH,header=None)
reports = df.loc[ : ]
ndata = np.array(reports)
dataList = ndata.tolist()

# Create Chla grid array
a = Dataset(NC_FILE)
chla_lat=(a.variables['lat'][:])
chla_lat=chla_lat.tolist()
chla_lon=(a.variables['lon'][:])
chla_lon=chla_lon.tolist()

# Confirm the Chla file for the specific day
def getdate(year, month, day):
    date = datetime.date(year, month, day)
    return date.strftime('%j')
file_name_list = os.listdir(Chla_NC_PATH)
def getfilename(days):
    i = int(days) % 8
    k = int(int(days) / 8)
    if i == 0:
        return file_name_list[k-1]
    else:
        return file_name_list[k]
count =0
for item in dataList:
    count = count + 1
    print(count)
    year = int(str(item[2])[ :4])
    month = int(str(item[2])[4:6])
    day = int(str(item[2])[6:8])
    date=int(getdate(year,month,day))
    filename = getfilename(date)
    nc_obj = Dataset(Chla_NC_PATH + '\\' + str(filename))
    chla_chla = (nc_obj.variables['chlor_a'][0])

# Retrieve Chla value for the given location
    lat_p = int((item[3] - int(item[3])) * 24)
    lon_p = int((item[4] - int(item[4])) * 24)
    lat = int(item[3]) * 24 + lat_p
    lon = int(item[4]) * 24 + lon_p
    a = chla_lat[2159-lat]
    b = chla_lon[lon+4320]
    c = chla_chla[2159-lat][lon+4320]
    item.append(c)

# Save the results
workbook = Workbook()
save_file = Ouput_PATH
worksheet = workbook.active
worksheet.title = "Sheet1"
for i in range(1,len(dataList)+1):
     for j in range(1,7):
         worksheet.cell(i, j, str(dataList[i - 1][j-1]))
workbook.save(filename=save_file)