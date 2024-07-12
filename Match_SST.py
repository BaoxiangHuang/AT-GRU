import datetime
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from openpyxl import Workbook

# Data paths
BGC_PATH = 'path/to/your/bgc'
SST_NC_PATH = 'path/to/your/SST/'
Ouput_PATH = 'path/to/your/finish/'
NC_FILE = 'path/to/your/SST/SST.nc'

# Obtain spatiotemporal information from BGC
df = pd.read_excel(BGC_PATH,header=None)
reports = df.loc[ : ]
ndata = np.array(reports)
dataList = ndata.tolist()

# Create SST grid array
a = Dataset(NC_FILE)
sst_lat=(a.variables['lat'][:])
sst_lat=sst_lat.tolist()
sst_lon=(a.variables['lon'][:])
sst_lon=sst_lon.tolist()

# Confirm the SST file for the specific day
def getdate(year, month, day):
    date = datetime.date(year, month, day)
    return date.strftime('%j')
count =0
for item in dataList:
    year = int(str(item[2])[ :4])
    month = int(str(item[2])[4:6])
    day = int(str(item[2])[6:8])
    date=int(getdate(year,month,day))
    nc_obj = Dataset(SST_NC_PATH)
    sst_sst = (nc_obj.variables['sst'][date-1])

# Retrieve Chla value for the given location
    lat_p = int((item[3] - int(item[3])) / 0.25)
    lon_p = int((item[4] - int(item[4])) / 0.25)
    lat = int(item[3]) * 4 + lat_p
    lon = int(item[4]) * 4 + lon_p
    a = sst_lat[360+lat]
    b = sst_lon[lon]
    c = sst_sst[lat+360][lon]
    count = count+1
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