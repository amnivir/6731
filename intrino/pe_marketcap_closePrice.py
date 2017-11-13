# First install the required package via this CLI command:
# pip install requests
 
import requests
import sys 
import datetime
import math
import itertools
import numpy as np
from collections import OrderedDict

import matplotlib.pyplot as plt


import numpy as np
import pylab as pl

# Setup common variables
 
api_username = "ed0b07bda40f6288be66f7b383beb4cb"
api_password = "19beb02c128021eb8d24997749c2b3ce"
base_url = "https://api.intrinio.com"
ticker=sys.argv[1]
K = int(sys.argv[2])

feature_list = [] 
feature_list.append("close_price")
feature_list.append("marketcap")
feature_list.append("pricetoearnings")
cp = OrderedDict()
mc = OrderedDict()
pe = OrderedDict()


class Values:
  closePrice = 0.0
  marketCap = 0.0
  priceToEarningRatio = 0.0


# Get the latest FY Income Statement for AAPL
 
def fetchFeature(featureName, featureArray):
	request_url = base_url + "/historical_data?"
	query_params = {
	    "page_size": '1000',
	    'page_number': '1',
	    'ticker': ticker,
	    'item': featureName,
	    'start_date': '2014-11-01',
	    'end_date': '2017-11-01',
	    'sort_order': 'asc'
	}
	 
	response = requests.get(request_url, params=query_params, auth=(api_username, api_password))
	if response.status_code == 401: print("Unauthorized! Check your username and password."); exit()
	 
	data = response.json()['data']
	 
	for row in data:
	    tag = row['date']
	    value = row['value']
	    d = datetime.datetime.strptime(tag, '%Y-%m-%d')
	    intDate = int(datetime.date.strftime(d, "%Y%m%d"))
	    if featureName == "marketcap": 
		    featureArray[intDate] = math.log10(float(value))
	    else:
		    featureArray[intDate] = float(value)


def calEuclideanDist(date1, date2):
	if date1 == date2:
		return -1
	else:
		return math.sqrt((cp[date1]-cp[date2])**2 + (mc[date1]-mc[date2])**2 + (pe[date1]-pe[date2])**2) 


fetchFeature(feature_list[0], cp)
fetchFeature(feature_list[1], mc)
fetchFeature(feature_list[2], pe)

'''
print "Market_Cap=", len(mc)
print "Closing_Price=", len(cp)
print "peratio=", len(pe)

for key in pe:
        print key , " market_cap=" , mc[key], " closing_price", cp[key], " peratio=", pe[key] 
'''
dates  = cp.keys()
listCp = cp.values()

datesListLength = len(dates)
datesLinear = np.arange(datesListLength)

''' DIsable plotting for now
plt.plot(datesLinear, listCp)
plt.xticks(datesLinear,dates,rotation='vertical')
plt.show()
'''

''' Training size is 70% of whole data '''
trainingDataSize = int(len(dates) * 0.70)
"""print "Training data size:", trainingDataSize"""



trainingDataList = []
testDataList = []

for key in pe:
	if key < dates[trainingDataSize]:
		trainingDataList.append(key)
	else:
		testDataList.append(key)

dictPredictedNextDayCp = OrderedDict()
dictActualNextDayCp = OrderedDict()

for i, currDate in enumerate(testDataList):
	if i+1 == len(testDataList):
		break
	''' increase trainingDataList'''
	distanceDict = {}
	for date in trainingDataList:
		distanceDict[date] = calEuclideanDist(currDate,date)

	"""print "distanceDict=", distanceDict"""

	"""print "Sorting.. """

	distanceDictSorted = sorted(distanceDict.items(), key=lambda x:x[1])

	"""print "distanceDict=", distanceDictSorted"""

	top5 = distanceDictSorted[:K]

	"""print "top5 Dates", top5"""

	nearestDates = [key for key, value in top5]

	"""print "Nearest Dates", nearestDates"""

	predictedAvgCp = 0
	for date in nearestDates:
		"""print "actual close price", cp[date]"""
		predictedAvgCp = predictedAvgCp + cp[date]

	predictedAvgCp = predictedAvgCp/K
	
	print "-----------------------------------------------------"
	print "Predicted Next day close price", predictedAvgCp
	print "Actual current day close price", cp[testDataList[i]] 
	print "Actual Next day close price", cp[testDataList[i+1]] 
	dictPredictedNextDayCp[i+1] = predictedAvgCp
	dictActualNextDayCp[i+1] = cp[testDataList[i+1]]

	
	trainingDataList.append(currDate)


print "date list size after finishing the algo", len(trainingDataList)
print "original list size", len(dates)


listDatesTestData  = dictPredictedNextDayCp.keys()
listPredictedNextDayCp = dictPredictedNextDayCp.values()
listActualNextDayCp = dictActualNextDayCp.values()

datesListLength = len(listDatesTestData)
datesLinear = np.arange(datesListLength)

plt.plot(datesLinear, listPredictedNextDayCp, 'r')
plt.plot(datesLinear, listActualNextDayCp, 'g')

plt.xticks(datesLinear,listDatesTestData,rotation='vertical')
plt.show()



