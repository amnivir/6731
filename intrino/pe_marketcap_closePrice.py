# First install the required package via this CLI command:
# pip install requests
import requests
import sys 
import datetime
import math
import itertools
import numpy as np
import csv
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
feature_list.append("pricetobook")
feature_list.append("adj_volume")
feature_list.append("percent_change")


#intialize the ordered dictunary which preserves the insertion order
cp = OrderedDict()
mc = OrderedDict()
pe = OrderedDict()
pb = OrderedDict()
av = OrderedDict()
pc = OrderedDict()


# Get the latest FY Income Statement for a feature name
 
def fetchFeature(featureName, featureArray):
	request_url = base_url + "/historical_data?"
	query_params = {
	    "page_size": '1000',
	    'page_number': '1',
	    'ticker': ticker,
	    'item': featureName,
	    'start_date': '2014-11-01',
	    'end_date': '2017-11-25',
	    'frequency': 'daily',
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
	    if featureName == "marketcap" or featureName == "adj_volume" : 
		    featureArray[intDate] = math.log10(float(value))
	    else:
		    featureArray[intDate] = float(value)


def calEuclideanDist(date1, date2):
	if date1 == date2:
		return -1
	else:
		return math.sqrt((cp[date1]-cp[date2])**2 + (mc[date1]-mc[date2])**2 + (pe[date1]-pe[date2])**2 + (pb[date1]-pb[date2])**2 + (av[date1]-av[date2])**2 + (pc[date1]-pc[date2])**2) 

def calManhattanDist(date1, date2):
	if date1 == date2:
		return -1
	else:
		return (abs(cp[date1]-cp[date2])+abs(mc[date1]-mc[date2]) + abs(pe[date1]-pe[date2]) + abs(pb[date1]-pb[date2]) + abs(av[date1]-av[date2]) + abs(pc[date1]-pc[date2])) 


fetchFeature(feature_list[0], cp)
fetchFeature(feature_list[1], mc)
fetchFeature(feature_list[2], pe)
fetchFeature(feature_list[3], pb)
fetchFeature(feature_list[4], av)
fetchFeature(feature_list[5], pc)


#write to csv
f = open(ticker, 'wt')
try:
    writer = csv.writer(f)
    writer.writerow( ('Date', 'marketcap(log)', 'pricetoearnings', 'pricetobook', 'adj_volume(log)', 'close_price', 'percent_change(class)') )
    for key in pe:
	# 0 means SELL and 1 means BUY
	classifier_category = 0
	if pc[key] > 0:
		classifier_category = 1
        writer.writerow((key, mc[key], pe[key], pb[key], av[key], cp[key], classifier_category))
finally:
    f.close()
    

print "Market_Cap=", len(mc)
print "Closing_Price=", len(cp)
print "peratio=", len(pe)

for key in pe:
        print key , " market_cap=" , mc[key], " closing_price", cp[key], " peratio=", pe[key], " pc=", pc[key]

dates  = cp.keys()
listCp = cp.values()

#datesListLength = len(dates)
#datesLinear = np.arange(datesListLength)

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

correctDecisionCounter = 0
correctDecisionCounterPc = 0

for i, currDate in enumerate(testDataList):
	if i+1 == len(testDataList):
		break
	''' increase trainingDataList'''
	distanceDict = {}
	for date in trainingDataList:
		distanceDict[date] = calManhattanDist(currDate,date)

	"""print "distanceDict=", distanceDict"""

	"""print "Sorting.. """

	distanceDictSorted = sorted(distanceDict.items(), key=lambda x:x[1])

	"""print "distanceDict=", distanceDictSorted"""

	topK = distanceDictSorted[:K]

	"""print "top5 Dates", top5"""

	nearestDates = [key for key, value in topK]

	print "\n-----------------------------------------------------"
	
	print "Current Date", currDate
	print "Nearest Dates", nearestDates
	nextDatesAtNearestDates = []
	
	#calucalate %change between next day of nearest days
	#percent_change = []
	#j=0
	#for nd in nearestDates:
	#	dateIndex = dates.index(nd)
	#	print "",dateIndex
	#	currentDayPrice =cp[dateIndex]
	#	nextDayPrice = cp[dateIndex+1]
	#	print "currentDayPrice:", currentDayPrice
	#	print "nextDayPrice:", nextDayPrice
		#pc = ( 100* (nextDayPrice - currentDayPrice) ) / currentDayPrice
		#percent_change.append(pc)
		#print "j:",j 
		#print "%change:",percent_change[j]
	        #j += 1


	pcCounterPositive = 0
	predictedAvgCp = 0
	for date in nearestDates:
		"""print "actual close price", cp[date]"""
		predictedAvgCp = predictedAvgCp + cp[date]

	predictedNextDayAvgCp = predictedAvgCp/K
	
	actualCurrentDayCp = cp[testDataList[i]]
	actualNextDayCp =	cp[testDataList[i+1]]
	print "Actual current day close price:", actualCurrentDayCp 
	print "Predicted Next day close price:", predictedNextDayAvgCp
	print "Actual Next day close price:", actualNextDayCp

	if predictedNextDayAvgCp > actualCurrentDayCp and actualNextDayCp > actualCurrentDayCp:
		print "CP Classifier Decision===> BUY" 
		correctDecisionCounter = correctDecisionCounter + 1
        
	if predictedNextDayAvgCp < actualCurrentDayCp and actualNextDayCp < actualCurrentDayCp:
		print "CP Classifier Decision===> SELL"
		correctDecisionCounter = correctDecisionCounter + 1
		pcCounterPositive += 1 
	
	dictPredictedNextDayCp[i+1] = predictedNextDayAvgCp
	dictActualNextDayCp[i+1] = cp[testDataList[i+1]]

	''' Add the test data into training sample'''
	trainingDataList.append(currDate)

	
	for date in nearestDates:
		print "actual percentChange", pc[date]
                if pc[date] > 0:
			pcCounterPositive += 1

	#positive trend
	if pcCounterPositive >= (K/2 + 1):
		if actualNextDayCp > actualCurrentDayCp:
			print "PC Classifier Decision===> BUY" 
			correctDecisionCounterPc += 1
	#negative trend	
	else:
		if actualNextDayCp < actualCurrentDayCp:
			print "PC Classifier Decision===> SELL" 
			correctDecisionCounterPc += 1
			
		
			
                  


#print "date list size after finishing the algo", len(trainingDataList)
#print "original list size", len(dates)



listDatesTestData  = dictPredictedNextDayCp.keys()
listPredictedNextDayCp = dictPredictedNextDayCp.values()
listActualNextDayCp = dictActualNextDayCp.values()

print "Accuracy of Stock Prediction by KNN(close price)=" , 100*correctDecisionCounter/len(listDatesTestData)
print "Accuracy of Stock Prediction by KNN(percent cng)=" , 100*correctDecisionCounterPc/len(listDatesTestData)

datesListLength = len(listDatesTestData)
datesLinear = np.arange(datesListLength)

plt.xlabel('Test Data Dates')
plt.ylabel('close_price')

plt.plot(datesLinear, listPredictedNextDayCp, 'r',label='Predicted')
plt.plot(datesLinear, listActualNextDayCp, 'g',label='Actual')

plt.xticks(datesLinear,listDatesTestData,rotation='vertical')

plt.legend()
plt.show()
plt.title(ticker)


