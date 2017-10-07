import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import time



totalStart = time.time()
date,bid,ask = np.loadtxt('/home/eshinig/comp/6731/project/python/GBPUSD/GBPUSD1d.txt', unpack=True,
                              delimiter=',',
                              converters={0:mdates.strpdate2num('%Y%m%d%H%M%S')})



def percentChange(startPoint, currentPoint):
    try:
        x= ((float(currentPoint) - startPoint)/abs(startPoint))*100.00
        if x == 0.0:
            return 0.0000000000001
        else:
            return x
    except:
        return 0.000000001
    

def patternStorage():
    startTime = time.time();
    x = len(avgLine) - 60

    y = 31
    
    while (y<x):
        pattern = []
        p = []
        for i in xrange(30):
            p.insert(i,percentChange(avgLine[y-30], avgLine[y-(29-i)]))
        
        outcomeRange = avgLine[y+30:y+60]
        currentPoint = avgLine[y]
        
        try:
            avgOutcome = reduce (lambda x, y: x+y, outcomeRange) / len(outcomeRange)
        except Exception, e:
            print str(e)
            avgOutcome = 0
                        
        futureOutcome = percentChange(currentPoint,avgOutcome)

        for i in xrange(30):
            pattern.append(p[i])
        
        patterArr.append(pattern)
        performanceArr.append(futureOutcome)
        y = y + 1

    stopTime = time.time();
    print "patternArr: future % change" , len(patterArr)
    print "future outcome: % change of current and avgOutcome", len(performanceArr)

    print "Time taken for patternStorage: ", (stopTime - startTime)
   

def currentPattern():
    cp = []
    for i in xrange(30,0,-1):
        cp.insert(30-i,percentChange(avgLine[-31],avgLine[-i]))


    for i in xrange(30):
        patForRec.append(cp[i])

    print "patternForRec" , patForRec


def patternRecognition():
    patternFound = False
    plotPatternArray = []
    predictedOutcomesAr = []
    for eachPattern in patterArr:
        similarPattern = []
        #this finds similarity of last 30 points in whole 1 day
        for i in xrange(30):
            similarPattern.insert(i,(100 - abs(percentChange(eachPattern[i],patForRec[i]))))

        avgSimilarity = reduce(lambda x, y: x+y, similarPattern)/len(similarPattern)

        if avgSimilarity > 70:
            patIndex = patterArr.index(eachPattern)

            patternFound = True
            '''print '###############################'
            print "currentPattern[10]:" ,patForRec
            print "pastPattern[10]:" ,eachPattern
            print "predicted outcome", performanceArr[patIndex]'''
            xp = list(range(1,31))
            plotPatternArray.append(eachPattern)

    if patternFound:
        fig  = plt.figure(figsize=(10,6))
        for eachPatt in plotPatternArray:
            futurePoints = patterArr.index(eachPatt)

            if performanceArr[futurePoints] > patForRec[28]:
                pcolor = '#24bc00'
            else:
                pcolor = '#d40000'
            plt.plot(xp,eachPatt)
            predictedOutcomesAr.append(performanceArr[futurePoints])
            plt.scatter(35,performanceArr[futurePoints])

        realOutcomeRange = allData[noOfPointsInOneBatch+20:noOfPointsInOneBatch+30]
        realAvgOutcome = reduce(lambda x, y: x+y, realOutcomeRange)/len(realOutcomeRange)
        realMovement = percentChange(allData[noOfPointsInOneBatch], realAvgOutcome)
        predictedAvgOutcome = reduce(lambda x, y: x+y, predictedOutcomesAr)/len(predictedOutcomesAr)
        plt.scatter(40,realMovement,c='#54fff7',s=25)
        plt.scatter(40,predictedAvgOutcome,c='b',s=25)
        plt.plot(xp,patForRec,'#54fff7',linewidth = 5)
        plt.grid(True)
        plt.title('Pattern Recognition')
        plt.show()
        
    
def graphRawFx():
    
    fig = plt.figure(figsize=(10,7))
    ax1 = plt.subplot2grid((40,40),(0,0), rowspan=40, colspan=40)
    
    ax1.plot(date,bid)
    ax1.plot(date,ask)
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%m:%S'))
    
    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)
    
    ax1_2 = ax1.twinx()
    ax1_2.fill_between(date,0, (ask-bid), facecolor='g', alpha=.3)
    
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.subplots_adjust(bottom=.23)
    
    plt.grid(True)
    plt.show()


dataLength = int (bid.shape[0])
print "dataLength:",dataLength

noOfPointsInOneBatch = 37000
allData = ((bid+ask)/2)

while noOfPointsInOneBatch < dataLength:
    avgLine = allData[:noOfPointsInOneBatch]
    patterArr= []
    performanceArr = []
    patForRec = []

    patternStorage()
    currentPattern()
    patternRecognition()

    totalEnd = time.time()
    print "Entire processing time taken:", totalEnd - totalStart

    #raw_input('press ENter to continue....')
    noOfPointsInOneBatch += 1

    
