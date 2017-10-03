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

avgLine = ((bid+ask)/2)
patterArr= []
performanceArr = []
patForRec = []


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
    for eachPattern in patterArr:
        similarPattern = []
        for i in xrange(30):
            similarPattern.insert(i,(100 - abs(percentChange(eachPattern[i],patForRec[i]))))

        avgSimilarity = reduce(lambda x, y: x+y, similarPattern)/len(similarPattern)

        if avgSimilarity > 40:
            patIndex = patterArr.index(eachPattern)
            print '###############################'
            print "currentPattern[10]:" ,patForRec
            print "pastPattern[10]:" ,eachPattern
            print "predicted outcome", performanceArr[patIndex]
            xp = list(range(1,31))
            fig  = plt.figure()
            plt.plot(xp,patForRec)
            plt.plot(xp,eachPattern)
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

patternStorage()
currentPattern()
patternRecognition()

totalEnd = time.time()

print "Entire processing time taken:", totalEnd - totalStart
