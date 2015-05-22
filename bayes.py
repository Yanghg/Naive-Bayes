# Name: 
# Date:
# Description:
#
#

import math, os, pickle, re, copy, sys


class Bayes_Classifier:

   def __init__(self):
      """This method initializes and trains the Naive Bayes Sentiment Classifier.  If a 
      cache of a trained classifier has been stored, it loads this cache.  Otherwise, 
      the system will proceed through training.  After running this method, the classifier 
      is ready to classify input text."""
      self.positive = {}
      self.negative = {}
      self.positiveNum = 0
      self.negativeNum = 0
      self.tp = 0
      self.fp = 0
      self.tn = 0
      self.fn = 0
      if os.path.isfile('store.pkl'):
         trainData = self.load('store.pkl')
         self.positive = trainData[0]
         self.negative = trainData[1]
         self.positiveNum = trainData[2]
         self.negativeNum = trainData[3]

      else:
         self.train(self.generateFileList(10)[0])
      # print self.positiveNum
      # print self.negativeNum

   def generateFileList(self,i):
      trainList = []
      validateList = []
      goodList = []
      badList = []
      for fFileObj in os.walk('movies_reviews/'):
         lFileList = fFileObj[2]
         break
      #return lFileList
      for filename in lFileList:
         if filename[0] == '.':
            continue
         rating = int(filename.split('-')[1])
         if rating == 1:
            badList.append(filename)
         else:
            goodList.append(filename)
      if i < 10:
         validateList = goodList[int(len(goodList)*i/10.0):int(len(goodList)*(i+1)/10.0)]+badList[int(len(badList)*i/10.0):int(len(badList)*(i+1)/10.0)]
         trainList = goodList[0:int(len(goodList)*i/10.0)]+goodList[int(len(goodList)*(i+1)/10.0):len(goodList)]+badList[0:int(len(badList)*i/10.0)]+badList[int(len(badList)*(i+1)/10.0):len(badList)]
         return trainList,validateList
      else:
         return goodList+badList,[]

   def train(self,fileList,isTenFold = False):   
      """Trains the Naive Bayes Sentiment Classifier."""
      positiveNum = 0
      negativeNum = 0
      self.positive = {}
      self.negative = {}
      trainData = []
      count = 0
      self.tp = 0
      self.fp = 0
      self.tn = 0
      self.fn = 0
      #put words of all documents into two dictionarys
      for filename in fileList:
         count += 1
         percentTraining = int(100*float(count)/len(fileList))
         if percentTraining > 100:
            percentTraining = 100
         sys.stdout.write( "Training Progress: %d%%\r" % percentTraining)  
         sys.stdout.flush()   
         if filename[0] == '.':
            continue
         rating = int(filename.split('-')[1])
         reviewStr = self.loadFile('movies_reviews/' + filename)
         reviewWords = self.tokenize(reviewStr)
         tempDic = {}
         for word in reviewWords:
            #word = word.lower()
            if not tempDic.has_key(word):
               tempDic[word] = True
         if rating == 5:
            positiveNum += 1
            for key in tempDic:
               if not self.positive.has_key(key):
                  self.positive[key] = 0
               self.positive[key] += 1
         else:
            negativeNum += 1
            for key in tempDic:
               if not self.negative.has_key(key):
                  self.negative[key] = 0
               self.negative[key] += 1
      self.positiveNum = positiveNum
      self.negativeNum = negativeNum
      self.addOneSmoothing()

      if not isTenFold:  
         trainData.append(self.positive)
         trainData.append(self.negative)
         trainData.append(positiveNum)
         trainData.append(negativeNum)     
         self.save(trainData,'store.pkl')
      print ""

   def classify(self, sText, isList = False, rating=1):
      """Given a target string sText, this function returns the most likely document
      class to which the target string belongs (i.e., positive, negative or neutral).
      """
      tokenList = self.tokenize(sText)
      #positive, negative = self.addOneSmoothing()
      #useless in this case
      positiveProb = float(self.positiveNum)/(self.positiveNum+self.negativeNum)
      negativeProb = float(self.negativeNum)/(self.positiveNum+self.negativeNum)

      positiveSum = 0
      negativeSum = 0
      difference = positiveSum - negativeSum
      for token in tokenList:
         token = token.lower()
         if self.positive.has_key(token):
            positiveSum += math.log(float(self.positive[token])/self.positiveNum,2)
            negativeSum += math.log(float(self.negative[token])/self.negativeNum,2)
      
      #print "positive result:" + str(positiveSum)
      #print "negative result:" + str(negativeSum)
      if not isList:
         print "Text Sample: "+sText
         if positiveSum - negativeSum > difference - 1.5 and positiveSum - negativeSum < difference + 1.5:
            return "Prediction Result: Neutral"
         elif positiveSum - negativeSum >= difference + 1.5:
            return "Prediction Result: Positive"
         else:
            return "Prediction Result: Negative"
      else:
         if positiveSum > negativeSum:
            if rating == 5:
               #print "right"
               self.tp += 1
            else:
               #print "wrong"
               self.fp += 1 
         else:
            if rating == 1:
               #print "right"
               self.tn += 1 
            else:
               #print "wrong"
               self.fn += 1 

   def loadFile(self, sFilename):
      """Given a file name, return the contents of the file as a string."""

      f = open(sFilename, "r")
      sTxt = f.read()
      f.close()
      return sTxt
   
   def save(self, dObj, sFilename):
      """Given an object and a file name, write the object to the file using pickle."""

      f = open(sFilename, "w")
      p = pickle.Pickler(f)
      p.dump(dObj)
      f.close()
   
   def load(self, sFilename):
      """Given a file name, load and return the object stored in the file."""

      f = open(sFilename, "r")
      u = pickle.Unpickler(f)
      dObj = u.load()
      f.close()
      return dObj

   #there is problem with tokenize, don't to don ' t
   def tokenize(self, sText): 
      """Given a string of text sText, returns a list of the individual tokens that 
      occur in that string (in order)."""

      lTokens = []
      sToken = ""
      for c in sText:
         if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\"" or c == "_" or c == "-" or c =="'":
            sToken += c
         else:
            if sToken != "":
               lTokens.append(sToken)
               sToken = ""
            if c.strip() != "":
               lTokens.append(str(c.strip()))
               
      if sToken != "":
         lTokens.append(sToken)

      return lTokens

   def addOneSmoothing(self):
      positiveSet = set()
      negativeSet = set()
      positive = copy.deepcopy(self.positive)
      negative = copy.deepcopy(self.negative)
      allSet = set()
      for item in positive:
         positiveSet.add(item)
      for item in negative:
         negativeSet.add(item)
      allSet = negativeSet | positiveSet
      #print len(allSet)
      for element in allSet:
         if not positive.has_key(element):
            positive[element] = 0
         if not negative.has_key(element):
            negative[element] = 0
         positive[element] += 1
         negative[element] += 1
      #return positive, negative
      self.positive = positive
      self.negative = negative

   def validate(self,validateList):
      validateDataList = [] 
      for filename in validateList:
         rating = int(filename.split('-')[1])
         reviewStr = self.loadFile('movies_reviews/' + filename)
         validateDataList.append((reviewStr,rating))
      return validateDataList

   def classifyList(self,validateDataList):
      """Given a target string sText, this function returns the most likely document
      class to which the target string belongs (i.e., positive, negative or neutral).
      """
      count = 0
      totalCount = len(validateDataList)
      #print len(validateDataList)
      for item in validateDataList:
         count += 1
         currentPercentage = int((float(count)*100)/totalCount)
         if currentPercentage > 100:
            currentPercentage = 100
         sys.stdout.write( "Validatoin Progress: %d%%\r" % currentPercentage)   
         sys.stdout.flush()
      accuracy = float(self.tp+self.tn)/(self.tp+self.tn+self.fp+self.fn)
      posPrecision = float(self.tp)/(self.tp+self.fp)
      posRecall = float(self.tp)/(self.tp+self.fn)
      negPrecision = float(self.tn)/(self.tn+self.fn)
      negRecall = float(self.tn)/(self.tn+self.fp)
      posF = 2 * posPrecision * posRecall/(posPrecision+posRecall)
      negF = 2 * negPrecision * negRecall/(negPrecision+negRecall)
      print ""
      print "Validation Result: " + str(accuracy) + str(posPrecision) + str(posRecall) + str(negPrecision) + str(negRecall)
      return accuracy, posPrecision, posRecall, negPrecision, negRecall, posF, negF

   def tenFoldValidation(self):
      result = []
      sum = 0
      for i in range(10):
         print "No."+str(i+1)+" Fold Validation:"
         trainList,validateList = self.generateFileList(i)
         self.train(trainList,True)
         validateDataList = self.validate(validateList)
         result.append(self.classifyList(validateDataList))
         print ""

      for i in result:
         averAccuracy += i[0]
         averPosPresicion += i[1]
         averPosRecall += i[2]
         averNegPrecision += i[3]
         averNegRecall += i[4]
         averPosF += i[5]
         averNegF += i[6]
      print "Average Validation Rate for Best Bayes: "+str(float(averAccuracy)/10)+str(float(averPosPresicion)/10)+str(float(averPosRecall)/10)+str(float(averNegPrecision)/10)+str(float(averNegRecall)/10)+str(float(averPosF)/10)+str(float(averNegF)/10)
      #return result





         



