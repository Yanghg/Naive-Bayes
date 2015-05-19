# Name: 
# Date:
# Description:
#
#

import math, os, pickle, re, copy
import nltk

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
      self.uselessword=['a','an','is','am','are','the','here','there','I','you','he','she','him','his','hers','her','it','its','this','that','which']
      self.checkset=set()
      for i in self.uselessword:
         self.checkset.add(i)
      if os.path.isfile('store.pkl'):
         trainData = self.load('store.pkl')
         self.positive = trainData[0]
         self.negative = trainData[1]
         self.positiveNum = trainData[2]
         self.negativeNum = trainData[3]

      else:
         self.train()
      '''
      print 'positive:'
      for p in self.positive:
         print p + ':' + str(self.positive[p])
      print '\nnegative:'
      for n in self.negative:
         print n + ':' + str(self.negative[n])
      '''
      print self.positiveNum
      print self.negativeNum

   def train(self):   
      """Trains the Naive Bayes Sentiment Classifier."""
      positiveNum = 0
      negativeNum = 0
      trainData = []
      lFileList = []
      rating = 0
      for fFileObj in os.walk('movies_reviews/'):
         lFileList = fFileObj[2]
         break
      #return lFileList

      #put words of all documents into two dictionarys
      for filename in lFileList:
         if filename[0] == '.':
            continue
         rating = int(filename.split('-')[1])
         reviewStr = self.loadFile('movies_reviews/' + filename)
         reviewWords = self.tokenize(reviewStr)
         tempDic = {}
         #put words of a document into two dictionarys
         i = 0
         while i < len(reviewWords):
            #turn letters of the word to lowercase
            word = reviewWords[i].lower()
            #extract stem of words
            if len(word) == 1 and ord(word) >= 128:
               break
            porter = nltk.PorterStemmer()
            word = str(porter.stem(word))
            if word in self.checkset :
               continue
            # cancel the useless part before the turn
            '''
            if word == 'but' or word == 'however':
               tempDic = {}
            '''
            #extract negative part
            if word[-3:] == "not" or word[-3:] == "n't":
               if i + 1 < len(reviewWords):
                  word = 'n-' + reviewWords[i+1]
                  i += 1 
            
            if not tempDic.has_key(word):
               tempDic[word] = True
            i += 1 
         if (rating == 5):
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

      trainData.append(self.positive)
      trainData.append(self.negative)
      trainData.append(positiveNum)
      trainData.append(negativeNum)     
      self.save(trainData,'store.pkl')
      '''
      dicP = self.load('store.pkl')
      if dicP[0] == self.positive:
         print True
      if dicP[1] == self.negative:
         print True
      '''
    
   def classify(self, sText):
      """Given a target string sText, this function returns the most likely document
      class to which the target string belongs (i.e., positive, negative or neutral).
      """
      tokenList = self.tokenize(sText)
      positive, negative = self.addOneSmoothing()
      #useless in this case
      positiveProb = float(self.positiveNum)/(self.positiveNum+self.negativeNum)
      negativeProb = float(self.negativeNum)/(self.positiveNum+self.negativeNum)

      positiveSum = 0
      negativeSum = 0
      difference = positiveSum - negativeSum
      i = 0

      while i < len(tokenList):
         #turn letters to lowercase
         token = tokenList[i].lower()
         #extract stem of words
         if len(token) == 1 and ord(token) >= 128:
            break
         porter = nltk.PorterStemmer()
         token = str(porter.stem(token))
         if token in self.checkset :
            continue
         #extract negative part
         if token[-3:] == "not" or token[-3:] == "n't":
               if i + 1 < len(tokenList):
                  token = 'n-' + tokenList[i+1]
                  i += 1
         i += 1 

         if positive.has_key(token):
            positiveSum += math.log(float(positive[token])/self.positiveNum,2)
            negativeSum += math.log(float(negative[token])/self.negativeNum,2)
      
      print positiveSum, negativeSum
      if positiveSum - negativeSum > difference - 1.5 and positiveSum - negativeSum < difference + 1.5:
         return 'Neutral'
      elif positiveSum - negativeSum >= difference + 1.5:
         return 'Positive'
      else:
         return 'Negative'

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

   def tokenize(self, sText): 
      """Given a string of text sText, returns a list of the individual tokens that 
      occur in that string (in order)."""

      lTokens = []
      sToken = ""
      for c in sText:
         if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\"" or c == "_" or c == "-":
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
      print len(allSet)
      for element in allSet:
         if not positive.has_key(element):
            positive[element] = 0
         if not negative.has_key(element):
            negative[element] = 0
         positive[element] += 1
         negative[element] += 1
      return positive, negative


