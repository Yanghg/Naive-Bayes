execfile('bayes.py')
bc = Bayes_Classifier()
result = bc.tenFoldValidation()
print result
#strRes = bc.loadFile('movies_reviews/movies-1-93.txt')
#print bc.tokenize(strRes)
#bc.train()

