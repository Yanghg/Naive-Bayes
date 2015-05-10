execfile('bayes.py')
bc = Bayes_Classifier()
strRes = bc.loadFile('movies_reviews/movies-1-93.txt')
print bc.tokenize(strRes)