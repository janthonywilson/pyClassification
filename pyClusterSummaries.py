'''
Created on Sep 1, 2015

@author: Tony
'''

import os
# import sys
# import csv
import time

import nltk
import numpy
import pandas
# import theano
import gensim
import scipy
# import itertools
import matplotlib.pyplot as pyplot

# from sklearn.kernel_ridge import KernelRidge
from sklearn.externals import joblib
from sklearn.decomposition import FastICA
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from sklearn import feature_extraction, tree
from sklearn.svm import SVC
# from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
# from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, FeatureHasher, TfidfVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import SGDClassifier

# http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
# https://github.com/jimmycallin/pydsm
# https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-3-more-fun-with-word-vectors

sTime = time.time()
Class = 'Genre'
ReTrainW2V = False
ICAKurt = False
W2V = True
GenW2VFeatures = True
PreprocStem = False
FeatureSelection = False
# Set print to be more reasonable...
numpy.set_printoptions(threshold=numpy.nan)
# pandas.set_option('display.height', 11300)
pandas.set_option('display.max_rows', 11300)

def preprocess(sentence):
    
    StopWordsSet = set(nltk.corpus.stopwords.words('english'))
    sentence = unicode(sentence.lower(), errors='replace')
    Tokenizer = nltk.RegexpTokenizer(r'\w+')
    SummaryToken = Tokenizer.tokenize(sentence)
    SummaryToken = [w for w in SummaryToken if not w in StopWordsSet]
    if PreprocStem:
        Stemmer = nltk.stem.porter.PorterStemmer()
        SummaryTokenStem = [Stemmer.stem(i) for i in SummaryToken]
        return( ' '.join( SummaryTokenStem ))  
    else:
        return SummaryToken

def featureSelection(features, classes, method):
    # Feature Selection...
    print features.shape
    
    if 'variance' in method:
        selector = VarianceThreshold(threshold=0.0001)
        features = selector.fit_transform(features)
#         pyplot.figure(), pyplot.hist(numpy.var(features, axis = 0), bins = 64), pyplot.show()
    elif 'trees' in method:
        forestFeatures = ExtraTreesClassifier(n_estimators = 512, random_state = 32)
        forestFeaturesFit = forestFeatures.fit(features, classes)
        featureImportance = 0.001
        featureBool = (forestFeaturesFit.feature_importances_ > featureImportance)
        features = features[:,featureBool]
    
    print features.shape
    return features

def genW2VFeatures(model, text):
    modelSet = set(model.index2word)
    icaFeatures = numpy.zeros( (len(text), model.vector_size) )
    fastICA = FastICA(n_components = 32, whiten = True, max_iter=2048, algorithm='deflation')

    
    for i in xrange(0, len(text)):
        textVec = text[i]
        features = []
        for word in textVec:
            if word in modelSet: 
                features.append(model[word])
    
        features = numpy.asarray(features, dtype = numpy.float32)
#         featuresNaN = numpy.isnan(features)
#         featuresInf = numpy.isinf(features)
#         print (featuresNaN == True), (featuresInf == True)
        
        if features.size > 0:
            U, s, V = numpy.linalg.svd(features, full_matrices=False)
            SVDComponents = U
            SVDVar = numpy.cumsum((s))
#             pyplot.subplot(1,3,1), pyplot.imshow(U, interpolation = 'none', vmin = 0.0, vmax = 1.0), pyplot.colorbar(), pyplot.subplot(1,3,2), pyplot.imshow(V, interpolation = 'none', vmin = 0.0, vmax = 1.0), pyplot.colorbar(), pyplot.subplot(1,3,3), pyplot.plot(SVDVar), pyplot.show()
#             print U.shape, V.shape, s.shape
            try:
                sourceICA = fastICA.fit(features).transform(features)
                fastICAComponents = fastICA.components_
                if ICAKurt:
                    kurtS = scipy.stats.kurtosis(fastICAComponents, axis = 1)
                    kurtIdx = numpy.argmax(kurtS)
                    icaFeatures[i, :] = fastICAComponents[kurtIdx, :]
                    print i, 'Kurtosis: ' +str(kurtS[kurtIdx]), 'S Shape: ' +str(sourceICA.shape), 'Kurt Shape: ' +str(kurtS.shape), 'ICA Shape: ' +str(fastICAComponents.shape)
                else:
                    print 'Loop ' +str(i)+ ' of ' +str(len(text)) 
                    icaFeatures[i, :] = numpy.mean(fastICAComponents, axis = 0)
                    
            except Exception, e:
                print i, e
                
#                 PCAIdx = numpy.argmax(PCAVar)
#                 icaFeatures[i, :] = PCAComponents[PCAIdx, :]

            
#             pyplot.clf(), pyplot.plot(icaFeatures[i, :]), pyplot.show()
            
        else:
            print i, type(features), features.shape, features.size
            icaFeatures[i, :] = numpy.zeros( (model.vector_size,) )
        
        
    pyplot.figure(), pyplot.imshow(icaFeatures, interpolation = 'none', vmin = 0.0, vmax = 1.0), pyplot.colorbar(), pyplot.show()
#     print icaFeatures[1:10, :]
    return icaFeatures
            
# DataFile = 'SummaryEpisodeActionComedyDrama.txt'
# DataFile = 'SummaryEpisodeAllGenre.txt'
DataFile = 'SummaryEpisode.txt'
# DataFile = 'SummaryDataFirstGenre.txt'
# DataFile = 'SummaryDataGroupedGenre.txt'
# DataFile = 'SummaryDataSplitGenre.txt'
DataFrameObject = pandas.read_csv('data' +os.sep+ DataFile, header = 0, delimiter='\t', na_values='.')

# print DataFrameObject.shape
# DataFrameObject = DataFrameObject[(DataFrameObject['Genre'] == 'Drama') | (DataFrameObject['Genre'] == 'Comedy') | (DataFrameObject['Genre'] == 'Action')]
# DataFrameObject = DataFrameObject.reindex()
# print DataFrameObject.shape


preprocSummaryStr = []
preprocSummary = []
preprocSummaryWords = []
preprocRating = []
for i in xrange(0, len(DataFrameObject.Summary)):
    SummaryToken = preprocess(DataFrameObject.Summary[i])
    preprocSummary.append( SummaryToken )
    preprocSummaryStr.append( ' '.join(SummaryToken)  )
    preprocSummaryWords +=  SummaryToken
    if DataFrameObject.Rating[i] not in 'none':
        preprocRating.append( int(numpy.round( float(DataFrameObject.Rating[i]) )) )
    else:
        preprocRating.append( -1 )


preprocSummaryWords = list(set(preprocSummaryWords))

if ReTrainW2V:
    modelW2V = gensim.models.Word2Vec(min_count = 8, size = 256, workers = 1, window = 5)
    modelW2V.build_vocab(preprocSummary)
    modelW2V.train(preprocSummary)
    joblib.dump(modelW2V, 'data' +os.sep+ 'W2VModel' +os.sep+ 'modelW2V.pkl') 
else:
    modelW2V = joblib.load('data' +os.sep+ 'W2VModel' +os.sep+ 'modelW2V.pkl')

if GenW2VFeatures:
    featuresW2V = genW2VFeatures(modelW2V, preprocSummary)
    numpy.savetxt('data' +os.sep+ 'ICAFeatures.txt', featuresW2V, delimiter = '\t')
else:
    featuresW2V = numpy.loadtxt('data' +os.sep+ 'ICAFeatures.txt', dtype = numpy.float32, delimiter = '\t')

vectorizer = TfidfVectorizer(analyzer='word', max_df=0.99, max_features=200000, min_df=0.01, use_idf=True, ngram_range=(1,3))
# vectorizer = HashingVectorizer(non_negative=True, n_features=200000)
# vectorizer = FeatureHasher(n_features=200000, input_type='string', non_negative=False)
# vectorizer = CountVectorizer(analyzer = 'word', tokenizer = None, preprocessor = None, stop_words = None, max_features = 1800) 


featuresSummary = vectorizer.fit_transform( preprocSummaryStr )
featuresNames  = vectorizer.get_feature_names()
# pyplot.figure(), pyplot.imshow(featuresSummary.todense()), pyplot.show()

if not W2V:
    npSummaryFeatures = featuresSummary.toarray()
elif W2V:
    npSummaryFeatures = featuresW2V

if Class is 'Genre':
    labelEncode = LabelEncoder()
    labelEncode.fit(DataFrameObject.Genre)
    classEncode = labelEncode.transform(DataFrameObject.Genre)
elif Class is 'Rating':
    classEncode = numpy.array(preprocRating)
    
# Count the classes...
SeriesGenre = pandas.Series(DataFrameObject.Genre)
SeriesGenreValues = SeriesGenre.values
SeriesGenreValuesUnique = numpy.unique(SeriesGenre.values)
vcSeriesGenre = SeriesGenre.value_counts()

SeriesRating = pandas.Series(preprocRating)
vcSeriesRating = SeriesRating.value_counts()
SeriesRatingValues = SeriesRating.values

SeriesGenreEncode = pandas.Series(classEncode)
vcSeriesGenreEncode = SeriesGenreEncode.value_counts()
# print vcSeriesGenre, vcSeriesRating, vcSeriesGenreEncode

AverageRating = numpy.zeros( (len(SeriesGenreValuesUnique), ) )
for i in xrange(0, len(SeriesGenreValuesUnique)):
    GenreBool = (SeriesGenreValues == SeriesGenreValuesUnique[i])
    Ratings = SeriesRatingValues[GenreBool]
    Ratings = Ratings[Ratings > -1]
#     print SeriesGenreValuesUnique[i], Ratings
    AverageRating[i] = numpy.mean( Ratings ) 

if FeatureSelection:
    npSummaryFeatures = featureSelection(npSummaryFeatures, classEncode, 'variance')
    
trainSummaryFeatures, testSummaryFeatures, trainClass, testClass = train_test_split(npSummaryFeatures, classEncode, test_size=0.33, random_state=32)


# Ensemble Parms...
nEstimators = 128

randForest = RandomForestClassifier(n_estimators = nEstimators, min_samples_split = 8) 
randForest = randForest.fit( trainSummaryFeatures, trainClass )
randForestPredict = randForest.predict( testSummaryFeatures )
randForestAccuracy = accuracy_score(testClass, randForestPredict)
randForestCM = confusion_matrix(testClass, randForestPredict)
pyplot.matshow(numpy.log(randForestCM+1)), pyplot.colorbar(), pyplot.show()

baggingClass = BaggingClassifier(n_estimators = nEstimators)
baggingClass = baggingClass.fit( trainSummaryFeatures, trainClass ) 
baggingClassPredict = baggingClass.predict( testSummaryFeatures )
baggingClassAccuracy = accuracy_score(testClass, baggingClassPredict)

boostingClass = AdaBoostClassifier(n_estimators = nEstimators)
boostingClass = boostingClass.fit( trainSummaryFeatures, trainClass ) 
boostingClassPredict = boostingClass.predict( testSummaryFeatures )
boostingClassAccuracy = accuracy_score(testClass, boostingClassPredict)

# bayesNaive = MultinomialNB()
# bayesNaive = bayesNaive.fit( trainSummaryFeatures, trainClass )
# bayesNaivePredict = bayesNaive.predict( testSummaryFeatures )
# bayesNaiveAccuracy = accuracy_score(testClass, bayesNaivePredict)
bayesNaiveAccuracy = 0.0


sgdClass = SGDClassifier(n_iter = 128)
sgdClass = sgdClass.fit(trainSummaryFeatures, trainClass)
sgdClassPredict = sgdClass.predict( testSummaryFeatures )
sgdClassAccuracy = accuracy_score(testClass, sgdClassPredict)


# C_range = numpy.logspace(-2, 10, 13)
# gamma_range = numpy.logspace(-9, 3, 13)
# param_grid = dict(gamma=gamma_range, C=C_range)
# # cv = StratifiedShuffleSplit(classEncode, n_iter = 5, test_size = 1, random_state = 32)
# grid = GridSearchCV(SVC(), param_grid=param_grid, cv=None)
# grid.fit(trainSummaryFeatures, trainClass)

svClass = SVC(C = 10.0, kernel='rbf', degree = 5, gamma = 1.0, coef0 = 0.0)
svClass = svClass.fit(trainSummaryFeatures, trainClass)
svClassPredict = svClass.predict(testSummaryFeatures)
svClassAccuracy = accuracy_score(testClass, svClassPredict)

dtClass = tree.DecisionTreeClassifier()
dtClass = dtClass.fit(trainSummaryFeatures, trainClass)
dtClassPredict = dtClass.predict(testSummaryFeatures)
dtClassAccuracy = accuracy_score(testClass, dtClassPredict)
dtClassScore = dtClass.score(testSummaryFeatures, testClass)

print ('Rand Forest: ' + str(randForestAccuracy))
print ('Bagging: ' + str(baggingClassAccuracy))
print ('Boosting: ' + str(boostingClassAccuracy))
print ('Bayes: ' + str(bayesNaiveAccuracy))
print ('SGD: ' + str(sgdClassAccuracy))
print ('SVC: ' + str(svClassAccuracy))
print ('DT: ' + str(dtClassAccuracy))

print('Duration: %s' % (time.time() - sTime))







