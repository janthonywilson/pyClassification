'''
Created on Oct 21, 2015

@author: Tony
'''

import os
import sys
import time

import nltk
import numpy
import scipy
import pandas
import gensim
import scipy.cluster.hierarchy 
import scipy.spatial.distance
import matplotlib.pyplot as pyplot

# from sklearn.kernel_ridge import KernelRidge
from sklearn import metrics
from sklearn import cluster
from sklearn.externals import joblib
from sklearn.decomposition import FastICA
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder



#===================================================================================
# pySummaryClassify CLASS
#===================================================================================
class pySummaryClassify(object):
    """Main Classification Class for Summary Data"""
    def __init__( self ):
        self.Sentence = []
        self.PreprocStem = False
        self.DataFile = []
        self.DataDir = 'data'
        self.DataFrameObject = pandas.DataFrame()
        self.Features = []
        self.FeatureNames = []
        self.Classes = []
        self.FeatureSelectionMethod = 'Variance'
        self.FeatureGenerationMethod = 'tf-idf'
        self.ReTrainW2V = False
        self.Sentences = []
        self.Summaries = []
        self.Vocab = []
        self.Ratings = []
        self.ICAKurt = False
        self.TFIDFAnalyzer = 'word'
        self.TFIDFMaxDF = 0.99
        self.TFIDFMaxFeatures = 200000
        self.TFIDFMinDF = 0.01
        self.TFIDFUseIDF= True
        self.TFIDFNgramRange = (1,3)
        self.ClassesValues = []
        self.ClassesValuesUnique = []
        self.ClassesValuesCount = 0
        self.ColumNames = ['ID',    'ShowName',    'EpisodeName',    'Class',    'Rating',    'Summary']



    #===============================================================================
    def LoadDataFrameObject( self ):
        """Load data from tab text file"""
        # DataFile = self.DataFile 
        # DataFile = 'SummaryEpisode.txt'
        # DataFile = 'SummaryDataFirstGenre.txt'
        # DataFile = 'SummaryDataGroupedGenre.txt'
        # DataFile = 'SummaryDataSplitGenre.txt'
        self.DataFrameObject = pandas.read_csv(self.DataDir +os.sep+ self.DataFile, header = 0, delimiter='\t', na_values='.', names = self.ColumNames)
    #===============================================================================
    def PreprocSentence( self ):
        """Main Preprocessing on a single sentence method"""
        
        StopWordsSet = set(nltk.corpus.stopwords.words('english'))
        if not self.Sentence:
            print 'No sentence initialized for Preprocessing'
            sys.exit()
        sentence = unicode(self.Sentence.lower(), errors='replace')
        Tokenizer = nltk.RegexpTokenizer(r'\w+')
        SummaryToken = Tokenizer.tokenize(sentence)
        SummaryToken = [w for w in SummaryToken if not w in StopWordsSet]
        if self.PreprocStem:
            Stemmer = nltk.stem.porter.PorterStemmer()
            SummaryTokenStem = [Stemmer.stem(i) for i in SummaryToken]
            self.Sentence = ( ' '.join( SummaryTokenStem ))  
        else:
            self.Sentence = SummaryToken
    #===============================================================================
    def FeatureSelection( self ):
        """Main feature selection method"""
        
        if 'Variance' in self.FeatureSelectionMethod:
            selector = VarianceThreshold(threshold=0.0001)
            self.Features = selector.fit_transform(self.Features)
    #         pyplot.figure(), pyplot.hist(numpy.var(features, axis = 0), bins = 64), pyplot.show()
        elif 'Trees' in self.FeatureSelectionMethod:
            forestFeatures = ExtraTreesClassifier(n_estimators = 512, random_state = 32)
            forestFeaturesFit = forestFeatures.fit(self.Features, self.Classes)
            featureImportance = 0.001
            featureBool = (forestFeaturesFit.feature_importances_ > featureImportance)
            self.Features = self.Features[:,featureBool]
    #===============================================================================
    def StackPreproc( self ):
        for i in xrange(0, len(self.DataFrameObject.Summary)):
            self.Sentence = self.DataFrameObject.Summary[i]
            self.PreprocSentence()
            self.Summaries.append( self.Sentence )
            self.Sentences.append( ' '.join(self.Sentence)  )
            self.Vocab +=  self.Sentence
            if self.DataFrameObject.Rating[i] not in 'none':
                self.Ratings.append( int(numpy.round( float(self.DataFrameObject.Rating[i]) )) )
            else:
                self.Ratings.append( -1 )
                
        self.Vocab = list(set(self.Vocab))
    #===============================================================================
    def GenFeatures( self ):
        """Main feature generation method"""
        
        if 'Word2Vec' in self.FeatureGenerationMethod:
            if self.ReTrainW2V:
                W2V = gensim.models.Word2Vec(min_count = 8, size = 256, workers = 1, window = 5)
                W2V.build_vocab(self.Summaries)
                W2V.train(self.Summaries)
                joblib.dump(W2V, 'data' +os.sep+ 'W2VModel' +os.sep+ 'modelW2V.pkl') 
            else:
                W2V = joblib.load('data' +os.sep+ 'W2VModel' +os.sep+ 'modelW2V.pkl')
    
            modelSet = set(W2V.index2word)
            icaFeatures = numpy.zeros( (len(self.Summaries), W2V.vector_size) )
            fastICA = FastICA(n_components = 32, whiten = True, max_iter=2048, algorithm='deflation')
        
            
            for i in xrange(0, len(self.Summaries)):
                textVec = self.Summaries[i]
                features = []
                for word in textVec:
                    if word in modelSet: 
                        features.append(W2V[word])
            
                features = numpy.asarray(features, dtype = numpy.float32)
                
                if features.size > 0:
                    U, s, V = numpy.linalg.svd(features, full_matrices=False)
                    SVDComponents = U
                    SVDVar = numpy.cumsum((s))
        #             pyplot.subplot(1,3,1), pyplot.imshow(U, interpolation = 'none', vmin = 0.0, vmax = 1.0), pyplot.colorbar(), pyplot.subplot(1,3,2), pyplot.imshow(V, interpolation = 'none', vmin = 0.0, vmax = 1.0), pyplot.colorbar(), pyplot.subplot(1,3,3), pyplot.plot(SVDVar), pyplot.show()
        #             print U.shape, V.shape, s.shape
                    try:
                        sourceICA = fastICA.fit(features).transform(features)
                        fastICAComponents = fastICA.components_
                        if self.ICAKurt:
                            kurtS = scipy.stats.kurtosis(fastICAComponents, axis = 1)
                            kurtIdx = numpy.argmax(kurtS)
                            icaFeatures[i, :] = fastICAComponents[kurtIdx, :]
                            print i, 'Kurtosis: ' +str(kurtS[kurtIdx]), 'S Shape: ' +str(sourceICA.shape), 'Kurt Shape: ' +str(kurtS.shape), 'ICA Shape: ' +str(fastICAComponents.shape)
                        else:
                            print 'Loop ' +str(i)+ ' of ' +str(len(self.Summaries)) 
                            icaFeatures[i, :] = numpy.mean(fastICAComponents, axis = 0)
                            
                    except Exception, e:
                        print i, e
                    
                else:
                    print i, type(features), features.shape, features.size
                    icaFeatures[i, :] = numpy.zeros( (W2V.vector_size,) )
                
                
            pyplot.figure(), pyplot.imshow(icaFeatures, interpolation = 'none', vmin = 0.0, vmax = 1.0), pyplot.colorbar(), pyplot.show()
            return icaFeatures
        
        elif 'tf-idf' in self.FeatureGenerationMethod:
            vectorizer = TfidfVectorizer(analyzer=self.TFIDFAnalyzer, max_df=self.TFIDFMaxDF, max_features=self.TFIDFMaxFeatures, min_df=self.TFIDFMinDF, use_idf=self.TFIDFUseIDF, ngram_range=self.TFIDFNgramRange)
#             vectorizer = TfidfVectorizer(self.TfidfAnalyzer, self.MaxDF, self.MaxFeatures, self.MinDF, self.UseIDF, self.NGramRange)
            self.Features = vectorizer.fit_transform( self.Sentences )
            self.FeatureNames  = vectorizer.get_feature_names()
    #===============================================================================
    def Vis( self, data ):
        dataShape = data.shape
        print dataShape, type(data)
        if (len(dataShape) > 1):
            if ( scipy.sparse.issparse(data) ):
                pyplot.imshow(data.toarray(), aspect = 'equal', interpolation = 'none')
            else:
                pyplot.imshow(data, aspect = 'equal', interpolation = 'none')
            pyplot.colorbar()
            pyplot.show()
    #===============================================================================
    def CountClasses( self ):
        pdClasses = pandas.Series(self.DataFrameObject.Class)
        self.ClassesValues = pdClasses.values
        self.ClassesValuesUnique = numpy.unique(pdClasses.values)
        self.ClassesValuesCount = pdClasses.value_counts()
    #===============================================================================
    def ClusterFeatures( self ):
        dataShape = self.Features.shape        
        print dataShape
        dists = scipy.spatial.distance.pdist(self.Features.todense(), 'euclidean')
        linkage = scipy.cluster.hierarchy.linkage(dists, method = 'complete')
#         dists = metrics.pairwise.pairwise_distances( data.astype(numpy.double), metric = 'euclidean' )
#         linkage = cluster.AgglomerativeClustering( dists, linkage='complete', affinity='euclidean', n_clusters=22, connectivity = False )
#         linkage.fit(data)
        hac = scipy.cluster.hierarchy.fcluster(linkage, 5, 'maxclust')
        
        
        
        
        
        
        
        
        
        
        
        
        
