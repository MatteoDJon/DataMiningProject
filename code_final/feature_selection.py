import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import mutual_info_classif
#from weka.attribute_selection import ASSearch
#from weka.attribute_selection import ASEvaluation
#from weka.attribute_selection import AttributeSelection



# https://machinelearningmastery.com/feature-selection-machine-learning-python/

def get_info_selection(features,labels,desiredNumberFeaturesSelected):

    sel_f = SelectKBest(mutual_info_classif, k=desiredNumberFeaturesSelected);
    sel_f.fit(features, labels);

    features_index = [];

    features_selected_info = sel_f.get_support();

    for i in range(0,len(features_selected_info)):
        if(features_selected_info[i]==True):
            features_index.append(i);

    return features_index

def get_f_selection(features,labels,desiredNumberFeaturesSelected):
    sel_f = SelectKBest(f_classif, k=desiredNumberFeaturesSelected);
    sel_f.fit(features, labels);

    features_index = [];

    features_selected_f = sel_f.get_support();

    for i in range(0,len(features_selected_f)):
        if(features_selected_f[i]==True):
            features_index.append(i);

    return features_index


def get_features_selected(features, labels,desiredNumberFeaturesSelected):
    best_features = [];

    actualFeaturesSelected=desiredNumberFeaturesSelected

    while(len(best_features)<desiredNumberFeaturesSelected):

        sel_f = SelectKBest(f_classif, k=actualFeaturesSelected);
        sel_f.fit(features,labels);

        features_selected_f = sel_f.get_support();

        self_info = SelectKBest(mutual_info_classif, k=actualFeaturesSelected);
        self_info.fit(features,labels);

        features_selected_info = self_info.get_support();

        best_features = []
        for i in range(0,actualFeaturesSelected):
            if(features_selected_f[i] == True and features_selected_info[i] ==True):
                best_features.append(i);

        actualFeaturesSelected+=1

    return best_features;


def run_feature_selection(features, labels, feature_selection, best_features):

   '''
    if feature_selection == 'select_K_Best':
        # feature extraction
        selector = SelectKBest(score_func=f_classif, k=4) # score_func=chi2 : only for non-negative features
        selector.fit(features, labels)
        # summarize scores
        scores = selector.scores_
        features_index_sorted = np.argsort(-scores)
        #features_selected = features[:, features_index_sorted[0:best_features]]
        np.set_printoptions(precision=3)
        print(selector.scores_)
   '''
   if feature_selection == 'select_chi':
       # feature extraction
       selector = SelectKBest(score_func=chi2, k=4) # score_func=chi2 : only for non-negative features
       selector.fit(features, labels)
       # summarize scores
       scores = selector.scores_
       #features_index_sorted = np.argsort(-scores)
       np.set_printoptions(precision=3)
       print(selector.scores_)
       #features_selected = features[:, features_index_sorted[0:best_features]]

   if feature_selection == 'select_fclassif':
       # feature extraction
       selector = SelectKBest(score_func=f_classif, k=4) # score_func=chi2 : only for non-negative features
       selector.fit(features, labels)
       # summarize scores
       scores = selector.scores_
       #features_index_sorted = np.argsort(-scores)
       np.set_printoptions(precision=3)
       print(selector.scores_)
       #features_selected = features[:, features_index_sorted[0:best_features]]

   # SelectFromModel and LassoCV

   # We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
   if feature_selection == 'LassoCV':
       clf = LassoCV()

       # Set a minimum threshold of 0.25
       sfm = SelectFromModel(clf, threshold=0.95)
       sfm.fit(features, labels)
       features_selected = sfm.transform(features).shape[1]

       """
       # Reset the threshold till the number of features equals two.
       # Note that the attribute can be set directly instead of repeatedly
       # fitting the metatransformer.
       while n_features > 2:
           sfm.threshold += 0.1
           X_transform = sfm.transform(X)
           n_features = X_transform.shape[1]
       """

   if feature_selection == 'info_Gain':
      # mutual_info_classif(features,labels,discrete_features=True)
       print(mutual_info_classif(features,labels,discrete_features=True))


   # Univariate feature selection
   # Univariate feature selection works by selecting the best features based on univariate statistical tests.
   # It can be seen as a preprocessing step to an estimator.
   # Scikit-learn exposes feature selection routines as objects that implement the transform method:
   #   - SelectKBest removes all but the k highest scoring features
   #   - SelectPercentile removes all but a user-specified highest scoring percentage of features
   #       common univariate statistical tests for each feature: false positive rate SelectFpr, false discovery rate SelectFdr, or family wise error SelectFwe.
   #   - GenericUnivariateSelect allows to perform univariate feature selection with a configurable strategy. This allows to select the best univariate selection strategy with hyper-parameter search estimator.

   if feature_selection == 'slct_percentile':
       selector = SelectPercentile(f_classif, percentile=10)
       selector.fit(features, labels)
       # The percentile not affect.
       # Just select in order the top features by number or threshold

       # Keep best 8 values?
       scores = selector.scores_
       features_index_sorted = np.argsort(-scores)
       np.set_printoptions(precision=3)
       print(selector.scores_)

       # scores = selector.scores_

       # scores = -np.log10(selector.pvalues_)
       # scores /= scores.max()

       #features_selected = features[:, features_index_sorted[0:best_features]]

   '''
    if feature_selection == "first_method": #cfsSubsetEval(attribute evaluator)+bestFirst(search method)
        search = ASSearch(classname="weka.attributeSelection.BestFirst", options=["-D", "1", "-N", "5"])
        evaluation = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval", options=["-P", "1", "-E", "1"])
        attsel = AttributeSelection()
        attsel.search(search)
        attsel.evaluator(evaluation)
        attsel.select_attributes(features)
        print("# attributes: " + str(attsel.number_attributes_selected))
        print("attributes (as numpy array): " + str(attsel.selected_attributes))
        print("attributes (as list): " + str(list(attsel.selected_attributes)))
        print("result string:\n" + attsel.results_string)

    if feature_selection == "second_method": #infoGain(attribute evaluator)+ranker(search method)
        helper.print_title("Attribute ranking (2-fold CV)")
        search = ASSearch(classname="weka.attributeSelection.Ranker", options=["-N", "-1"])
        evaluation = ASEvaluation("weka.attributeSelection.InfoGainAttributeEval")
        attsel = AttributeSelection()
        attsel.ranking(True)
        #attsel.folds(2)
        attsel.crossvalidation(False)
        #attsel.seed(42)
        attsel.search(search)
        attsel.evaluator(evaluation)
        attsel.select_attributes(features)
        print("ranked attributes:\n" + str(attsel.ranked_attributes))
        print("result string:\n" + attsel.results_string)

    if feature_selection == "third method": #wrappedC45(attribute evaluator)+bestFirst(search method)
        classifier = Classifier(classname="weka.classifiers.meta.AttributeSelectedClassifier")
        aseval = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval")
        assearch = ASSearch(classname="weka.attributeSelection.WrapperSubsetEval", options=["-B"])
        base = Classifier(classname="weka.classifiers.trees.J48")
        # setting nested options is always a bit tricky, getting all the escaped double quotes right
        # simply using the bean property for setting Java objects is often easier and less error prone
        classifier.set_property("classifier", base.jobject)
        classifier.set_property("evaluator", aseval.jobject)
        classifier.set_property("search", assearch.jobject)
        evaluation = Evaluation(data)
        evaluation.crossvalidate_model(classifier, data, 10, Random(1))
        print(evaluation.summary())
    '''


    #print("Selected only " + str(features_selected.shape) + " features ")

    #return features_selected, features_index_sorted