% steps of my documentation

## (I) Classification and Results

In this section it is explaned the classification  flow and the classification results. 

At this point the system gives us 2 pair of signals representing the same beats, from MLII and V1, and the correct 
classifications of them.

In this article five of the more popular classification algorithms avaible are tested. 
Two classifier for each algorithm are trained with the MLII and V1 beat-signals, then a voting strategy has been 
provided  to decide the final classification.

The algorithms selected in this first classification task are:
1. Naive Bayesian
2. Random Forest
3. C4.5
4. K-NN
5. SVM 

The main problem for classification is that the signals are strongly imbalanced, for this reason the more interesting 
classifiers would be 2. and 5. for their good performance in case of imbalanced data. Furthmore classifiers have been 
tested with and without oversampling of the datas, in order to understand if an oversampling technique would be 
effective in our scope.

Finally a voting strategy has been provided inserting another classification step in the flow: in order to handle all 
the possible combination of classifiers results trained with MLII and V1, a decision tree approach has been used,
with a **Random Forest** classifier ( due to its good performance with imbalanced data ). This classifier has been 
trained with the couples of class-prediction of the two model trained before.

