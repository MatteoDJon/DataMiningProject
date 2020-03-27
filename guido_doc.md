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
classifiers would be 2. and 5. for their good performance in case of imbalanced data. Furthermore classifiers have been 
tested with and without oversampling of the datas, in order to understand if an oversampling technique would be 
effective in our scope.

Finally a voting strategy has been provided inserting another classification step in the flow: in order to handle all 
the possible combination of classifiers results trained with MLII and V1, a decision tree approach has been used,
with a **Random Forest** classifier ( due to its good performance with imbalanced data ). This classifier has been 
trained with the couples of class-prediction of the two model trained before.

### 6-Fold Validation
In order to obtain significant test values a 6-Fold Validation has been provided. Like it is metioned before each of this classification steps have been maded two times: one with oversampling and one without. 

Preliminarily the dataset is randomly divided in 2 samples: one for training and one for testing. Then each of the 5 algorithms has been trained and after that has been forced to make predictions on the testing-set. 

For each algorithm 2 models have been trained with MLII and V1 beats-data; then all of their predictions in couple with the actual classes of the testing set have been utilized as training-set for **the Random Forest "voting" model**. Even this step has been repeated two times with and without oversampling of the beats.

After the creation of the voting model the beats-database has been shuffled and then a 6-fold validation process has been provided: the database consist of 48 ECG signals, so 8 signals in each step are selected as testing set and the other 40 signals as training set, finally the mean of the 6 results has been calculated in order to provide relevant results.





