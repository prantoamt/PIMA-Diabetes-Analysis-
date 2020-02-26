# PIMA-Diabetes-Analysis-

##### This Repository contains all the experimental reports of analyses on PIMA Indian Diabetes dataset. Based on our own analyses, we built machine learning models using different algorithms such as, K-Nearest Neighbor, Decision Tree, Naive Bayes & Random Forest.


## Relations between attributes: (Correlation Matrix)
<img src="IMG/co_relation.png" width="1000" height="600" />
## Decision Tree:

Decision Tree Hyper-parameter Tuning | Tree Visualization 
:-------------------------:|:-------------------------:
<img src="IMG/Decision Tree Hyperparameter Tuning.png" width="500" height="300" align="center" /> | <img src="/IMG/tree.png" width="500" height="300" align="center" />
Max Depth: 2 | Impurity Measurement: Entrophy 
Test Accuracy: 0.7316017316017316 | Train Accuracy: 0.7560521415270018 
Cross Validation Score: 0.7225325884543762 | Accuracy on  Kurmitola Hospital: 0.7955801104972375 <br/>

## K-Nearest Neighbor:

#### K-NN accuracy:

Hyper-parameter Tuning

<img src="/IMG/knn Hyperparameter Tuning1.png" width="500" height="300" align="center" />
K=16 (best fit) <br/>
Test Accuracy: 0.7575757575757576 <br/>
Train Accuracy: 0.7821229050279329 <br/>
Cross Validation Score: 0.776536312849162 <br/>
Accuracy on  Kurmitola Hospital: 0.8121546961325967 <br/>

## Random Forest:

#### Random Forest Accuracy:
'n_estimators': 800, <br/>
'min_samples_split': 10, <br/>
'min_samples_leaf': 4, <br/>
'max_features': 'sqrt', <br/>
'max_depth': 50, <br/>
'bootstrap': True <br/>

Train Accuracy: 0.8845437616387337 <br/>
Test Accuracy: 0.7792207792207793 <br/>
Accuracy on Kurmitola Hospital: 0.7734806629834254 <br/>


## Naive Bayes:

#### Naive Bayes Accuracy:

Train Accuracy: 0.7746741154562383 <br/>
Test Accuracy: 0.7229437229437229 <br/>
KTH Accuracy: 0.7790055248618785

## Accuracy Comparison:

<img src="IMG/Accuracybar_with_voting.png" width="600" height="400" align="center" />



## Confusion Matrix of classifiers

On PIMA test set | On Kurmitola Hospital Dataset
:-------------------------:|:-------------------------:
<img src="IMG/Confusion_Matrix.png" width="500" height="300" align="center" /> | <img src="/IMG/Confusion_Matrix_KTH.png" width="500" height="300" align="center" />


## ROC Curve of classifiers

On PIMA test set | On Kurmitola Hospital Dataset
:-------------------------:|:-------------------------:
<img src="IMG/ROC_AUC.png" width="500" height="300" align="center" /> | <img src="/IMG/ROC_AUC_KTH.png" width="500" height="300" align="center" />
