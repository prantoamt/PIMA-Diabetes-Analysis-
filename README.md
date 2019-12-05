# PIMA-Diabetes-Analysis-

##### This Repository contains all the experimental reports of analyses on PIMA Indian Diabetes dataset. Based on our own analyses, we built machine learning models using different algorithms such as, K-Nearest Neighbor, Decision Tree, Naive Bayes.


## Relations between attributes:
Scatter with all Attributes |  Scatter with most important atrributes
:-------------------------:|:-------------------------:
<img src="IMG/scatter_kaggle_raw.png" width="500" height="400" /> | <img src="IMG/scatter_kaggle_normalized(5 attr).png" width="500" height="400" /> 
## Decision Tree:

Decision Tree Accuracy test by varying tree depth | Tree Visualization 
:-------------------------:|:-------------------------:
<img src="IMG/Decision Tree Accuracy Variation With Depth.png" width="500" height="300" align="center" /> | <img src="/IMG/tree.png" width="500" height="300" align="center" />
| |Max Depth: 4 </br> 
| |Impurity Measurement: Entrophy </br>
| |Test Accuracy: 0.7705627705627706 </br>
| |Train Accuracy: 0.7746741154562383 </br> 


## K-Nearest Neighbor:


#### K-NN accuracy:

Raw Data with all Attributes |  With Our Selected Attributes
:-------------------------:|:-------------------------:
<img src="/IMG/knn_accuracy_kaggle_raw_data.png" width="500" height="300" align="center" />  |  <img src="/IMG/knn_accuracy_kaggle_5_attr_raw_data.png" width="500" height="300" align="center" />
<img src="/IMG/knn_accuracy_kaggle_normalized_data.png" width="500" height="300" align="center" /> | <img src="/IMG/knn_accuracy_kaggle_normalized_data(5 attr).png" width="500" height="300" align="center" />
| | K=19 (best fit)
| |Test Accuracy: 0.7792207792207793
| |Train Accuracy: 0.7728119180633147
