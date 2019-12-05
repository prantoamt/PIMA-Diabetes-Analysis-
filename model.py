#### %matplotlib inline
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import mglearn
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus


plt.figure(figsize=(12, 10), dpi=80) 
plt.style.use('ggplot')


## Load Dataset
def loadDataset(path):
    dataframe = pd.read_csv(path)
    return dataframe


## Normalize all attributes
def normalize(df):
    pregnancies_min = df['Pregnancies'].min()
    pregnancies_max = df['Pregnancies'].max()    
    glucose_max = df['Glucose'].max()
    glucose_min = df['Glucose'].min()    
    blood_pressure_max = df['BloodPressure'].max()
    blood_pressure_min = df['BloodPressure'].min()
#     skinthickness_max = df['SkinThickness'].max()
#     skinthickness_min = df['SkinThickness'].min()
#     insulin_max = df['Insulin'].max()
#     insulin_min = df['Insulin'].min()
    bmi_max = df['BMI'].max()
    bmi_min = df['BMI'].min()
#     diabetes_pedigree_function_max = df['DiabetesPedigreeFunction'].max()
#     diabetes_pedigree_function_min = df['DiabetesPedigreeFunction'].min()
    age_max = df['Age'].max()
    age_min = df['Age'].min()
    max_min_dict = {
        'pregnancies_min': pregnancies_min,
        'pregnancies_max': pregnancies_max,    
        'glucose_max': glucose_max,
        'glucose_min': glucose_min,    
        'blood_pressure_max': blood_pressure_max,
        'blood_pressure_min': blood_pressure_min,
        'bmi_max': bmi_max,
        'bmi_min': bmi_min,
        'age_max': age_max,
        'age_min': age_min
    }
    
    for i in range(0,df.index.size):
        df.iloc[i,0] = (df.iloc[i,0] - pregnancies_min) / (pregnancies_max-pregnancies_min)
        df.iloc[i,1] = (df.iloc[i,1] - glucose_min) / (glucose_max-glucose_min)
        df.iloc[i,2] = (df.iloc[i,2] - blood_pressure_min) / (blood_pressure_max-blood_pressure_min)
#         df.iloc[i,3] = (df.iloc[i,3] - skinthickness_min) / (skinthickness_max-skinthickness_min)
#         df.iloc[i,4] = (df.iloc[i,4] - insulin_min) / (insulin_max-insulin_min)
        df.iloc[i,3] = (df.iloc[i,3] - bmi_min) / (bmi_max-bmi_min)
#         df.iloc[i,6] = (df.iloc[i,6] - diabetes_pedigree_function_min) / (diabetes_pedigree_function_max-diabetes_pedigree_function_min)
        df.iloc[i,4] = (df.iloc[i,4] - age_min) / (age_max-age_min)
    return df, max_min_dict



def scatter_plot(X_train, y_train):
    colors_palette = {0: 'green', 1: 'red'}
    colors = [colors_palette[c] for c in y_train]
    grr = pd.plotting.scatter_matrix(X_train, c=colors, marker=".", figsize =(20,15), alpha = 0.8, range_padding=0.05, diagonal="kde" , s=60, grid=True)
#     plt.savefig("scatter_kaggle_normalized(5 attr).png")
    #scatter_plot_ends




##Knn Accuracy Graph
def knn_accuracy_graph(max_neighbor, X_train, y_train, X_test, y_test, title):
    neighbors_settings = range(1, max_neighbor)
    training_acuracy = []
    test_acuracy = []
    for neighbors in neighbors_settings:
        clf = KNeighborsClassifier(n_neighbors=neighbors)
        clf.fit(X_train, y_train)
        training_acuracy.append(clf.score(X_train,y_train))
        test_acuracy.append(clf.score(X_test,y_test))
        
    plt.plot(neighbors_settings, training_acuracy, label='Training Acuracy')    
    plt.plot(neighbors_settings, test_acuracy, 'g', label='Test Acuracy')
    plt.xticks(np.arange(min(neighbors_settings), max(neighbors_settings)+2, 1.0))
#     plt.yticks(np.arange(min([0,1]), max([0,1])+1, 0.05))
    plt.xlabel("Neighbors")
    plt.ylabel("Accuray")
    plt.title(title)
    plt.legend()
    plt.show()
    # plt.savefig('knn_accuracy_kaggle_normalized_data(5 attr).png')
    #knn_accuracy_graph ends


def knn_model(X_train, y_train, max_min_dict, neighbor):
    knn = KNeighborsClassifier(n_neighbors = neighbor)
    knn.fit(X_train, y_train)
    print("Predicting Based on K-NN:")
    print("----------------------------------")
    pregnancy = input("Number of Pregnency: ")
    glucose = input("Glucose level: ")
    blood_pressure = input("Blood Pressure: ")
    bmi = input("BMI: ")
    age = input("Age: ")
    pregnancy = (float(pregnancy) - max_min_dict['pregnancies_min']) / (max_min_dict['pregnancies_max'] - max_min_dict['pregnancies_min'])
    glucose = (float(glucose) - max_min_dict['glucose_min']) / (max_min_dict['glucose_max'] - max_min_dict['glucose_min'])
    blood_pressure = (float(blood_pressure) - max_min_dict['blood_pressure_min']) / (max_min_dict['blood_pressure_max'] - max_min_dict['blood_pressure_min'])
    bmi = (float(bmi) - max_min_dict['bmi_min']) / (max_min_dict['bmi_max'] - max_min_dict['bmi_min'])
    age = (float(age) - max_min_dict['age_min']) / (max_min_dict['age_max'] - max_min_dict['age_min'])
    df_dic = {
        'Pregnancies': [pregnancy],
        'Glucose': [glucose],    
        'BloodPressure': [blood_pressure],
        'BMI': [bmi],    
        'Age': [age]
    }
    df = pd.DataFrame(data=df_dic)
    result = knn.predict(df)
    if(result == 0):
        print("Congratulations! You don't have diabetes.")
    elif(result == 1):
        print("Opps! Seems like you have diabetes. Take care of yourself.")
    ##Ends knn_model

    
    
def print_decision_tree(tree_clf, features):
    dot_data = StringIO()
    export_graphviz(tree_clf, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True,feature_names = features,class_names=["Doesn't have diabetes","Has diabetes"])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('tree.png')
    Image(graph.create_png())
    

def decision_tree_accuracy_graph(tree, X_train, y_train, X_test, y_test, max_depth):
    test_accuracy = []
    train_accuracy = []
    depth_range = range(1,max_depth)
    for i in depth_range:
        tree.max_depth = i
        tree.fit(X_train, y_train)
        train_accuracy.append(tree.score(X_train, y_train))
        test_accuracy.append(tree.score(X_test, y_test))
    plt.plot(depth_range, train_accuracy, label="Train Accuracy")    
    plt.plot(depth_range, test_accuracy, label="Test Accuracy")
    plt.xlabel("Tree Depth")
    plt.ylabel("Accuracy")
    plt.title("Decision Tree Accuracy Variation With Depth")
    plt.legend()
    plt.savefig("Decision Tree Accuracy Variation With Depth")
    
    
def decision_tree_model(X_train, y_train, max_min_dict, depth):
    tree_clf = DecisionTreeClassifier(criterion="entropy")
    tree_clf.max_depth=depth
    tree_clf.fit(X_train, y_train)
    print("Predicting Based on Decision Tree:")
    print("----------------------------------")
    pregnancy = input("Number of Pregnency: ")
    glucose = input("Glucose level: ")
    blood_pressure = input("Blood Pressure: ")
    bmi = input("BMI: ")
    age = input("Age: ")
    pregnancy = (float(pregnancy) - max_min_dict['pregnancies_min']) / (max_min_dict['pregnancies_max'] - max_min_dict['pregnancies_min'])
    glucose = (float(glucose) - max_min_dict['glucose_min']) / (max_min_dict['glucose_max'] - max_min_dict['glucose_min'])
    blood_pressure = (float(blood_pressure) - max_min_dict['blood_pressure_min']) / (max_min_dict['blood_pressure_max'] - max_min_dict['blood_pressure_min'])
    bmi = (float(bmi) - max_min_dict['bmi_min']) / (max_min_dict['bmi_max'] - max_min_dict['bmi_min'])
    age = (float(age) - max_min_dict['age_min']) / (max_min_dict['age_max'] - max_min_dict['age_min'])
    df_dic = {
        'Pregnancies': [pregnancy],
        'Glucose': [glucose],    
        'BloodPressure': [blood_pressure],
        'BMI': [bmi],    
        'Age': [age]
    }
    df = pd.DataFrame(data=df_dic)
    result = tree_clf.predict(df)
    if(result == 0):
        print("Congratulations! You don't have diabetes.")
    elif(result == 1):
        print("Opps! Seems like you have diabetes. Take care of yourself.")
    
    

def main():
    df = loadDataset("diabetes.csv")
    classes = df["Outcome"].to_numpy()
    df.drop(['Outcome','SkinThickness','Insulin','DiabetesPedigreeFunction'], axis = 1, inplace = True)

    data_frame = pd.DataFrame(df.values, columns = df.columns)

    data_frame, max_min_dict = normalize(data_frame)

    x_train, x_test, y_train, y_test = train_test_split(data_frame, classes, test_size = 0.30, shuffle=False)
#     knn_accuracy_graph(20,x_train, y_train, x_test, y_test, 'knn Accuracy with Kaggle normalized data(5 attr)')
#     scatter_plot(x_train, y_train)
#     knn_model(x_train, y_train, max_min_dict, 18)
#     decision_tree_accuracy_graph(tree_clf, x_train, y_train, x_test, y_test, 10)
    decision_tree_model(x_train, y_train, max_min_dict, 4)
    
#main ends


main()
