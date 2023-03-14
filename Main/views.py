from django.shortcuts import render
# from matplotlib.style import context
from numpy import dtype
import pandas as pd
import numpy as np
from Main.models import Data, Model
import json
import os


import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.svm import SVC
import xgboost as xg
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score

import shutil

# Create your views here.

file_name = ''
final_file_name = ''


file_to_export = open("./static/Machine_learning_model_builder.txt", "r")
# file_to_export = open('Machine_learning_model_builder.txt', 'w')


def intro(request):

    try:
        shutil.rmtree("./media")
    except:
        print()

    try:
        os.mkdir("./media")
    except:
        print()

    return render(request, 'intro.html')


def home2(request):
    return render(request, 'home2.html')


def plots(request):
    filename = 'default'
    if request.method == "POST":
        filename = request.POST.get("filename")

    context = {
        "filename": filename,
    }
    return render(request, 'plots.html', context=context)


def ml(request):

    # directory = "./media/"
    # for filename in os.scandir(directory):
    #     if filename.is_file():
    #         # print(filename.path)
    #         print(filename.name)

    name = 'iris.csv'

    file = None
    if request.method == 'POST':
        file = request.FILES.get('file')
        # name = request.POST['name']
        if file != None:
            name = file.name
            print(name)
            document = Data.objects.create(file=file, name=name)
            document.save()
            final_file_name = name

    dataframe = pd.read_csv('media/'+name)
    # print(dataframe.columns)

    #################################### Plots #####################################

    # columns = dataframe.columns
    # print(columns)

    # plots = []

    # for i in columns:
    #     for j in columns:
    #         if i != j:
    #             if dataframe[i].dtype != object and dataframe[j].dtype != object:
    #                 # sn.lineplot(data=dataframe, x=i, y=j)
    #                 # plt.show()
    #                 continue

    # for i in columns:
    #     if dataframe[i].dtype != object:
    #         print(dataframe[i])
    # sn.lineplot(list(dataframe[i]))
    # plt.show()

    # sn.lineplot(data=dataframe, x=dataframe.columns[0], y=dataframe.columns[1])
    # plt.show()

    # https://towardsdatascience.com/14-data-visualization-plots-of-seaborn-14a7bdd16cd7

    if request.method == 'POST':
        plot_type = request.POST.get("plot_type")
        plot_parameter1 = request.POST.get("plot_parameter1")
        plot_parameter2 = request.POST.get("plot_parameter2")
        print(plot_parameter1)
        print(plot_parameter2)

        if plot_parameter1 != None and plot_parameter2 != None:
            sn.set_style('dark')
            if (plot_type == "lineplot"):
                sn.lineplot(data=dataframe, x=str(
                    plot_parameter1), y=str(plot_parameter2), color="#f05454")
            elif (plot_type == "scatterplot"):
                sn.scatterplot(data=dataframe, x=str(
                    plot_parameter1), y=str(plot_parameter2), color="#f05454")
            elif (plot_type == "rugplot"):
                sn.rugplot(data=dataframe, x=str(
                    plot_parameter1), y=str(plot_parameter2), color="#f05454")
            elif (plot_type == "jointplot"):
                sn.jointplot(data=dataframe, x=str(plot_parameter1),
                             y=str(plot_parameter2), kind="hex", color="#f05454")
            elif (plot_type == "pairplot"):
                sn.pairplot(data=dataframe, hue=str(
                    plot_parameter1), color="#f05454")

            elif (plot_type == "stripplot"):
                sn.stripplot(data=dataframe, x=str(
                    plot_parameter1), y=str(plot_parameter2), color="#f05454")
            elif (plot_type == "swarmplot"):
                sn.swarmplot(data=dataframe, x=str(
                    plot_parameter1), y=str(plot_parameter2), color="#f05454")
            elif (plot_type == "histogram"):
                sn.histplot(data=dataframe, x=str(
                    plot_parameter1), color="#f05454")
            elif (plot_type == "barchart"):
                sn.barplot(data=dataframe, x=str(
                    plot_parameter1), y=str(plot_parameter2), color="#f05454")
            elif (plot_type == "boxplot"):
                sn.boxplot(data=dataframe, x=str(
                    plot_parameter1), y=str(plot_parameter2), color="#f05454")
            elif (plot_type == "violinplot"):
                sn.violinplot(data=dataframe, x=str(
                    plot_parameter1), y=str(plot_parameter2), color="#f05454")

            elif (plot_type == "distplot"):
                sn.distplot(data=dataframe, x=str(
                    plot_parameter1), color="#f05454")
            elif (plot_type == "countplot"):
                sn.countplot(data=dataframe, x=str(
                    plot_parameter1), color="#f05454")
            elif (plot_type == "piechart"):
                sn.pairplot(data=dataframe, hue=str(
                    plot_parameter1), color="#f05454")
            plt.show()

    #########################################################################

    if os.path.exists("./static/plots/heatmap.png"):
        os.remove("./static/plots/heatmap.png")
    else:
        print("The file does not exist")

    corr_table = dataframe.corr()

    correlation_table = []
    temp_list = []

    for i in range(len(corr_table)):
        for j in range(len(corr_table.columns)):
            temp_list.append(corr_table.iloc[i, j])
        correlation_table.append(temp_list)
        temp_list = []

    corr_columns = dataframe.columns

    # sn.heatmap(data=corr_table)
    # plt.savefig('./static/plots/heatmap.png')
    # displaying the plotted heatmap
    # plt.show()

    ########################################################################

    # json_records = dataframe.reset_index().to_json(orient ='records')
    # data = []
    # data = json.loads(json_records)
    # data_table = {'data': data}

    data_table = []
    temp_list = []

    for i in range(len(dataframe)):
        for j in range(len(dataframe.columns)):
            temp_list.append(dataframe.iloc[i, j])
        data_table.append(temp_list)
        temp_list = []

    file_to_export = open("./static/Machine_learning_model_builder.txt", "r+")
    file_to_export.truncate(0)
    file_to_export.close()

    file_to_export = open("./static/Machine_learning_model_builder.txt", "w")

    file_to_export.write(f"Dataset : {name}.csv\n")
    file_to_export.write(f"Total Rows : {len(dataframe.index)}\n")
    file_to_export.write(f"Total Columns : {len(dataframe.columns)}\n")
    file_to_export.write(f"Columns : {dataframe.columns.tolist()}\n")
    file_to_export.write(
        f"Column Types : {dataframe[dataframe.columns].dtypes}\n")

    file_to_export.close()

    context = {
        'filename': name,
        'dataframe': dataframe,
        'columns': dataframe.columns,
        'rowscount': len(dataframe.index),
        # 'columns':dataframe.columns.tolist(),
        'columncount': len(dataframe.columns),
        'columntype': dataframe[dataframe.columns].dtypes,
        'data_table': data_table,
        'correlation_table': correlation_table,
    }

    return render(request, 'ml.html', context)


def seeavailabledatasets(request):

    file_name_list = []

    for i in Data.objects.all():
        file_name_list.append(i.name)

    file_path_list = []

    for i in file_name_list:
        temp_path = 'media/' + i + '.csv'
        file_path_list.append(temp_path)

    return render(request, 'seeavailabledatasets.html', {'file_name_list': file_name_list, 'file_path_list': file_path_list})


def documentations(request):
    return render(request, 'documentations.html')


def rateus(request):
    return render(request, 'rateus.html')


# name = 'default'
predict_contest = {}
fullModelName = ""


def predict(request):

    y_predict = 'none'
    y_test = 'none'
    predict_contest = {}
    regression_list = ["lr", "rlr", "knnr", "dtr", "rfr", "svmr", "xr"]
    classification_list = ["lrc", "nbc", "knnc", "dtc", "rfc", "svmc", "xc"]
    name = ''
    if request.method == 'POST':
        name = request.POST.get('name')
        print("\n\n\n")
        print(name)
        print("\n\n\n")
        # name2 = final_file_name
        modelType = request.POST.get('modelType')
        print(f"\n\n\n\n\nModel Type is {modelType} \n\n\n\n\n\n")
        model2 = request.POST.get('algorithmType')
        print(f"\n\n\n\n\nModel Type is {model2} \n\n\n\n\n\n")
        # features2 = request.POST.get('features')
        # model2 = request.POST.get('model')
        # print(features2)
        # print(type(features2))
        # labels2 = request.POST.get('labels')
        features2 = request.POST.getlist('features')
        labels2 = request.POST.get('labels')

        print(features2)

        temp1 = ""
        for i in features2:
            temp1 = temp1 + i + "*"
        # temp1[-1] =  ""
        print(temp1)

        temp2 = ""
        for i in labels2:
            temp2 = temp2 + i + "*"
        # temp2[-1] =  ""
        print(temp2)

        print("feature List type : ", type(features2))
        print("feature List  : ", features2)

        testdata = 0.3

        model = Model(name=name, model=model2, features=temp1, labels=labels2)
        # model.save()

        # seemodel = Model.objects.filter(name=name2,model=model2,features=features2,labels=labels2)

        # features4 = seemodel[0].features
        # features5 = features4.split("*")

        filepath = 'media/' + name
        dataframe = pd.read_csv(filepath)
        # print('Dataframe : \n',dataframe[features5])
        # print('Dataframe Type : \n',type(dataframe[features5]))

        algo = LinearRegression()

        if model2 == 'lr':
            fullModelName = "Linear Regression"
            algo = LinearRegression()
        elif model2 == 'rlr':
            fullModelName = "Ridge & Lasso (Regression)"
            algo = Ridge(alpha=1)
        elif model2 == 'knnr':
            fullModelName = "K-Nearest Neighbors Regressor"
            algo = KNeighborsRegressor()
        elif model2 == 'dtr':
            fullModelName = "Decision Tree Regressor"
            algo = DecisionTreeRegressor()
        elif model2 == 'rfr':
            fullModelName = "Random Forest Regressor"
            algo = RandomForestRegressor()
        elif model2 == 'svmr':
            fullModelName = "Support Vector Machine Regressor"
            algo = SVR()
        elif model2 == 'xr':
            fullModelName = "Xgboost Regressor"
            algo = xg.XGBRegressor()
        elif model2 == 'lrc':
            fullModelName = "Logistic Regression (Classification)"
            algo = Ridge()
        elif model2 == 'nbc':
            fullModelName = "Naive Baye's (Classification)"
            algo = GaussianNB()
        elif model2 == 'knnc':
            fullModelName = "K-Nearest Neighbors Classifier"
            algo = KNeighborsClassifier()
        elif model2 == 'dtc':
            fullModelName = "Decision Tree Classifier"
            algo = DecisionTreeClassifier()
        elif model2 == 'rfc':
            fullModelName = "Random Forest Classifier"
            algo = RandomForestClassifier()
        elif model2 == 'svmc':
            fullModelName = "Support Vector Machine Classifier"
            algo = SVC()
        elif model2 == 'xc':
            fullModelName = "Xgboost Classifier"
            algo = xg.XGBClassifier()

        filepath = 'media/' + name
        dataframe = pd.read_csv(filepath)

        features_data = dataframe[features2]
        label_data = dataframe[labels2]

        # print(label_data)

        labels3 = list(set(np.array(label_data)))
        labels3.sort()
        # print(labels3)

        # converting to number column
        # for i in features_data:
        #     if type(i) == object:
        #         set_list = list(set(features_data[i]))
        #         set_list.sort()
        #         print( "set list : ", set_list)
        #         for j in range(len(features_data[i])):
        #             features_data[i][j] = set_list.index(features_data[i][j])
        #         print(i, " : ", dataframe[i])

        # for j in features_data.columns:
        #     if features_data[j].dtypes == object:
        #         temp_list = list(set(features_data[j]))
        #         temp_list.sort()

        #         for i in range(len(features_data[j])):
        #             features_data[j][i] = temp_list.index(features_data[j][i])

        if modelType == "classification" and label_data.dtypes == object:
            temp_label_list = list(np.array(dataframe[labels2]))
            # print("::::")
            # print(temp_label_list)
            # print("::::")

            labels3 = list(set(temp_label_list))
            labels3.sort()

            for i in range(len(temp_label_list)):
                temp_label_list[i] = labels3.index(temp_label_list[i])

            # print("::::")
            # print(temp_label_list)
            # print("::::")

            label_data = pd.DataFrame(temp_label_list)

        x_train, x_test, y_train, y_test = train_test_split(
            features_data, label_data, test_size=testdata)

        algo.fit(x_train, y_train)
        y_predict = algo.predict(x_test)

        if modelType == "classification":
            y_test = list(y_test[(y_test.columns)[0]])

        y_predict = list((np.array(y_predict)).flatten())

        if modelType == "classification":
            for i in range(len(y_test)):
                y_test[i] = labels3[y_test[i]]
                y_predict[i] = labels3[y_predict[i]]

        modelscore = 100

        if modelType == "regression":
            modelscore = metrics.r2_score(y_test, y_predict)*100
            modelscore = float("{:.2f}".format(modelscore))

        if modelType == "classification":
            modelscore = accuracy_score(y_test, y_predict)*100
            modelscore = float("{:.2f}".format(modelscore))

        space_saperated_features = '0 0 0'

        file_to_export = open(
            "./static/Machine_learning_model_builder.txt", "a+")

        file_to_export.write(f"Test:Train Data Split : {testdata}\n")
        file_to_export.write(f"Features : {features2}\n")
        file_to_export.write(f"Label : {labels2}\n")
        file_to_export.write(f"Model : {fullModelName}\n")
        file_to_export.write(f"Model Score : {modelscore}\n")

        file_to_export.close()

        print("HERE IS ANS")
        print(y_test)
        print(y_predict)

        predict_contest = {
            'y_test': y_test,
            'y_predict': y_predict,
            'modelscore': modelscore,
            'fullModelName': fullModelName,
            'features':  features2,
            'labels': labels2,
            # 'seemodel' : seemodel[0],
            # 'manual_predicted_value': manual_predicted_value
        }

    return render(request, 'predict.html', predict_contest)


def manualpredict(request):

    if request.method == 'POST':
        generated_modelname = request.POST.get('generated_modelname')
        space_saperated_features = request.POST.get('space_saperated_features')
        print(type(generated_modelname))
        print(space_saperated_features)
        print(type(space_saperated_features))

        space_saperated_features2 = space_saperated_features.split()
        print(space_saperated_features2)
        float_space_saperated_features = []

        for i in space_saperated_features2:
            float_space_saperated_features.append(float(i))

        print(float_space_saperated_features)
        print(type(float_space_saperated_features))

        temp_list = generated_modelname.split('_')
        print(temp_list)
        [dataset_name, model_name] = temp_list

        filter_model = Model.objects.filter(
            name=dataset_name, model=model_name)
        filter_model2 = filter_model[0]

        dataset_name = filter_model2.name
        model_name = filter_model2.model

        dataset_path = 'media/'+dataset_name+'.csv'
        dataset = pd.read_csv(dataset_path)

        if model_name == 'lr':
            Lr = LinearRegression()

        Lr.fit(dataset[(filter_model2.features).split()],
               dataset[filter_model2.labels])

        predicted_values = Lr.predict(
            pd.DataFrame(float_space_saperated_features))

        # for i in model_obj:
        #     if i == generated_modelname:

        #         print("I IM IN BROOOOOO")
        #         break
        #     # print(i)

        # print("MOdel Obj : ",model_obj)

        context = {
            'space_saperated_features': space_saperated_features,
            'predicted_values': predicted_values
        }

    return render(request, 'manualpredict.html', context)
