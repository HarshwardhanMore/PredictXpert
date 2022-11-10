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
 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
# Create your views here.

file_name=''

file_to_export = open("./static/Machine_learning_model_builder.txt", "r")
# file_to_export = open('Machine_learning_model_builder.txt', 'w')

def intro(request):
    return render(request, 'intro.html')

def home2(request):
    return render(request, 'home2.html')

def ml(request):


    name = 'default'
    
    if request.method == 'POST':
        file = request.FILES['file']
        name = request.POST['name']
        document = Data.objects.create(file=file,name=name)
        document.save()

    dataframe = pd.read_csv('media/'+name+'.csv')


    if os.path.exists("./static/plots/heatmap.png"):
        os.remove("./static/plots/heatmap.png")
    else:
        print("The file does not exist")


    corr_table =  dataframe.corr()

    correlation_table = []
    temp_list = []

    for i in range(len(corr_table)):
        for j in range(len(corr_table.columns)):
            temp_list.append(corr_table.iloc[i,j])
        correlation_table.append(temp_list)
        temp_list = []

    corr_columns = dataframe.columns


    fig = sn.heatmap(data = corr_table)
    plt.savefig('./static/plots/heatmap.png')
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
            temp_list.append(dataframe.iloc[i,j])
        data_table.append(temp_list)
        temp_list = []

    file_to_export = open("./static/Machine_learning_model_builder.txt","r+")
    file_to_export.truncate(0)
    file_to_export.close()

    file_to_export = open("./static/Machine_learning_model_builder.txt", "w")

    file_to_export.write(f"Dataset : {name}.csv\n")
    file_to_export.write(f"Total Rows : {len(dataframe.index)}\n")
    file_to_export.write(f"Total Columns : {len(dataframe.columns)}\n")
    file_to_export.write(f"Columns : {dataframe.columns.tolist()}\n")
    file_to_export.write(f"Column Types : {dataframe[dataframe.columns].dtypes}\n")
    
    file_to_export.close()

    context={
        'filename':name,
        'dataframe':dataframe,
        'rowscount':len(dataframe.index),
        'columns':dataframe.columns.tolist(),
        'columncount':len(dataframe.columns),
        'columntype' : dataframe[dataframe.columns].dtypes,
        'data_table' : data_table,
        'correlation_table' : correlation_table,
    }
    
    return render(request, 'ml.html', context)


def seeavailabledatasets(request):

    file_name_list = []

    for i in Data.objects.all():
        file_name_list.append(i.name)
    
    file_path_list = []

    for i in file_name_list:
        temp_path = 'media/'+ i +'.csv'
        file_path_list.append(temp_path)

    return render(request,'seeavailabledatasets.html',{'file_name_list':file_name_list,'file_path_list':file_path_list})

def documentations(request):
    return render(request, 'documentations.html')

def rateus(request):
    return render(request, 'rateus.html')


# name = 'default'
predict_contest = {}

def predict(request):

    y_predict = 'none'
    y_test = 'none'
    predict_contest = {}

    if request.method == 'POST':
        name2 = request.POST.get('name')
        model2 = request.POST.get('model')
        features2 = request.POST.get('features')
        # print(features2)
        # print(type(features2))
        labels2 = request.POST.get('labels')

        testdata = 0.3

        model = Model(name=name2,model=model2,features=features2,labels=labels2)
        model.save()

        seemodel = Model.objects.filter(name=name2,model=model2,features=features2,labels=labels2)

        features4 = seemodel[0].features
        # print("features4 : ",features4)
        # print("features4 Type : ",type(features4))
        # print("features4 Splitted : ",features4.split())
        features5 = features4.split()
        # print("Feature 5 : ", features5)

        filepath = 'media/' + name2 + '.csv'
        dataframe = pd.read_csv(filepath)
        # print('Dataframe : \n',dataframe[features5])
        # print('Dataframe Type : \n',type(dataframe[features5]))


        if model2 == 'LinearRegression':

            # print("\n\nInside : If Statement \n\n")

            Lr = LinearRegression()
            filepath = 'media/' + name2 + '.csv'
            dataframe = pd.read_csv(filepath)

            features_data = dataframe[features5]
            label_data = dataframe[labels2]

            # print("OP : ",features_data)
            # print(label_data)

            x_train, x_test, y_train, y_test = train_test_split(features_data,label_data, test_size=testdata)

            Lr.fit(x_train, y_train)

            y_predict = Lr.predict(x_test)

            modelscore = metrics.r2_score(y_test,y_predict)*100
            modelscore = float("{:.2f}".format(modelscore))

            space_saperated_features = '0 0 0'

            
            file_to_export = open("./static/Machine_learning_model_builder.txt", "a+")

            file_to_export.write(f"Test:Train Data Split : {testdata}\n")
            file_to_export.write(f"Features : {features5}\n")
            file_to_export.write(f"Label : {labels2}\n")
            file_to_export.write(f"Model : {seemodel[0]}\n")
            file_to_export.write(f"Model Score : {modelscore}\n")

            file_to_export.close()
    
    
        predict_contest = {
            'y_test' :y_test, 
            'y_predict' : y_predict, 
            'modelscore' : modelscore,
            'model': model2,
            'features' :  features5, 
            'labels' : labels2,
            'seemodel' : seemodel[0],
            # 'manual_predicted_value': manual_predicted_value
        }

    return render(request,'predict.html', predict_contest) 


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

        filter_model =  Model.objects.filter(name=dataset_name, model=model_name)
        filter_model2 = filter_model[0]
        
        dataset_name = filter_model2.name
        model_name = filter_model2.model

        dataset_path = 'media/'+dataset_name+'.csv'
        dataset = pd.read_csv(dataset_path)

        if model_name == 'LinearRegression':
            Lr = LinearRegression()

        Lr.fit(dataset[(filter_model2.features).split()],dataset[filter_model2.labels])

        predicted_values = Lr.predict(pd.DataFrame(float_space_saperated_features))

        # for i in model_obj:
        #     if i == generated_modelname:
                
        #         print("I IM IN BROOOOOO")
        #         break
        #     # print(i)

        # print("MOdel Obj : ",model_obj)


        context={
            'space_saperated_features' : space_saperated_features,
            'predicted_values' : predicted_values
        }

    return render(request,'manualpredict.html',context)

