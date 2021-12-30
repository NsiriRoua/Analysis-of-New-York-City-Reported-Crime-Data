# Import numpy, pandas, matpltlib.pyplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px #graphic library
import datetime
df = pd.read_csv("NYPD_Complaint_Data_Historic_2019.csv")
df_backup = df
columns_remove = ['PREM_TYP_DESC','LOC_OF_OCCUR_DESC','CRM_ATPT_CPTD_CD','LAW_CAT_CD','ADDR_PCT_CD','PD_CD','PD_DESC', 'CMPLNT_NUM', 'OFNS_DESC', 'JURIS_DESC', 'JURISDICTION_CODE', 'PATROL_BORO', 'CMPLNT_TO_DT', 'CMPLNT_TO_TM', 'HADEVELOPT', 'HOUSING_PSA', 'PARKS_NM', 'RPT_DT', 'STATION_NAME', 'TRANSIT_DISTRICT', 'X_COORD_CD','Y_COORD_CD', 'Lat_Lon', 'SUSP_AGE_GROUP', 'SUSP_SEX', 'SUSP_RACE']
df = df.drop(columns_remove, axis=1)
df.dropna(subset=['CMPLNT_FR_DT'], inplace=True)
df.dropna(subset=['CMPLNT_FR_TM'], inplace=True)
df.replace('UNKNOWN', np.NaN, inplace=True)
df.replace('E', np.NaN, inplace=True)
df.replace('D', np.NaN, inplace=True)
df.replace('U', np.NaN, inplace=True)
print('Number of rows before removing rows with missing values: ' + str(df.shape[0]))
df.dropna(axis=0, inplace=True)
print('Number of rows after removing rows with missing values: ' + str(df.shape[0]))
df['CMPLNT_FR_Year'] = df['CMPLNT_FR_DT'].map(lambda x: int(str(x).split('-')[0]))
df['CMPLNT_FR_Month'] = df['CMPLNT_FR_DT'].map(lambda x: int(x.split('-')[1]))
df['CMPLNT_FR_Day'] = df['CMPLNT_FR_DT'].map(lambda x: int(x.split('-')[2]))
df['CMPLNT_FR_Hour'] =df['CMPLNT_FR_TM'].map(lambda x: int(x.split(':')[0]))
df['CMPLNT_FR_DT'] = pd.to_datetime(df['CMPLNT_FR_DT'], errors='coerce')
df['Weekday']=df['CMPLNT_FR_DT'].dt.strftime('%a')
df['Weekday'].unique()
from sklearn.preprocessing import LabelEncoder
df_sel1 = pd.DataFrame(df, columns=['Weekday'])
labelencoder = LabelEncoder()
df_sel1['Weekday_cat'] = labelencoder.fit_transform(df['Weekday'])
df['Weekday']=df_sel1.iloc[:,1:]
columns_remove = ['CMPLNT_FR_TM', 'CMPLNT_FR_DT']
df = df.drop(columns_remove, axis=1)
def ky_cat(ky_cd):
    if ky_cd in [101,102,103]:
        return "HOMICIDE"
    elif ky_cd in [104,115,116,233,234,356,460]:
        return "SEXCRIME"
    elif ky_cd in [105,107,109,110,111,112,113,231,238,340,341,342,343,358]:
        return "THEFTFRAUD"
    elif ky_cd in [106,114,124,344]:
        return "OTHERVIOLENT"
    elif ky_cd in [117,118,119,232,235,236,346,347,577]:
        return "DRUGS"
    elif ky_cd in [120, 121, 125, 126, 345, 345, 348, 349, 351, 352, 353, 354, 355, 357, 359, 360, 361, 362, 363, 364, 364, 364, 365, 366, 455, 571, 572, 578, 672, 675, 676, 677, 677, 678, 685, 881] :
        return "OTHER"
df['KY_CD'] = df['KY_CD'].map(lambda x: ky_cat(x))
dum_df = pd.get_dummies(df, columns=["KY_CD","BORO_NM","VIC_SEX","VIC_RACE","VIC_AGE_GROUP"], prefix=["KY_CD","BORO_NM","VIC_SEX","VIC_RACE","VIC_AGE_GROUP"] )
# Get the feature vector
X = dum_df.drop(['KY_CD_DRUGS', 'KY_CD_HOMICIDE', 'KY_CD_OTHER', 'KY_CD_OTHERVIOLENT', 'KY_CD_SEXCRIME', 'KY_CD_THEFTFRAUD'], axis = 1)

# Get the target vector
y = dum_df[['KY_CD_DRUGS', 'KY_CD_HOMICIDE', 'KY_CD_OTHER', 'KY_CD_OTHERVIOLENT', 'KY_CD_SEXCRIME', 'KY_CD_THEFTFRAUD']]

print('X shape: ' + str(X.shape))
print('y shape: ' + str(y.shape))
from sklearn.model_selection import train_test_split

# Randomly choose 30% of the data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# Show the shape of the data
print('y train shape: ' + str(np.unique(y_train, return_counts=True)))
print('y test shape: ' + str(np.unique(y_test, return_counts=True)))
from sklearn.model_selection import train_test_split

# Randomly choose 30% of the data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# Show the shape of the data
print('y train shape: ' + str(np.unique(y_train, return_counts=True)))
print('y test shape: ' + str(np.unique(y_test, return_counts=True)))
from sklearn.utils.multiclass import type_of_target
print(type_of_target(y_train.idxmax(axis=1)))
print(type_of_target(y_train))
y_train.idxmax(axis=1)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
MCP = ModelCheckpoint('Best_points.h5',verbose=1,save_best_only=True,monitor='val_accuracy',mode='max')
ES = EarlyStopping(monitor='val_accuracy',min_delta=0,verbose=1,restore_best_weights = True,patience=3,mode='max')
RLP = ReduceLROnPlateau(monitor='val_loss',patience=3,factor=0.2,min_lr=0.0001)
from keras.layers.core import Dense, Dropout
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
import keras
LR = 0.0001
# Create model
model = Sequential()
model.add(Dense(512, input_dim=55, activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(6, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit model to training data
model.fit(X_train, y_train, epochs=10, batch_size=512)

# Evaluate model on test data
scores = model.evaluate(X_test, y_test)
print("\n%s: %.14f%%" % (model.metrics_names[1], scores[1]*100))

# Import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import math
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100, # Number of trees
                            min_samples_split = 20,
                            bootstrap = True, 
                            max_depth = 50, 
                            min_samples_leaf = 25)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

# Get the accuracy score
acc_rf=accuracy_score(y_test, y_pred)

# Model Accuracy, how often is the classifier correct?
print("[Random forest algorithm] accuracy_score: {:.3f}.".format(acc_rf))
cols_results=['family','model','classification_rate','runtime']
results = pd.DataFrame(columns=cols_results)


from sklearn.neighbors import KNeighborsClassifier
import time
kVals = range(1,10)
knn_names = ['KNN-'+str(k) for k in kVals]
for k in kVals:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    
    time_start = time.time()
    y_pred = knn.predict(X_test)
    time_run = time.time()-time_start
    
    results = results.append(pd.DataFrame([['KNN',knn_names[k-1],accuracy_score(y_test,y_pred),time_run]],columns=cols_results),ignore_index=True)
results[results.family=='KNN']
from sklearn.dummy import DummyClassifier
clf = DummyClassifier(strategy='stratified',random_state=0)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print('Accuracy of a random classifier is: %.2f%%'%(accuracy_score(y_test,y_pred)*100))

from flask import Flask , render_template , request , flash , redirect , url_for
from folium.plugins import Draw
import json
import folium
import datetime

def getCoords(coords):  
    for i in range(len(coords)):
        if(coords[i] == '['):
            start = i 
        if(coords[i] == ']'):
            end = i+1
            break 

    coords = coords[start+1 : end-1]
    pair = coords.split(",")
    x = float(pair[0])
    y = float(pair[1])        
    return x , y 
def getAgeGroup(age):
    try : 
        age = int(age)
        if(age>64):
            return 5
        if(age<18):
            return 2
        if(age>19 and age<25):
            return 3
        if(age>24 and age<45):
            return 0
        if(age>44 and age<65):
            return 1

    except : 
        return 4  
def getSex(sexe):
    d = 0 
    m = 0 
    f = 0
    e = 0
    if(sexe=='D'):
        d=1
    if(sexe=='F'):
        f=1
    if(sexe=='M'):
        m=1
    return d,m,f,e
def getRaces(race):
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    f = 0
    g = 0
    if(race=='a'):
        a = 1
    if(race=='b'):
        b = 1
    if(race=='c'):
        c = 1
    if(race=='d'):
        d = 1
    if(race=='e'):
        e = 1
    if(race=='f'):
        f = 1
    if(race=='g'):
        g = 1
    return a , b, c , d, e , f , g
def getTime(time):
    time = str(time)
    tmp = time.split(':')[0]
    if int(tmp)>6 and int(tmp)<=12:
        return 1
    elif int(tmp)>12 and int(tmp)<=17 :
        return 0
    elif int(tmp)<20:
        return 2
    else : 
        return 3
def getKeyFromValue(arg):
    t = []
    for k , v in df_backup.items():
        if v == arg :
            t.append(k)
    return t 
app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def index():
    coords = ''
    if request.method == "POST":
        coords = request.form.get('coords')
        name = request.form.get('name')
        race = request.form.get('race')
        sexe = request.form.get('sexe')
        age = request.form.get('age')
        trip_time_str = request.form.get('trip_time')
        trip_time = datetime.datetime.strptime(trip_time_str, '%Y-%m-%dT%H:%M')
        Latitude, Longitude = getCoords(coords)
        VIC_AGE_GROUP = getAgeGroup(age)
        VIC_SEX_D, VIC_SEX_M, VIC_SEX_F, VIC_SEX_E = getSex(sexe)
        a , b , c , d , e , f , g  = getRaces(race)
        time = getTime(trip_time.time())
        input = [Latitude, Longitude, VIC_AGE_GROUP, VIC_SEX_D, VIC_SEX_E, VIC_SEX_F,
       VIC_SEX_M, a , b , c , d , e , f , g , trip_time.month, trip_time.year, trip_time.day, time]
        coords = [Latitude , Longitude ]
        input = [input]
        result1 = clf.predict(input)
        proba = clf.predict_proba(input)
        CrimesList = dict(zip(y, proba[0]))
        return render_template("alert.html", crimes = CrimesList)
        
    return render_template("main_page.html", coords = coords )

if __name__ == '__main__':
    app.run(debug=True)