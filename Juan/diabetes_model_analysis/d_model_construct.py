import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle


df_d = pd.read_csv('./diabetes_model_analysis/archive/diabetes_binary_health_indicators_BRFSS2015.csv')

df_d.drop(columns=['Income','Education','Age','Sex','PhysHlth','MentHlth','NoDocbcCost',
                   'AnyHealthcare','HvyAlcoholConsump','Veggies','Fruits','PhysActivity',
                   'HeartDiseaseorAttack','Stroke','Smoker','CholCheck'])

X = df_d.iloc[:,1:]
y = df_d.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.10, random_state = 32)

clf = RandomForestClassifier(max_depth=5,random_state= 32)
clf.fit(X_train,y_train)

filename = 'model.sav'

model = pickle.dump(clf,open(filename,'wb'))