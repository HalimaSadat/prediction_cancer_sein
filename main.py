import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle as pickle

def training_model(data):
#split la data en predicteur et target colonne
  X = data.drop(['diagnostic'], axis=1)
  y= data['diagnostic']  

#j'ai utilisé le StandardScaler pour uniformiser et standardiser le données
  scaler=StandardScaler()
  X=scaler.fit_transform(X)

#je split en test et train
  X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)

#entrainer le model
  model=LogisticRegression()
  model.fit(X_train, y_train)

  #test le model
  y_pred = model.predict(X_test)
  print('Accuracy of our model:', accuracy_score(y_test, y_pred))
  print("Classification report: \n", classification_report(y_test, y_pred))
  
  return model, scaler



def cleaned_data():
  data = pd.read_csv("base.csv")
  
#nettoyage de données
  data=data.drop(['Unnamed: 32', 'patient'], axis=1) 

#encodage: moi je veux tester en fonction du diagnostic malin
  data['diagnostic'] = data['diagnostic'].map({'M':1, 'B':0})
    
  return data


def main():
  data = cleaned_data()
  print(data.info())

#entrainer le model
  model, scaler=training_model(data)
#quand on l'entraine on remarque une Accuracy du model de: 0.9736842105263158

  with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
  with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)




  

if __name__ == '__main__':
  main()