import streamlit as st
import pickle as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def cleaned_data():
  data = pd.read_csv("base.csv")
  
#nettoyage de données
  data=data.drop(['Unnamed: 32', 'patient'], axis=1) 

#encodage: moi je veux tester en fonction du diagnostic malin
  data['diagnostic'] = data['diagnostic'].map({'M':1, 'B':0})
    
  return data


#je crée le sidebar de mon appliavec tout mes input pour enfin générer le chart et les prediction
# le sidebar regroupe un curseur pour chaque variable de ma data. on a 30 variables = 30 curseurs       
def ajout_sidebar():
   st.sidebar.header("Mesures des noyaux cellulaires")

   data = cleaned_data()
   
   sliders_labels = [
          ("rayon (moyen)", "rayon_moyen"),
          ("texture (moyen)", "texture_moyen"),
          ("perimetere (moyen)", "perimetere_moyen"),
          ("aire (moyen)", "aire_moyen"),
          ("longueur_rayon (moyen)", "longueur_rayon_moyen"),
          ("compacite (moyen)", "compacite_moyen"),
          ("concavite (moyen)", "concavite_moyen"),
          ("points_concaves (moyen)", "points concaves_moyen"),
          ("symetrie (moyen)", "symetrie_moyen"),
          ("dimension_fractale (moyen)", "dimension_fractale_moyen"),
          ("rayon (es)", "rayon_es"),
          ("texture (es)", "texture_es"),
          ("perimetere (es)", "perimetere_es"),
          ("aire (es)", "aire_es"),
          ("longueur_rayon (es)", "longueur_rayon_es"),
          ("compacite (es)", "compacite_es"),
          ("concavite (es)", "concavite_es"),
          ("points_concaves (es)", "points concaves_es"),
          ("symetrie (es)", "symetrie_es"),
          ("dimension_fractale (es)", "dimension_fractale_es"),
          ("rayon (pire)", "rayon_pire"),
          ("texture (pire)", "texture_pire"),
          ("perimetere (pire)", "perimetere_pire"),
          ("aire (pire)", "aire_pire"),
          ("longueur_rayon (pire)", "longueur_rayon_pire"),
          ("compacite (pire)", "compacite_pire"),
          ("concavite (pire)", "concavite_pire"),
          ("points concaves (pire)", "points concaves_pire"),
          ("symetrie ((pire)", "symetrie_pire"),
          ("dimension_fractale (pire)", "dimension_fractale_pire"),

   ]

 #créer un dictionnaire ou je vais attribuer pour chaque clés la valeur moyenne de chaque variable 
 # calculés deja (the value ci dessous) 
    
   diction_input = {}  
   for label, key in sliders_labels:
    diction_input[key] = st.sidebar.slider(
      label,
      min_value=float(0),
      max_value=float(data[key].max()),
      value=float(data[key].mean()) #avoir la valeur moyenne pour chaque var
    )
     
   return diction_input
      
#je scale toute mes valeurs de mes variables pour avoir des valeurs qui seront comprises entre 0 et 1
# je crée une fonction to scale all my values 
def scaled_value(diction_input):
  data = cleaned_data()

  Z= data.drop(['diagnostic'], axis=1)

  scaled_dict = {}

  for key, value in diction_input.items():
    max_value = Z[key].max()
    min_value = Z[key].min()
    scaled_value = (value - min_value)/ (max_value - min_value)
    scaled_dict[key]= scaled_value

  return scaled_dict  


#afin de créer un Multiple Trace Radar Chart: je map toute les valeurs des moyennes, erreurs starndars et les pires 
# je vais utiliser une des librairie de javascript qui s'appelle plotly parce que 
# c'est une librairie qui génére d'intercatives charts
# #https://plotly.com/python/radar-chart/ 


def carte_radar(input_data):
  
  input_data = scaled_value(input_data) 
  categories = ['rayon','texture','périmètre','aire','longueur de rayon','compacité',
                'concavité', 'points concaves','symetrie','dimension fractale']

  fig = go.Figure()

#je map les clés valeur qui sont associés à la moyenne
  fig.add_trace(go.Scatterpolar(
        r=[input_data['rayon_moyen'], input_data['texture_moyen'], input_data['perimetere_moyen'],
           input_data['aire_moyen'], input_data['longueur_rayon_moyen'], input_data['compacite_moyen'],
           input_data['concavite_moyen'], input_data['points concaves_moyen'], input_data['symetrie_moyen'],
           input_data['dimension_fractale_moyen']],
        theta=categories,
        fill='toself',
        name='Mean Value'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[input_data['rayon_es'], input_data['texture_es'], input_data['perimetere_es'],
           input_data['aire_es'], input_data['longueur_rayon_es'], input_data['compacite_es'],
           input_data['concavite_es'], input_data['points concaves_es'], input_data['symetrie_es'],
           input_data['dimension_fractale_es']],
        theta=categories,
        fill='toself',
        name='Standard Error'
  ))

  fig.add_trace(go.Scatterpolar(
        r=[input_data['rayon_pire'], input_data['texture_pire'], input_data['perimetere_pire'],
           input_data['aire_pire'], input_data['longueur_rayon_pire'], input_data['compacite_pire'],
           input_data['concavite_pire'], input_data['points concaves_pire'], input_data['symetrie_pire'],
           input_data['dimension_fractale_pire']],
        theta=categories,
        fill='toself',
        name='Worst Value'
  ))  

  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
  )
  return fig

# définir une fonction pour visualiser la prediction tt en important le model de pred vu auparavant
def ajout_predic(input_data):
  model = pickle.load(open("model.pkl", "rb")) #binaire mode
  scaler = pickle.load(open("scaler.pkl", "rb"))
  
#convertir le diction de input_data into a simple array
  input_array = np.array(list(input_data.values())).reshape(1, -1) #chaque valeur de variable est supposee etre dauns une colonne

  scaled_input_array = scaler.transform(input_array)

  pred=model.predict(scaled_input_array)

  st.subheader("Prédictions du type de la tumeur")
  st.write("le type de la tumeur est:")

  if pred[0] == 0:
    st.write("<span class='diagnostic benin'>Bénigne</span>", unsafe_allow_html=True)
  else:
    st.write("<span class='diagnostic malin'>Maligne</span>", unsafe_allow_html=True)

  st.write("Probabilité qu'elle soit une tumeur bénigne:", model.predict_proba(scaled_input_array)[0][0])
  st.write("Probabilité qu'elle soit une tumeur maligne:", model.predict_proba(scaled_input_array)[0][1])


def main():
    #set the page of configuration for my app
    st.set_page_config(
       page_title="Prédiction du cancer de sein",
       page_icon=":female-doctor",
       layout="wide",
       initial_sidebar_state="expanded"
    )
    
    with open("app/assets/style.css") as f:
      st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
    input_data = ajout_sidebar() #j'update a chaue fois input data avec les diff changement que
# jeffectue pour structurer mon application

    with st.container():
      st.title("Une application prédictive du cancer de sein")  
      st.write("Cette application utilise un modèle d'apprentissage automatique pour prédire si une masse mammaire est bénigne ou maligne en fonction des mesures qu'elle reçoit de notre laboratoire. On peut également mettre à jour les mesures manuellement à l'aide des curseurs dans la barre latérale.")


#définir une liste qui prends mes colonnes dont jen aurais besoin et leurs ratios
#(les premiers sont 4plus garndes que les secondes)
    col1, col2 = st.columns([4,1])

    with col1:
      radar_chart = carte_radar(input_data)
      st.plotly_chart(radar_chart)
    with col2:
      ajout_predic(input_data)


if __name__ == '__main__':
    main()

