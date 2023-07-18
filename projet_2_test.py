import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity

# import des base de données
acteurs = pd.read_csv('https://datasets.imdbws.com/name.basics.tsv.gz', sep ='\t', nrows=3000000)
titre = pd.read_csv('https://datasets.imdbws.com/title.basics.tsv.gz', sep = '\t', dtype={'isAdult': str}).query('titleType == "movie"')
distribution = pd.read_csv('https://datasets.imdbws.com/title.principals.tsv.gz', sep ='\t', nrows=3000000)
note = pd.read_csv('https://datasets.imdbws.com/title.ratings.tsv.gz', sep ='\t')

# Nettoyage de données
titre['startYear'] = pd.to_numeric(titre['startYear'], errors='coerce')
titre['runtimeMinutes'] = pd.to_numeric(titre['runtimeMinutes'], errors='coerce')
titre['genres'] = titre['genres'].replace({'\\N':'inconnus'})
titre['runtimeMinutes'] = titre['runtimeMinutes'].replace({'\\N':0})
titre['startYear'] = titre['startYear'].replace({'\\N':0})
titre = titre[titre['runtimeMinutes'] > 40]
titre = titre[titre['genres'].isin(['Adult','Game-Show','Music','News','Reality-TV','Short','Talk-Show']) == False]
titre = titre.loc[(titre['startYear'] >= 1990)]
distribution = distribution[['tconst', 'nconst', 'category']]
acteurs = acteurs[['nconst', 'primaryName']]

# préparation de données
df_titre = pd.merge(titre, note, left_on='tconst', right_on='tconst')
df_titre = pd.merge(df_titre, distribution, on = "tconst")
df_titre_clean = df_titre.loc[df_titre['category'].isin(['actor','actress','director'])]
df_titre_clean = df_titre[['tconst','primaryTitle','startYear','runtimeMinutes','genres','averageRating','numVotes','nconst']]

#TOP 1000 des meilleurs films
titre2 = titre

moyenne = note['numVotes'].mean()
ecart = note['numVotes'].std()
note['IsPopular'] = note['numVotes'].apply(lambda x: True if x >= moyenne + 2*ecart  else False)
note_titre = pd.merge(titre2, note, on = 'tconst')
best_film = note_titre.loc[note_titre['IsPopular'] == True]
best_film = best_film.sort_values('averageRating', ascending = False)
best_film['rank'] = best_film['numVotes'].rank(ascending = False)
best_film = best_film.loc[best_film['rank'] < 2000]
best_film.reset_index(inplace=True)
best_film = best_film.iloc[0:1000, :]
best_film.head()

# Top acteurs
distri_best_film_1000 = pd.merge(best_film, distribution, on = "tconst", how = "left")
acteurs_best_1000 = pd.merge(distri_best_film_1000, acteurs, on = "nconst", how = "inner")
personnage_best_1000 = acteurs_best_1000.groupby(["primaryName","category", "nconst"])["tconst"].count().sort_values(ascending=False).reset_index(name="nombrefilm")
df_actrices = personnage_best_1000.loc[personnage_best_1000["category"]=="actress"].nlargest(50, "nombrefilm")
df_acteurs = personnage_best_1000.loc[personnage_best_1000["category"]=="actor"].nlargest(50, "nombrefilm")
df_real = personnage_best_1000.loc[personnage_best_1000["category"]=="director"].nlargest(50, "nombrefilm")
df_selection_acteurs = pd.concat([df_acteurs, df_actrices, df_real])
df_selection_acteurs = df_selection_acteurs.drop(columns = ['nombrefilm'], axis=1)
df_individu_selection = df_selection_acteurs.pivot_table(index='nconst', columns='category', values='primaryName', aggfunc='first').reset_index()

#Joiture de table
titre_selection_acteurs = pd.merge(df_titre_clean, df_individu_selection, how='left', on='nconst')
titre_selection_acteurs = titre_selection_acteurs[['tconst','primaryTitle','startYear','runtimeMinutes','genres','averageRating','numVotes','actor','actress','director']]
def join_sans_nan(values):
  valeur_non_nan = [str(string) for string in values if pd.notnull(string)]
  return ','.join(valeur_non_nan)
# Aggrégation des lignes avec même tconst
film_selection = titre_selection_acteurs.groupby(['tconst','primaryTitle'], as_index=False).agg({'actor':join_sans_nan,
                                                                                                 'actress':join_sans_nan,
                                                                                                 'director':join_sans_nan,
                                                                                                 'startYear':'first',
                                                                                                 'runtimeMinutes':'first',
                                                                                                 'genres':'first',
                                                                                                 'averageRating':'first',
                                                                                                 'numVotes':'first'})
film_selection = film_selection.rename(columns={'primaryTitle':'title'})

dm_film_selection = pd.concat([film_selection, film_selection['genres'].str.get_dummies(sep=',')],axis=1)
dm_film_selection = pd.concat([dm_film_selection, dm_film_selection['actor'].str.get_dummies(sep=',')],axis=1)
dm_film_selection = pd.concat([dm_film_selection, dm_film_selection['actress'].str.get_dummies(sep=',')],axis=1)
dm_film_selection = pd.concat([dm_film_selection, dm_film_selection['director'].str.get_dummies(sep=',')],axis=1)

# création de model
X = dm_film_selection.select_dtypes(include = 'number')
y = dm_film_selection['title']
y = pd.DataFrame(y)
X_scaled = StandardScaler().fit_transform(X)
X_scaled = pd.DataFrame(X_scaled)
X_scaled.columns = X.columns
model_Classifier = KNeighborsClassifier(weights="distance", n_neighbors = 10, metric='cosine').fit(X_scaled, y)

# Foction de recommandaton

st.title('Recommandation de films')

df = dm_film_selection
critere_titre = st.text_input("Insérez le titre de film :")
critere_genre = st.text_input("Insérez le genre de film :")
critere_acteur = st.text_input("Insérez les acteurs/actrices de film :")

def reco(df, critere_titre, critere_genre, critere_acteur):
  # Traitement du critère Titre
  try:
    recherche_titre = df.loc[(df['title'] == critere_titre)]
    recherche_titre_scaled = X_scaled.iloc[recherche_titre.index]
  except :
    recherche_titre_scaled = pd.DataFrame()

  # Traitement du critère Genre
  try:
    recherche_genre = df.loc[(df[critere_genre] == 1)]
    recherche_genre_scaled = X_scaled.iloc[recherche_genre.index]
  except :
    recherche_genre_scaled = pd.DataFrame()

  # # Traitement du critère Acteur
  try:
    recherche_acteur = df.loc[(df[critere_acteur] == 1)]
    recherche_acteur_scaled = X_scaled.iloc[recherche_acteur.index]
  except :
    recherche_acteur_scaled = pd.DataFrame()

  # Calcul du nombre de résultats
  nb_results = len(recherche_titre_scaled) + len(recherche_genre_scaled) + len(recherche_acteur_scaled)

  # Ajout des pondérations selon les critères
  recherche_titre_scaled = recherche_titre_scaled * nb_results
  print(recherche_titre_scaled)
  # recherche_acteur_scaled[critere_acteur] = recherche_acteur_scaled[critere_acteur] * 2000

  # Créarion d'un DataFrame avec les résultats pondérés
  recherche_scaled = pd.concat([recherche_titre_scaled, recherche_genre_scaled, recherche_acteur_scaled])

  # Recommandation selon la méthode de la somme vectorielle
  somme_vec = pd.DataFrame(recherche_scaled.sum()).T
  recommandation = model_Classifier.kneighbors(somme_vec)[1][0]
  return df.iloc[recommandation,[1,2,3,4,5,6,7,8]]

st.write(reco(df, critere_titre, critere_genre, critere_acteur))
