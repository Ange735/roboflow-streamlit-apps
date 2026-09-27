# 🎯 Roboflow Streamlit Apps

Interfaces web [Streamlit](https://streamlit.io) pour tester des modèles de **détection d'objets** et de **classification d'images** entraînés sur [Roboflow](https://roboflow.com) : sur une image, un lot d'images, une vidéo ou la webcam.

## Les applications

### 1. `api_rest/`: détection avec l'API REST Roboflow

`app_detection.py` n'utilise pas le SDK : il appelle directement l'API hébergée de Roboflow (`detect.roboflow.com`). C'est la version la plus légère, et elle se déploie facilement sur Streamlit Cloud.

- **Image unique** : détection, boîtes annotées, statistiques (nombre d'objets, classes, confiance moyenne) et téléchargement du résultat
- **Groupe d'images** : analyse de plusieurs images d'un coup, avec des statistiques globales
- **Webcam** : prise de photo, puis détection instantanée

```bash
cd api_rest
pip install -r requirements.txt
streamlit run app_detection.py
```

### 2. `sdk_roboflow/`: détection et classification avec le SDK Python

| App | Rôle |
|---|---|
| `app_detection.py` | Détection d'objets : seuils de confiance et de chevauchement réglables, épaisseur des boîtes |
| `app_classification.py` | Classification : affichage des Top-K prédictions |

Les deux apps proposent trois onglets :
- **Image** : résultat annoté et export en JSON ou en PNG
- **Dossier** : plusieurs images ou une archive ZIP, avec la distribution des classes
- **Vidéo** : analyse d'une image sur N, puis export en JSON (et, pour la détection, vidéo annotée en MP4)

```bash
cd sdk_roboflow
pip install -r requirements.txt
streamlit run app_detection.py        # ou app_classification.py
```

## Configuration

Dans la barre latérale de l'app, renseignez :
- votre **clé API Roboflow** (Roboflow → Settings → API Keys) ;
- le **Project / Model ID** et la **version** du modèle (onglet *Deploy* de votre projet Roboflow).

La clé n'est jamais écrite dans le code : elle est saisie dans l'interface.

## Technologies

Python · Streamlit · Roboflow (API REST et SDK) · Pillow · OpenCV · pandas
