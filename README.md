# SoundScapify - Song Recommender based on Soundscape
A song recommender integrated with Spotify API based on the surrounding acoustic scene

![Screenrecord.gif](gif/screen-capture.gif)

## Introduction
### Background
Music has been along our side throughout our life. When we are babies, our parents would sing lullabies at night to make us asleep. When we are students, music accompaniments help us go through vicious cycles of exams and projects. Even as an adult, music help us go through a lot when we are bored, happy, sad or angry. Therefore, music plays a major role in our livelihood. 

In a [survey](https://www.plansponsor.com/SURVEY-SAYS-How-You-Spend-Your-Commute/?layout=print) back in 2015, almost half of the respondents listen to music during their commute. In Singapore alone, 40% of Singaporeans listen to music during their commute according to NAC [survey.](https://www.nac.gov.sg/about-us/media-centre/press-releases/singaporeans-are-avid-music-listeners-says-first-national-music-consumption-survey)
This also elevated the reason due to a [study](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(13)00049-1) that revealed music helps to reduce stress and improve our overall well being. 

Therefore it is important have songs that fit with the ambience of our commute in general.     

### Problem Statement
Based on the information above, the project aims to build a **song recommender** based on the **current ambience and similar mood**. To accomplish the application, we first need to build a **classifier model** to classify the **ambient soundscape** in order for the recommender to give relevant and best result. Our target for model is to achieve more than **80%** accuracy. Based on the classified ambience, we then obtained a range of **audio features** that will be utilised to obtain similar songs.

### Proposed Approach
For this project, we will be utilising a deep learning model, which is **Long Short-Term Memory (LSTM) Neural Network**, which is one type of **Recurrent Neural Network** that has the ability to utilise order dependence. This means that the model will train based on the time sequence and "remember" the previous time sequence info to be incorporated in the current time sequence learning. This type of model works well in sequence prediction problems, and in our case, identifying the ambience sound.

**The Valence-Arousal plane** is also utilised in this project. The great thing of the model is it is able to identify mood of an audio within the 2-axis plane. The two features that are utilised are valence and arousal(energy). **Arousal** is an emotional dimension of musically energy level, while **Valence** is an emotional dimension of the comfortable level of the listener, which can be translated to whether the audio sounds happier or not [[*source*]](https://ieeexplore.ieee.org/document/8982519).

We would also be using **K-Means clustering** algorithm to make cluster of songs that will be fitted to our acoustic scenes classified. This will help us to obtain the range of valence and energy value that will be inputs to the recommender.

For the recommender, we uses Spotify's own recommender that we can access from **Spotify API** with additional information provided based on the classification made earlier. The info that is feeded including recently played tracks, minimum & maximum valence value and minimum & maximum energy value for the tracks recommended.

Overall, the **workflow** is as per below:
1. Find soundscape dataset
2. Explore and preprocess the dataset
3. Build a LSTM model and tune it
4. Build dataset of songs with audio features info
5. Explore the songs dataset
5. Create a criteria range for valence and energy
6. Create an app as a proof of concept (POC)

For this project, we will have 4 parts of notebook as follows:  
1. Introduction and Train Dataset EDA
2. Model Training
3. Model Prediction with Test Dataset
4. Song Dataset Retrieval

Apart from notebooks, we also needed a couple of .py scripts that is meant for the app development:
1. `app.py` -  contains the app function and features that has been developed
2. `authorization.py` - contains the authorization key of your Spotify API, such as client id, client secret and redirect url which can be retrieved from Spotify API upon registration 
3. `spotify.py` - contains collection of functions that are utilizing the Spotify API 

### Data Dictionary
This project involves the following datasets:  

|Dataset|Description|
|:---|:---|
|`fold1_train.csv`|Original dataset from [TAU Urban Acoustic Scenes 2022 Mobile, development dataset](https://zenodo.org/record/6337421) that contains filename and label of 1-second clips from multiple cities across Europe and recorded in 10 different acoustic scenes meant for training purposes| 
|`fold1_test.csv`|Original dataset from [TAU Urban Acoustic Scenes 2022 Mobile, development dataset](https://zenodo.org/record/6337421) that contains filename and label of 1-second clips from multiple cities across Europe and recorded in 10 different acoustic scenes meant for test purposes, the size of this dataset is 1/10th of the train dataset| 
|`valence_arousal_dataset.csv`|Dataset of songs from multiple genres that is scraped using [Spotify API](https://developer.spotify.com/documentation/web-api/) which includes the valence and energy value of the songs|
|`recommend_criteria.csv`|Dataset of criteria for the valence and energy range based on the label, which is extracted from `valence_arousal_dataset.csv`|

There are also 2 pickle models that are utilised in this project which can be found in the link below: <https://drive.google.com/drive/folders/1tX8-UGQFAby8Gm2CAJGE4Ep9LBZlRut0?usp=sharing>

The datasets contains the following information/features:  

|Feature|Type|Dataset|Description|
|:---|:---:|:---:|:---|
|filename|*str*|`fold1_train.csv` & `fold1_test.csv`|The filename of 1-second clips from the dataset| 
|scene_label|*str*|`fold1_train.csv`|The labels of the acoustic scene where the audio is captured|
|id|*str*|`valence_arousal_dataset.csv`|The Spotify id of the songs in the dataset|
|genre|*str*|`valence_arousal_dataset.csv`|The genre of the songs in the dataset|
|track_name|*str*|`valence_arousal_dataset.csv`|The track name of the songs in the dataset|
|artist_name|*str*|`valence_arousal_dataset.csv`|The artist name of the songs in the dataset|
|valence|*float*|`valence_arousal_dataset.csv`|The valence value of the songs in the dataset|
|energy|*float*|`valence_arousal_dataset.csv`|The energy value of the songs in the dataset|
|label|*str*|`recommend_criteria.csv`|The label of acoustic scenes that we are classifying|
|valence_min|*float*|`recommend_criteria.csv`|The minimum valence value for the label classified|
|valence_max|*float*|`recommend_criteria.csv`|The maximum valence value for the label classified|
|energy_min|*float*|`recommend_criteria.csv`|The minimum energy value for the label classified|
|energy_2nd|*float*|`recommend_criteria.csv`|The energy value in 33th percentile for the label classified|
|energy_3rd|*float*|`recommend_criteria.csv`|The energy value in 66th percentile for the label classified|
|energy_max|*float*|`recommend_criteria.csv`|The maximum energy value for the label classified|

## Overview
From the dataset of TAU Urban Acoustic Scene 2022, there are 10 labels of acoustic scene provided. For this project, we will be focusing on 4 labels that is common during our commute around which consists of: 
- metro, 
- bus, 
- street_traffic and
- park 

From the .wav files provided, we converted the audio files to mel-spectrogram using **librosa** package that became the input of the model created.

Next, we preprocessed the input further, such as train-test split, label encoding the label and creating keras Sequence type of class for data generator, before fitted to the model created. We also set accuracy and loss as our main metrics to determine the best model.

After few epochs, we obtained the accuracy of train and validation dataset to be fairly high (around 0.9). Furthermore, we tested the model using unseen data (which is the test dataset) to check the robustness of the model. The evaluation of the unseen data turns out to have  relatively low accuracy (around 0.6), but still acceptable for deep learning model. Therefore, we proceed to obtain the song dataset.

The song dataset is built by utilising **spotipy** package to access the Spotify API recommender that recommends for every genres available. Then, we used K-Means clustering to cluster the dataset into 4 cluster which is then fitted to our 4 acoustic scene. The valence and energy value range also identified that will be inputted to the recommended as well.

Finally, an app is created as a proof of concept. The flow of the app is after inputting the audio files, it can display the mel-spectrogram before classfying the audio using our model. The next step, it can opened up the Spotify and start playing the recommended tracks based on the audio classification. 

The app can be access using terminal and **streamlit** package with the command below:  
`streamlit run app.py`

## Summary
The model that we are using which is a **Long Short-Term Memory Neural Network** algorithm. Upon fitting the train dataset, we obtained the accuracy of prediction as follows:  
  

|Dataset|Accuracy|Loss|
|:---|:---:|:---|
|train|0.9542|0.137204|
|validation|0.8959|0.341624|
|test(unseen)|0.5711|1.9916|

The accuracy and loss is quite good for the train and validation dataset. However, for the test dataset, the model accuracy drops by substantial percentage (30%). This maybe due to the emergence of overfitting to the train dataset. Another aspect, that may contribute to these is the metrics value chosen.

It is an interesting findings since the audio files originate from the same file which has been splitted to a 1-second clips. This may happened due to the different audio profile of different clips within the same file.

## Limitation
- The acoustic scene that we are trying to classify only consists of 4 scenes. It does not cover other acoustic scene that also utilise music on the scene, such as cafe, restaurant, lobby, club, gym, etc
- Limitation of time to explore more on the model tuning or even other type of deep learning model that may have better prediction for the dataset.
- Limited dataset on the valence and energy relation to acoustic scene. Most of the dataset only has the feature values for songs.
- The microphone and machine could not detect the ambience sound that is needed to be classified from the app. Load audio features only is functional at the time being.

## Future Works
- Improve the model accuracy by tuning the current the model or introduce different type of deep learning model. We may choose different metrics to be selected as well.
- Increase the number of acoustic scene that can be classified to have a more better classification model that fits every acoustic scene.
- Collect data that represent the valence and energy value based on the acoustic scene to achieve a better representation of the value range that is inputted to the Spotify recommender.
- Develop an Android software to deploy the app in order for us to use the recording features, since the phone microphone can capture a better audio. 