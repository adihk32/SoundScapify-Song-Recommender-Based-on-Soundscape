import spotipy
from spotipy.oauth2 import SpotifyOAuth

import authorization
import pandas as pd
import time
import datetime
import numpy as np
import webbrowser

client_id = authorization.SPOTIFY_CLIENT_ID
client_secret = authorization.SPOTIFY_CLIENT_SECRET
redirect_uri = authorization.SPOTIFY_REDIRECT_URI

# setup the scope of permission needed
scope = ['user-modify-playback-state',
         'user-read-playback-state',
         'user-library-read',
         'user-read-recently-played']

# setup authorization manager
auth_manager = SpotifyOAuth(client_id=client_id,
                           client_secret=client_secret,
                           redirect_uri=redirect_uri,
                           scope=scope)

sp = spotipy.Spotify(auth_manager=auth_manager)

df_criteria = pd.read_csv('csv/recommend_criteria.csv')

def IsMorning(x):
    start = datetime.time(2, 0, 0)
    end = datetime.time(12, 0, 0)
    
    if start <= end:
        return start <= x <= end
    else:
        return start <= x or x <= end
    
def IsNoon(x):
    start = datetime.time(12, 0, 0)
    end = datetime.time(18, 0, 0)
    
    if start <= end:
        return start <= x <= end
    else:
        return start <= x or x <= end

def IsNight(x):
    start = datetime.time(18, 0, 0)
    end = datetime.time(2, 0, 0)
    
    if start <= end:
        return start <= x <= end
    else:
        return start <= x or x <= end

def GetEnergyRange(label):
    current = datetime.datetime.now().time()

    if IsMorning(current):
        energy_min = df_criteria[df_criteria['label'] == label]['energy_min']
        energy_max = df_criteria[df_criteria['label'] == label]['energy_2nd']
    
    if IsNoon(current):
        energy_min = df_criteria[df_criteria['label'] == label]['energy_3rd']
        energy_max = df_criteria[df_criteria['label'] == label]['energy_max']
        
    if IsNight(current):
        energy_min = df_criteria[df_criteria['label'] == label]['energy_2nd']
        energy_max = df_criteria[df_criteria['label'] == label]['energy_3rd']
    
    return energy_min, energy_max

def GetValenceRange(label):
    valence_min = df_criteria[df_criteria['label'] == label]['valence_min']
    valence_max = df_criteria[df_criteria['label'] == label]['valence_max']
    
    return valence_min, valence_max

def GetRecommendation(label):
    recent_id = []
    recommended_uri = []
    
    recent_playback = sp.current_user_recently_played(limit=5)
    
    for track in recent_playback['items']:
        track_id = track['track']['id']
        recent_id.append(track_id)
    
    val_min, val_max = GetValenceRange(label)
    ene_min, ene_max = GetEnergyRange(label)
    
    recs = sp.recommendations(seed_tracks=recent_id, limit=10, 
                              min_valence= val_min, max_valence= val_max,
                              min_energy= ene_min, max_energy= ene_max,
                              country='SG')
    
    for track in recs['tracks']:
        track_uri = track['uri']
        recommended_uri.append(track_uri)
    
    return recommended_uri

def OpenSpotify():
    chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe %s'
    webbrowser.get(chrome_path).open('https://open.spotify.com')

    time.sleep(3)
    
    devices = sp.devices()
    
    return devices

def StartPlayback(uris):
    devices = sp.devices()
    
    if devices != {}:
        
        if devices['devices'] == []:
            devices = OpenSpotify()

    else: devices = OpenSpotify()


    sp.start_playback(device_id=devices['devices'][0]['id'], 
                      uris=uris)