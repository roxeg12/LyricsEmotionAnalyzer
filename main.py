# Main File for this program
from collections import defaultdict
import lyricsgenius
import os
from dotenv import load_dotenv
import re
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def genius_setup() -> lyricsgenius.Genius:
    """
    Completes set up and authentication of a genius object
    instance with a token stored externally (in a .env file for privacy)

    Inputs: none
    Returns: an authenticated instance of the Genius class

    Requires: that a .env file with the variable 'GENIUS_ACCESS_TOKEN'
    exists, and its value is set to a valid genius access token
    """
    # Load env global variables
    load_dotenv()

    genius_token = os.getenv("GENIUS_ACCESS_TOKEN")

    # Instantiate genius with token
    genius = lyricsgenius.Genius(genius_token)
    return genius

def clean_lyrics(lyrics: str) -> str:
    """
    Preprocess lyric data by removing unnecessary details,
    like descriptors such as '[Chorus]' or '[Verse 1]'

    Inputs:
        - lyrics: an unprocessed string from a lyricsgenius Song object
    Returns:
        the same lyrics, with non-lyric text removed and white space minimized

    """
    # remove conributor and song title information
    lyrics = lyrics.partition(" Lyrics")[2]

    # remove text in brackets
    lyrics = re.sub(r'\[.*?\]', '', lyrics)

    # replace multiple line breaks with a single space
    lyrics = re.sub(r'\n+', ' ', lyrics)
    return lyrics.strip()


def load_nrc_lexicon(filepath: str) -> dict[str, dict[str, int]]:
    """
    Parse the raw NRC lexicon file into a dictionary

    Inputs:
        - filepath: a string describing where the lexicon file is stored
    Returns:
        A dictionary of each word mapped to a dictionary of each emotion and 
        the word's score

    Requires that the filepath be a valid location where the lexicon file is stored
    """
    emotion_dict = {}
    
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            word, emotion, association = line.strip().split('\t')

            # only consider words with association
            if association == '1': 
                if word not in emotion_dict:
                    emotion_dict[word] = {e: 0 for e in [
                        'anger', 'anticipation', 'disgust', 'fear',
                        'joy', 'sadness', 'surprise', 'trust',
                        'positive', 'negative'
                    ]}
                emotion_dict[word][emotion] = 1
    return emotion_dict

def normalize_scores(scores: dict, total_words: int) -> dict:
    """
    Converts raw emotion word counts to percentage of total words

    """
    if total_words == 0:
        return {k: 0.0 for k in scores.keys()}

    normalized_scores = {emotion: round(count / total_words, 4) for emotion, count in scores.items()}
    return normalized_scores

def analyze_lyric_emotions(lyrics: str, nrc_lexicon: dict) -> dict[str, int]:

    # clean lyrics into lowercase words
    words = re.findall(r'\b\w+\b', lyrics.lower())

    total_words = len(words)

    # initialize emotion counters
    emotion_scores = defaultdict(int)

    # score each word
    for word in words:
        if word in nrc_lexicon:
            for emotion, value in nrc_lexicon[word].items():
                emotion_scores[emotion] += value

    raw_scores = dict(emotion_scores)

    normalized_scores = normalize_scores(raw_scores, total_words)

    scores = {
        "raw": raw_scores,
        "normalized": normalized_scores
    }
    
    return scores



def fetch_and_analyze_album_lyrics(genius: lyricsgenius.Genius, album_name: str, artist_name: str, nrc_lexicon: dict):
    """
    Retrieves the lyrics from the album with name album_name by album_artist and stores each
    song's data in a joint JSON file, including the cleaned song lyrics

    Inputs:
        - genius: the lyricsgenius.Genius object associated with a valid API token
        - album_name: a valid album name
        - artist_name: a valid artist name that has produced the album denoted by album_name
    Returns:
        Nothing

    Requires:
        genius has already been instantiated
    """
    album = genius.search_album(album_name, artist_name)
    
    
    songs_data = []

    # store data for each song as JSON
    for track1 in album.tracks:
        track = track1.to_dict()
        cleaned_lyrics = clean_lyrics(track['song']['lyrics'])
        emotion_analysis = analyze_lyric_emotions(cleaned_lyrics, nrc_lexicon)
        song_data = {
            "artist": artist_name,
            "album": album_name,
            "song_title": track['song']['title'],
            "lyrics": track['song']['lyrics'],
            "cleaned_lyrics": cleaned_lyrics,
            "emotion_scores": emotion_analysis['raw'],
            "normalized_emotion_scores": emotion_analysis['normalized'],
            "line_by_line_analysis": None,
            "spotify_features": None
        }
        songs_data.append(song_data)

        # print(f"{track['song']['title']} lyrics: {track['song']['lyrics']}")

    with open('songs_lyrics_data1.json', 'w', encoding='utf-8') as f:
        json.dump(songs_data, f, ensure_ascii=False, indent=4)
    return songs_data

def plot_emotion_journey(album_data, emotions, use_norm=True, include_sent=False):

    sentiments = ['positive', 'negative']

    if include_sent:
        emotions.extend(sentiments)

    
    emotion_series = {emotion: [] for emotion in emotions}
    song_titles = []
    
    for song in album_data:
        song_titles.append(song['song_title'])
        
        for emotion in emotions:
            if use_norm:
                score = song.get('normalized_emotion_scores', {}).get(emotion, 0)
            else:
                score = song.get('emotion_scores', {}).get(emotion, 0)

            emotion_series[emotion].append(score)

    plt.figure(figsize=(12, 6))
    for emotion, scores in emotion_series.items():
        plt.plot(song_titles, scores, marker='o', label=emotion)
    plt.xticks(rotation=45)
    plt.title(f"Emotion Journey Across Album ({'Normalized' if use_norm else 'Raw'})")
    plt.xlabel("Songs")
    plt.ylabel(f"{'Normalized' if use_norm else 'Raw'} Emotion Score")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_stacked_emotion_composite(album_data, emotions, use_norm=True):

    
    data = []
    song_titles = []

    for song in album_data:
        song_titles.append(song['song_title'])

        if use_norm:
            row = [song['normalized_emotion_scores'].get(e, 0) for e in emotions]
        else:
            row = [song['emotion_scores'].get(e, 0) for e in emotions]
        
        data.append(row)
    
    df = pd.DataFrame(data, columns=emotions, index=song_titles)

    ax = df.plot(kind='area', stacked=True, figsize=(12, 6), alpha=0.7)
    plt.title(f"Emotional Composition across {album_data[0]['album']} ({'Normalized' if use_norm else 'Raw'})")
    plt.xlabel('Song')
    plt.ylabel('Emotion Score')
    plt.xticks(range(len(song_titles)), song_titles, rotation=45)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_emotion_heatmap(album_data, emotions, use_norm=True):
    
    data = []
    song_titles = []

    for song in album_data:
        song_titles.append(song['song_title'])
        if use_norm:
            row = [song['normalized_emotion_scores'].get(e, 0) for e in emotions]
        else:
            row = [song['emotion_scores'].get(e, 0) for e in emotions]
        data.append(row)
    
    df = pd.DataFrame(data, columns=emotions, index=song_titles)

    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5)
    plt.title(f"Emotion Heatmap Across {album_data[0]['album']} ({'Normalized' if use_norm else 'Raw'})")
    plt.xlabel("Emotions")
    plt.ylabel("Songs")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_stacked_emotion_dist(album_data, emotions, use_norm=True):
    
    song_titles = []
    emotion_values = {emotion: [] for emotion in emotions}

    for song in album_data:
        song_titles.append(song['song_title'])
        for emotion in emotions:
            if use_norm:
                score = song.get('normalized_emotion_scores', {}).get(emotion, 0)
            else:
                score = song.get('emotion_scores', {}).get(emotion, 0)
            emotion_values[emotion].append(score)

    bottom_vals = np.zeros(len(song_titles))
    plt.figure(figsize=(12, 6))

    for emotion in emotions:
        plt.bar(song_titles, emotion_values[emotion], bottom=bottom_vals, label=emotion)
        bottom_vals += np.array(emotion_values[emotion])

    plt.xticks(rotation=45)
    plt.ylabel(f"{'Normalized' if use_norm else 'Raw'} Emotion Score")
    plt.xlabel("Songs")
    plt.title(f"Emotion Distribution Across {album_data[0]['album']} by {album_data[0]['artist']} ({'Normalized' if use_norm else 'Raw'})")
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    genius = genius_setup()
    album_name = input("Search for album: ")
    artist_name = input("By artist: ")

    emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']

    lex = load_nrc_lexicon("./NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")

    album_data = fetch_and_analyze_album_lyrics(genius, album_name, artist_name, lex)

    # plot_emotion_journey(album_data)

    # plot_stacked_emotion_composite(album_data, emotions)

    # plot_emotion_heatmap(album_data, emotions)

    plot_stacked_emotion_dist(album_data, emotions)
