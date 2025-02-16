# Main File for this program
from collections import defaultdict
import lyricsgenius
import os
from dotenv import load_dotenv
import re
import json


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

'''
def fetch_album_lyrics(genius: lyricsgenius.Genius, album_name: str, artist_name: str):
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
        song_data = {
            "artist": artist_name,
            "album": album_name,
            "song_title": track['song']['title'],
            "lyrics": track['song']['lyrics'],
            "cleaned_lyrics": clean_lyrics(track['song']['lyrics']),
            "emotion_scores": None,
            "line_by_line_analysis": None,
            "spotify_features": None
        }
        songs_data.append(song_data)

        print(f"{track['song']['title']} lyrics: {track['song']['lyrics']}")

    with open('songs_lyrics_data1.json', 'w', encoding='utf-8') as f:
        json.dump(songs_data, f, ensure_ascii=False, indent=4)
    return
'''

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

def analyze_lyric_emotions(lyrics: str, nrc_lexicon: dict) -> dict[str, int]:

    # clean lyrics into lowercase words
    words = re.findall(r'\b\w+\b', lyrics.lower())

    # initialize emotion counters
    emotion_scores = defaultdict(int)

    # score each word
    for word in words:
        if word in nrc_lexicon:
            for emotion, value in nrc_lexicon[word].items():
                emotion_scores[emotion] += value
    
    return dict(emotion_scores)

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
        song_data = {
            "artist": artist_name,
            "album": album_name,
            "song_title": track['song']['title'],
            "lyrics": track['song']['lyrics'],
            "cleaned_lyrics": clean_lyrics(track['song']['lyrics']),
            "emotion_scores": analyze_lyric_emotions(track['song']['lyrics'], nrc_lexicon),
            "line_by_line_analysis": None,
            "spotify_features": None
        }
        songs_data.append(song_data)

        # print(f"{track['song']['title']} lyrics: {track['song']['lyrics']}")

    with open('songs_lyrics_data1.json', 'w', encoding='utf-8') as f:
        json.dump(songs_data, f, ensure_ascii=False, indent=4)
    return


if __name__ == "__main__":
    genius = genius_setup()
    album_name = input("Search for album: ")
    artist_name = input("By artist: ")

    lex = load_nrc_lexicon("./NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")

    fetch_and_analyze_album_lyrics(genius, album_name, artist_name, lex)
