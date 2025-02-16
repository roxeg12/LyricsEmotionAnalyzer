# Main File for this program
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
    # remove text in brackets
    lyrics = re.sub(r'\[.*?\]', '', lyrics)

    # replace multiple line breaks with a single space
    lyrics = re.sub(r'\n+', ' ', lyrics)
    return lyrics.strip()


def fetch_album_lyrics(genius: lyricsgenius.Genius, album_name: str, artist_name: str):
    """
    
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

    with open('songs_lyrics_data.json', 'w', encoding='utf-8') as f:
        json.dump(songs_data, f, ensure_ascii=False, indent=4)
    return

if __name__ == "__main__":
    genius = genius_setup()
    album_name = input("Search for album: ")
    artist_name = input("By artist: ")

    fetch_album_lyrics(genius, album_name, artist_name)
