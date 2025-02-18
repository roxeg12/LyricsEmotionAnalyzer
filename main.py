# Main File for this program
from collections import defaultdict
import lyricsgenius
import os
from dotenv import load_dotenv
import re
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity


def genius_setup() -> lyricsgenius.Genius:
    """
    Completes set up and authentication of a genius object
    instance with a token stored externally (in a .env file for privacy)

    Inputs: none
    Returns: an authenticated instance of the Genius class

    Requires that a .env file with the variable 'GENIUS_ACCESS_TOKEN'
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

    Inputs:
        - scores: a dictionary mapping emotions to the count of words associated with that emotion
        - total_words: the number of words in the set of lyrics being represented by scores
    
    Returns:
        A dictionary mapping emotions to the percentage of words associated with that emotion

    """
    if total_words == 0:
        return {k: 0.0 for k in scores.keys()}

    normalized_scores = {emotion: round(count / total_words, 4) for emotion, count in scores.items()}
    return normalized_scores

def analyze_lyric_emotions(lyrics: str, nrc_lexicon: dict) -> dict[str, int]:
    """
    Analyzes the emotion words in 'lyrics' using data from the nrc_lexicon, keeping track of
    emotion words found in the lyrics

    Inputs:
        - lyrics: a string that contains all the lyrics to be analyzed in this function call
        - nrc_lexicon: a dictionary of words mapped to a dictionary of each emotion and 
          the word's score
    
    Returns:
        A dictionary of the raw and normalized emotion scores for these lyrics - 
        the value of each entry is a dictionary mapping each emotion to its score
        for these lyrics
    
    Requires that the nrc_lexicon be a valid dictionary, preprocessed by the function load_nrc_lexicon
    """
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

    # calculate the normalized scores
    normalized_scores = normalize_scores(raw_scores, total_words)

    # return both raw and normalized scores
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
        A list of the data for each song in the album

    Requires that genius has already been instantiated
    """
    # fetch the album data from Genius
    album = genius.search_album(album_name, artist_name)
    
    
    songs_data = []

    # store data for each song as JSON
    for track1 in album.tracks:
        track = track1.to_dict()

        # preprocess the lyrics
        cleaned_lyrics = clean_lyrics(track['song']['lyrics'])
        emotion_analysis = analyze_lyric_emotions(cleaned_lyrics, nrc_lexicon)

        # create the dictionary object
        song_data = {
            "artist": artist_name,
            "album": album_name,
            "song_title": track['song']['title'],
            "lyrics": track['song']['lyrics'],
            "cleaned_lyrics": cleaned_lyrics,
            "emotion_scores": emotion_analysis['raw'],
            "normalized_emotion_scores": emotion_analysis['normalized'],
        }
        songs_data.append(song_data)

        # print(f"{track['song']['title']} lyrics: {track['song']['lyrics']}")

    # store in a json file for easy access (instead of printing)
    with open('songs_lyrics_data1.json', 'w', encoding='utf-8') as f:
        json.dump(songs_data, f, ensure_ascii=False, indent=4)
    return songs_data

def plot_emotion_journey(album_data, emotions, use_norm=True, include_sent=False):
    """
    Creates and displays a line graph that plots the score of each emotion in emotions across the length of the album
    (from song to song). Be default, the scores presented are normalized, each emotion is a different color-coded line, the x-axis contains 
    the names of each song on the album, and the y-axis is the emotion score.

    Inputs:
        - album_data: a list of the data of each song on the album, created by fetch_and_analyze_album_lyrics
        - emotions: a list of all the emotions being analyzed for
        - use_norm: a boolean indicating whether to use the normalized or raw emotion scores; true by default
        - include_sent: a boolean indicating whether to include the positive/negative sentiment descriptors in the analysis; true by default
    
    Returns:
        nothing
    
    Requires that album_data was create by the function fetch_and_analyze_album_lyrics, or at least follows the schema defined
    within that function.
    """
    sentiments = ['positive', 'negative']

    if include_sent:
        emotions.extend(sentiments)

    # set up data for emotions
    emotion_series = {emotion: [] for emotion in emotions}
    song_titles = []
    
    # iterate through the data for each song in the album
    for song in album_data:
        song_titles.append(song['song_title'])
        
        for emotion in emotions:
            if use_norm:
                score = song.get('normalized_emotion_scores', {}).get(emotion, 0)
            else:
                score = song.get('emotion_scores', {}).get(emotion, 0)

            emotion_series[emotion].append(score)

    # plot the line graph
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
    """
    Creates and displays a stacked composition graph that plots the score of each emotion in emotions across the length of the album
    (from song to song). Be default, the scores presented are normalized, each emotion has a different color-coded area, the x-axis contains 
    the names of each song on the album, and the y-axis is the emotion score.

    Inputs:
        - album_data: a list of the data of each song on the album, created by fetch_and_analyze_album_lyrics
        - emotions: a list of all the emotions being analyzed for
        - use_norm: a boolean indicating whether to use the normalized or raw emotion scores; true by default
    
    Returns:
        nothing
    
    Requires that album_data was create by the function fetch_and_analyze_album_lyrics, or at least follows the schema defined
    within that function.
    """
    
    data = []
    song_titles = []

    # retrieve the emotion score for each song 
    for song in album_data:
        song_titles.append(song['song_title'])

        if use_norm:
            row = [song['normalized_emotion_scores'].get(e, 0) for e in emotions]
        else:
            row = [song['emotion_scores'].get(e, 0) for e in emotions]
        
        data.append(row)
    
    # create a data frame for the emotion score data
    df = pd.DataFrame(data, columns=emotions, index=song_titles)

    # plot the graph
    ax = df.plot(kind='area', stacked=True, figsize=(12, 6), alpha=0.7)
    plt.title(f"Emotional Composition across {album_data[0]['album']} ({'Normalized' if use_norm else 'Raw'})")
    plt.xlabel('Song')
    plt.ylabel('Emotion Score')
    plt.xticks(range(len(song_titles)), song_titles, rotation=45)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_emotion_heatmap(album_data, emotions, use_norm=True):
    """
    Creates and displays a heatmap that plots the score of each emotion in emotions across the length of the album
    (from song to song). Be default, the scores presented are normalized, the x-axis contains the emotions, and the 
    y-axis contains the names of the songs on the album. The emotion score for each song for each emotion is displayed
    at the corresponding intersecting cell.

    Inputs:
        - album_data: a list of the data of each song on the album, created by fetch_and_analyze_album_lyrics
        - emotions: a list of all the emotions being analyzed for
        - use_norm: a boolean indicating whether to use the normalized or raw emotion scores; true by default
    
    Returns:
        nothing
    
    Requires that album_data was create by the function fetch_and_analyze_album_lyrics, or at least follows the schema defined
    within that function.
    """


    data = []
    song_titles = []

    # retrieve emotion scores for each song
    for song in album_data:
        song_titles.append(song['song_title'])
        if use_norm:
            row = [song['normalized_emotion_scores'].get(e, 0) for e in emotions]
        else:
            row = [song['emotion_scores'].get(e, 0) for e in emotions]
        data.append(row)
    
    # create data frame
    df = pd.DataFrame(data, columns=emotions, index=song_titles)

    # plot the graph
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5)
    plt.title(f"Emotion Heatmap Across {album_data[0]['album']} ({'Normalized' if use_norm else 'Raw'})")
    plt.xlabel("Emotions")
    plt.ylabel("Songs")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_stacked_emotion_dist(album_data, emotions, use_norm=True):
    """
    Creates and displays a stacked bar graph that plots the score of each emotion in emotions across the length of the album
    (from song to song). Be default, the scores presented are normalized, each emotion has a different color-coded area within each bar, 
    the x-axis contains the names of each song on the album, and the y-axis is the emotion score.

    Inputs:
        - album_data: a list of the data of each song on the album, created by fetch_and_analyze_album_lyrics
        - emotions: a list of all the emotions being analyzed for
        - use_norm: a boolean indicating whether to use the normalized or raw emotion scores; true by default
    
    Returns:
        nothing
    
    Requires that album_data was create by the function fetch_and_analyze_album_lyrics, or at least follows the schema defined
    within that function.
    """
    song_titles = []
    emotion_values = {emotion: [] for emotion in emotions}

    # retrieve emotions scores for each song
    for song in album_data:
        song_titles.append(song['song_title'])
        for emotion in emotions:
            if use_norm:
                score = song.get('normalized_emotion_scores', {}).get(emotion, 0)
            else:
                score = song.get('emotion_scores', {}).get(emotion, 0)
            emotion_values[emotion].append(score)

    # plot graph
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

def plot_emotion_radar(song_title, album_data, emotions, use_norm=True):
    """
    Creates and displays a radar/spider graph that plots the score of each emotion in emotions for one song. 
    By default, the scores presented are normalized, each emotion is a different axis, and the distance from the center
    on each axis represents the emotion score for that emotion.

    Inputs:
        - album_data: a list of the data of each song on the album, created by fetch_and_analyze_album_lyrics
        - emotions: a list of all the emotions being analyzed for
        - use_norm: a boolean indicating whether to use the normalized or raw emotion scores; true by default
    
    Returns:
        nothing
    
    Requires that album_data was create by the function fetch_and_analyze_album_lyrics, or at least follows the schema defined
    within that function.
    """
    song_data = {}

    # get song data
    for song in album_data:
        if song['song_title'] == song_title:
            song_data = song

    values = []
    for emotion in emotions:
        if use_norm:
            values.append(song_data['normalized_emotion_scores'].get(emotion, 0))
        else:
            values.append(song_data['emotion_scores'].get(emotion, 0))
    values += values[:1]

    # preprocess score data to fit radial graph
    angles = np.linspace(0, 2*np.pi, len(emotions), endpoint=False).tolist()
    angles += angles[:1]

    # plot graph
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.fill(angles, values, color='cyan', alpha=0.25)
    ax.plot(angles, values, marker='o', color='blue')
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(emotions)

    plt.title(f"Emotional Profile: {song_title}")
    plt.show()

def plot_emotion_wordcloud(album_data):
    """
    Creates and displays a wordcloud, where the size of the word displayed corresponds with its frequency within
    the lyrics.

    ** Note: no emotion analysis on this function yet, not included in options for CLI

    Inputs:
        - album_data: a list of the data of each song on the album, created by fetch_and_analyze_album_lyrics
    
    Returns:
        nothing
    
    Requires that album_data was create by the function fetch_and_analyze_album_lyrics, or at least follows the schema defined
    within that function.
    """
    lyrics = ""

    # get lyrics
    for song in album_data:
        lyric = song['cleaned_lyrics'] + " "
        lyrics += lyric

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(lyrics)

    # plot graph
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Emotion Word Cloud for {album_data[0]['album']} by {album_data[0]['artist']}")
    plt.show()

def calculate_album_similarity(album_data_1, album_data_2, emotions, use_norm=True):
    """
    Calculates the cosine similarity between the emotion scores of the two albums

    Inputs:
        - album_data_1: a list of the data of each song on the first album, created by fetch_and_analyze_album_lyrics 
        - album_data_2: a list of the data of each song on the second album, created by fetch_and_analyze_album_lyrics
        - emotions: a list of all the emotions being analyzed for
        - use_norm: a boolean indicating whether to use the normalized or raw emotion scores; true by default

    Returns:
        The cosine similarity between the two albums' emotion scores

    Requires that album_data_1 and album_data_2 were created by the function fetch_and_analyze_album_lyrics, or at least follows the schema defined
    within that function.
    
    """
    # get average emotion scores for each album
    album_avg_1 = np.mean([[song['normalized_emotion_scores' if use_norm else 'emotion_scores'].get(e, 0) for e in emotions] for song in album_data_1], axis=0)
    album_avg_2 = np.mean([[song['normalized_emotion_scores' if use_norm else 'emotion_scores'].get(e, 0) for e in emotions] for song in album_data_2], axis=0)

    # calculate cosine similariy
    similarity = cosine_similarity([album_avg_1], [album_avg_2])[0][0]

    return similarity

def plot_comparison(album_data_1, album_data_2, emotions, use_norm=True):
    """
    Creates and displays a comparison of the emotion data of the two albums. 
    Displays multiple graphs:
        - Multi-line graph: combines the visuals of a line graph for each album onto one graph
        - Heatmaps: displays a heatmap for each album side by sied
        - Cosine similarity text display: displays the value that respresents the cosine similarity
                                          between the two albums
                                        
    Inputs:
        - album_data_1: a list of the data of each song on the first album, created by fetch_and_analyze_album_lyrics 
        - album_data_2: a list of the data of each song on the second album, created by fetch_and_analyze_album_lyrics
        - emotions: a list of all the emotions being analyzed for
        - use_norm: a boolean indicating whether to use the normalized or raw emotion scores; true by default

    Returns:
        Nothing

    Requires that album_data_1 and album_data_2 were created by the function fetch_and_analyze_album_lyrics, or at least follows the schema defined
    within that function.
    
    """
    # a custom color scheme
    emotion_colors = {
        'anger': 'red', 
        'anticipation': 'orange',
        'disgust': 'green', 
        'fear': 'purple',
        'joy': 'gold', 
        'sadness': 'blue',
        'surprise': 'magenta',
        'trust': 'brown'
    }

    # store these reused values
    album_name_1 = f"{album_data_1[0]['album']}"
    artist_1 = f"{album_data_1[0]['artist']}"
    album_name_2 = f"{album_data_2[0]['album']}"
    artist_2 = f"{album_data_2[0]['artist']}"

    # Extract emotion scores for each album
    album_scores_1 = {e: [] for e in emotions}
    album_scores_2 = {e: [] for e in emotions}

    for song in album_data_1:
        for e in emotions:
            score = song['normalized_emotion_scores' if use_norm else 'emotion_scores'].get(e, 0)
            album_scores_1[e].append(score)
    
    for song in album_data_2:
        for e in emotions:
            score = song['normalized_emotion_scores' if use_norm else 'emotion_scores'].get(e, 0)
            album_scores_2[e].append(score)
    
    # create DataFrame for each album
    df_1 = pd.DataFrame(album_scores_1)
    df_2 = pd.DataFrame(album_scores_2)

    avg_1 = df_1.mean().values.reshape(1, -1)
    avg_2 = df_2.mean().values.reshape(1, -1)
    similarity = cosine_similarity(avg_1, avg_2)[0][0]

    # Begin plotting
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f"A Lyrical Emotion Comparison: {album_name_1} by {artist_1} vs {album_name_2} by {artist_2}")

    # Mutli-line plot
    for emotion in emotions:
        axs[0, 0].plot(df_1.index, df_1[emotion], label=f"{album_name_1} - {emotion}", linestyle='-', marker='o', color=emotion_colors[emotion])
        axs[0, 0].plot(df_2.index, df_2[emotion], label=f"{album_name_2} - {emotion}", linestyle="--", marker='x', color=emotion_colors[emotion])

    axs[0, 0].set_title("Emotional Journey Comparison: Multi-line Plot")
    axs[0, 0].set_xlabel("Track Number")
    axs[0, 0].set_ylabel(f"Emotion Score ({'Normalized' if use_norm else 'Raw'})")
    legend_elems = []
    for emotion in emotions:
        legend_elems.append(Line2D([0], [0], color=emotion_colors[emotion], lw=2, label=emotion))
    axs[0, 0].legend(handles=legend_elems, loc='upper right', title=f"Emotions\nSolid line - {album_name_1}\nDashed line - {album_name_2}", fontsize=8)
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)

    # Heatmap for Album 1
    sns.heatmap(df_1.T, cmap="coolwarm", ax=axs[1, 0], cbar=True, annot=False, linewidths=0.5)
    axs[1, 0].set_title(f"Emotion Heatmap: {album_name_1} by {artist_1}")
    axs[1, 0].set_xlabel("Track Number")
    axs[1, 0].set_ylabel("Emotion")

    # Heatmap for Album 2
    sns.heatmap(df_2.T, cmap="coolwarm", ax=axs[1, 1], cbar=True, annot=False, linewidths=0.5)
    axs[1, 1].set_title(f"Emotion Heatmap: {album_name_2} by {artist_2}")
    axs[1, 1].set_xlabel("Track Number")
    axs[1, 1].set_ylabel("Emotion")

    # Cosine Similarity display
    axs[0, 1].axis("off")
    axs[0, 1].text(0.5, 0.5, f"Cosine Similarity\n{similarity:.4f}", fontsize=18, ha='center', va='center', bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=1'))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# Simple CLI for easy use
if __name__ == "__main__":

    # initialize genius object with API key
    genius = genius_setup()

    # Prompt 1: choose 1 album or two albums -- TODO: Enforce 2 options
    action = input("What would you like to do today?\n1) Analyze one album\n2) Compare two albums\nType 1 or 2: ")

    # set up reused collections
    emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    lex = load_nrc_lexicon("./NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")


    if action == '1': # 1 album

        # Retrieve search info -- TODO: Error handling!! 
        album_name = input("Search for album: ")
        artist_name = input("By artist: ")

        # Fetch album data and emotion scores
        album_data = fetch_and_analyze_album_lyrics(genius, album_name, artist_name, lex)

        # Prompt 2.1: Choose graph type 
        method = input(f"{album_data[0]['album']} by {album_data[0]['artist']} retrieved. How would you like to analyze this data?\n1) Line Graph\n2) Stacked Composition\n3) Heatmap\n4) Radar (of a single song)\nType a number 1-4: ")

        # Display correct graph -- TODO: Enforce 4 options
        if method == '1':
            plot_emotion_journey(album_data, emotions)
        elif method == '2':
            plot_stacked_emotion_composite(album_data, emotions)
        elif method == '3':
            plot_emotion_heatmap(album_data, emotions)
        elif method == '4':
            song = input("What song on this album would you like to analye? ")
            plot_emotion_radar(song, album_data, emotions)

    elif action == '2': # 2 albums
        # Retrieve search info
        album_one = input("Album 1: ")
        artist_one = input("By artist: ")

        album_two = input("Album 2: ")
        artist_two = input("By artist: ")

        # Fetch album data and emotion scores
        album_data_1 = fetch_and_analyze_album_lyrics(genius, album_one, artist_one, lex)
        album_data_2 = fetch_and_analyze_album_lyrics(genius, album_two, artist_two, lex)

        # Plot graphs
        plot_comparison(album_data_1, album_data_2, emotions)