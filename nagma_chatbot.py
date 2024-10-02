import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from rapidfuzz import process, fuzz  # Using RapidFuzz for better performance
from responses import responses, intents
from utils import parse_release_date, format_release_date, extract_numeric_range

class NagmaChatbot:
    """
    A chatbot for providing music recommendations and information.
    """

    def __init__(self, data_path):
        """
        Initializes the NagmaChatbot with a dataset.

        Parameters:
            data_path (str): Path to the CSV file containing music data.
        """
        self.df = self.load_dataset(data_path)
        self.user_preferences = {}
        self.context = {}

    def load_dataset(self, path):
        """
        Loads the dataset from a CSV file.

        Parameters:
            path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded dataset.
        """
        try:
            df = pd.read_csv(path)
            if 'release_date' in df.columns:
                df['release_date'] = df['release_date'].apply(parse_release_date)
            return df
        except FileNotFoundError:
            print(f"Error: The file at {path} was not found.")
            return pd.DataFrame()
        except pd.errors.ParserError:
            print("Error: The file could not be parsed as a CSV.")
            return pd.DataFrame()

    # ... [Other methods remain the same] ...

    def get_artist_stats(self, artist_name):
        """
        Retrieves statistics and information about a specific artist.

        Parameters:
            artist_name (str): Name of the artist.

        Returns:
            dict or str: Dictionary containing artist stats or an error message.
        """
        artist_songs = self.df[self.df['artists'].str.contains(artist_name, case=False, na=False)]
        if artist_songs.empty:
            return f"I couldn't find any songs by {artist_name} in my database."

        stats = {
            'total_songs': len(artist_songs),
            'avg_popularity': artist_songs['popularity'].mean() if 'popularity' in self.df.columns else None,
            'most_recent': artist_songs['release_date'].max() if 'release_date' in self.df.columns else None,
        }

        feature_stats = {}
        for feature in ['valence', 'energy', 'danceability', 'tempo', 'acousticness', 'instrumentalness', 'speechiness', 'loudness']:
            if feature in self.df.columns:
                feature_stats[feature] = {
                    'mean': artist_songs[feature].mean(),
                    'min': artist_songs[feature].min(),
                    'max': artist_songs[feature].max()
                }

        return {'basic_stats': stats, 'feature_stats': feature_stats}

    def format_artist_info(self, artist_info, artist_name):
        """
        Formats artist statistics into a readable string.

        Parameters:
            artist_info (dict or str): Artist statistics or error message.
            artist_name (str): Name of the artist.

        Returns:
            str: Formatted artist information.
        """
        if isinstance(artist_info, str):
            return artist_info  # Return the error message

        basic_stats = artist_info['basic_stats']
        feature_stats = artist_info['feature_stats']

        response = f"Here's what I know about {artist_name}:\n"
        response += f"- Total songs in the database: {basic_stats['total_songs']}\n"

        if basic_stats['avg_popularity'] is not None:
            response += f"- Average popularity: {basic_stats['avg_popularity']:.2f}/100\n"

        if basic_stats['most_recent'] is not None and pd.notnull(basic_stats['most_recent']):
            response += f"- Most recent release: {format_release_date(basic_stats['most_recent'])}\n"

        response += "\nMusical features:\n"
        for feature, stats in feature_stats.items():
            response += f"- {feature.capitalize()}:\n"
            response += f"  Average: {stats['mean']:.2f}\n"
            response += f"  Range: {stats['min']:.2f} - {stats['max']:.2f}\n"

        return response

    def get_response(self, user_input):
        """
        Generates a response based on user input.

        Parameters:
            user_input (str): User input.

        Returns:
            str: Chatbot response.
        """
        user_input = user_input.lower()

        # Iterate over intents
        for intent_name, phrases in intents.items():
            # Perform fuzzy matching between user input and intent phrases
            matched_phrase, score, _ = process.extractOne(
                user_input, phrases, scorer=fuzz.token_set_ratio
            )
            if score >= 70:
                if intent_name == 'recommend_songs':
                    # Handle song recommendation
                    print("Great! I'd be happy to recommend some songs for you.")
                    preferences = self.get_user_preferences_via_chat()
                    recommended_songs = self.recommend_songs(preferences)
                    if isinstance(recommended_songs, str):
                        return recommended_songs
                    response = "Based on your preferences, here are some song recommendations:\n"
                    for song in recommended_songs:
                        response += f"- {song['name']} by {song['artists']}"
                        if song.get('album'):
                            response += f" from the album '{song['album']}'"
                        if song.get('release_date'):
                            response += f" (Released: {format_release_date(song['release_date'])})"
                        response += "\n"
                    return response

                elif intent_name == 'artist_information':
                    # Extract artist name from user input
                    artist_name = user_input
                    for phrase in phrases:
                        if phrase in user_input:
                            artist_name = user_input.replace(phrase, '').strip()
                            break
                    artist_info = self.get_artist_stats(artist_name)
                    return self.format_artist_info(artist_info, artist_name)

                elif intent_name == 'song_information':
                    # Extract song name from user input
                    song_name = user_input
                    for phrase in phrases:
                        if phrase in user_input:
                            song_name = user_input.replace(phrase, '').strip()
                            break
                    song_info = self.get_song_info(song_name)
                    return song_info

                elif intent_name == 'trending_songs':
                    # Handle trending songs request
                    trending_songs = self.get_trending_songs()
                    if isinstance(trending_songs, str):
                        return trending_songs
                    response = "Here are the current trending songs:\n"
                    for song in trending_songs:
                        response += f"- {song['name']} by {song['artists']}"
                        if 'popularity' in song:
                            response += f" (Popularity: {song['popularity']})"
                        response += "\n"
                    return response

                # Add more intent handling here as needed

        # Fallback response using the responses dictionary
        matched_intent, score, _ = process.extractOne(
            user_input, responses.keys(), scorer=fuzz.token_set_ratio
        )
        if score >= 70:
            return responses[matched_intent]

        return "I'm not sure what you're asking. Could you rephrase your question?"

    def run(self):
        """
        Starts the chatbot interaction loop.
        """
        print("Welcome to the Enhanced Music Chatbot!")
        print("I can help you with:")
        print("- Finding similar songs")
        print("- Getting artist statistics")
        print("- Finding songs by musical features")
        print("- Release date information")
        print("And more! Type 'exit' to end the chat.")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Chatbot: Goodbye!")
                break
            response = self.get_response(user_input)
            print("Chatbot:", response)