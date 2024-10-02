import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from rapidfuzz import process, fuzz  # Using RapidFuzz for better performance
from responses import responses
from utils import parse_release_date, format_release_date, extract_numeric_range

class NagmaChatbot:
    """
    A chatbot for providing music recommendations and information.
    """

    def __init__(self, data_path):
        """
        Initializes the MusicChatbot with a dataset.

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

    def parse_feature_query(self, query):
        """
        Parses queries about musical features.

        Parameters:
            query (str): User input query.

        Returns:
            tuple: (feature, range_values)
        """
        features = {
            'energy': {'high': (0.7, 1.0), 'medium': (0.4, 0.7), 'low': (0, 0.4)},
            'valence': {'high': (0.7, 1.0), 'medium': (0.4, 0.7), 'low': (0, 0.4)},
            'danceability': {'high': (0.7, 1.0), 'medium': (0.4, 0.7), 'low': (0, 0.4)},
            'acousticness': {'high': (0.7, 1.0), 'medium': (0.4, 0.7), 'low': (0, 0.4)},
            'instrumentalness': {'high': (0.7, 1.0), 'medium': (0.4, 0.7), 'low': (0, 0.4)},
            'speechiness': {'high': (0.7, 1.0), 'medium': (0.4, 0.7), 'low': (0, 0.4)},
            'tempo': {'high': (120, 300), 'medium': (76, 120), 'low': (0, 76)}
        }
        
        for feature, ranges in features.items():
            if feature in query:
                # Check for exact numeric range
                numeric_range = extract_numeric_range(query)
                if numeric_range:
                    return feature, numeric_range
                
                # Check for qualitative descriptors
                for descriptor, range_values in ranges.items():
                    if descriptor in query:
                        return feature, range_values
                
                # Default to 'high' if no range specified
                return feature, ranges['high']
        
        return None, None

    def improve_song_matching(self, query):
        """
        Matches song names in queries using fuzzy matching.

        Parameters:
            query (str): User input query.

        Returns:
            pd.Series: Matched song row.
        """
        # Clean up the query
        release_indicators = ['when', 'release', 'came out']
        clean_query = query
        for indicator in release_indicators:
            clean_query = clean_query.replace(indicator, '').strip()
        
        # Try exact match
        exact_match = self.df[self.df['name'].str.lower() == clean_query.lower()]
        if not exact_match.empty:
            return exact_match.iloc[0]
        
        # Fuzzy matching
        matches = process.extract(
            clean_query.lower(), 
            self.df['name'].str.lower(), 
            scorer=fuzz.token_set_ratio, 
            limit=5
        )
        for match in matches:
            if match[1] >= 70:
                song = self.df.iloc[match[2]]
                return song
        
        return None

    def get_user_preferences_via_chat(self):
        """
        Collects user preferences through conversational prompts.

        Returns:
            dict: User preferences for features.
        """
        preferences = {}
        features = [
            'valence', 
            'energy', 
            'danceability', 
            'acousticness',
            'instrumentalness',
            'speechiness',
            'tempo',
            'loudness'
        ]
        
        print("To recommend songs, I need to know your preferences.")
        for feature in features:
            while True:
                user_input = input(f"Please rate {feature} on a scale (e.g., 0.5): ")
                try:
                    value = float(user_input)
                    if feature == 'tempo':
                        if 0 <= value <= 300:
                            preferences[feature] = value
                            break
                        else:
                            print("Tempo should be between 0 and 300 BPM.")
                    elif feature == 'loudness':
                        if -60 <= value <= 0:
                            preferences[feature] = value
                            break
                        else:
                            print("Loudness should be between -60 and 0 dB.")
                    else:
                        if 0 <= value <= 1:
                            preferences[feature] = value
                            break
                        else:
                            print(f"{feature.capitalize()} should be between 0 and 1.")
                except ValueError:
                    print("Please enter a valid number.")
        return preferences

    def recommend_songs(self, preferences, n=5):
        """
        Recommends songs based on user preferences.

        Parameters:
            preferences (dict): User preferences for features.
            n (int): Number of songs to recommend.

        Returns:
            list: List of recommended songs.
        """
        available_features = [col for col in preferences.keys() if col in self.df.columns]
        if not available_features:
            return "I'm sorry, but I don't have enough feature information to make recommendations."

        # Normalize data
        scaler = MinMaxScaler()
        df_normalized = pd.DataFrame(scaler.fit_transform(self.df[available_features]), columns=available_features)
        user_profile = pd.DataFrame([preferences])[available_features]
        user_normalized = pd.DataFrame(scaler.transform(user_profile), columns=available_features)

        # Calculate cosine similarity
        similarity = cosine_similarity(user_normalized, df_normalized)
        similar_indices = similarity.argsort()[0][-2*n:][::-1]
        recommended_indices = np.random.choice(similar_indices, n, replace=False)
        recommendations = self.df.iloc[recommended_indices]
        return recommendations[['name', 'artists', 'album', 'release_date']].to_dict('records')

    def get_trending_songs(self):
        """
        Retrieves trending songs based on popularity.

        Returns:
            list: List of trending songs.
        """
        if 'popularity' not in self.df.columns:
            return "I'm sorry, I can't determine trending songs without popularity data."

        if 'release_date' in self.df.columns:
            current_date = datetime.now()
            last_month = current_date - pd.DateOffset(months=1)
            recent_songs = self.df[self.df['release_date'] > last_month]
            if not recent_songs.empty:
                trending = recent_songs.nlargest(5, 'popularity')
            else:
                trending = self.df.nlargest(5, 'popularity')
        else:
            trending = self.df.nlargest(5, 'popularity')

        return trending[['name', 'artists', 'popularity', 'album', 'release_date']].to_dict('records')

    def get_response(self, user_input):
        """
        Generates a response based on user input.

        Parameters:
            user_input (str): User input.

        Returns:
            str: Chatbot response.
        """
        user_input = user_input.lower()

        if "recommend" in user_input and "song" in user_input:
            print("Great! I'd be happy to recommend some songs for you.")
            preferences = self.get_user_preferences_via_chat()
            recommended_songs = self.recommend_songs(preferences)
            response = "Based on your preferences, here are some song recommendations:\n"
            for song in recommended_songs:
                response += f"- {song['name']} by {song['artists']}"
                if song.get('album'):
                    response += f" from the album '{song['album']}'"
                if song.get('release_date'):
                    response += f" (Released: {format_release_date(song['release_date'])})"
                response += "\n"
            return response

        # Additional handlers for other intents...

        # Fallback response
        matched_intent, score = process.extractOne(user_input, responses.keys(), scorer=fuzz.token_set_ratio)
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