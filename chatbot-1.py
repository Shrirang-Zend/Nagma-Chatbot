import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from fuzzywuzzy import process, fuzz
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from collections import defaultdict
from response_1 import responses
from datetime import datetime

# Uncomment these lines if you need to download the required NLTK data files for the first time
# nltk.download('punkt')
# nltk.download('stopwords')

# Load the music dataset
def load_dataset():
    df = pd.read_csv('./data.csv')
    if 'release_date' in df.columns:
        df['release_date'] = df['release_date'].apply(parse_release_date)
    return df

# Helper functions
def parse_release_date(date_str):
    try:
        return pd.to_datetime(date_str)
    except ValueError:
        return pd.to_datetime(f"{date_str}-01-01")
    
def parse_feature_query(query):
    """Parse queries about musical features, handling both ranges and qualitative descriptors."""
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
            # First, check for exact range
            numeric_range = extract_numeric_range(query)
            if numeric_range:
                return feature, numeric_range
            
            # Then, check for qualitative descriptors
            for descriptor, range_values in ranges.items():
                if descriptor in query:
                    return feature, range_values
            
            # If feature is mentioned but no range, assume high
            return feature, ranges['high']
    
    return None, None

def improve_song_matching(query, df):
    """Improved function to match song names in queries."""
    # List of words that might indicate a release date query
    release_indicators = ['when', 'release', 'came out']
    
    # Clean up the query by removing release date indicators
    clean_query = query
    for indicator in release_indicators:
        clean_query = clean_query.replace(indicator, '').strip()
    
    # Try exact match first (after cleaning the query)
    exact_match = df[df['name'].str.lower() == clean_query.lower()]
    if not exact_match.empty:
        return exact_match.iloc[0]
    
    # If no exact match, try fuzzy matching
    best_match = None
    best_score = 0
    
    for _, row in df.iterrows():
        song_name = row['name'].lower()
        artist_name = row['artists'].lower()
        
        # Fuzzy match the song name and artist name from the query
        song_ratio = fuzz.ratio(clean_query.lower(), song_name)
        partial_ratio = fuzz.partial_ratio(clean_query.lower(), song_name)
        token_sort_ratio = fuzz.token_sort_ratio(clean_query.lower(), song_name)
        token_set_ratio = fuzz.token_set_ratio(clean_query.lower(), song_name)
        
        # Also consider artist name in matching
        artist_ratio = fuzz.token_set_ratio(clean_query.lower(), artist_name)
        
        # Use the highest score from the fuzzy matching methods
        max_score = max(song_ratio, partial_ratio, token_sort_ratio, token_set_ratio, artist_ratio)
        
        # Update best match if the current score is higher
        if max_score > best_score and max_score >= 70:  # You can adjust this threshold
            best_score = max_score
            best_match = row
    
    return best_match

def format_release_date(date):
    if pd.notnull(date):
        if isinstance(date, str):
            date = pd.to_datetime(date)
        if date.strftime('%m-%d') == '01-01':
            return date.strftime('%Y')
        else:
            return date.strftime('%Y-%m-%d')
    return 'Unknown'

def fuzzy_match_song(query, df, threshold=80):
    matches = []
    for _, row in df.iterrows():
        song_name = row['name'].lower()
        artist_name = row['artists'].lower()
        
        song_score = fuzz.token_set_ratio(query.lower(), song_name)
        artist_score = fuzz.token_set_ratio(query.lower(), artist_name)
        
        max_score = max(song_score, artist_score)
        
        if max_score >= threshold:
            matches.append({
                'row': row,
                'score': max_score
            })
    
    return sorted(matches, key=lambda x: x['score'], reverse=True)

def extract_numeric_range(text):
    patterns = [
        r'(\d+)\s*-\s*(\d+)',
        r'between\s+(\d+)\s+and\s+(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return sorted([int(match.group(1)), int(match.group(2))])
    return None

# Feature-based functions
def get_user_preferences():
    print("To recommend songs, I need to know your preferences. Please rate the following on a scale of 0 to 1:")
    preferences = {}
    features = [
        'valence', 
        'energy', 
        'danceability', 
        'acousticness',
        'instrumentalness',
        'speechiness'
    ]
    
    for feature in features:
        while True:
            try:
                value = float(input(f"{feature.capitalize()} (0-1): "))
                if 0 <= value <= 1:
                    preferences[feature] = value
                    break
                else:
                    print("Please enter a value between 0 and 1.")
            except ValueError:
                print("Please enter a valid number.")
    
    # Handle tempo separately as it uses a different scale
    while True:
        try:
            tempo = float(input("Tempo (typically 50-200 BPM): "))
            if 0 <= tempo <= 300:  # allowing a wide range for tempo
                preferences['tempo'] = tempo
                break
            else:
                print("Please enter a tempo between 0 and 300 BPM.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Loudness is typically between -60 and 0 dB
    while True:
        try:
            loudness = float(input("Loudness (-60 to 0 dB): "))
            if -60 <= loudness <= 0:
                preferences['loudness'] = loudness
                break
            else:
                print("Please enter a loudness value between -60 and 0 dB.")
        except ValueError:
            print("Please enter a valid number.")
    
    return preferences

def get_songs_by_feature_range(feature, min_val, max_val, df):
    if feature not in df.columns:
        return f"I'm sorry, I don't have information about {feature} for the songs."
    
    filtered_songs = df[(df[feature] >= min_val) & (df[feature] <= max_val)]
    return filtered_songs.sort_values('popularity', ascending=False).head(5)

# Song recommendation functions
def recommend_songs(preferences, df, n=5):
    # Only use the features that are present in both the dataset and user preferences
    available_features = [col for col in preferences.keys() if col in df.columns]
    
    if not available_features:
        return "I'm sorry, but I don't have enough feature information to make recommendations."
    
    # Normalize the data
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df[available_features]), columns=available_features)
    
    # Prepare the user preferences for normalization
    user_profile = pd.DataFrame([{k: v for k, v in preferences.items() if k in available_features}])
    
    # Normalize user preferences
    user_normalized = pd.DataFrame(scaler.transform(user_profile), columns=available_features)
    
    # Calculate cosine similarity between user preferences and songs in the dataset
    similarity = cosine_similarity(user_normalized, df_normalized)
    
    # Get the indices of the most similar songs
    similar_indices = similarity.argsort()[0][-2*n:][::-1]
    
    # Select `n` random songs from the most similar ones
    recommended_indices = np.random.choice(similar_indices, n, replace=False)
    
    # Return the song recommendations
    recommendations = df.iloc[recommended_indices]
    return recommendations[['name', 'artists'] + (['album'] if 'album' in df.columns else []) + (['release_date'] if 'release_date' in df.columns else [])].to_dict('records')

def find_similar_songs(song_name, df, n=5):
    song = df[df['name'].str.lower() == song_name.lower()]
    if song.empty:
        fuzzy_matches = fuzzy_match_song(song_name, df)
        if not fuzzy_matches:
            return f"I couldn't find '{song_name}' or any similar songs."
        song = pd.DataFrame([fuzzy_matches[0]['row']])
    
    features = ['valence', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness', 'speechiness', 'tempo']
    available_features = [f for f in features if f in df.columns]
    
    if not available_features:
        return "I don't have enough feature information to find similar songs."
    
    song_vector = song[available_features].values
    df_features = df[available_features].values
    
    similarities = cosine_similarity(song_vector, df_features)
    similar_indices = similarities[0].argsort()[-n-1:-1][::-1]
    similar_songs = df.iloc[similar_indices]
    
    return similar_songs[['name', 'artists'] + (['album'] if 'album' in df.columns else [])].to_dict('records')

# Query handling functions
def get_songs_by_release_date(date_str, df):
    if 'release_date' not in df.columns:
        return "I'm sorry, I don't have release date information for the songs."
    
    try:
        date = pd.to_datetime(date_str)
    except ValueError:
        return f"I'm sorry, '{date_str}' is not a valid date format. Please use YYYY-MM-DD."
    
    released_songs = df[df['release_date'] == date]
    
    if released_songs.empty:
        return f"I couldn't find any songs released exactly on {date.strftime('%Y-%m-%d')}. Would you like to search for songs released in the same month or year?"
    
    return released_songs[['name', 'artists'] + (['album'] if 'album' in df.columns else [])].to_dict('records')

def get_trending_songs(df):
    if 'popularity' not in df.columns:
        return "I'm sorry, I can't determine trending songs without popularity data."
    
    if 'release_date' in df.columns:
        current_date = datetime.now()
        last_month = current_date - pd.DateOffset(months=1)
        recent_songs = df[df['release_date'] > last_month]
        
        if len(recent_songs) > 0:
            trending = recent_songs.sort_values('popularity', ascending=False).head(5)
        else:
            songs_2020 = df[(df['release_date'] >= '2020-01-01') & (df['release_date'] <= '2020-12-31')]
            trending = songs_2020.sort_values('popularity', ascending=False).head(5)
    else:
        trending = df.sort_values('popularity', ascending=False).head(5)
    
    return trending[['name', 'artists', 'popularity'] + (['album'] if 'album' in df.columns else []) + (['release_date'] if 'release_date' in df.columns else [])].to_dict('records')

def get_popular_songs_by_artist(artist_name, df):
    artist_songs = df[df['artists'].str.contains(artist_name, case=False)]
    if artist_songs.empty:
        return f"I'm sorry, I couldn't find any songs by {artist_name}."
    
    popular_songs = artist_songs.sort_values('popularity', ascending=False).head(5)
    
    return popular_songs[['name', 'artists'] + (['album'] if 'album' in df.columns else []) + (['release_date'] if 'release_date' in df.columns else [])].to_dict('records')

def get_song_info(song_name, df):
    song = df[df['name'].str.lower() == song_name.lower()]
    if song.empty:
        fuzzy_matches = fuzzy_match_song(song_name, df)
        if not fuzzy_matches:
            return f"I'm sorry, I couldn't find information about {song_name}."
        song = pd.DataFrame([fuzzy_matches[0]['row']])
    
    song = song.iloc[0]
    info = f"Song: {song['name']}\n"
    info += f"Artist: {song['artists']}\n"
    if 'album' in df.columns:
        info += f"Album: {song['album']}\n"
    if 'release_date' in df.columns:
        info += f"Release Date: {format_release_date(song['release_date'])}\n"
    if 'popularity' in df.columns:
        info += f"Popularity: {song['popularity']}/100\n"
    
    features = ['tempo', 'speechiness', 'valence', 'energy', 'danceability', 'acousticness']
    for feature in features:
        if feature in df.columns:
            info += f"{feature.capitalize()}: {song[feature]:.2f}\n"
    
    return info

def get_artist_stats(artist_name, df):
    artist_songs = df[df['artists'].str.contains(artist_name, case=False)]
    if artist_songs.empty:
        return f"I couldn't find any songs by {artist_name}."
    
    stats = {
        'total_songs': len(artist_songs),
        'avg_popularity': artist_songs['popularity'].mean() if 'popularity' in df.columns else None,
        'most_recent': artist_songs['release_date'].max() if 'release_date' in df.columns else None,
    }
    
    feature_stats = {}
    for feature in ['valence', 'energy', 'danceability', 'tempo']:
        if feature in df.columns:
            feature_stats[feature] = {
                'mean': artist_songs[feature].mean(),
                'min': artist_songs[feature].min(),
                'max': artist_songs[feature].max()
            }
    
    return {'basic_stats': stats, 'feature_stats': feature_stats}

def format_artist_info(artist_info, artist_name):
    if isinstance(artist_info, str):
        return artist_info
    
    basic_stats = artist_info['basic_stats']
    feature_stats = artist_info['feature_stats']
    
    response = f"Here's what I know about {artist_name}:\n"
    response += f"- Total songs in the database: {basic_stats['total_songs']}\n"
    
    if basic_stats['avg_popularity'] is not None:
        response += f"- Average popularity: {basic_stats['avg_popularity']:.2f}/100\n"
    
    if basic_stats['most_recent'] is not None:
        response += f"- Most recent release: {format_release_date(basic_stats['most_recent'])}\n"
    
    response += "\nMusical features:\n"
    for feature, stats in feature_stats.items():
        response += f"- {feature.capitalize()}:\n"
        response += f"  Average: {stats['mean']:.2f}\n"
        response += f"  Range: {stats['min']:.2f} - {stats['max']:.2f}\n"
    
    return response

def get_response(user_input, df):
    user_input = user_input.lower()
    
    #recommendation
    if "recommend" in user_input and "song" in user_input:
        print("Great! I'd be happy to recommend some songs for you.")
        user_preferences = get_user_preferences()
        recommended_songs = recommend_songs(user_preferences, df)
        response = "Based on your preferences, here are some song recommendations:\n"
        for song in recommended_songs:
            response += f"- {song['name']} by {song['artists']}"
            if 'album' in song:
                response += f" from the album '{song['album']}'"
            if 'release_date' in song:
                response += f" (Released: {format_release_date(song['release_date'])})"
            response += "\n"
        return response
    
    # Handle release date queries
    if any(word in user_input for word in ['when', 'release date', 'came out']):
        song = improve_song_matching(user_input, df)
        if song is not None:
            if 'release_date' in df.columns:
                return f"{song['name']} by {song['artists']} was released on {format_release_date(song['release_date'])}"
            else:
                return f"I found {song['name']} by {song['artists']}, but I don't have release date information."
    
    # Handle trending songs query
    if "trending" in user_input or "current popular songs" in user_input:
        trending_songs = get_trending_songs(df)
        if isinstance(trending_songs, str):
            return trending_songs
        response = "Here are some currently trending songs:\n"
        for song in trending_songs:
            response += f"- {song['name']} by {song['artists']}"
            if 'popularity' in song:
                response += f" (Popularity: {song['popularity']})"
            if 'release_date' in song:
                response += f" Released on: {format_release_date(song['release_date'])}"
            response += "\n"
        return response
    
    # Handle feature queries
    feature, range_values = parse_feature_query(user_input)
    if feature:
        songs = get_songs_by_feature_range(feature, range_values[0], range_values[1], df)
        if isinstance(songs, str):
            return songs
        
        response = f"Here are some songs with {feature} between {range_values[0]} and {range_values[1]}:\n"
        for _, song in songs.iterrows():
            response += f"- {song['name']} by {song['artists']} ({feature}: {song[feature]:.2f})\n"
        return response
    
    # Handle artist queries
    if "who is" in user_input or "tell me about artist" in user_input:
        artist_name = user_input.split("who is")[-1].strip() if "who is" in user_input else user_input.split("tell me about artist")[-1].strip()
        artist_info = get_artist_stats(artist_name, df)
        return format_artist_info(artist_info, artist_name)
    
    # Handle song information queries
    if "tell me about" in user_input:
        song_name = user_input.split("tell me about")[-1].strip()
        return get_song_info(song_name, df)
    
    # Fallback: Attempt to match song names
    potential_song = improve_song_matching(user_input, df)
    if potential_song is not None:
        return f"I found this song: {potential_song['name']} by {potential_song['artists']}. What would you like to know about it?"
    
    # Existing fallback: use response dictionary or fuzzy match
    matched_intent, score = process.extractOne(user_input, responses.keys())
    if score >= 70:
        return responses[matched_intent]
    
    return "I'm not sure what you're asking. Could you rephrase your question?"

# Main chatbot function
def chatbot():
    print("Welcome to the Enhanced Music Chatbot!")
    print("I can help you with:")
    print("- Finding similar songs")
    print("- Getting artist statistics")
    print("- Finding songs by musical features")
    print("- Release date information")
    print("And more! Type 'exit' to end the chat.")
    
    df = load_dataset()
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        
        response = get_response(user_input, df)
        print("Chatbot:", response)

if __name__ == "__main__":
    chatbot()
