# Music Streaming Service Chatbot

An intelligent chatbot that helps users discover music, get song recommendations, and explore artist statistics based on various musical features and preferences.

## Features

- Song recommendations based on user preferences
- Artist statistics and information
- Finding similar songs
- Song search by musical features (energy, valence, danceability, etc.)
- Release date information
- Trending songs discovery

## Dependencies

To run this chatbot, you'll need Python 3.7+ and the following Python packages:

```
nltk==3.8.1
pandas==2.1.1
scikit-learn==1.3.0
numpy==1.24.3
fuzzywuzzy==0.18.0
python-Levenshtein==0.21.1  # Optional, but improves fuzzywuzzy performance
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/music-streaming-chatbot.git
cd music-streaming-chatbot
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage

1. Ensure your music dataset (data.csv) is in the correct location specified in the code.

2. Run the chatbot:
```bash
python chatbot-1.py
```

3. Follow the prompts to interact with the chatbot. You can:
   - Ask for song recommendations
   - Get information about artists
   - Find songs with specific musical features
   - Get release date information
   - Discover trending songs

## Data Format

The chatbot expects a CSV file with the following columns:
- name (song name)
- artists
- album (optional)
- release_date (optional)
- popularity
- valence
- energy
- danceability
- acousticness
- instrumentalness
- loudness
- speechiness
- tempo

## Example Interactions

```
You: recommend me songs
Chatbot: Great! I'd be happy to recommend some songs for you.
[Follows with questions about your music preferences]

You: tell me about artist The Beatles
Chatbot: [Provides statistics and information about The Beatles]

You: what are some high energy songs?
Chatbot: [Lists songs with high energy ratings]
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset source: [Add your dataset source here]
- Inspired by [Add any inspirations or references here]
