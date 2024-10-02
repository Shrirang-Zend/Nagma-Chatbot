# Nagma Music Chatbot

This project is an enhanced music chatbot designed to interact with users, providing song recommendations, artist information, trending songs, and more. It leverages a music dataset to answer queries related to songs, artists, and musical features.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Dataset](#dataset)
- [Contributing](#contributing)

## Features

- **Song Recommendations**: Recommends songs based on user preferences for various musical features.
- **Artist Information**: Provides statistics and information about artists.
- **Song Information**: Retrieves details about specific songs.
- **Trending Songs**: Lists trending songs based on popularity and release date.
- **Feature-Based Queries**: Handles queries about songs with specific musical features.
- **Interactive Chat**: Engages in a conversational manner with users.

## Prerequisites

- Python 3.7 or higher
- A music dataset in CSV format (see [Dataset](#dataset) section)

## Installation

1. **Clone the Repository**

   ```bash
   git clone [https://github.com/yourusername/your-repo.git](https://github.com/Shrirang-Zend/Nagma-Chatbot)
   cd your-repo
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   ```

   - **On Windows:**

     ```bash
     venv\Scripts\activate
     ```

   - **On macOS/Linux:**

     ```bash
     source venv/bin/activate
     ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Place Your Dataset**

   - Ensure your music dataset CSV file is placed in the project directory.
   - Update the `DATA_PATH` in `config.py` to point to your dataset file.

2. **Run the Chatbot**

   ```bash
   python chatbot.py
   ```

3. **Interact with the Chatbot**

   - Type your queries into the console.
   - Examples of things you can ask:
     - "Recommend some songs for me."
     - "Tell me about [song name]."
     - "Who is [artist name]?"
     - "What are the current trending songs?"
   - Type `'exit'` to end the chat session.

## Project Structure

- `chatbot.py`: Main script to run the chatbot.
- `music_chatbot.py`: Contains the `MusicChatbot` class with all chatbot functionalities.
- `utils.py`: Utility functions used by the chatbot.
- `responses.py`: Predefined responses and explanations.
- `config.py`: Configuration file for setting paths and other constants.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.

## Configuration

- **`config.py`**

  Update the `DATA_PATH` variable in `config.py` to point to your dataset CSV file.

  ```python
  # config.py

  DATA_PATH = 'data.csv'  # Update this path to the location of your dataset
  ```

## Dataset

You can use the same dataset as me OR need a music dataset in CSV format containing at least the following columns:

- `name`: Song name
- `artists`: Artist name(s)
- `release_date`: Release date of the song
- `popularity`: Popularity metric (0-100)
- Musical feature columns such as `valence`, `energy`, `danceability`, `acousticness`, `instrumentalness`, `speechiness`, `tempo`, etc.

**Note**: The dataset should be compatible with the features used in the chatbot code. If your dataset has different column names, you may need to adjust the code accordingly.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements, bug fixes, or new features.
