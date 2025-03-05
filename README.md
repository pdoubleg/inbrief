# Weather Agent with MonsterUI

A modern, user-friendly weather assistant application built with MonsterUI and powered by OpenAI's GPT-4 model. Ask questions about the weather in natural language and get concise, accurate responses.

## Features

- 🌤️ Get real-time weather information for any location in the world
- 🤖 Natural language processing powered by OpenAI's GPT-4
- 🎨 Modern, responsive UI built with MonsterUI
- 🧪 API connection testing functionality
- 🌐 Geocoding API integration for location search
- 📊 Weather API integration for weather data

## Installation

### Prerequisites

- Python 3.10 or higher
- OpenAI API key
- Weather API key (from Tomorrow.io)
- Geocoding API key (from Maps.co)

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/weather-agent.git
cd weather-agent
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Set up environment variables

Create a `.env` file in the project root directory with the following content:

```
OPENAI_API_KEY=your_openai_api_key
WEATHER_API_KEY=your_weather_api_key
GEO_API_KEY=your_geo_api_key
```

Replace the placeholders with your actual API keys.

## Usage

### Running the Gradio version

```bash
python weather_agent_gradio.py
```

### Running the MonsterUI version

```bash
python weather_agent_monster.py
```

## Application Structure

- `weather_agent_gradio.py`: Original application using Gradio UI
- `weather_agent_monster.py`: New implementation using MonsterUI
- `requirements.txt`: List of Python dependencies

## API Keys

### Weather API (Tomorrow.io)

Sign up for a free API key at [Tomorrow.io](https://www.tomorrow.io/weather-api/).

### Geocoding API (Maps.co)

Sign up for a free API key at [Maps.co](https://geocode.maps.co/).

## How It Works

1. User enters a weather-related question
2. The agent processes the query using OpenAI's GPT-4 model
3. The agent uses the geocoding API to determine the location's coordinates
4. The agent retrieves weather data from the weather API
5. The result is displayed in a conversational format

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the GPT-4 model
- Tomorrow.io for the Weather API
- Maps.co for the Geocoding API
- The creators of MonsterUI and Gradio for the UI frameworks
