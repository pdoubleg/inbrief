
from dataclasses import dataclass
import os
import re
import base64
import openai
import hashlib
import requests
import numpy as np
from PIL import Image
import streamlit as st
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import streamlit_shadcn_ui as ui
from urllib.parse import urlparse
from typing import Any, Dict, Literal, Optional
from httpx import AsyncClient
from pydantic_ai import Agent, ModelRetry, RunContext
from openai.types.images_response import ImagesResponse
from pydantic import BaseModel, ConfigDict, HttpUrl, model_validator
from streamlit_extras.add_vertical_space import add_vertical_space as avs


# ------------------ weather agent ------------------

# Load environment variables
load_dotenv()


@dataclass
class Deps:
    """
    Dependencies container for the weather agent.
    
    Attributes:
        client: Async HTTP client for making API requests
        weather_api_key: API key for the weather service
        geo_api_key: API key for the geocoding service
    """
    client: AsyncClient
    weather_api_key: str | None
    geo_api_key: str | None
    


# Initialize the weather agent with OpenAI GPT-4
weather_agent = Agent(
    'openai:gpt-4o',
    # System prompt to guide the agent's behavior
    system_prompt=(
        'You are a helpful weather assistant. '
        'Use the `get_lat_lng` tool to get the latitude and longitude of the locations, '
        'then use the the other tools to help with the user weather needs. '
        'Output should be in markdown format.'
    ),
    deps_type=Deps,
    result_type=str,
    retries=4,
)


@weather_agent.tool
async def get_lat_lng(
    ctx: RunContext[Deps], location_description: str
) -> dict[str, float]:
    """
    Get the latitude and longitude of a location.

    Args:
        location_description: A description of a location (city name, address, etc.)

    Returns:
        A dictionary containing the latitude and longitude

    Raises:
        ModelRetry: If the location cannot be found
        
    Example:
        >>> await get_lat_lng(ctx, "New York City")
        {'lat': 40.7128, 'lng': -74.0060}
    """
    if ctx.deps.geo_api_key is None:
        # If no API key is provided, return a dummy response (London)
        return {'lat': 51.1, 'lng': -0.1}

    params = {
        'q': location_description,
        'api_key': ctx.deps.geo_api_key,
    }

    r = await ctx.deps.client.get('https://geocode.maps.co/search', params=params)
    r.raise_for_status()
    data = r.json()

    if data:
        # Extract coordinates from the first result
        return {'lat': float(data[0]['lat']), 'lng': float(data[0]['lon'])}
    else:
        raise ModelRetry('Could not find the location')


@weather_agent.tool
async def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
    """
    Get the weather at a location.

    Args:
        lat: Latitude of the location
        lng: Longitude of the location

    Returns:
        A dictionary containing weather information including temperature and description

    Example:
        >>> await get_weather(ctx, 40.7128, -74.0060)
        {'temperature': '21°C', 'description': 'Mostly Clear'}
    """
    if ctx.deps.weather_api_key is None:
        # If no API key is provided, return a dummy response
        return {'temperature': '21 °C', 'description': 'Sunny'}

    params = {
        'apikey': ctx.deps.weather_api_key,
        'location': f'{lat},{lng}',
        'units': 'metric',
    }
    r = await ctx.deps.client.get(
        'https://api.tomorrow.io/v4/weather/realtime', params=params
    )
    r.raise_for_status()
    data = r.json()

    values = data['data']['values']
    # Map weather codes to human-readable descriptions
    # https://docs.tomorrow.io/reference/data-layers-weather-codes
    code_lookup = {
        1000: 'Clear, Sunny',
        1100: 'Mostly Clear',
        1101: 'Partly Cloudy',
        1102: 'Mostly Cloudy',
        1001: 'Cloudy',
        2000: 'Fog',
        2100: 'Light Fog',
        4000: 'Drizzle',
        4001: 'Rain',
        4200: 'Light Rain',
        4201: 'Heavy Rain',
        5000: 'Snow',
        5001: 'Flurries',
        5100: 'Light Snow',
        5101: 'Heavy Snow',
        6000: 'Freezing Drizzle',
        6001: 'Freezing Rain',
        6200: 'Light Freezing Rain',
        6201: 'Heavy Freezing Rain',
        7000: 'Ice Pellets',
        7101: 'Heavy Ice Pellets',
        7102: 'Light Ice Pellets',
        8000: 'Thunderstorm',
    }
    return {
        'temperature': f'{values["temperatureApparent"]:0.0f}°C',
        'description': code_lookup.get(values['weatherCode'], 'Unknown'),
    }

@weather_agent.tool
def get_local_alerts(tx: RunContext[Deps], lat: float, lng: float) -> str:
    """
    Get local weather alerts for a location.

    Args:
        lat: Latitude of the location
        lng: Longitude of the location

    Returns:
        str: Markdown-formatted string containing the weather alerts.
    """
    response = requests.get(f'https://api.weather.gov/alerts?point={lat},{lng}').json()
    markdown_output = ""
    for x in response['features'][:1]:
        markdown_output += f"# {x['properties']['headline']}\n\n"
        markdown_output += f"### {x['properties']['areaDesc']}\n\n"
        markdown_output += f"{x['properties']['description']}\n\n"
        markdown_output += "---\n\n"
    return markdown_output


@weather_agent.tool
def get_detailed_weather(ctx: RunContext[Deps], search_query: str, search_type: Literal["city", "zip"]) -> str:
    """
    Fetch detailed weather for a city or zip code. City name must be city only, no state or country.
    
    Args:
        search_query: The city name or zip code
        search_type: The type of location to search for
        
    Returns:
        WeatherData: The weather for the location
    """
    API_key = os.getenv("OPENWEATHERMAPAPIKEY")
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    
    if search_type == "city":
        final_url = f"{base_url}appid={API_key}&q={search_query}"
    elif search_type == "zip":
        final_url = f"{base_url}appid={API_key}&zip={search_query}"
    else:
        raise ValueError(f"Invalid search type: {search_type}. Must be either 'city' or 'zip'.")
    
    owm_response_json = requests.get(final_url).json()
    
    sunset_utc = datetime.fromtimestamp(owm_response_json["sys"]["sunset"])
    sunset_local = sunset_utc.strftime("%I:%M %p")
    
    temp_celsius = owm_response_json["main"]["temp"] - 273.15
    temp_max_celsius = owm_response_json["main"]["temp_max"] - 273.15
    temp_min_celsius = owm_response_json["main"]["temp_min"] - 273.15
    temp_feels_like_celsius = owm_response_json["main"]["feels_like"] - 273.15

    temp_fahrenheit = round(celsius_to_fahrenheit(temp_celsius), 2)
    temp_max_fahrenheit = round(celsius_to_fahrenheit(temp_max_celsius), 2)
    temp_min_fahrenheit = round(celsius_to_fahrenheit(temp_min_celsius), 2)
    temp_feels_like_fahrenheit = round(celsius_to_fahrenheit(temp_feels_like_celsius), 2)

    rain = owm_response_json.get("rain", "No rain")
    
    owm_dict = {
        "temp": temp_fahrenheit,
        "temp_max": temp_max_fahrenheit,
        "temp_min": temp_min_fahrenheit,
        "feels_like": temp_feels_like_fahrenheit,
        "description": owm_response_json["weather"][0]["description"],
        "icon": owm_response_json["weather"][0]["icon"],
        "wind_speed": owm_response_json["wind"]["speed"],
        "wind_direction": owm_response_json["wind"]["deg"],
        "humidity": owm_response_json["main"]["humidity"],
        "rain": rain,
        "cloud_cover": owm_response_json["clouds"]["all"],
        "sunset_local": sunset_local,
        "city_name": owm_response_json["name"],
        "date_stamp": datetime.utcnow().strftime("%A, %B %d, %Y")
    }
    result = WeatherData(**owm_dict)
    return result.to_markdown



# ------------------ utility functions ------------------


def celsius_to_fahrenheit(temp_celsius: float) -> float:
    return (temp_celsius * 9/5) + 32

def safe_get(data, dot_chained_keys):
    """
    {'a': {'b': [{'c': 1}]}}
    safe_get(data, 'a.b.0.c') -> 1
    """
    keys = dot_chained_keys.split(".")
    for key in keys:
        try:
            if isinstance(data, list):
                data = data[int(key)]
            else:
                data = data[key]
        except (KeyError, TypeError, IndexError):
            return None
    return data


def response_parser(response: Dict[str, Any]):
    return safe_get(response, "choices.0.message.content")


def is_url(image_path: str) -> bool:
    """
    Check if the given string is a valid URL.

    Args:
        image_path (str): The string to check.

    Returns:
        bool: True if the string is a valid URL, False otherwise.
    """
    try:
        result = urlparse(image_path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def encode_image_to_base64(image_path: str) -> str:
    """
    Encode a local image file to a base64 string.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def thumbnail(image, scale=3):
    return image.resize(np.array(image.size)//scale)

def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()    
    
# ------------------ pydantic models ------------------


class WeatherData(BaseModel):
    temp: float
    temp_max: float
    temp_min: float
    feels_like: float
    description: str
    icon: str
    wind_speed: float
    wind_direction: int
    humidity: int
    rain: str
    cloud_cover: int
    sunset_local: str
    city_name: str
    date_stamp: str

    def __str__(self):
        return (
            f"{self.date_stamp}\n"
            f"In {self.city_name}, the weather is currently:\n"
            f"Status: {self.description.title()}\n"
            f"Wind speed: {self.wind_speed} m/s, direction: {self.wind_direction}°\n"
            f"Humidity: {self.humidity}%\n"
            f"Temperature: \n"
            f"  - Current: {self.temp}°F\n"
            f"  - High: {self.temp_max}°F\n"
            f"  - Low: {self.temp_min}°F\n"
            f"  - Feels like: {self.feels_like}°F\n"
            f"Rain: {self.rain if self.rain else 'No rain'}\n"
            f"Cloud cover: {self.cloud_cover}%"
        )
    
    @property
    def to_markdown(self):
        return (
            f"{self.date_stamp}\n\n"
            f"The weather in **{self.city_name}** is currently:\n\n"
            f"Status: {self.description.title()}\n\n"
            f"Wind speed: {self.wind_speed} m/s, direction: {self.wind_direction}°\n\n"
            f"Humidity: {self.humidity}%\n\n"
            f"Temperature: \n\n"
            f"  - Current: {self.temp}°F\n"
            f"  - High: {self.temp_max}°F\n"
            f"  - Low: {self.temp_min}°F\n"
            f"  - Feels like: {self.feels_like}°F\n\n"
            f"Rain: {self.rain if self.rain else 'No rain'}\n\n"
            f"Cloud cover: {self.cloud_cover}%"
        )

class TheWeather(BaseModel):
    model_config = ConfigDict(
        extra='allow'
    )
    image_url: HttpUrl
    query: Optional[str] = None
    timestamp: Optional[datetime | str] = None
    hash_id: Optional[str] = None
    
    
    @model_validator(mode="before")
    def generate_tiemstamp(cls, values):
        timestamp = datetime.now()
        values['timestamp'] = timestamp.isoformat(timespec='minutes')
        return values
    
    @property
    def filename(self):
        filename = re.sub(
            r'[\\/*?:"<>|]', "", str(self.item.title)
        )  # Remove invalid file name characters
        filename = re.sub(r"\s+", "_", filename)  # Replace spaces with underscores
        filename += ".jpg"  # Append file extension
        return filename

    @property
    def full_path(self):
        folder_path: str = "./data/theweatherapp"
        # Ensure the folder exists
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        # Combine folder path and filename
        file_path = Path(folder_path) / self.filename
        return file_path
    
    @property
    def image(self) -> Image.Image:
        if not Path(self.full_path).exists():
            self.download_image()
        return Image.open(self.full_path)

    def download_image(
        self,
        folder_path: str = "./data/theweatherapp",
    ) -> None:
        """
        Downloads an image from a given URL and saves it to a specified folder with a filename
        based on the cleaned title attribute.

        Args:
            folder_path (str): The path to the folder where the image will be saved.

        Returns:
            None
        """
        # Ensure the folder exists
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        # Combine folder path and filename
        file_path = Path(folder_path) / self.filename

        # Download and save the image
        response = requests.get(self.item.image_url)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
        else:
            print(f"Failed to download image from {self.image_url}")


# ------------------ content generators ------------------

# def get_local_alerts(max_alerts: int = 1) -> str:
#     """
#     Get local weather alerts for the current IP address location.

#     Args:
#         max_alerts (int): Maximum number of alerts to include in the output. Defaults to 5.

#     Returns:
#         str: Markdown-formatted string containing the weather alerts.
#     """
#     ip_address = requests.get('http://api.ipify.org').text
#     geo_data = requests.get(f'http://ip-api.com/json/{ip_address}').json()
#     lat = geo_data['lat']
#     lon = geo_data['lon']
#     response = requests.get(f'https://api.weather.gov/alerts?point={lat},{lon}').json()
#     markdown_output = ""
#     for x in response['features'][:max_alerts]:
#         markdown_output += f"# {x['properties']['headline']}\n\n"
#         markdown_output += f"### {x['properties']['areaDesc']}\n\n"
#         markdown_output += f"{x['properties']['description']}\n\n"
#         markdown_output += "---\n\n"
#     return markdown_output


# def fetch_weather(search_query: str, search_type: str = "city") -> WeatherData:
#     API_key = os.getenv("OPENWEATHERMAPAPIKEY")
#     base_url = "http://api.openweathermap.org/data/2.5/weather?"
    
#     if search_type == "city":
#         final_url = f"{base_url}appid={API_key}&q={search_query}"
#     elif search_type == "zip":
#         final_url = f"{base_url}appid={API_key}&zip={search_query}"
#     else:
#         raise ValueError(f"Invalid search type: {search_type}. Must be either 'city' or 'zip'.")
    
#     owm_response_json = requests.get(final_url).json()
    
#     sunset_utc = datetime.fromtimestamp(owm_response_json["sys"]["sunset"])
#     sunset_local = sunset_utc.strftime("%I:%M %p")
    
#     temp_celsius = owm_response_json["main"]["temp"] - 273.15
#     temp_max_celsius = owm_response_json["main"]["temp_max"] - 273.15
#     temp_min_celsius = owm_response_json["main"]["temp_min"] - 273.15
#     temp_feels_like_celsius = owm_response_json["main"]["feels_like"] - 273.15

#     temp_fahrenheit = round(celsius_to_fahrenheit(temp_celsius), 2)
#     temp_max_fahrenheit = round(celsius_to_fahrenheit(temp_max_celsius), 2)
#     temp_min_fahrenheit = round(celsius_to_fahrenheit(temp_min_celsius), 2)
#     temp_feels_like_fahrenheit = round(celsius_to_fahrenheit(temp_feels_like_celsius), 2)

#     rain = owm_response_json.get("rain", "No rain")
    
#     owm_dict = {
#         "temp": temp_fahrenheit,
#         "temp_max": temp_max_fahrenheit,
#         "temp_min": temp_min_fahrenheit,
#         "feels_like": temp_feels_like_fahrenheit,
#         "description": owm_response_json["weather"][0]["description"],
#         "icon": owm_response_json["weather"][0]["icon"],
#         "wind_speed": owm_response_json["wind"]["speed"],
#         "wind_direction": owm_response_json["wind"]["deg"],
#         "humidity": owm_response_json["main"]["humidity"],
#         "rain": rain,
#         "cloud_cover": owm_response_json["clouds"]["all"],
#         "sunset_local": sunset_local,
#         "city_name": owm_response_json["name"],
#         "date_stamp": datetime.utcnow().strftime("%A, %B %d, %Y")
#     }
    
#     return WeatherData(**owm_dict)


def prompt(
    prompt: str,
    model: str = "gpt-4o-mini",
    instructions: str = "You are a helpful assistant.",
):
    return openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": instructions,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        stream=True,
    )

def prompt_image_gen(
    prompt: str,
    openai_key: str = os.getenv("OPENAI_API_KEY"),
    model: str = "dall-e-3", # or 'dall-e-2'
    size_category: str = "square",
    style: str = "vivid", # or 'natural'
    quality: str = "standard",
) -> Dict[str, str]:
    """
    Generate an image from a prompt using the OpenAI API, dynamically save it to a file path based on the prompt and current datetime,
    and return a dictionary with the file path and URL for display purposes.

    Args:
        prompt (str): The prompt to generate the image from.
        openai_key (str): The OpenAI API key.
        model (str, optional): The model to use for image generation. Defaults to "dall-e-3".
        size (str, optional): The size of the generated image. Defaults to "512x512".
        quality (str, optional): The quality of the generated image. Defaults to "standard".

    Returns:
        Dict[str, str]: A dictionary containing the file path and URL of the generated image.
    """   
    d2_size_mapping = {
        "small": "256x256",
        "medium": "512x512",
        "large": "1024x1024",
    }
    d3_size_mapping = {
        "square": "1024x1024",
        "wide": "1792x1024",
        "tall": "1024x1792"
    }
    if model == "dall-e-2":
            size_mapping = d2_size_mapping
    elif model == "dall-e-3":
        size_mapping = d3_size_mapping
    else:
        raise ValueError("Unsupported model. Choose either 'dall-e-2' or 'dall-e-3'.")
    
    # Set the OpenAI API key
    client = openai.OpenAI(
    )
    
    # Get the size from the mapping
    size = size_mapping.get(size_category, "512x512")

    # Generate the image
    response: ImagesResponse = client.images.generate(
        prompt=prompt,
        model=model,
        n=1,
        quality=quality,
        size=size,
        style=style,
    )

    # Extract the image URL from the response
    image_data = response.data[0]
    image_url = image_data.url

    # Create a sanitized version of the prompt for the file name
    sanitized_prompt = re.sub(r'[^A-Za-z0-9]+', '', prompt)[:8]
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "data/dalle_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = f"{output_dir}/{sanitized_prompt}_{datetime_str}.jpeg"

    # Download and save the image
    with requests.get(image_url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    return {"file_path": file_path, "image_url": image_url}

def convert_timestamp_to_datetime(timestamp: str) -> str:
    return datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d")

timestamp = datetime.timestamp(datetime.now())

timestamp_string = convert_timestamp_to_datetime(timestamp)

def image_gen_prompt(
    prompt: str,
    model: str = "gpt-4o-mini",
):
    return openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a master of the visual arts, adept at vividly describing the effects of the weather for a given location. You will receive a CITY and a WEATHER REPORT, along with optional USER_NOTES. Use them to make am awesome Dalle-3 prompt that emphasizes the weather, make it extreme and try to incorporate elements of the city if they are known. We really want the user to **feel** the weather in their home town. Make sure to keep it under 4000 characters",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        stream=True
    )



def run_weather_app():
    """ Run the Streamlit app for weather forecasting. """
    # Initialize the AsyncClient and dependencies
    load_dotenv()
    client = AsyncClient()      
    weather_api_key = os.getenv('WEATHER_API_KEY')
    geo_api_key = os.getenv('GEO_API_KEY')

    deps = Deps(client=client, weather_api_key=weather_api_key, geo_api_key=geo_api_key)
    
    st.markdown('# Weather Forecast', unsafe_allow_html=True)
    # TODO: Add option to search by zip code
    user_input = ui.input(default_value=None, type='text', placeholder="", key="user_input")

    if ui.button('Get Weather', key="clk_btn", className="bg-cyan-950 text-white"):
        if user_input:
            # Get basic weather forecast
            weather_data = weather_agent.run_sync(
                f"Please provide a detailed weather report for {user_input}.",
                deps=deps,
            )
            report = weather_data.data
            st.write(report)
            full_report = ""
            report_message_placeholder = st.empty()
            weather_report_stream = prompt(
                prompt=f"Please concisely summarize the following weather report. Include a concise narrative with highlights and end with a markdown table summary.\n\nHere is the report: {report}"
            )
            for chunk in weather_report_stream:
                if chunk.choices[0].delta.content is not None:
                    response = chunk.choices[0].delta.content
                    full_report += response
                    report_message_placeholder.markdown(full_report + "▌", unsafe_allow_html=True)
            # Capture the completed response tokens
            report_message_placeholder.markdown(full_report, unsafe_allow_html=True)
            avs(1)
            # Create prompt for image gen and stream in response
            with st.status(label="dreaming about the weather ...", expanded=False) as status:
                full_response = ""
                message_placeholder = st.empty()
                dalle3_prompt_stream = image_gen_prompt(f"Please help write an awesome prompt for Dalle-3 that depicts the weather in given location. Here is the weather report: {report}")
                for chunk in dalle3_prompt_stream:
                    if chunk.choices[0].delta.content is not None:
                        response = chunk.choices[0].delta.content
                        full_response += response
                        message_placeholder.markdown(full_response + "▌")
                status.update(label=' ', expanded=False)
            # Capture the completed response tokens
            message_placeholder.markdown(full_response)
            # Send image gen prompt to dalle-3
            gen_image = prompt_image_gen(prompt=full_response)
            # Render image using native url
            avs(1)
            st.image(gen_image['image_url'], caption=f"{timestamp_string}")
            # Render button link to the image
            avs(1)
            st.link_button("image link", gen_image['image_url'], use_container_width=True)
        else:
            st.error("Please enter a city name to check the weather.")

if __name__ == '__main__':
    run_weather_app()