import os
import platform
import subprocess
import time
import webbrowser

# Imports - Spotify API
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.exceptions import SpotifyException

from dotenv import load_dotenv # TO load env Variables from .env file in the folder

load_dotenv()

class SpotifyTool:
    def __init__(self, client_id=None, client_secret=None, redirect_uri=None, scope=None):
        self.client_id = client_id or os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET")
        self.redirect_uri = redirect_uri or os.getenv("SPOTIFY_REDIRECT_URI")

        if not all([self.client_id, self.client_secret, self.redirect_uri]):
            raise ValueError("Spotify credentials are not fully set in environment variables.")
        
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope="user-modify-playback-state user-read-playback-state"
        ))
    
    def _open_spotify_app(self):
        try:
            print(f"Opening Spotify application...")
            msg = "Opening Spotify app on the Device"
            system = platform.system()
            if system == "Windows":
                subprocess.Popen(["spotify.exe"])
            elif system == "Darwin":
                subprocess.Popen(["open", "-a", "Spotify"])
            time.sleep(5)
            return True, "Spotify App Exists"

        except Exception as e:
            return False, f"Couldn't Open Spotify App."

    def _fallback_web_player(self):
        try:
            webbrowser.open("https://open.spotify.com")
            time.sleep(5)
            return True, "Spotify Web Player Opened"
        except Exception as e:
            return False, "Couldn't Open Spotify Web Player."

    def _wait_for_device(self, max_attempts=10, wait_time=2):
        """
        Waits for a Spotify device to become available.
        Returns device_id if found, None otherwise.
        """
        for attempt in range(max_attempts):
            try:
                devices = self.sp.devices()
                if devices and devices.get('devices'):
                    return devices['devices'][0]['id']
            except SpotifyException as e:
                if e.http_status == 401:
                    return None  # Not authenticated
            except Exception as e:
                print(f"Error checking devices (attempt {attempt + 1}): {e}")
            
            time.sleep(wait_time)
        return None

    def play_song(self, song_name):
        """
        Searches for a song by name and plays it on the user's active device.
        """

        try:
            results = self.sp.search(q=song_name, type='track', limit=1)
            tracks = results['tracks']['items']

            if not tracks:
                raise f"Error: No tracks found for '{song_name}'"
            
            track_uri = tracks[0]['uri']
            track_title = tracks[0]['name']
            artist_name = tracks[0]['artists'][0]['name']

            devices = self.sp.devices() # Get Active devices active devices

            if not devices['devices']:
                app_open, _ = self._open_spotify_app() # Try to open Spotify App
                if not app_open:
                    web_open, _ = self._fallback_web_player() # Fallback to web player
                    if not web_open:
                        raise Exception("No active Spotify device found and unable to open Spotify app or web player.")
                
                device_id = self._wait_for_device()
            else:
                device_id = devices['devices'][0]['id'] # Use the first available device
            self.sp.start_playback(device_id=device_id, uris=[track_uri])

            return f"Now Playing: '{track_title}' by {artist_name}'"
        except Exception as e:
            return f"An error occured while trying to play the song: {str(e)}"
    
    def stop_playback(self):
        """
        Stops playback on the user's active device.
        """
        try:
            self.sp.pause_playback()
            return "Playback stopped."
        except Exception as e:
            return f"An error occured while trying to stop playback: {str(e)}"

if __name__ == "__main__":
    try:
        music_tool = SpotifyTool()
        response = music_tool.play_song("WRONG by Chris Grey")
        print(response)

    except ValueError as ve:
        print(f"Configuration Error: {str(ve)}")