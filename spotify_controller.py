import os
import time

import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()

SCOPES = (
    "user-read-playback-state "
    "user-modify-playback-state "
    "playlist-read-private"
)


class SpotifyController:

    def __init__(self):
        self.spotify = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=os.getenv("SPOTIFY_CLIENT_ID"),
                client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
                redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI"),
                scope=SCOPES,
                cache_path=".spotify_cache",
                open_browser=True
            )
        )

    # -------------------------
    # DEVICES
    # -------------------------

    def list_devices(self):
        devices = self.spotify.devices()["devices"]

        print("\n=== APPAREILS SPOTIFY ===")

        for device in devices:
            print(
                f"{device['name']} | "
                f"type={device['type']} | "
                f"active={device['is_active']} | "
                f"id={device['id']}"
            )

        return devices

    def get_device(self, name):
        devices = self.spotify.devices()["devices"]

        for device in devices:
            if name.lower() in device["name"].lower():
                return device

        return None

    # -------------------------
    # PLAYLISTS
    # -------------------------

    def list_playlists(self):
        result = self.spotify.current_user_playlists(limit=50)

        playlists = []

        while result:
            for playlist in result["items"]:
                playlists.append(playlist)

                print(
                    f"{playlist['name']} | "
                    f"{playlist['uri']}"
                )

            if result["next"]:
                result = self.spotify.next(result)
            else:
                break

        return playlists

    def get_playlist(self, name):
        result = self.spotify.current_user_playlists(limit=50)

        while result:
            for playlist in result["items"]:

                if playlist["name"].lower() == name.lower():
                    return playlist

            if result["next"]:
                result = self.spotify.next(result)
            else:
                break

        return None

    # -------------------------
    # PLAYER
    # -------------------------

    def play_playlist(self, playlist_name, device_name):

        device = self.get_device(device_name)

        if device is None:
            print(f"❌ Appareil '{device_name}' introuvable")
            return False

        playlist = self.get_playlist(playlist_name)

        if playlist is None:
            print(f"❌ Playlist '{playlist_name}' introuvable")
            return False

        print()
        print("🎵 Playlist :", playlist["name"])
        print("📺 Device   :", device["name"])
        print("URI         :", playlist["uri"])

        device_id = device["id"]

        # Rend la TV active
        self.spotify.transfer_playback(
            device_id=device_id,
            force_play=False
        )

        time.sleep(1)

        # Lance directement la playlist
        self.spotify.start_playback(
            device_id=device_id,
            context_uri=playlist["uri"]
        )

        print("▶️ Lecture lancée")

        return True

    def pause(self, device_name=None):

        device_id = None

        if device_name:
            device = self.get_device(device_name)

            if device:
                device_id = device["id"]

        self.spotify.pause_playback(
            device_id=device_id
        )

    def play(self, device_name=None):

        device_id = None

        if device_name:
            device = self.get_device(device_name)

            if device:
                device_id = device["id"]

        self.spotify.start_playback(
            device_id=device_id
        )

    def next(self, device_name=None):

        device_id = None

        if device_name:
            device = self.get_device(device_name)

            if device:
                device_id = device["id"]

        self.spotify.next_track(
            device_id=device_id
        )

    def previous(self, device_name=None):

        device_id = None

        if device_name:
            device = self.get_device(device_name)

            if device:
                device_id = device["id"]

        self.spotify.previous_track(
            device_id=device_id
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    spotify = SpotifyController()

    spotify.play_playlist(
        playlist_name="GoFRap",
        device_name="[LG] webOS TV NANO826QB"
    )