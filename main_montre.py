import asyncio
import threading
import time

from bleak import BleakClient
from wakeonlan import send_magic_packet

from lg_controller import LGController
from spotify_controller import SpotifyController


# ============================================================
# CONFIGURATION CW700
# ============================================================

WATCH_MAC = "C9:F2:BA:70:EF:8A"

AF6 = "00000af6-0000-1000-8000-00805f9b34fb"
AF7 = "00000af7-0000-1000-8000-00805f9b34fb"

RECONNECT_DELAY = 3


# ============================================================
# CONFIGURATION LG
# ============================================================

TV_MAC = "20:28:BC:89:9D:EA"

NETFLIX_APP = "netflix"
SPOTIFY_APP = "spotify-beehive"

TV_BOOT_DELAY = 8
APP_LOAD_DELAY = 8


# ============================================================
# CONFIGURATION SPOTIFY
# ============================================================

SPOTIFY_TV_ID = "f2a3a10e-4dfe-4fdf-a1df-19d2e51c903f"

PLAYLIST_FRAP = "spotify:playlist:6wi5jWo1QEfwITLEvex4T6"
PLAYLIST_SYMPA = "spotify:playlist:2UCpFUh0dYuOqKXg4N6FvT"


# ============================================================
# ACTIVITÉS CW700
# ============================================================

WALKING_ID = 0x01
CYCLING_ID = 0x03
HIKING_ID = 0x04

EVENT_START = 0x11
EVENT_STOP = 0x13
EVENT_PAUSE = 0x14
EVENT_PLAY = 0x15


# ============================================================
# COMMANDES MENU MUSIQUE CW700
# ============================================================

# Mapping BLE confirmé expérimentalement :
#
# 07 01 01 00 00 00 -> PREVIOUS
# 07 01 04 00 00 00 -> PLAY/PAUSE
# 07 01 05 00 00 00 -> NEXT
#
# Fonctions que nous leur attribuons :
#
# PREVIOUS   -> VOLUME +
# PLAY/PAUSE -> VOLUME -
# NEXT       -> MORCEAU SUIVANT
MEDIA_PREVIOUS = 0x01
MEDIA_PLAY_PAUSE = 0x04
MEDIA_NEXT = 0x05


# ============================================================
# CONTROLLERS
# ============================================================

tv = LGController()
spotify = SpotifyController()


# ============================================================
# ÉTAT GLOBAL
# ============================================================

active_mode = None
last_event = None

tv_starting = False

state_lock = threading.Lock()


# ============================================================
# NOTIFICATION CW700
# ============================================================

async def watch_notify(client, sender="Ada", message="Connectee"):

    sender_bytes = sender.encode("utf-8")
    message_bytes = message.encode("utf-8")

    packet = (
        bytes([
            0x05,
            0x03,
            0x01,
            0x01,
            0x14,
            len(message_bytes),
            0x00,
            len(sender_bytes),
        ])
        + sender_bytes
        + message_bytes
    )

    await client.write_gatt_char(
        AF6,
        packet,
        response=True
    )


# ============================================================
# TV
# ============================================================

def wake_tv():

    print("📺 Wake-on-LAN LG...")

    try:

        send_magic_packet(TV_MAC)

        return True

    except Exception as e:

        print(
            "❌ Erreur Wake-on-LAN :",
            e
        )

        return False


def ensure_tv():
    """
    Réveille la TV et assure une connexion WebOS.
    """

    # On envoie toujours le WOL.
    # Une ancienne connexion WebOS peut encore exister alors que
    # la TV vient d'être rallumée.

    wake_tv()

    if tv.client is not None:

        print(
            "📺 Connexion LG déjà présente"
        )

        return True

    print(
        f"⏳ Démarrage TV ({TV_BOOT_DELAY}s)..."
    )

    time.sleep(
        TV_BOOT_DELAY
    )

    for attempt in range(1, 6):

        print(
            f"🔌 Connexion LG ({attempt}/5)..."
        )

        try:

            if tv.start(timeout=5):

                print(
                    "✅ LG prête"
                )

                return True

        except Exception as e:

            print(
                "⚠️ LG :",
                e
            )

        time.sleep(3)

    print(
        "❌ Impossible de joindre la LG"
    )

    return False


def wait_future(future, timeout=10):

    if future is None:
        return False

    try:

        future.result(
            timeout=timeout
        )

        return True

    except Exception as e:

        print(
            "❌ Erreur commande LG :",
            e
        )

        return False


# ============================================================
# NETFLIX
# ============================================================

def launch_netflix():

    global tv_starting

    with state_lock:

        if tv_starting:

            print(
                "⚠️ Démarrage TV déjà en cours"
            )

            return

        tv_starting = True

    try:

        print()
        print(
            "🚶 WALKING"
        )
        print(
            "🎬 Lancement Netflix"
        )

        if not ensure_tv():
            return

        print(
            "🎬 Ouverture Netflix..."
        )

        future = tv.launch_app(
            NETFLIX_APP
        )

        if not wait_future(future):
            return

        time.sleep(
            APP_LOAD_DELAY
        )

        print(
            "👤 Sélection profil Hugo"
        )

        future = tv.ok()

        wait_future(
            future
        )

        print(
            "✅ Netflix prêt"
        )

    finally:

        with state_lock:
            tv_starting = False


# ============================================================
# SPOTIFY
# ============================================================

SPOTIFY_DEVICE_TIMEOUT = 30
SPOTIFY_DEVICE_POLL = 1


def wait_for_spotify_device(
    timeout=SPOTIFY_DEVICE_TIMEOUT
):

    print(
        "🔎 Attente de la LG dans Spotify Connect..."
    )

    start = time.time()

    while time.time() - start < timeout:

        try:

            devices = spotify.spotify.devices()["devices"]

            for device in devices:

                print(
                    f"   Spotify device : "
                    f"{device['name']} "
                    f"| active={device['is_active']}"
                )

                if device["id"] == SPOTIFY_TV_ID:

                    print(
                        "✅ LG détectée par Spotify Connect"
                    )

                    return True

        except Exception as e:

            print(
                "⚠️ Vérification Spotify :",
                e
            )

        elapsed = int(
            time.time() - start
        )

        print(
            f"⏳ Spotify pas encore prêt "
            f"({elapsed}/{timeout}s)"
        )

        time.sleep(
            SPOTIFY_DEVICE_POLL
        )

    print(
        "❌ Timeout : LG absente de Spotify Connect"
    )

    return False


def launch_spotify(
    playlist_uri,
    playlist_name
):

    global tv_starting

    with state_lock:

        if tv_starting:

            print(
                "⚠️ Démarrage TV déjà en cours"
            )

            return

        tv_starting = True

    try:

        print()
        print(
            f"🎵 Spotify → {playlist_name}"
        )

        # ====================================================
        # TV
        # ====================================================

        if not ensure_tv():
            return

        # ====================================================
        # APPLICATION SPOTIFY LG
        # ====================================================

        print(
            "🟢 Ouverture Spotify sur LG..."
        )

        future = tv.launch_app(
            SPOTIFY_APP
        )

        if not wait_future(future):
            return

        # ====================================================
        # ATTENTE SPOTIFY CONNECT
        # ====================================================

        if not wait_for_spotify_device():

            print(
                "❌ Spotify LG indisponible"
            )

            return

        # ====================================================
        # LANCEMENT PLAYLIST
        # ====================================================

        try:

            print(
                "🎵 Transfert lecture → LG"
            )

            spotify.spotify.transfer_playback(
                device_id=SPOTIFY_TV_ID,
                force_play=False
            )

            time.sleep(1)

            print(
                f"▶️ Lancement {playlist_name}"
            )

            spotify.spotify.start_playback(
                device_id=SPOTIFY_TV_ID,
                context_uri=playlist_uri
            )

            print(
                f"✅ {playlist_name} lancée"
            )

        except Exception as e:

            print(
                "❌ Erreur Spotify :",
                e
            )

    finally:

        with state_lock:
            tv_starting = False


def launch_frap():

    launch_spotify(
        PLAYLIST_FRAP,
        "GoFRap"
    )


def launch_sympa():

    launch_spotify(
        PLAYLIST_SYMPA,
        "Sympa"
    )


# ============================================================
# BOUTONS DE L'ACTIVITÉ
# ============================================================

# Le bouton principal de la montre conserve son comportement :
#
# PAUSE ACTIVITÉ  -> pause Spotify / Netflix
# PLAY ACTIVITÉ   -> reprise Spotify / Netflix


def pause():

    mode = active_mode

    print(
        "⏸ PAUSE"
    )

    try:

        if mode in (
            "frap",
            "sympa"
        ):

            spotify.spotify.pause_playback(
                device_id=SPOTIFY_TV_ID
            )

            print(
                "✅ Spotify pause"
            )

        elif mode == "netflix":

            future = tv.pause()

            wait_future(
                future
            )

            print(
                "✅ Netflix pause"
            )

    except Exception as e:

        print(
            "❌ Erreur pause :",
            e
        )


def play():

    mode = active_mode

    print(
        "▶️ PLAY"
    )

    try:

        if mode in (
            "frap",
            "sympa"
        ):

            spotify.spotify.start_playback(
                device_id=SPOTIFY_TV_ID
            )

            print(
                "✅ Spotify play"
            )

        elif mode == "netflix":

            future = tv.play()

            wait_future(
                future
            )

            print(
                "✅ Netflix play"
            )

    except Exception as e:

        print(
            "❌ Erreur play :",
            e
        )


# ============================================================
# MENU MUSIQUE CW700
# ============================================================

def media_volume_up():

    if active_mode not in (
        "frap",
        "sympa"
    ):
        return

    print(
        "🔊 VOLUME +"
    )

    try:

        future = tv.volume_up()

        if wait_future(future):

            print(
                "✅ Volume LG augmenté"
            )

    except Exception as e:

        print(
            "❌ Volume + :",
            e
        )


def media_volume_down():

    if active_mode not in (
        "frap",
        "sympa"
    ):
        return

    print(
        "🔉 VOLUME -"
    )

    try:

        future = tv.volume_down()

        if wait_future(future):

            print(
                "✅ Volume LG diminué"
            )

    except Exception as e:

        print(
            "❌ Volume - :",
            e
        )


def media_next_track():

    if active_mode not in (
        "frap",
        "sympa"
    ):
        return

    print(
        "⏭ MORCEAU SUIVANT"
    )

    try:

        spotify.spotify.next_track(
            device_id=SPOTIFY_TV_ID
        )

        print(
            "✅ Spotify → morceau suivant"
        )

    except Exception as e:

        print(
            "❌ Spotify next :",
            e
        )


# ============================================================
# POWER OFF
# ============================================================

def power_off():

    print(
        "🔴 Extinction TV"
    )

    try:

        future = tv.power_off()

        if future is not None:

            future.result(
                timeout=5
            )

        print(
            "✅ Commande extinction envoyée"
        )

    except Exception as e:

        print(
            "⚠️ Erreur extinction :",
            e
        )

    finally:

        time.sleep(2)

        try:

            tv.stop()

        except Exception as e:

            print(
                "⚠️ Fermeture connexion LG :",
                e
            )

        print(
            "🔌 Connexion LG réinitialisée"
        )


# ============================================================
# DÉTECTION ACTIVITÉ
# ============================================================

def get_activity(data):

    # Format observé :
    #
    # 10 11 XX XX XX XX ID
    #
    # Les octets intermédiaires changent.
    # Le dernier octet contient l'ID de l'activité.

    if len(data) != 7:
        return None

    if data[0] != 0x10:
        return None

    if data[1] != EVENT_START:
        return None

    activity_id = data[-1]

    print(
        f"🏷 Activity ID = "
        f"0x{activity_id:02X}"
    )

    if activity_id == WALKING_ID:
        return "walking"

    if activity_id == CYCLING_ID:
        return "cycling"

    if activity_id == HIKING_ID:
        return "hiking"

    return None


# ============================================================
# ACTIVATION DES MODES
# ============================================================

def activate_walking():

    global active_mode

    active_mode = "netflix"

    print()
    print(
        "================================"
    )
    print(
        "🚶 WALKING détecté"
    )
    print(
        "➡ Netflix / Hugo"
    )
    print(
        "================================"
    )

    threading.Thread(
        target=launch_netflix,
        daemon=True
    ).start()


def activate_cycling():

    global active_mode

    active_mode = "frap"

    print()
    print(
        "================================"
    )
    print(
        "🚴 CYCLING détecté"
    )
    print(
        "➡ Spotify / GoFRap"
    )
    print(
        "================================"
    )

    threading.Thread(
        target=launch_frap,
        daemon=True
    ).start()


def activate_hiking():

    global active_mode

    active_mode = "sympa"

    print()
    print(
        "================================"
    )
    print(
        "🥾 HIKING détecté"
    )
    print(
        "➡ Spotify / Sympa"
    )
    print(
        "================================"
    )

    threading.Thread(
        target=launch_sympa,
        daemon=True
    ).start()


# ============================================================
# CALLBACK BLE
# ============================================================

def notification(sender, data):

    global active_mode
    global last_event

    data = bytes(data)

    if len(data) < 2:
        return

    print(
        "AF7 ->",
        data.hex(" ")
    )

    # ========================================================
    # MENU MUSIQUE CW700
    # ========================================================
    #
    # Mapping BLE :
    #
    # 07 01 01 -> PREVIOUS
    # 07 01 04 -> PLAY/PAUSE
    # 07 01 05 -> NEXT
    #
    # Mapping Ada :
    #
    # PREVIOUS   -> VOLUME +
    # PLAY/PAUSE -> VOLUME -
    # NEXT       -> MORCEAU SUIVANT
    #
    # Ces commandes ne sont PAS dédoublonnées.
    # Chaque paquet représente un clic utilisateur.


    if (
            len(data) == 6
            and data[0] == 0x07
            and data[1] == 0x01
    ):

        command = data[2]

        # PREVIOUS -> MORCEAU SUIVANT
        if command == MEDIA_PREVIOUS:

            print("🎵 PREVIOUS → morceau suivant")

            threading.Thread(
                target=media_next_track,
                daemon=True
            ).start()

        # PLAY/PAUSE -> VOLUME -
        elif command == MEDIA_PLAY_PAUSE:

            print("🎵 PLAY/PAUSE → volume -")

            threading.Thread(
                target=media_volume_down,
                daemon=True
            ).start()

        # NEXT -> VOLUME +
        elif command == MEDIA_NEXT:

            print("🎵 NEXT → volume +")

            threading.Thread(
                target=media_volume_up,
                daemon=True
            ).start()

        else:

            print(
                f"⚠️ Commande média inconnue : "
                f"0x{command:02X}"
            )

        return

    # ========================================================
    # ÉVÉNEMENTS ACTIVITÉ
    # ========================================================

    if data[0] != 0x10:
        return

    event = data[1]

    print(
        f"   event=0x{event:02X}"
    )

    # La CW700 répète les événements d'activité environ
    # toutes les deux secondes.
    #
    # On ignore donc les répétitions.

    if event == last_event:
        return

    last_event = event

    # ========================================================
    # START
    # ========================================================

    if event == EVENT_START:

        activity = get_activity(
            data
        )

        if activity == "walking":

            activate_walking()

        elif activity == "cycling":

            activate_cycling()

        elif activity == "hiking":

            activate_hiking()

        else:

            print(
                "⚠️ Activité inconnue :",
                data.hex(" ")
            )

        return

    # ========================================================
    # AUCUN MODE ACTIF
    # ========================================================

    if active_mode is None:
        return

    # ========================================================
    # PAUSE ACTIVITÉ
    # ========================================================

    if event == EVENT_PAUSE:

        threading.Thread(
            target=pause,
            daemon=True
        ).start()

        return

    # ========================================================
    # REPRISE ACTIVITÉ
    # ========================================================

    if event == EVENT_PLAY:

        threading.Thread(
            target=play,
            daemon=True
        ).start()

        return

    # ========================================================
    # STOP ACTIVITÉ
    # ========================================================

    if event == EVENT_STOP:

        previous_mode = active_mode

        active_mode = None

        print(
            f"⏹ STOP activité "
            f"({previous_mode})"
        )

        threading.Thread(
            target=power_off,
            daemon=True
        ).start()

        return


# ============================================================
# CONNEXION CW700
# ============================================================

async def connect_watch():

    attempt = 0

    while True:

        attempt += 1

        print(
            f"⌚ Connexion CW700 HR "
            f"(tentative {attempt})..."
        )

        client = BleakClient(
            WATCH_MAC
        )

        try:

            await client.connect()

            if client.is_connected:

                print(
                    "✅ CW700 connectée"
                )

                return client

        except asyncio.CancelledError:

            raise

        except Exception as e:

            print(
                "⚠️ Montre indisponible :",
                e
            )

        try:

            if client.is_connected:

                await client.disconnect()

        except Exception:

            pass

        print(
            f"🔄 Nouvelle tentative dans "
            f"{RECONNECT_DELAY}s..."
        )

        await asyncio.sleep(
            RECONNECT_DELAY
        )


# ============================================================
# MAIN
# ============================================================

async def main():

    global last_event

    print()
    print(
        "=========================================="
    )
    print(
        "        ADA - CW700 LG REMOTE"
    )
    print(
        "=========================================="
    )

    print()

    print(
        "ACTIVITÉS :"
    )

    print(
        "🚶 Walking → Netflix / Hugo"
    )

    print(
        "🚴 Cycling → Spotify / GoFRap"
    )

    print(
        "🥾 Hiking  → Spotify / Sympa"
    )

    print()

    print(
        "BOUTON ACTIVITÉ :"
    )

    print(
        "⏸ Pause → pause média"
    )

    print(
        "▶️ Play  → reprise média"
    )

    print(
        "⏹ Stop  → extinction LG"
    )

    print()

    print(
        "MENU MUSIQUE (Spotify) :"
    )

    print(
        "⏮ Previous   → volume LG +"
    )

    print(
        "⏯ Play/Pause → volume LG -"
    )

    print(
        "⏭ Next       → morceau suivant"
    )

    print()

    print(
        "=========================================="
    )

    print()

    while True:

        client = None

        try:

            # =================================================
            # CONNEXION CW700
            # =================================================

            client = await connect_watch()

            # On réinitialise uniquement le dernier événement.
            #
            # active_mode n'est volontairement PAS remis à None
            # afin de conserver l'activité en cours après une
            # déconnexion/reconnexion BLE.

            last_event = None

            # =================================================
            # NOTIFICATIONS AF7
            # =================================================

            await client.start_notify(
                AF7,
                notification
            )

            print(
                "📡 Écoute AF7 active"
            )

            # =================================================
            # VIBRATION / NOTIFICATION DE CONNEXION
            # =================================================

            try:

                await watch_notify(
                    client,
                    sender="Ada",
                    message="Connectee"
                )

                print(
                    "📳 Confirmation envoyée "
                    "à la CW700"
                )

            except Exception as e:

                print(
                    "⚠️ Notification CW700 :",
                    e
                )

            print(
                "⌚ En attente d'une activité...\n"
            )

            # =================================================
            # RESTER CONNECTÉ
            # =================================================

            while client.is_connected:

                await asyncio.sleep(1)

            print(
                "\n❌ CW700 déconnectée"
            )

        except asyncio.CancelledError:

            raise

        except Exception as e:

            print(
                "⚠️ Erreur BLE :",
                e
            )

        finally:

            if client is not None:

                try:

                    if client.is_connected:

                        try:

                            await client.stop_notify(
                                AF7
                            )

                        except Exception:

                            pass

                        await client.disconnect()

                except Exception:

                    pass

            # IMPORTANT :
            #
            # On ne touche pas à active_mode ici.
            #
            # Si la montre se déconnecte pendant Cycling ou
            # Hiking, Spotify continue et le mode sera encore
            # connu après reconnexion.

            last_event = None

        print(
            f"🔄 Reconnexion dans "
            f"{RECONNECT_DELAY}s..."
        )

        await asyncio.sleep(
            RECONNECT_DELAY
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print(
            "\n🛑 Ada arrêtée"
        )

    finally:

        try:

            tv.stop()

        except Exception:

            pass