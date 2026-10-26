import asyncio
import json
import threading
from pathlib import Path

from bscpylgtv import WebOsClient


class LGController:

    def __init__(
        self,
        ip="192.168.0.24",
        key_file="lg_tv_key.json",
        manifest_file="manifest.json",
        pairing_type="PIN"
    ):
        self.ip = ip

        # Dossier contenant lg_controller.py
        base_dir = Path(__file__).resolve().parent

        self.key_file = base_dir / key_file
        self.manifest_file = base_dir / manifest_file

        self.pairing_type = pairing_type

        self.client = None
        self.loop = None
        self.thread = None

        self.ready = threading.Event()
        self.connection_error = None

    # ============================================================
    # CLIENT KEY
    # ============================================================

    def _load_key(self):

        if not self.key_file.exists():
            return None

        try:
            with open(self.key_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            return data.get("client_key")

        except Exception as e:
            print("Erreur lecture client key :", e)
            return None

    def _save_key(self, key):

        with open(self.key_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ip": self.ip,
                    "client_key": key
                },
                f,
                indent=4
            )

    # ============================================================
    # CONNEXION ASYNC
    # ============================================================

    async def _connect(self):

        key = self._load_key()

        self.client = await WebOsClient.create(
            self.ip,
            client_key=key,

            # Nécessaire avec ton nouveau firmware LG
            manifest_file_path=str(self.manifest_file),
            pairing_type=self.pairing_type,

            get_hello_info=False,
            states=[]
        )

        await self.client.connect()

        print("LG TV connectée")

        # Si nouvel appairage → sauvegarde de la clé
        if self.client.client_key != key:

            self._save_key(self.client.client_key)

            print("Nouvelle client key sauvegardée")

    # ============================================================
    # THREAD ASYNCIO
    # ============================================================

    def start(self, timeout=10):

        if self.thread is not None and self.thread.is_alive():
            return True

        self.ready.clear()
        self.connection_error = None

        self.loop = asyncio.new_event_loop()

        def run():

            asyncio.set_event_loop(self.loop)

            try:
                self.loop.run_until_complete(self._connect())

            except Exception as e:
                self.connection_error = e
                print("Impossible de connecter la TV :", e)

                self.ready.set()
                return

            self.ready.set()

            self.loop.run_forever()

        self.thread = threading.Thread(
            target=run,
            daemon=True
        )

        self.thread.start()

        # Attend que la connexion soit établie
        self.ready.wait(timeout)

        if self.connection_error is not None:
            return False

        if self.client is None:
            print("Timeout connexion TV")
            return False

        return True

    # ============================================================
    # EXÉCUTION DES COMMANDES ASYNC
    # ============================================================

    def _run(self, coroutine):

        if (
            self.loop is None
            or self.client is None
            or not self.loop.is_running()
        ):
            print("TV non connectée")

            # Évite le warning "coroutine was never awaited"
            coroutine.close()
            return None

        future = asyncio.run_coroutine_threadsafe(
            coroutine,
            self.loop
        )

        return future

    # ============================================================
    # VOLUME
    # ============================================================

    def volume_up(self):
        return self._run(
            self.client.volume_up()
        )

    def volume_down(self):
        return self._run(
            self.client.volume_down()
        )

    def mute(self):
        return self._run(
            self.client.set_mute(True)
        )

    def unmute(self):
        return self._run(
            self.client.set_mute(False)
        )

    # ============================================================
    # MEDIA
    # ============================================================

    def play(self):
        return self._run(
            self.client.play()
        )

    def pause(self):
        return self._run(
            self.client.pause()
        )

    # ============================================================
    # POWER
    # ============================================================

    def power_off(self):
        return self._run(
            self.client.power_off()
        )

    # ============================================================
    # APPLICATIONS
    # ============================================================

    def launch_app(self, app_id):
        return self._run(
            self.client.launch_app(app_id)
        )

    # ============================================================
    # TÉLÉCOMMANDE
    # ============================================================

    def button(self, name):
        return self._run(
            self.client.button(name)
        )

    def up(self):
        return self.button("UP")

    def down(self):
        return self.button("DOWN")

    def left(self):
        return self.button("LEFT")

    def right(self):
        return self.button("RIGHT")

    def ok(self):
        return self.button("ENTER")

    def back(self):
        return self.button("BACK")

    def home(self):
        return self.button("HOME")

    # ============================================================
    # DÉCONNEXION
    # ============================================================

    def stop(self):

        if self.loop is None:
            return

        if self.client is not None and self.loop.is_running():

            future = asyncio.run_coroutine_threadsafe(
                self.client.disconnect(),
                self.loop
            )

            try:
                future.result(timeout=3)
            except Exception:
                pass

        if self.loop.is_running():
            self.loop.call_soon_threadsafe(
                self.loop.stop
            )

        self.client = None

        print("LG TV déconnectée")