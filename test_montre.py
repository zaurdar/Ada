import asyncio
from datetime import datetime
from bleak import BleakClient

WATCH_MAC = "C9:F2:BA:70:EF:8A"

AF7 = "00000af7-0000-1000-8000-00805f9b34fb"
AF2 = "00000af2-0000-1000-8000-00805f9b34fb"


def callback(name):
    def handler(sender, data):
        data = bytes(data)

        now = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        print(
            f"{now} | {name} | "
            f"{len(data):2d} bytes | "
            f"{data.hex(' ')}"
        )

    return handler


async def main():

    print("⌚ Connexion CW700...")

    async with BleakClient(WATCH_MAC) as client:

        print("✅ Connectée")
        print()

        # AF7 = canal que nous utilisons déjà pour les activités
        await client.start_notify(
            AF7,
            callback("AF7")
        )

        # AF2 est également une caractéristique NOTIFY.
        # On l'écoute au cas où les commandes média passent ici.
        await client.start_notify(
            AF2,
            callback("AF2")
        )

        print("📡 Sniffer actif sur AF7 + AF2")
        print()
        print("Fais maintenant dans cet ordre :")
        print("  1. PLAY/PAUSE")
        print("  2. NEXT")
        print("  3. PREVIOUS")
        print()
        print("Ctrl+C pour arrêter.")
        print()

        while True:
            await asyncio.sleep(1)


if __name__ == "__main__":

    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print("\n🛑 Sniffer arrêté")