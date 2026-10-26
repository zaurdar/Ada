import asyncio
import json
from bscpylgtv import WebOsClient

TV_IP = "192.168.0.24"

async def main():

    client = await WebOsClient.create(
        TV_IP,
        manifest_file_path="manifest.json",
        pairing_type="PIN",
        get_hello_info=False,
        states=[]
    )

    try:
        await client.connect()

        print("TV connectée")

        key = client.client_key

        print("Client key :", key)

        with open("lg_tv_key.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ip": TV_IP,
                    "client_key": key
                },
                f,
                indent=4
            )

        print("lg_tv_key.json créé !")

    finally:
        await client.disconnect()


asyncio.run(main())