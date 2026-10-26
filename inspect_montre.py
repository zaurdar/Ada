import asyncio
from bleak import BleakClient

WATCH_ADDRESS = "C9:F2:BA:70:EF:8A"


async def main():

    print("Connexion à la montre...")

    async with BleakClient(WATCH_ADDRESS) as client:

        print("Connecté :", client.is_connected)
        print()

        for service in client.services:

            print("=" * 70)
            print("SERVICE")
            print("UUID :", service.uuid)
            print("Description :", service.description)

            for char in service.characteristics:

                print()
                print("   CHARACTERISTIC")
                print("   UUID :", char.uuid)
                print("   Description :", char.description)
                print("   Propriétés :", char.properties)

                for descriptor in char.descriptors:
                    print(
                        "      Descriptor :",
                        descriptor.uuid,
                        descriptor.description
                    )


asyncio.run(main())