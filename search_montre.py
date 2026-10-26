import asyncio
from bleak import BleakScanner


def detection_callback(device, advertisement_data):
    print(
        f"{device.name!s:25} "
        f"{device.address:20} "
        f"RSSI={advertisement_data.rssi:4} dBm "
        f"{advertisement_data.service_uuids}"
    )


async def main():

    scanner = BleakScanner(
        detection_callback=detection_callback
    )

    print("Scan BLE en continu...")
    print("Allume maintenant la montre.\n")

    await scanner.start()

    try:
        while True:
            await asyncio.sleep(1)

    finally:
        await scanner.stop()


asyncio.run(main())