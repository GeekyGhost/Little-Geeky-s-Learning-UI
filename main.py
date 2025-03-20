# main.py
import asyncio
import logging
from ui.app import LittleGeekyApp

if __name__ == "__main__":
    try:
        app = LittleGeekyApp()
        iface = app.create_interface()
        iface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            show_error=True
        )
    except Exception as e:
        logging.error(f"Failed to launch the WebUI: {e}")
        print(f"Failed to launch the WebUI: {e}")