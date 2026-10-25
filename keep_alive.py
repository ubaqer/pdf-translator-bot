from flask import Flask
from threading import Thread
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "Bot is running perfectly!"

def run():
    # Render.com provides the PORT environment variable dynamically
    port = int(os.environ.get('PORT', 8080))
    # We run the flask app in host='0.0.0.0' so Render can access it
    app.run(host='0.0.0.0', port=port)

def keep_alive():
    """
    Starts a lightweight web server in a background thread.
    This satisfies Render's 'Web Service' requirement to bind to a port,
    and gives us a URL we can ping via UptimeRobot to keep the bot awake.
    """
    server = Thread(target=run)
    server.start()
