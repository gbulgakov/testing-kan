import requests
from utils import load_secrets

secrets = load_secrets()
TELEGRAM_TOKEN = secrets.get('telegram', {}).get('bot_token')
GEORGY_ID = secrets.get('telegram', {}).get('georgy_id')
DANIL_ID = secrets.get('telegram', {}).get('danil_id')


# --- 2. Отправляем уведомление и файл в Telegram ---
def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for CHAT_ID in [GEORGY_ID, DANIL_ID]:
        params = {"chat_id": CHAT_ID, "text": text,  "parse_mode": "MarkdownV2"}
        requests.post(url, params=params)
    
def send_telegram_file(file_path):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
    for CHAT_ID in [GEORGY_ID, DANIL_ID]:
        with open(file_path, 'rb') as f:
            files = {"document": f}
            data = {"chat_id": CHAT_ID}
            requests.post(url, files=files, data=data)
