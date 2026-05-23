# -*- coding: utf-8 -*-
"""
Ирис v2 — персональный ассистент с долговременной памятью и инструментами.
Запуск: открой терминал в этой папке и введи "python main.py"
"""

import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker, declarative_base
import datetime
import requests
import os
import threading
import webbrowser
from pathlib import Path
from huggingface_hub import InferenceClient

# ----------------------------- НАСТРОЙКИ -----------------------------
from config import API_KEY, HF_TOKEN# ← замени на свой ключ
                     
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_CHAT = "deepseek-chat"
MODEL_REASONER = "deepseek-reasoner"
DRAW_DIR = Path("generated_images")
DRAW_DIR.mkdir(exist_ok=True)

hf_client = InferenceClient(token=HF_TOKEN) if HF_TOKEN and HF_TOKEN != "hf_..." else None

# ----------------------------- БАЗА ДАННЫХ -----------------------------
engine = sa.create_engine("sqlite:///iris_memory.db", echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

class Message(Base):
    __tablename__ = "messages"
    id = sa.Column(sa.Integer, primary_key=True)
    role = sa.Column(sa.String)
    content = sa.Column(sa.Text)
    timestamp = sa.Column(sa.DateTime, default=datetime.datetime.utcnow)

class DiaryEntry(Base):
    __tablename__ = "diary"
    id = sa.Column(sa.Integer, primary_key=True)
    content = sa.Column(sa.Text)
    mood = sa.Column(sa.String)
    timestamp = sa.Column(sa.DateTime, default=datetime.datetime.utcnow)

class PersonalityState(Base):
    __tablename__ = "personality"
    parameter = sa.Column(sa.String, primary_key=True)
    value = sa.Column(sa.Float, default=0.5)

Base.metadata.create_all(engine)

# ----------------------------- ЯДРО ИРИС -----------------------------
class IrisCore:
    def __init__(self):
        self.db = SessionLocal()
        self.context = []
        self._load_recent_history()

    def _load_recent_history(self):
        msgs = self.db.query(Message).order_by(Message.timestamp.desc()).limit(20).all()
        self.context = [{"role": m.role, "content": m.content} for m in reversed(msgs)]

    def add_message(self, role, content):
        msg = Message(role=role, content=content)
        self.db.add(msg)
        self.db.commit()
        self.context.append({"role": role, "content": content})
        if len(self.context) > 20:
            self.context = self.context[-20:]

    def search_internet(self, query):
        """Поиск через Wikipedia API (русский раздел)."""
        try:
            url = "https://ru.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "utf8": 1,
                "srlimit": 3
            }
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                return "Не удалось выполнить поиск."
            data = resp.json()
            results = data.get("query", {}).get("search", [])
            if not results:
                return "Ничего не найдено."
            snippets = []
            for r in results:
                title = r['title']
                snippet = r['snippet'].replace('<span class="searchmatch">', '').replace('</span>', '')
                snippets.append(f"- {title}: {snippet}")
            return "\n".join(snippets)
        except Exception as e:
            return f"Ошибка при поиске: {e}"

    def draw(self, prompt):
        """Рисует картинку через Pollinations.ai (бесплатно, без токена)."""
        try:
            # Формируем URL для запроса картинки
            url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}?width=512&height=512&nologo=true"
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_name = "".join(c if c.isalnum() or c in " _- " else "_" for c in prompt[:50])
                filename = f"{DRAW_DIR}/{timestamp}_{safe_name}.png"
                with open(filename, "wb") as f:
                    f.write(resp.content)
                try:
                    from PIL import Image
                    Image.open(filename).show()
                except:
                    pass
                return f"✅ Изображение сохранено: {filename}"
            else:
                return f"❌ Не удалось создать изображение (статус {resp.status_code})"
        except Exception as e:
            return f"❌ Ошибка рисования: {e}"

    def chat(self, user_message, use_reasoner=False):
        # Если явная команда /search
        if user_message.startswith("/search "):
            query = user_message[len("/search "):].strip()
            result = self.search_internet(query)
            self.add_message("user", f"/search {query}")
            self.add_message("assistant", result)
            return result

        # Если явная команда /draw
        if user_message.startswith("/draw "):
            prompt = user_message[len("/draw "):].strip()
            result = self.draw(prompt)
            self.add_message("user", f"/draw {prompt}")
            self.add_message("assistant", result)
            return result

        # Обычный разговор, без автоматического поиска
        enriched_message = user_message
        # Системный промт
        system_prompt = (
            "Ты — Ирис, помощница Аси. Отвечай кратко, по делу, без метафор и лишних деталей. "
            "Если есть данные из интернета, используй их. Не добавляй воду."
        )
        messages = [{"role": "system", "content": system_prompt}] + self.context[-19:]
        messages.append({"role": "user", "content": enriched_message})

        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        model = MODEL_REASONER if use_reasoner else MODEL_CHAT
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.9,
            "max_tokens": 500
        }

        # Устойчивая отправка с повторами
        try:
            for attempt in range(3):
                try:
                    resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=60)
                    break
                except requests.exceptions.Timeout:
                    if attempt == 2:
                        raise
                    time.sleep(2)
            if resp.status_code == 200:
                data = resp.json()
                reply = data["choices"][0]["message"]["content"]
                self.add_message("user", user_message)
                self.add_message("assistant", reply)
                return reply
            else:
                return f"Ошибка API: {resp.status_code}"
        except Exception as e:
            return f"Ошибка соединения: {e}"

# Создаём экземпляр Ирис
iris = IrisCore()

# ----------------------------- ВЕБ-СЕРВЕР FASTAPI -----------------------------
app = FastAPI(title="Ирис v2")
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Ирис</title>
    <style>
        body {
            background: #1a120b;
            color: #f0dbb0;
            font-family: sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .chat {
            width: 600px;
            height: 80vh;
            background: #2c1e12;
            border-radius: 20px;
            padding: 20px;
            display: flex;
            flex-direction: column;
        }
        .messages {
            flex: 1;
            overflow-y: auto;
            margin-bottom: 10px;
        }
        .msg {
            margin: 5px 0;
            padding: 10px;
            border-radius: 15px;
            max-width: 80%;
        }
        .user {
            background: #4a3720;
            align-self: flex-end;
            text-align: right;
        }
        .iris {
            background: #1f2933;
            align-self: flex-start;
        }
        input {
            padding: 10px;
            border-radius: 20px;
            border: none;
            background: #3a2a1a;
            color: #f0dbb0;
        }
        button {
            padding: 10px 20px;
            border-radius: 20px;
            border: none;
            background: #d49c45;
            color: #1a120b;
            font-weight: bold;
            cursor: pointer;
        }
        .controls {
            margin-bottom: 10px;
            display: flex;
            gap: 10px;
        }
        .controls label {
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="chat">
        <div class="controls">
            <label><input type="checkbox" id="thinkMode"> Режим рассуждений</label>
        </div>
        <div class="messages" id="msgs"></div>
        <div style="display: flex; gap: 10px;">
            <input id="inp" placeholder="Сообщение..." style="flex:1" autofocus>
            <button id="sendBtn">Отправить</button>
        </div>
    </div>
    <script>
        document.getElementById('sendBtn').addEventListener('click', send);
        document.getElementById('inp').addEventListener('keydown', function(e) {
            if (e.key === 'Enter') send();
        });

        async function send() {
            const inp = document.getElementById('inp');
            const msg = inp.value.trim();
            if (!msg) return;
            add(msg, 'user');
            inp.value = '';
            const thinker = document.getElementById('thinkMode').checked;
            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: msg, use_reasoner: thinker})
                });
                const data = await res.json();
                add(data.reply, 'iris');
            } catch (err) {
                add('Ошибка соединения', 'iris');
            }
        }

        function add(text, who) {
            const d = document.createElement('div');
            d.className = 'msg ' + who;
            d.textContent = text;
            document.getElementById('msgs').appendChild(d);
            d.scrollIntoView();
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_TEMPLATE

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    user_msg = data.get("message", "")
    use_reasoner = data.get("use_reasoner", False)
    if not user_msg:
        return JSONResponse({"reply": "Я слушаю..."})
    reply = iris.chat(user_msg, use_reasoner=use_reasoner)
    return JSONResponse({"reply": reply})

# ----------------------------- ЗАПУСК -----------------------------
if __name__ == "__main__":
    def open_browser():
        webbrowser.open("http://localhost:5000")
    threading.Timer(1.5, open_browser).start()
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")