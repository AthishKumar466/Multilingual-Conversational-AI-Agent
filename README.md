# Multilingual-Conversational-AI-Agent*
This document contains a complete, ready-to-run skeleton for a multilingual chatbot using FastAPI backend and a simple frontend. It demonstrates:

* FastAPI backend with WebSocket-based real-time chat.
* Language detection and translation using Hugging Face `transformers` (MarianMT / mBART pipelines).
* LangChain wrapper (simple example) to call a GPT-style LLM for responses.
* A lightweight static frontend (HTML + JS) that connects via WebSocket and shows multilingual UI.

> **Notes & placeholders:**
>
> * You must provide your own OpenAI (or compatible) API key and set it in environment variables.
> * Model downloads (Hugging Face) will occur at runtime; ensure internet access.

---

## Project structure

```
multilingual-chatbot/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── translation.py
│   │   └── llm_agent.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   └── static/
│       └── app.js
├── docker-compose.yml
└── README.md
```

---

## backend/requirements.txt

```text
fastapi==0.95.2
uvicorn[standard]==0.22.0
transformers==4.40.0
sentencepiece==0.1.99
torch>=1.12.0
langchain==0.0.420
openai==1.0.0
python-multipart==0.0.6
pydantic==1.10.7
httpx==0.24.1
python-dotenv==1.0.0
```

---

## backend/app/translation.py

```python
# translation.py
# Utilities for language detection and translation using HuggingFace pipelines

from typing import Tuple

from transformers import pipeline

# We'll lazily load the models to avoid startup cost.
_translation_pipelines = {}

LANGUAGE_MODELS = {
    # MarianMT examples (source-target): use model names appropriate for your languages.
    # For production, pick specific source->target models or use mBART for many-to-many.
    # Here are a few example model IDs (may require updating):
    'en->hi': 'Helsinki-NLP/opus-mt-en-hi',
    'hi->en': 'Helsinki-NLP/opus-mt-hi-en',
    'en->ja': 'Helsinki-NLP/opus-mt-en-jap',
    'ja->en': 'Helsinki-NLP/opus-mt-ja-en',
}


def _get_pipeline(key: str):
    if key in _translation_pipelines:
        return _translation_pipelines[key]
    model_id = LANGUAGE_MODELS.get(key)
    if not model_id:
        raise ValueError(f"No translation model configured for {key}")
    pipe = pipeline('translation', model=model_id)
    _translation_pipelines[key] = pipe
    return pipe


def translate_text(text: str, src: str, tgt: str) -> str:
    """Translate text from `src` to `tgt`. src/tgt are ISO codes like 'en','hi','ja'.
    This function chooses a pipeline by building a key 'src->tgt' and calling the model.
    """
    key = f"{src}->{tgt}"
    pipe = _get_pipeline(key)
    out = pipe(text)
    # pipeline returns a list of dicts with 'translation_text'
    return out[0]["translation_text"]


# Optional helper: fallback to mBART-like many-to-many (commented for brevity)
# from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
# ... implement many-to-many if you prefer
```

---

## backend/app/llm\_agent.py

```python
# llm_agent.py
# Minimal LangChain + OpenAI wrapper to take an input prompt and return a reply.

import os
from typing import Optional

from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"


def get_llm(api_key: Optional[str] = None):
    api_key = api_key or os.environ.get(OPENAI_API_KEY_ENV)
    if not api_key:
        raise RuntimeError("OpenAI API key is not set. Set OPENAI_API_KEY environment variable.")
    # This will use the environment key; adjust parameters as needed.
    llm = OpenAI(openai_api_key=api_key, temperature=0.2)
    return llm


SYSTEM_PROMPT = """
You are a helpful multilingual assistant. When a user message is received (in English), respond concisely and politely.
If the input was translated from another language, try to preserve tone and formality of the original message.
"""

PROMPT_TEMPLATE = """
{system}
User: {user}
Assistant:
"""


def generate_reply(user_text: str, system: str = SYSTEM_PROMPT, api_key: Optional[str] = None) -> str:
    llm = get_llm(api_key=api_key)
    prompt = PromptTemplate(input_variables=["system", "user"], template=PROMPT_TEMPLATE)
    chain = LLMChain(llm=llm, prompt=prompt)
    resp = chain.run({"system": system, "user": user_text})
    return resp.strip()
```

---

## backend/app/main.py

```python
# main.py
# FastAPI app that ties translation + llm agent together and exposes a WebSocket for real-time chat.

import os
import json
from typing import Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .translation import translate_text
from .llm_agent import generate_reply

app = FastAPI()

# serve frontend files if you place them at backend/static
app.mount('/static', StaticFiles(directory=os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static')), name='static')


class ChatMessage(BaseModel):
    text: str
    source_language: str  # e.g. 'hi', 'en', 'ja'


@app.websocket('/ws/chat')
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)
            # expected keys: text, source_language, target_language
            text = payload.get('text')
            src = payload.get('source_language', 'en')
            tgt = payload.get('bot_language', 'en')

            # Step 1: translate user message to English for LLM (if not already English)
            if src != 'en':
                try:
                    user_en = translate_text(text, src, 'en')
                except Exception as e:
                    await websocket.send_text(json.dumps({'error': f'Translation failed: {e}'}))
                    continue
            else:
                user_en = text

            # Step 2: generate reply in English
            try:
                bot_en = generate_reply(user_en)
            except Exception as e:
                await websocket.send_text(json.dumps({'error': f'LLM call failed: {e}'}))
                continue

            # Step 3: translate bot reply back to user's language (or requested bot_language)
            out_lang = src if tgt == 'source' else tgt
            if out_lang != 'en':
                try:
                    bot_out = translate_text(bot_en, 'en', out_lang)
                except Exception as e:
                    await websocket.send_text(json.dumps({'error': f'Reverse translation failed: {e}'}))
                    continue
            else:
                bot_out = bot_en

            # Step 4: send response
            resp_obj = {
                'reply': bot_out,
                'reply_en': bot_en,
                'detected_source': src,
            }
            await websocket.send_text(json.dumps(resp_obj))

    except WebSocketDisconnect:
        print('Client disconnected')


@app.get('/')
async def index():
    html = HTMLResponse(open(os.path.join(os.path.dirname(__file__), '..', 'frontend', 'index.html')).read())
    return html


@app.post('/translate')
async def translate_api(msg: ChatMessage, target: str = 'en'):
    try:
        out = translate_text(msg.text, msg.source_language, target)
        return {'translated': out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## frontend/index.html

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Multilingual Chatbot</title>
  <style>
    body { font-family: Arial, sans-serif; padding: 1rem; }
    #messages { border: 1px solid #ddd; height: 300px; overflow: auto; padding: .5rem; }
    .msg { margin: .25rem 0 }
    .user { font-weight: 600 }
  </style>
</head>
<body>
  <h1>Multilingual Chatbot</h1>
  <label>Language: <select id="lang"><option value="en">English</option><option value="hi">Hindi</option><option value="ja">Japanese</option></select></label>
  <div id="messages"></div>
  <textarea id="input" rows="3" style="width:100%"></textarea>
  <button id="send">Send</button>

  <script src="/static/app.js"></script>
</body>
</html>
```

---

## frontend/static/app.js

```javascript
const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + window.location.host + '/ws/chat');

const messages = document.getElementById('messages');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const langSel = document.getElementById('lang');

ws.onopen = () => {
  appendSystem('Connected to server');
}

ws.onmessage = (evt) => {
  let data = JSON.parse(evt.data);
  if (data.error) {
    appendSystem('Error: ' + data.error);
    return;
  }
  appendBot(data.reply);
}

ws.onclose = () => appendSystem('Disconnected');

function appendSystem(text) {
  const d = document.createElement('div'); d.className='msg'; d.textContent = text; messages.appendChild(d);
}
function appendUser(text) {
  const d = document.createElement('div'); d.className='msg user'; d.textContent = 'You: ' + text; messages.appendChild(d);
}
function appendBot(text) {
  const d = document.createElement('div'); d.className='msg bot'; d.textContent = 'Bot: ' + text; messages.appendChild(d);
}

sendBtn.onclick = () => {
  const text = input.value.trim();
  if (!text) return;
  const payload = { text, source_language: langSel.value, bot_language: langSel.value };
  ws.send(JSON.stringify(payload));
  appendUser(text);
  input.value = '';
}
```

---

## docker-compose.yml (optional)

```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - '8000:8000'
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./frontend:/app/frontend
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## README.md (quick start)

````markdown
# Multilingual Chatbot — Quick start

1. Create a virtual env and install python packages from `backend/requirements.txt`.

2. Set environment variable `OPENAI_API_KEY`.

3. Put the `frontend` folder next to `backend` folder, or mount it.

4. From the `backend` folder run:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
````

5. Open [http://localhost:8000](http://localhost:8000) in a browser.

Notes:

* The translations expect the transformer model IDs in `translation.py` to be valid. Replace or add models suited for your exact languages.
* For production, consider caching loaded models, batching translation requests, and using paid inference endpoints.

```

---

## Next steps & improvements

- Add language detection (e.g., `langdetect` or `fasttext`) to auto-detect user language.
- Use a many-to-many model (mBART) for broader language coverage.
- Add authentication, rate limiting, and usage logging.
- Build a React frontend and use WebRTC / socket.io if you prefer richer real-time features.

---

If you'd like, I can now:
- Split these files into separate downloadable files (ZIP) and provide them.
- Replace LangChain/OpenAI use with direct OpenAI API calls or another LLM provider.
- Convert frontend into a React app.

Tell me which one and I will lay out the files for download.

```
