import os, json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from .translation import translate_text
from .llm_agent import generate_reply

app = FastAPI()

app.mount('/static', StaticFiles(directory=os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static')), name='static')

class ChatMessage(BaseModel):
    text: str
    source_language: str

@app.websocket('/ws/chat')
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)
            text = payload.get('text')
            src = payload.get('source_language', 'en')
            tgt = payload.get('bot_language', 'en')

            if src != 'en':
                user_en = translate_text(text, src, 'en')
            else:
                user_en = text

            bot_en = generate_reply(user_en)
            out_lang = src if tgt == 'source' else tgt
            if out_lang != 'en':
                bot_out = translate_text(bot_en, 'en', out_lang)
            else:
                bot_out = bot_en

            resp_obj = { 'reply': bot_out, 'reply_en': bot_en, 'detected_source': src }
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