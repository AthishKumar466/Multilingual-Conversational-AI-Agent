const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + window.location.host + '/ws/chat');

const messages = document.getElementById('messages');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const langSel = document.getElementById('lang');

ws.onopen = () => { appendSystem('Connected to server'); }

ws.onmessage = (evt) => {
  let data = JSON.parse(evt.data);
  if (data.error) {
    appendSystem('Error: ' + data.error);
    return;
  }
  appendBot(data.reply);
}

ws.onclose = () => appendSystem('Disconnected');

function appendSystem(text) { const d=document.createElement('div');d.className='msg';d.textContent=text;messages.appendChild(d);}
function appendUser(text) { const d=document.createElement('div');d.className='msg user';d.textContent='You: '+text;messages.appendChild(d);}
function appendBot(text) { const d=document.createElement('div');d.className='msg bot';d.textContent='Bot: '+text;messages.appendChild(d);}

sendBtn.onclick = () => {
  const text = input.value.trim();
  if (!text) return;
  const payload = { text, source_language: langSel.value, bot_language: langSel.value };
  ws.send(JSON.stringify(payload));
  appendUser(text);
  input.value = '';
}