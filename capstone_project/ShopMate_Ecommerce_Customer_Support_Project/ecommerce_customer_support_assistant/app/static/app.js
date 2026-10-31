
const chat = document.getElementById('chat');
const form = document.getElementById('form');
const input = document.getElementById('input');
const send = document.getElementById('send');
const userHistory = [];
function addMessage(text, who, sources=[]) {
  const wrap = document.createElement('div'); wrap.className = `message ${who}`;
  const bubble = document.createElement('div'); bubble.className = 'bubble';
  // Text-only rendering intentionally avoids injecting model output as HTML.
  bubble.textContent = text;
  if (sources.length) {
    const src = document.createElement('div'); src.className = 'sources';
    src.textContent = 'Help-center sources: ' + sources.map(s => s.title).join(', ');
    bubble.appendChild(document.createElement('br')); bubble.appendChild(src);
  }
  wrap.appendChild(bubble); chat.appendChild(wrap); chat.scrollTop = chat.scrollHeight;
}
async function sendMessage(text) {
  text = (text || '').trim(); if (!text) return;
  addMessage(text, 'user'); input.value = ''; send.disabled = true;
  const history = userHistory.slice(-20); userHistory.push(text);
  const loading = document.createElement('div'); loading.className = 'message bot';
  loading.innerHTML = '<div class="bubble">Checking the help center…</div>'; chat.appendChild(loading);
  try {
    const r = await fetch('/api/chat', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text, history})});
    const data = await r.json(); loading.remove();
    addMessage(data.answer + (data.escalation ? '\\n\\nEscalation requested.' : ''), 'bot', data.sources || []);
  } catch(e) { loading.remove(); addMessage('Sorry, the support service is temporarily unavailable. Please try again.', 'bot'); }
  finally { send.disabled = false; input.focus(); }
}
form.addEventListener('submit', e => {e.preventDefault();sendMessage(input.value);});
document.querySelectorAll('[data-msg]').forEach(b => b.addEventListener('click', () => sendMessage(b.dataset.msg)));
input.addEventListener('keydown', e => {if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();form.requestSubmit();}});
