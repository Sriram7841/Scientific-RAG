/* GraphRAG — Shared JS utilities */

function getCsrf() {
  return (document.cookie.split(';')
    .map(c => c.trim())
    .find(c => c.startsWith('csrftoken=')) || '').split('=')[1] || '';
}

function esc(s) {
  const d = document.createElement('div');
  d.appendChild(document.createTextNode(s || ''));
  return d.innerHTML;
}

function showToast(msg, type = 'success') {
  const el = document.createElement('div');
  el.className = `fixed bottom-6 right-6 z-50 px-5 py-3 rounded-xl shadow-xl
    text-white text-sm font-medium transition-all
    ${type === 'error' ? 'bg-red-500' : 'bg-gradient-to-r from-brand-DEFAULT to-sky-DEFAULT'}`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3000);
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 180) + 'px';
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    showToast('Copied to clipboard!');
  } catch {
    showToast('Copy failed', 'error');
  }
}

function renderMd(text) {
  return (text || '')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>')
    .replace(/\*(.*?)\*/g,'<em>$1</em>')
    .replace(/`(.*?)`/g,'<code>$1</code>')
    .replace(/\n\n/g,'</p><p class="mb-2">')
    .replace(/\n/g,'<br>');
}
