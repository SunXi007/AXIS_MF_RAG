/* Frontend for the Axis MF Facts Chatbot.
   Owns no answer logic: it renders whatever /api/chat returns, including the
   guardrail verdict, so a refusal renders as a refusal rather than being
   smoothed over into an answer. */

const STORAGE_KEY = 'axis_mf_session_id';
const CHIPS = ['help', 'verified', 'account_balance', 'savings'];
const CHIP_COLORS = ['', 'chip-b', 'chip-c', 'chip-d'];

const el = {
  stream: document.getElementById('stream'),
  form: document.getElementById('composer'),
  input: document.getElementById('chat-input'),
  send: document.getElementById('send-btn'),
  reset: document.getElementById('reset-btn'),
  memory: document.getElementById('memory-label'),
  healthPill: document.getElementById('health-pill'),
  modelPill: document.getElementById('model-pill'),
  disclaimer: document.getElementById('disclaimer'),
  brandSub: document.getElementById('brand-sub'),
  suggestionRow: document.getElementById('suggestion-row'),
  suggestions: document.getElementById('suggestions'),
};

function sessionId() {
  let id = localStorage.getItem(STORAGE_KEY);
  if (!/^[0-9a-f]{32}$/.test(id || '')) {
    const bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);
    id = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
    localStorage.setItem(STORAGE_KEY, id);
  }
  return id;
}

function stamp() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function addUserTurn(question) {
  const turn = document.createElement('div');
  turn.className = 'turn user';
  turn.innerHTML =
    '<div class="bubble-user"><p></p></div>' +
    '<div class="stamp"><span></span>' +
    '<span class="material-symbols-outlined">done_all</span></div>';
  turn.querySelector('p').textContent = question;
  turn.querySelector('.stamp span').textContent = stamp();
  el.stream.appendChild(turn);
  scrollDown();
}

function typingTurn() {
  const turn = document.createElement('div');
  turn.className = 'turn bot';
  turn.innerHTML =
    '<div class="bot-head"><div class="bot-avatar">' +
    '<span class="material-symbols-outlined">auto_awesome</span></div>' +
    '<span class="bot-name">Axis Facts Assistant</span></div>' +
    '<div class="bubble-bot"><div class="typing"><span></span><span></span><span></span></div></div>';
  el.stream.appendChild(turn);
  scrollDown();
  return turn;
}

function noticeBlock(title, detail, verdict) {
  const node = document.createElement('div');
  node.className = 'notice';
  node.innerHTML =
    '<span class="material-symbols-outlined">gpp_maybe</span>' +
    '<div class="notice-body"><strong></strong><span></span>' +
    '<code></code></div>';
  node.querySelector('strong').textContent = title;
  node.querySelector('.notice-body span').textContent = detail;
  node.querySelector('code').textContent = verdict ? `verdict: ${verdict}` : '';
  return node;
}

function provenanceBlock(data) {
  const wrap = document.createElement('details');
  wrap.className = 'prov';
  const hits = data.hits || [];
  const summary = document.createElement('summary');
  summary.innerHTML =
    `<span class="material-symbols-outlined">receipt_long</span>` +
    `<span>${hits.length} retrieved chunk${hits.length === 1 ? '' : 's'}</span>`;
  wrap.appendChild(summary);

  const list = document.createElement('div');
  list.className = 'prov-list';
  hits.forEach((hit) => {
    const item = document.createElement('div');
    item.className = 'prov-item';
    const label = hit.scheme || hit.source_id || 'chunk';
    item.innerHTML =
      '<div class="prov-top"><span class="prov-id"></span>' +
      '<span class="score"><span class="material-symbols-outlined">trending_up</span><span></span></span></div>' +
      '<div class="prov-meta"></div><p class="prov-excerpt"></p>';
    item.querySelector('.prov-id').textContent = hit.chunk_id;
    item.querySelector('.score span:last-child').textContent = hit.score;
    item.querySelector('.prov-meta').textContent = label;
    item.querySelector('.prov-excerpt').textContent = hit.excerpt || '';
    list.appendChild(item);
  });
  wrap.appendChild(list);
  return wrap;
}

function renderAnswer(data) {
  const turn = document.createElement('div');
  turn.className = 'turn bot';

  const head = document.createElement('div');
  head.className = 'bot-head';
  head.innerHTML =
    '<div class="bot-avatar"><span class="material-symbols-outlined">auto_awesome</span></div>' +
    '<span class="bot-name">Axis Facts Assistant</span>' +
    '<span class="bot-tag"></span>';
  turn.appendChild(head);
  head.querySelector('.bot-tag').textContent =
    data.template_id === 'FACTUAL' ? 'grounded' : data.template_id;

  const bubble = document.createElement('div');
  bubble.className = 'bubble-bot';

  const body = document.createElement('p');
  body.textContent = data.answer;
  bubble.appendChild(body);

  if (data.rewritten_question) {
    const rw = document.createElement('div');
    rw.className = 'rewrite';
    rw.innerHTML =
      '<span class="material-symbols-outlined">link</span><span></span>';
    rw.querySelector('span:last-child').textContent =
      `Follow-up resolved. Searched for: ${data.rewritten_question}`;
    bubble.appendChild(rw);
  }

  if (data.citation_url) {
    const cite = document.createElement('div');
    cite.className = 'citation';
    const link = document.createElement('a');
    link.href = data.citation_url;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.textContent = data.source_title || 'Official Axis MF source';
    const stampNode = document.createElement('small');
    stampNode.textContent = data.last_updated
      ? `Last updated from sources: ${data.last_updated}`
      : 'Last updated from sources: date not stated on page';
    cite.appendChild(link);
    cite.appendChild(stampNode);
    bubble.appendChild(cite);
  }

  if (data.template_id !== 'FACTUAL') {
    const detail = (data.matched || []).join(', ') || 'no grounded answer';
    bubble.appendChild(
      noticeBlock(
        'This answer was not generated from the sources',
        detail,
        data.verdict
      )
    );
  }

  if ((data.hits || []).length) {
    bubble.appendChild(provenanceBlock(data));
  }

  if (data.latency_ms && Object.keys(data.latency_ms).length) {
    const lat = document.createElement('div');
    lat.className = 'latency';
    lat.textContent = Object.entries(data.latency_ms)
      .map(([k, v]) => `${k} ${v}ms`)
      .join(' · ');
    bubble.appendChild(lat);
  }

  turn.appendChild(bubble);
  el.stream.appendChild(turn);
  scrollDown();
}

function renderEmpty() {
  const node = document.createElement('div');
  node.className = 'empty';
  node.innerHTML =
    '<h2>Ask a factual question about an Axis Mutual Fund</h2>' +
    '<p>Exit loads, expense ratios, risk profile, ELSS lock-in and plan details. ' +
    'Answers are quoted from official Axis MF pages and statutory documents, with the source linked. ' +
    'If the sources do not cover it, the assistant says so instead of guessing.</p>';
  el.stream.appendChild(node);
}

function scrollDown() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
}

function clearStream() {
  el.stream.innerHTML = '';
  renderEmpty();
  el.memory.textContent = 'no history';
}

function setBusy(busy) {
  el.send.disabled = busy;
  el.input.disabled = busy;
}

async function ask(question) {
  if (!question || el.send.disabled) return;
  addUserTurn(question);
  el.input.value = '';
  setBusy(true);
  const placeholder = typingTurn();

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, session_id: sessionId() }),
    });
    const payload = await response.json();
    placeholder.remove();
    if (!response.ok) {
      renderAnswer({
        answer: payload.detail || 'The assistant is unavailable right now.',
        template_id: 'ERROR',
        verdict: 'server',
        hits: [],
      });
      return;
    }
    renderAnswer(payload);
    if (payload.memory) el.memory.textContent = payload.memory;
  } catch (err) {
    placeholder.remove();
    renderAnswer({
      answer: 'Could not reach the server.',
      template_id: 'ERROR',
      verdict: 'network',
      hits: [],
    });
  } finally {
    setBusy(false);
    el.input.focus();
  }
}

function renderSuggestions(list) {
  el.suggestionRow.innerHTML = '';
  (list || []).forEach((text, index) => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = `chip ${CHIP_COLORS[index % CHIP_COLORS.length]}`;
    const icon = document.createElement('span');
    icon.className = 'material-symbols-outlined';
    icon.textContent = CHIPS[index % CHIPS.length];
    const label = document.createElement('span');
    label.textContent = text;
    chip.appendChild(icon);
    chip.appendChild(label);
    chip.addEventListener('click', () => ask(text));
    el.suggestionRow.appendChild(chip);
  });
}

async function boot() {
  clearStream();
  try {
    const [configRes, healthRes] = await Promise.all([
      fetch('/api/config'),
      fetch('/api/health'),
    ]);
    const cfg = await configRes.json();
    el.disclaimer.textContent = cfg.disclaimer || '';
    el.modelPill.textContent = cfg.model || '';
    renderSuggestions(cfg.suggestions);

    const health = await healthRes.json();
    if (health.ok) {
      el.healthPill.textContent = 'index ready';
      el.healthPill.className = 'pill pill-live';
    } else {
      el.healthPill.textContent = 'not ready';
      el.healthPill.className = 'pill pill-down';
      el.brandSub.textContent = health.problem || '';
      el.suggestions.style.display = 'none';
    }
  } catch (err) {
    el.healthPill.textContent = 'offline';
    el.healthPill.className = 'pill pill-down';
  }
}

el.form.addEventListener('submit', (event) => {
  event.preventDefault();
  ask(el.input.value.trim());
});

el.reset.addEventListener('click', async () => {
  el.reset.disabled = true;
  try {
    await fetch('/api/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId() }),
    });
  } catch (err) {
    /* the local view is cleared regardless */
  }
  clearStream();
  el.reset.disabled = false;
  el.input.focus();
});

boot();