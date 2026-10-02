/* Frontend for the Axis MF Facts Chatbot.
   Owns no answer logic: it renders whatever /api/chat returns, including the
   guardrail verdict, so a refusal renders as a refusal rather than being
   smoothed over into an answer. */

const STORAGE_KEY = 'axis_mf_session_id';
const HISTORY_KEY = 'axis_mf_history';
const HISTORY_MAX = 100;
const CHIPS = ['help', 'verified', 'account_balance', 'savings'];
const CHIP_COLORS = ['', 'chip-b', 'chip-c', 'chip-d'];

const el = {
  stream: document.getElementById('stream'),
  form: document.getElementById('composer'),
  input: document.getElementById('chat-input'),
  send: document.getElementById('send-btn'),
  reset: document.getElementById('reset-btn'),
  healthPill: document.getElementById('health-pill'),
  disclaimer: document.getElementById('disclaimer'),
  brandSub: document.getElementById('brand-sub'),
  suggestionRow: document.getElementById('suggestion-row'),
  suggestions: document.getElementById('suggestions'),
  sidebar: document.getElementById('sidebar'),
  scrim: document.getElementById('scrim'),
  menuBtn: document.getElementById('menu-btn'),
  sidebarClose: document.getElementById('sidebar-close'),
  newChat: document.getElementById('new-chat'),
  historyList: document.getElementById('history-list'),
  historyEmpty: document.getElementById('history-empty'),
  historyNote: document.getElementById('history-note'),
  aboutDisclaimer: document.getElementById('about-disclaimer'),
  explorer: document.getElementById('explorer'),
  catPills: document.getElementById('cat-pills'),
  fundCard: document.getElementById('fund-card'),
  fundIcon: document.getElementById('fund-icon'),
  fundName: document.getElementById('fund-name'),
  fundCatPill: document.getElementById('fund-cat-pill'),
  fundTagline: document.getElementById('fund-tagline'),
  fundDocs: document.getElementById('fund-docs'),
  fundStarters: document.getElementById('fund-starters'),
  heroTitle: document.getElementById('hero-title'),
  scopeBar: document.getElementById('scope-bar'),
  scopeChip: document.getElementById('scope-chip'),
  scopeIcon: document.getElementById('scope-icon'),
  scopeLabel: document.getElementById('scope-label'),
  scopeClear: document.getElementById('scope-clear'),
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

/* ---------- history (this browser only) ----------
   The server keeps a rolling window in memory and a free-tier container can be
   evicted at any time, so history is deliberately local: it is an archive of
   what was asked and what was answered, not a replay of live conversation
   state. Replaying turns locally would imply the server still remembers the
   subject, and after a cold start it does not. */

function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((e) => e && typeof e.question === 'string' && e.question);
  } catch (err) {
    return [];
  }
}

function saveHistory(entries) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(entries.slice(-HISTORY_MAX)));
  } catch (err) {
    /* private mode or quota: history is a convenience, never a hard failure */
  }
}

function recordHistory(question, data) {
  const entries = loadHistory();
  entries.push({
    question,
    answer: data.answer || '',
    templateId: data.template_id || '',
    source: data.source_title || '',
    url: data.citation_url || '',
    at: new Date().toISOString(),
  });
  saveHistory(entries);
  renderHistory();
}

function historyTime(iso) {
  const then = new Date(iso);
  if (Number.isNaN(then.getTime())) return '';
  const today = new Date();
  const sameDay = then.toDateString() === today.toDateString();
  return sameDay
    ? then.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : then.toLocaleDateString([], { day: 'numeric', month: 'short' });
}

function renderHistory() {
  const entries = loadHistory();
  el.historyList.innerHTML = '';
  el.historyEmpty.hidden = entries.length > 0;
  el.historyNote.textContent = entries.length
    ? `${entries.length} question${entries.length === 1 ? '' : 's'} · this browser only`
    : 'Saved in this browser only.';

  entries.slice().reverse().forEach((entry) => {
    const item = document.createElement('li');
    item.className = 'hist';

    const details = document.createElement('details');
    const summary = document.createElement('summary');

    const question = document.createElement('span');
    question.className = 'hist-q';
    question.textContent = entry.question;

    const meta = document.createElement('span');
    meta.className = 'hist-meta';
    const tag =
      entry.templateId === 'FACTUAL'
        ? 'grounded'
        : entry.templateId === 'ERROR'
          ? 'error'
          : entry.templateId || 'note';
    meta.textContent = `${historyTime(entry.at)} · ${tag}`;

    summary.appendChild(question);
    summary.appendChild(meta);

    const body = document.createElement('div');
    body.className = 'hist-body';

    const answer = document.createElement('p');
    answer.textContent = entry.answer;
    body.appendChild(answer);

    if (entry.url) {
      const link = document.createElement('a');
      link.className = 'hist-link';
      link.href = entry.url;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = entry.source || 'Official Axis MF source';
      body.appendChild(link);
    }

    details.appendChild(summary);
    details.appendChild(body);
    item.appendChild(details);
    el.historyList.appendChild(item);
  });
}

function clearHistory() {
  try {
    localStorage.removeItem(HISTORY_KEY);
  } catch (err) {
    /* nothing to do */
  }
  renderHistory();
}

/* ---------- sidebar ---------- */

const TABS = [
  { button: document.getElementById('tab-history'), panel: document.getElementById('panel-history') },
  { button: document.getElementById('tab-about'), panel: document.getElementById('panel-about') },
];

function selectTab(index) {
  TABS.forEach((tab, i) => {
    const active = i === index;
    tab.button.classList.toggle('is-active', active);
    tab.button.setAttribute('aria-selected', String(active));
    tab.panel.hidden = !active;
  });
}

function setSidebar(open) {
  el.sidebar.classList.toggle('is-open', open);
  el.scrim.hidden = !open;
  el.menuBtn.setAttribute('aria-expanded', String(open));
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

  if (state.active) {
    const note = document.createElement('div');
    note.className = 'scope-note';
    note.textContent = `Asking about ${state.active.title}`;
    turn.appendChild(note);
  }

  el.stream.appendChild(turn);
  scrollDown();
}

function typingTurn() {
  const turn = document.createElement('div');
  turn.className = 'turn bot';
  turn.innerHTML =
    '<div class="bot-head"><div class="bot-avatar">' +
    '<span class="material-symbols-outlined">auto_awesome</span></div>' +
    '<span class="bot-name">IND Money AI</span></div>' +
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
    '<span class="bot-name">IND Money AI</span>' +
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
  return node;
}

function scrollDown() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
}

function clearStream() {
  el.stream.innerHTML = '';
  showExplorer(true);
  closeCategory();
}

function setBusy(busy) {
  el.send.disabled = busy;
  el.input.disabled = busy;
}

async function ask(question) {
  if (!question || el.send.disabled) return;
  addUserTurn(question);
  el.input.value = '';
  showExplorer(false);
  closeCategory();
  setBusy(true);
  const placeholder = typingTurn();

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: scopedQuestion(question),
        session_id: sessionId(),
      }),
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
    if (payload.template_id !== 'ERROR') recordHistory(question, payload);
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

const WARM_POLL_MS = 2000;
const WARM_MAX_POLLS = 45;

function setHealthPill(text, modifier) {
  el.healthPill.textContent = text;
  el.healthPill.className = `pill ${modifier}`;
}

/* The container warms the embedding model in the background at boot. A free
   instance is slow enough that the first question can otherwise arrive before
   the model is resident, so poll rather than leaving "warming" on screen. */
/* ---------- fund category browser ----------
   Categories come from /api/config, which the server builds from
   config.FUND_CATALOG. Only schemes that are actually indexed are listed: a
   card the corpus cannot answer would return NOT_FOUND and read as a bug.

   Selecting a card scopes later questions. The scoping is done by prefixing the
   scheme name onto the text sent to the API, which routes through the existing
   SCHEME_ALIASES metadata filter rather than adding a second, divergent path for
   choosing a filter. */

const state = {
  categories: [],
  active: null,
};

function showExplorer(show) {
  el.explorer.hidden = !show;
  el.suggestions.hidden = show;
}

function setScope(category) {
  state.active = category || null;
  if (!category) {
    el.scopeBar.hidden = true;
    return;
  }
  el.scopeLabel.textContent = category.title;
  el.scopeIcon.textContent = category.icon;
  el.scopeBar.hidden = false;
}

/* The text actually sent upstream, so retrieval filters on the right scheme. */
function scopedQuestion(question) {
  if (!state.active) return question;
  const scheme = SCHEME_PHRASE[state.active.scheme];
  return scheme ? `${scheme}: ${question}` : question;
}

const SCHEME_PHRASE = {
  large_cap: 'Axis Large Cap Fund',
  flexi_cap: 'Axis Flexi Cap Fund',
  elss: 'Axis ELSS Tax Saver Fund',
  midcap: 'Axis Midcap Fund',
  amc_wide: 'Axis Mutual Fund',
};

/* DESIGN.md calls these "interactive badges and chips": fully pill-shaped quick
   filters, so a fund reads as a filter you can tap rather than a report card. */
function renderCategories() {
  el.catPills.innerHTML = '';
  state.categories.forEach((category) => {
    const pill = document.createElement('button');
    pill.type = 'button';
    pill.className = 'chip';
    pill.dataset.scheme = category.scheme;

    const glyph = document.createElement('span');
    glyph.className = 'material-symbols-outlined';
    glyph.textContent = category.icon;

    const label = document.createElement('span');
    label.textContent = category.category;

    pill.appendChild(glyph);
    pill.appendChild(label);
    el.catPills.appendChild(pill);
  });
}

const DOC_LABEL = {
  scheme_page: 'Scheme page',
  sid: 'SID',
  kim: 'Key Information Memorandum',
  factsheet: 'Fund factsheet',
  efactsheet: 'E-Factsheet',
  downloads: 'Downloads',
  homepage: 'AMC homepage',
};

function openCategory(category) {
  el.fundIcon.firstElementChild.textContent = category.icon;
  el.fundName.textContent = category.title;
  el.fundCatPill.textContent = category.category;
  el.fundTagline.textContent = category.tagline;

  el.fundDocs.innerHTML = '';
  (category.documents || []).forEach((doc) => {
    const pill = document.createElement('span');
    pill.className = 'pill pill-ok';
    pill.textContent = DOC_LABEL[doc] || doc;
    el.fundDocs.appendChild(pill);
  });

  el.fundStarters.innerHTML = '';
  category.starters.forEach((text) => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'chip chip-q';
    const glyph = document.createElement('span');
    glyph.className = 'material-symbols-outlined';
    glyph.textContent = 'north_east';
    const label = document.createElement('span');
    label.textContent = text;
    chip.appendChild(glyph);
    chip.appendChild(label);
    chip.addEventListener('click', () => ask(text));
    el.fundStarters.appendChild(chip);
  });

  Array.from(el.catPills.children).forEach((pill) => {
    pill.classList.toggle('is-active', pill.dataset.scheme === category.scheme);
  });

  el.fundCard.hidden = false;
  el.heroTitle.textContent = 'Here is what you can ask';
  setScope(category);
  el.fundCard.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

function closeCategory() {
  el.fundCard.hidden = true;
  el.heroTitle.textContent = 'Where should we start?';
  Array.from(el.catPills.children).forEach((pill) => pill.classList.remove('is-active'));
  setScope(null);
}

async function waitForWarm() {
  for (let attempt = 0; attempt < WARM_MAX_POLLS; attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, WARM_POLL_MS));
    let health;
    try {
      const response = await fetch('/api/health');
      if (!response.ok) continue;
      health = await response.json();
    } catch (err) {
      continue;
    }
    if (health.pipeline === 'ready') {
      setHealthPill('index ready', 'pill-live');
      el.suggestions.style.display = '';
      return;
    }
    if (health.pipeline === 'failed') {
      setHealthPill('index failed', 'pill-down');
      el.brandSub.textContent = health.pipeline_problem || 'The search index could not be loaded.';
      return;
    }
  }
  setHealthPill('index slow to load', 'pill-amber');
}

async function boot() {
  clearStream();
  renderHistory();
  selectTab(0);
  try {
    const [configRes, healthRes] = await Promise.all([
      fetch('/api/config'),
      fetch('/api/health'),
    ]);
    const cfg = await configRes.json();
    el.disclaimer.textContent = cfg.disclaimer || '';
    el.aboutDisclaimer.textContent = cfg.disclaimer || '';
    state.categories = cfg.categories || [];
    renderCategories();
    renderSuggestions(cfg.suggestions);

    const health = await healthRes.json();
    if (!health.ok) {
      setHealthPill('not ready', 'pill-down');
      el.brandSub.textContent = health.problem || '';
      el.suggestions.style.display = 'none';
      return;
    }
    if (health.pipeline === 'ready') {
      setHealthPill('index ready', 'pill-live');
      return;
    }
    if (health.pipeline === 'failed') {
      setHealthPill('index failed', 'pill-down');
      el.brandSub.textContent = health.pipeline_problem || 'The search index could not be loaded.';
      return;
    }
    setHealthPill('warming index', 'pill-amber');
    waitForWarm();
  } catch (err) {
    setHealthPill('offline', 'pill-down');
  }
}

el.form.addEventListener('submit', (event) => {
  event.preventDefault();
  ask(el.input.value.trim());
});

TABS.forEach((tab, index) => {
  tab.button.addEventListener('click', () => selectTab(index));
});

el.menuBtn.addEventListener('click', () => setSidebar(true));
el.sidebarClose.addEventListener('click', () => setSidebar(false));
el.scrim.addEventListener('click', () => setSidebar(false));

el.catPills.addEventListener('click', (event) => {
  /* Clicking the already-selected fund clears the context, which makes the
     quick filter behave like a toggle the way a filter chip should. */
  const pill = event.target.closest('.chip');
  if (!pill) return;
  const category = state.categories.find((item) => item.scheme === pill.dataset.scheme);
  if (!category) return;
  if (state.active && state.active.scheme === category.scheme) closeCategory();
  else openCategory(category);
});

el.scopeClear.addEventListener('click', () => {
  closeCategory();
  el.input.focus();
});

el.newChat.addEventListener('click', async () => {
  el.newChat.disabled = true;
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
  clearHistory();
  setSidebar(false);
  el.newChat.disabled = false;
  el.input.focus();
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