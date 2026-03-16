/* ============================================================
   FakeJob Shield — app.js
   ============================================================ */

// ── Example data ──────────────────────────────────────────────
const EXAMPLES = [
  {
    type:  'fake',
    title: 'Registration Fee Scam',
    text:  'Work from home internship. No experience required. Pay Rs.999 registration fee to get started. Earn Rs.50,000 per month guaranteed. Send your Aadhar card photo to confirm your seat. WhatsApp us immediately. Limited slots!',
    tags:  ['Registration fee', 'Guaranteed salary', 'Aadhar request', 'Urgency'],
  },
  {
    type:  'fake',
    title: 'Data Entry Too-Good-To-Be-True',
    text:  'URGENT HIRING! Online data entry job from home. Work 2-3 hours daily. Earn $5000 per week. No interview needed. No qualification required. Just pay a small security deposit of Rs.1999 which will be refunded. Call now: +91-9999999999.',
    tags:  ['Urgency', 'Unrealistic pay', 'Security deposit', 'No qualification'],
  },
  {
    type:  'legit',
    title: 'Software Engineering Intern',
    text:  'Software Engineering Intern at TechCorp Solutions Pvt. Ltd. Seeking undergraduate students from CS/IT background with knowledge of Python, JavaScript, or Java. Intern will work on real product features, attend sprint meetings, and receive mentoring. Duration: 3 months. Stipend: Rs.15,000/month. Location: Bangalore.',
    tags:  ['Clear company', 'Skill requirement', 'Structured stipend', 'Real location'],
  },
  {
    type:  'legit',
    title: 'Marketing Intern at Startup',
    text:  'Marketing Intern at Meesho Inc. Responsibilities: managing social media accounts, creating content calendars, running paid campaigns, tracking analytics. Requirements: BBA/MBA student, good communication, knowledge of Canva. Duration: 6 months. Stipend: Rs.12,000/month. Hybrid mode.',
    tags:  ['Defined role', 'Educational req.', 'Hybrid mode', 'Specific tools'],
  },
];

// ── Build example cards ──────────────────────────────────────
function buildExamples() {
  const grid = document.getElementById('examplesGrid');
  grid.innerHTML = EXAMPLES.map((ex, i) => `
    <div class="example-card ${ex.type}" onclick="runExample(${i})">
      <div class="ex-badge ${ex.type}">${ex.type === 'fake' ? '🔴 FAKE' : '🟢 LEGIT'}</div>
      <div class="ex-title">${ex.title}</div>
      <div class="ex-text">${ex.text.slice(0, 140)}…</div>
      <div class="ex-tags">${ex.tags.map(t => `<span class="ex-tag">${t}</span>`).join('')}</div>
      <button class="ex-btn">▶ Run Detection</button>
    </div>
  `).join('');
}

// ── Word count live update ────────────────────────────────────
const inputEl   = document.getElementById('descriptionInput');
const countEl   = document.getElementById('wordCount');

inputEl.addEventListener('input', () => {
  const words = inputEl.value.trim().split(/\s+/).filter(Boolean).length;
  countEl.textContent = `${words} word${words !== 1 ? 's' : ''}`;
  countEl.style.color = words < 10 ? '#fbbf24' : '#4ade80';
});

// ── Clear ─────────────────────────────────────────────────────
function clearInput() {
  inputEl.value = '';
  countEl.textContent = '0 words';
  countEl.style.color = '';
  hideTip();
  resetResult();
}

// ── Run example ───────────────────────────────────────────────
function runExample(i) {
  inputEl.value = EXAMPLES[i].text;
  inputEl.dispatchEvent(new Event('input'));
  window.scrollTo({ top: 0, behavior: 'smooth' });
  setTimeout(() => analyzeText(), 400);
}

// ── Main analyze function ─────────────────────────────────────
async function analyzeText() {
  const text = inputEl.value.trim();
  if (!text) { showTip('warn', '⚠️ Please paste a job description first.'); return; }
  if (text.split(/\s+/).length < 5) { showTip('warn', '⚠️ Add more text for a reliable prediction (at least 10 words).'); return; }

  hideTip();
  setLoading(true);

  try {
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ description: text }),
    });
    const data = await res.json();
    if (data.error) { showTip('warn', '⚠️ ' + data.error); setLoading(false); return; }
    renderResult(data);
  } catch (err) {
    showTip('warn', '⚠️ Could not reach the server. Is flask_api.py running?');
  } finally {
    setLoading(false);
  }
}

// ── Render result ─────────────────────────────────────────────
function renderResult(data) {
  const isFake = data.prediction === 1;
  const risk   = data.risk_score; // 0–100

  // Show result state, hide idle
  document.getElementById('idleState').classList.add('hidden');
  const resultState = document.getElementById('resultState');
  resultState.classList.remove('hidden');
  resultState.classList.add('fade-in');

  // Meter
  animateMeter(risk, isFake);

  // Label card
  const labelCard = document.getElementById('resultLabelCard');
  labelCard.classList.remove('is-fake', 'is-legit');
  labelCard.classList.add(isFake ? 'is-fake' : 'is-legit');

  document.getElementById('resultIcon').textContent  = isFake ? '🚨' : '✅';
  document.getElementById('resultLabel').textContent = isFake ? 'FAKE INTERNSHIP' : 'LEGITIMATE POSTING';
  document.getElementById('resultLabel').style.color = isFake ? '#f87171' : '#4ade80';
  document.getElementById('resultAdvice').textContent = isFake
    ? 'This posting shows signs of fraud. Do NOT apply, pay any fee, or share personal documents.'
    : 'This posting appears genuine. Still verify the company independently before sharing sensitive info.';

  // KPIs
  document.getElementById('kpiWords').textContent  = data.word_count;
  document.getElementById('kpiTokens').textContent = data.token_count;
  document.getElementById('kpiFlags').textContent  = data.flagged_count;
  const modelShort = { LogisticRegression:'LR', RandomForestClassifier:'RF',
                        SVC:'SVM', MultinomialNB:'NB' };
  document.getElementById('kpiModel').textContent  = modelShort[data.model_type] || data.model_type.slice(0,4);

  // Red flags
  renderRedFlags(data.red_flags, data.flagged_count);

  // Tip
  showTip(isFake ? 'warn' : 'info',
    isFake ? `🚨 ${data.flagged_count} red flag(s) detected. Fraud risk: ${risk}%`
           : `✅ Looks safe! Fraud risk score: ${risk}%. Always do independent research.`);
}

// ── Animate SVG arc + needle ──────────────────────────────────
function animateMeter(risk, isFake) {
  const arc    = document.getElementById('meterArc');
  const needle = document.getElementById('meterNeedle');
  const valEl  = document.getElementById('meterValue');

  const total  = 251.2;                         // full arc circumference approx
  const offset = total - (risk / 100) * total;
  const color  = risk > 65 ? '#ef4444' : risk > 40 ? '#fbbf24' : '#22c55e';

  arc.style.stroke          = color;
  arc.style.strokeDashoffset = offset;
  valEl.style.color          = color;

  // Needle: rotate from -90° (left) to +90° (right)
  const deg = -90 + (risk / 100) * 180;
  needle.style.transform = `rotate(${deg}deg)`;

  // Animate number
  let current = 0;
  const step  = risk / 60;
  const timer = setInterval(() => {
    current = Math.min(current + step, risk);
    valEl.textContent = Math.round(current) + '%';
    if (current >= risk) clearInterval(timer);
  }, 16);
}

// ── Red flag cards ────────────────────────────────────────────
function renderRedFlags(flags, count) {
  const section = document.getElementById('redFlagSection');
  const grid    = document.getElementById('rfGrid');
  const summary = document.getElementById('rfSummary');

  grid.innerHTML = Object.entries(flags).map(([flag, hit]) => `
    <div class="rf-item ${hit ? 'hit' : 'clear'}">
      <span>${hit ? '🔴' : '🟢'}</span>
      <span>${flag}</span>
    </div>
  `).join('');

  if (count >= 3) {
    summary.className = 'rf-summary danger';
    summary.textContent = `⚠️ ${count} red flags detected — this posting is very likely fraudulent.`;
  } else if (count >= 1) {
    summary.className = 'rf-summary warning';
    summary.textContent = `⚠️ ${count} red flag(s) found — proceed with caution.`;
  } else {
    summary.className = 'rf-summary ok';
    summary.textContent = '✅ No major red flags found in the text.';
  }

  section.classList.remove('hidden');
  section.classList.add('fade-in');
}

// ── Reset result view ─────────────────────────────────────────
function resetResult() {
  document.getElementById('idleState').classList.remove('hidden');
  document.getElementById('resultState').classList.add('hidden');
  document.getElementById('redFlagSection').classList.add('hidden');
}

// ── Loading state ─────────────────────────────────────────────
function setLoading(on) {
  const btn    = document.getElementById('analyzeBtn');
  const text   = document.querySelector('.btn-text');
  const loader = document.getElementById('btnLoader');
  btn.disabled = on;
  text.classList.toggle('hidden', on);
  loader.classList.toggle('hidden', !on);
}

// ── Tip helper ────────────────────────────────────────────────
function showTip(type, msg) {
  const box = document.getElementById('tipBox');
  box.style.display = 'block';
  box.className = `tip-box ${type}`;
  document.getElementById('tipText').textContent = msg;
}
function hideTip() {
  document.getElementById('tipBox').style.display = 'none';
}

// ── Init ──────────────────────────────────────────────────────
buildExamples();

// Enter key shortcut (Ctrl+Enter)
inputEl.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') analyzeText();
});
