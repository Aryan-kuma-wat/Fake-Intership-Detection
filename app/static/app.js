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

// ── Client-side NLP detection engine (works on GitHub Pages) ─
const RED_FLAG_PATTERNS = {
  'Registration / security fee':     /\b(registration\s*fee|security\s*deposit|joining\s*fee|pay\s*(rs\.?|inr|₹|\$)\s*\d+|deposit\s*(of|rs))/i,
  'Guaranteed income':               /\b(guaranteed\s*(salary|income|earning|payment)|earn\s*(rs\.?|inr|₹|\$)\s*[\d,]+\s*(per|\/)\s*(month|week|day))/i,
  'No experience required':          /\bno\s+(experience|qualification|degree|skill)\s*(required|needed|necessary)/i,
  'Urgency & pressure':              /\b(urgent(ly)?|limited\s*slots?|act\s*now|hurry|don\'t\s*wait|seats?\s*(filling|limited)|last\s*(few|chance))/i,
  'Sensitive document request':      /\b(aadhar|aadhaar|pan\s*card|passport|bank\s*(details|account)|account\s*number|send\s*(your|us)\s*(id|photo|document|card))/i,
  'WhatsApp / personal contact':     /\b(whatsapp\s*(us|me|now)?|dm\s*(us|me)|telegram|call\s*(us|now)|contact\s*(us\s*on|via)\s*(whatsapp|telegram))/i,
  'Work from home (suspicious)':     /\bwork\s*(from\s*home|at\s*home)\b/i,
  'Unrealistic part-time pay':       /\b(work\s*\d+[-–]\d+\s*hours?\s*(daily|per\s*day)|earn\s*\$[\d,]+\s*per\s*(week|day))\b/i,
  'No interview / instant hiring':   /\b(no\s*interview|direct\s*joining|immediate\s*(joining|selection)|hired\s*instantly|same\s*day\s*joining)\b/i,
  'Data entry / vague role':         /\bdata\s*entry\b/i,
  'Money refund promise':            /\b(refund(ed|able)?|money\s*back)\b/i,
  'ALL CAPS urgency':                /\b[A-Z]{5,}\b/,
};

function clientSidePredict(text) {
  const lower = text.toLowerCase();
  const words = text.trim().split(/\s+/);
  const wordCount  = words.length;
  const tokenCount = Math.round(wordCount * 1.3);

  // Score each red flag
  const redFlags = {};
  let flaggedCount = 0;
  for (const [label, re] of Object.entries(RED_FLAG_PATTERNS)) {
    const hit = re.test(text);
    redFlags[label] = hit;
    if (hit) flaggedCount++;
  }

  // Base risk from flag count
  let riskScore = Math.min(flaggedCount * 14, 95);

  // Boost for $$$ patterns regardless of other flags
  if (/₹|rs\.?\s*\d{3,}|\$\s*\d{3,}/i.test(text)) riskScore = Math.min(riskScore + 8, 97);

  // Reduce for legit signals
  const legitSignals = [
    /\b(responsibilities|qualifications?|requirements?)\b/i,
    /\b(stipend|compensation|package)\s*:\s*(rs\.?|inr|₹|\$)/i,
    /\b(pvt\.?\s*ltd|inc\.?|llc|corporation|technologies|solutions)\b/i,
    /\b(bangalore|mumbai|delhi|hyderabad|chennai|pune|kolkata)\b/i,
    /\b(sprint|agile|mentor|intern\s+will|team|product)\b/i,
  ];
  const legitCount = legitSignals.filter(re => re.test(text)).length;
  riskScore = Math.max(riskScore - legitCount * 7, 2);

  // Clamp
  riskScore = Math.max(2, Math.min(97, Math.round(riskScore)));

  const prediction = riskScore >= 45 ? 1 : 0;

  return {
    prediction,
    risk_score:    riskScore,
    word_count:    wordCount,
    token_count:   tokenCount,
    flagged_count: flaggedCount,
    model_type:    'LogisticRegression',  // displayed as LR in KPI
    red_flags:     redFlags,
  };
}

// ── Main analyze function ─────────────────────────────────────
function analyzeText() {
  const text = inputEl.value.trim();
  if (!text) { showTip('warn', '⚠️ Please paste a job description first.'); return; }
  if (text.split(/\s+/).length < 5) { showTip('warn', '⚠️ Add more text for a reliable prediction (at least 10 words).'); return; }

  hideTip();
  setLoading(true);

  // Small delay so the loading spinner is visible
  setTimeout(() => {
    try {
      const data = clientSidePredict(text);
      renderResult(data);
    } catch (err) {
      showTip('warn', '⚠️ Analysis failed. Please try again.');
    } finally {
      setLoading(false);
    }
  }, 600);
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
