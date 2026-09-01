// Same-origin in production; falls back to localhost:8000 for local dev.
const API_BASE = (["localhost", "127.0.0.1", ""].includes(location.hostname))
  ? "http://localhost:8000"
  : window.location.origin;
document.getElementById('apiBaseLabel').textContent = API_BASE.replace(/^https?:\/\//, '');

let categoryChart = null;
let timelineChart = null;
let liveFeedTimer = null;
let logsState = { page: 1, page_size: 25, total_pages: 1 };
let logsDebounceTimer = null;
let activeTab = 'overview';

// ===== Auth state — JWT session (see auth.py). Token + user info persist
// in localStorage (same pattern already used here for the theme
// preference) so a page refresh doesn't force a re-login. =====
let authToken = localStorage.getItem('siem-auth-token') || null;
let currentUser = null;
try {
  const savedUser = localStorage.getItem('siem-auth-user');
  if (savedUser) currentUser = JSON.parse(savedUser);
} catch (e) { currentUser = null; }

function applyAuthUI() {
  const loggedIn = !!authToken && !!currentUser;
  document.getElementById('loginOverlay').style.display = loggedIn ? 'none' : 'flex';
  document.getElementById('topbarUser').style.display = loggedIn ? 'flex' : 'none';
  document.body.classList.toggle('role-admin', loggedIn && currentUser.role === 'admin');
  if (loggedIn) {
    document.getElementById('topbarUserName').textContent = currentUser.display_name || currentUser.username;
    document.getElementById('topbarUserRole').textContent = currentUser.role;
  }
}

async function handleLoginSubmit(evt) {
  evt.preventDefault();
  const username = document.getElementById('loginUsername').value.trim();
  const password = document.getElementById('loginPassword').value;
  const errEl = document.getElementById('loginError');
  const btn = document.getElementById('loginSubmitBtn');
  errEl.textContent = '';
  btn.disabled = true;
  btn.textContent = 'SIGNING IN…';
  try {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || 'Invalid username or password');
    }
    const data = await res.json();
    authToken = data.access_token;
    currentUser = data.user;
    localStorage.setItem('siem-auth-token', authToken);
    localStorage.setItem('siem-auth-user', JSON.stringify(currentUser));
    applyAuthUI();
    document.getElementById('loginPassword').value = '';
    loadOverview();
    loadAdminUsersIfAdmin();
  } catch (e) {
    errEl.textContent = e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = 'SIGN IN';
  }
  return false;
}

function logout() {
  authToken = null;
  currentUser = null;
  localStorage.removeItem('siem-auth-token');
  localStorage.removeItem('siem-auth-user');
  applyAuthUI();
}

// Wraps fetch with the Authorization header attached whenever we have a
// token -- used for every write (POST/PATCH/DELETE) against alerts,
// cases, and webhooks. A 401 means the session expired or was revoked;
// drop back to the login screen rather than showing a confusing error.
async function authFetch(url, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
  const res = await fetch(url, { ...options, headers });
  if (res.status === 401) {
    logout();
  }
  return res;
}

const SEVERITY_COLORS = { critical: 'var(--critical)', high: 'var(--high)', medium: 'var(--medium)', low: 'var(--low)' };
const SEVERITY_HEX = { critical: '#F0465B', high: '#F5A623', medium: '#F2C94C', low: '#34D399' };

// ===== Tab switching =====
document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', () => {
    document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    item.classList.add('active');
    document.getElementById('tab-' + item.dataset.tab).classList.add('active');
    activeTab = item.dataset.tab;
    loadTabData(activeTab);
    document.getElementById('sidebar').classList.remove('open');
    document.getElementById('sidebarBackdrop').classList.remove('open');
  });
});

function selectTab(tabId) {
  const item = document.querySelector(`.nav-item[data-tab="${tabId}"]`);
  if (item) item.click();
}

function loadTabData(tab) {
  switch (tab) {
    case 'overview': loadOverview(); break;
    case 'threats': loadThreats(); break;
    case 'cases': loadCases(); break;
    case 'fraud': loadFraud(); break;
    case 'user-behavior': loadUserBehavior(); break;
    case 'admin-activity': loadAdminActivity(); break;
    case 'ai-insights': loadAiInsights(); break;
    case 'logs': loadLogs(); break;
    case 'settings': loadWebhooks(); loadAdminUsersIfAdmin(); break;
    default: break; // reports is static
  }
}

function toggleMobileNav() {
  document.getElementById('sidebar').classList.toggle('open');
  document.getElementById('sidebarBackdrop').classList.toggle('open');
}

// ===== Top bar: clock, search, notifications =====
function tickClock() {
  const now = new Date();
  const pad = n => String(n).padStart(2, '0');
  const val = `${now.getFullYear()}-${pad(now.getMonth()+1)}-${pad(now.getDate())} ${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
  document.getElementById('clockValue').textContent = val;
}
setInterval(tickClock, 1000);
tickClock();

function runGlobalSearch() {
  const q = document.getElementById('globalSearch').value.trim();
  if (!q) return;
  selectTab('logs');
  document.getElementById('logsSearch').value = q;
  logsState.page = 1;
  loadLogs();
}

function jumpToOpenAlerts() {
  selectTab('threats');
  document.getElementById('threatsStatusFilter').value = 'open';
  loadThreats();
}

async function refreshNotifBadge() {
  try {
    const data = await apiGet('/threats?limit=1&group_incidents=true&status=open');
    const count = data.count ?? 0;
    const ping = document.getElementById('notifPing');
    if (count > 0) {
      ping.style.display = 'flex';
      ping.textContent = count > 99 ? '99+' : count;
    } else {
      ping.style.display = 'none';
    }
  } catch (e) { /* silent — notification badge is a nice-to-have */ }
}

// ===== Connection status =====
async function checkStatus() {
  const dot = document.getElementById('statusDot');
  const label = document.getElementById('statusLabel');
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error();
    dot.className = 'status-dot live';
    label.textContent = 'All systems operational';
  } catch (e) {
    dot.className = 'status-dot offline';
    label.textContent = 'API unreachable';
  }
}

async function apiGet(path) {
  const res = await authFetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  return res.json();
}

function severityBadge(sev) {
  const level = (sev || 'low').toLowerCase();
  const map = { low: 'low', medium: 'medium', high: 'high', critical: 'critical' };
  const cls = map[level] || 'low';
  return `<span class="badge ${cls}">${level}</span>`;
}

function renderOriginChip(intel) {
  if (!intel) return '';
  if (intel.is_private) return `<span class="intel-chip internal">internal</span>`;
  const rep = intel.reputation || 'unknown';
  const repClass = ['malicious', 'suspicious', 'clean'].includes(rep) ? rep : 'unknown';
  const label = intel.country_code ? `${intel.country_code}${intel.city ? ' · ' + intel.city : ''}` : rep;
  const title = (intel.reputation_reason || intel.country || '').replace(/"/g, '&quot;');
  return `<span class="intel-chip ${repClass}" title="${title}">${label}</span>`;
}

function renderMitreChip(mitre) {
  if (!mitre) return '';
  return `<a href="${mitre.url}" target="_blank" rel="noopener" class="mitre-chip" title="${mitre.tactic}">${mitre.technique_id}</a>`;
}

function emptyRow(colspan, title, sub) {
  return `<tr><td colspan="${colspan}"><div class="empty-state"><div class="empty-icon">∅</div><div class="empty-title">${title}</div><div class="empty-sub">${sub}</div></div></td></tr>`;
}
function errorRow(colspan, msg) {
  return `<tr><td colspan="${colspan}"><div class="empty-state"><div class="empty-icon" style="color:var(--critical);">!</div><div class="empty-title" style="color:var(--critical);">Couldn't load data</div><div class="empty-sub">${msg}</div></div></td></tr>`;
}
function formatFeedTime(ts) {
  if (!ts) return '—';
  const d = new Date(ts);
  if (isNaN(d.getTime())) return ts;
  const diffMin = Math.round((Date.now() - d.getTime()) / 60000);
  if (diffMin < 1) return 'just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.round(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  return d.toLocaleDateString();
}

// ===== Gauge renderer (SVG ring) =====
function renderGauge(holderId, valueId, percent, colorVar) {
  const holder = document.getElementById(holderId);
  const r = 34, c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(100, percent));
  const offset = c - (pct / 100) * c;
  holder.innerHTML = `
    <svg width="84" height="84" viewBox="0 0 84 84">
      <circle cx="42" cy="42" r="${r}" fill="none" stroke="var(--border)" stroke-width="8"></circle>
      <circle cx="42" cy="42" r="${r}" fill="none" stroke="${colorVar}" stroke-width="8"
        stroke-linecap="round" stroke-dasharray="${c}" stroke-dashoffset="${offset}"></circle>
    </svg>`;
  document.getElementById(valueId).textContent = Math.round(pct) + '%';
  document.getElementById(valueId).style.color = colorVar;
}

// ===== KPI trend chip =====
function trendChip(deltaPct) {
  if (deltaPct === null || deltaPct === undefined || !isFinite(deltaPct)) {
    return `<span class="kpi-sub-text">last 24h</span>`;
  }
  const rounded = Math.round(deltaPct * 10) / 10;
  if (Math.abs(rounded) < 0.5) return `<span class="kpi-trend flat">flat</span><span class="kpi-sub-text">vs prior 24h</span>`;
  const cls = rounded > 0 ? 'up' : 'down';
  const arrow = rounded > 0 ? '↑' : '↓';
  return `<span class="kpi-trend ${cls}">${arrow} ${Math.abs(rounded)}%</span><span class="kpi-sub-text">vs prior 24h</span>`;
}

function kpiCard(icon, iconBg, iconColor, label, value, valueClass, trendHtml) {
  return `
    <div class="kpi-card-v2">
      <div class="kpi-card-v2-top">
        <div class="kpi-icon" style="background:${iconBg}; color:${iconColor};">${icon}</div>
        <div class="kpi-label-v2">${label}</div>
      </div>
      <div class="kpi-value-v2 ${valueClass || ''}">${value}</div>
      <div class="kpi-sub-row">${trendHtml}</div>
    </div>`;
}

// ===== Overview =====
async function loadOverview() {
  const kpiGrid = document.getElementById('overviewKpisV2');

  try {
    const [overview, timeline48, topEntities] = await Promise.all([
      apiGet('/overview'),
      apiGet('/overview/timeline?hours=48&buckets=2').catch(() => null),
      apiGet('/overview/top-entities?scan_limit=500&top_n=100').catch(() => ({ top_ips: [], top_users: [] })),
    ]);

    let totalDelta = null, anomalyDelta = null;
    if (timeline48 && timeline48.total && timeline48.total.length === 2) {
      const [prior, recent] = timeline48.total;
      totalDelta = prior > 0 ? ((recent - prior) / prior) * 100 : (recent > 0 ? 100 : null);
      const [priorA, recentA] = timeline48.anomalies;
      anomalyDelta = priorA > 0 ? ((recentA - priorA) / priorA) * 100 : (recentA > 0 ? 100 : null);
    }

    const highRiskUsers = (topEntities.top_users || []).filter(u => ['High', 'Critical'].includes(u.max_level)).length;
    const activeSources = (topEntities.top_ips || []).length + (topEntities.top_users || []).length;

    kpiGrid.innerHTML = [
      kpiCard('▤', 'var(--blue-dim)', 'var(--blue)', 'Total Logs', (overview.total_events ?? 0).toLocaleString(), '', trendChip(totalDelta)),
      kpiCard('▲', 'var(--orange-dim)', 'var(--orange)', 'Threats Detected', (overview.events_by_category?.threat ?? 0).toLocaleString(), '', `<span class="kpi-sub-text">in category "threat"</span>`),
      kpiCard('!', 'var(--red-dim)', 'var(--red)', 'Critical Alerts', (overview.critical_events ?? 0).toLocaleString(), 'critical', `<span class="kpi-sub-text">risk level: critical</span>`),
      kpiCard('✦', 'var(--purple-dim)', 'var(--purple)', 'AI Anomalies', (overview.anomalies ?? 0).toLocaleString(), '', trendChip(anomalyDelta)),
      kpiCard('◉', 'var(--teal-dim)', 'var(--teal)', 'High Risk Users', highRiskUsers.toLocaleString(), '', `<span class="kpi-sub-text">High/Critical, scanned</span>`),
      kpiCard('◆', 'var(--green-dim)', 'var(--green)', 'Active Sources', activeSources.toLocaleString(), '', `<span class="kpi-sub-text">distinct IPs + users</span>`),
    ].join('');

    renderTopEntityTables(topEntities.top_ips || [], topEntities.top_users || []);

    const cats = overview.events_by_category || {};
    renderEventTypes(cats, overview.total_events || 0);
  } catch (e) {
    kpiGrid.innerHTML = `<div class="panel" style="grid-column:1/-1;">${errorRow(1, e.message).replace('<tr>','').replace('</tr>','').replace(/<td colspan="1">|<\/td>/g,'')}</div>`;
  }

  loadTimeline();
  loadSeverityDonut();
  loadAiGauges();
  loadRecentAlerts();
  loadLiveFeed();
  loadEntityRisk();
  loadEscalatingRisk();
  startLiveFeedPolling();
  refreshNotifBadge();
}

// ===== Entity Risk (UEBA) — persistent, time-decaying risk per IP/user =====
function riskScoreColor(score) {
  if (score >= 70) return SEVERITY_HEX.critical;
  if (score >= 45) return SEVERITY_HEX.high;
  if (score >= 20) return SEVERITY_HEX.medium;
  return SEVERITY_HEX.low;
}

function sparklineSvg(trend, color) {
  if (!trend || trend.length < 2) {
    return `<span class="lstm-hint">not enough history</span>`;
  }
  const w = 90, h = 26, pad = 3;
  const min = Math.min(...trend), max = Math.max(...trend);
  const range = (max - min) || 1;
  const step = (w - pad * 2) / (trend.length - 1);
  const points = trend.map((v, i) => {
    const x = pad + i * step;
    const y = h - pad - ((v - min) / range) * (h - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
    <polyline points="${points}" fill="none" stroke="${color}" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>
  </svg>`;
}

async function loadEntityRisk() {
  const tbody = document.getElementById('entityRiskTable');
  try {
    const data = await apiGet('/entity-risk/top?limit=8&min_score=1');
    const results = data.results || [];
    if (!results.length) {
      tbody.innerHTML = emptyRow(6, 'No entity risk data yet', 'Risk accumulates as events are logged for each IP/user.');
      return;
    }
    tbody.innerHTML = results.map(r => {
      const color = riskScoreColor(r.risk_score);
      return `
        <tr>
          <td><code class="inline">${r.entity_id}</code></td>
          <td style="color:var(--text-muted); text-transform:capitalize;">${r.entity_type}</td>
          <td><span class="badge" style="background:${color}22; color:${color};">${r.risk_score}</span></td>
          <td>${sparklineSvg(r.trend, color)}</td>
          <td>${r.event_count}×</td>
          <td>${formatFeedTime(r.last_updated)}</td>
        </tr>`;
    }).join('');
  } catch (e) {
    tbody.innerHTML = errorRow(6, e.message);
  }
}

// ===== Early Warning (predictive) — entities whose risk is climbing
// fast but hasn't crossed the alert threshold yet. See
// entity_risk.py's get_escalating_entities() for the naive-linear-ETA
// caveat. =====
async function loadEscalatingRisk() {
  const tbody = document.getElementById('escalatingRiskTable');
  if (!tbody) return; // panel not present on this build yet
  try {
    const data = await apiGet('/entity-risk/predictive?limit=8&min_velocity=2');
    const results = data.results || [];
    if (!results.length) {
      tbody.innerHTML = emptyRow(6, 'No fast-rising entities right now', 'This is a good sign — nothing is escalating quickly.');
      return;
    }
    tbody.innerHTML = results.map(r => {
      const color = riskScoreColor(r.risk_score);
      const eta = r.eta_hours_to_threshold != null ? `~${r.eta_hours_to_threshold}h` : '—';
      return `
        <tr>
          <td><code class="inline">${r.entity_id}</code></td>
          <td style="color:var(--text-muted); text-transform:capitalize;">${r.entity_type}</td>
          <td><span class="badge" style="background:${color}22; color:${color};">${r.risk_score}</span></td>
          <td style="color:var(--orange);">+${r.velocity_per_hour}/hr</td>
          <td>${eta}</td>
          <td>${sparklineSvg(r.trend, color)}</td>
        </tr>`;
    }).join('');
  } catch (e) {
    tbody.innerHTML = errorRow(6, e.message);
  }
}

async function loadTimeline() {
  try {
    const data = await apiGet('/overview/timeline?hours=24&buckets=24');
    renderTimelineChart(data.labels || [], data.total || [], data.anomalies || []);
  } catch (e) { console.error('Timeline load failed:', e.message); }
}

function renderTimelineChart(labels, total, anomalies) {
  const ctx = document.getElementById('timelineChart');
  if (timelineChart) timelineChart.destroy();
  const legendColor = getComputedStyle(document.body).getPropertyValue('--chart-legend-color').trim() || '#9297AB';
  const isLight = document.body.classList.contains('light-mode');
  const gridColor = isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.06)';

  timelineChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Events', data: total, borderColor: '#5B7FFF', backgroundColor: 'rgba(91,127,255,0.12)', fill: true, tension: 0.35, pointRadius: 0, borderWidth: 2 },
        { label: 'Anomalies', data: anomalies, borderColor: '#F0465B', backgroundColor: 'rgba(240,70,91,0.12)', fill: true, tension: 0.35, pointRadius: 0, borderWidth: 2 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { position: 'bottom', labels: { color: legendColor, font: { family: 'IBM Plex Mono', size: 10 }, padding: 12, boxWidth: 10 } } },
      scales: {
        x: { ticks: { color: legendColor, maxTicksLimit: 6, font: { size: 10 } }, grid: { color: gridColor } },
        y: { beginAtZero: true, ticks: { color: legendColor, precision: 0, font: { size: 10 } }, grid: { color: gridColor } },
      },
    },
  });
}

// ===== Severity donut — derived from the raw `severity` field via /logs counts =====
async function loadSeverityDonut() {
  const legendEl = document.getElementById('severityLegend');
  try {
    const [critical, high, medium, low] = await Promise.all([
      apiGet('/logs?severity=critical&page_size=1'),
      apiGet('/logs?severity=high&page_size=1'),
      apiGet('/logs?severity=medium&page_size=1'),
      apiGet('/logs?severity=low&page_size=1'),
    ]);
    const counts = { critical: critical.total || 0, high: high.total || 0, medium: medium.total || 0, low: low.total || 0 };
    const total = counts.critical + counts.high + counts.medium + counts.low;

    document.getElementById('severityDonutTotal').textContent = total.toLocaleString();

    const ctx = document.getElementById('categoryChart');
    if (categoryChart) categoryChart.destroy();
    const isLight = document.body.classList.contains('light-mode');
    categoryChart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Critical', 'High', 'Medium', 'Low'],
        datasets: [{
          data: total > 0 ? [counts.critical, counts.high, counts.medium, counts.low] : [1],
          backgroundColor: total > 0 ? [SEVERITY_HEX.critical, SEVERITY_HEX.high, SEVERITY_HEX.medium, SEVERITY_HEX.low] : ['#232838'],
          borderColor: isLight ? '#FFFFFF' : '#131722',
          borderWidth: 3,
        }]
      },
      options: { responsive: true, maintainAspectRatio: false, cutout: '72%', plugins: { legend: { display: false } } }
    });

    if (total === 0) {
      legendEl.innerHTML = '<p class="lstm-hint">No events yet.</p>';
      return;
    }
    const rows = [
      ['critical', 'Critical', counts.critical], ['high', 'High', counts.high],
      ['medium', 'Medium', counts.medium], ['low', 'Low', counts.low],
    ];
    legendEl.innerHTML = rows.map(([key, label, count]) => `
      <div class="donut-legend-row">
        <span class="donut-dot" style="background:${SEVERITY_HEX[key]};"></span>
        <span class="donut-legend-label">${label} (${total ? Math.round(count/total*100) : 0}%)</span>
        <span class="donut-legend-value">${count}</span>
      </div>`).join('');
  } catch (e) {
    legendEl.innerHTML = `<p class="lstm-hint">Couldn't load: ${e.message}</p>`;
  }
}

// ===== AI Detection gauges — honest averages over recently-scored events,
// not a single opaque "AI score". Fraud AI = mean risk_score of recent
// fraud-category events; Behavioral AI = mean risk_score of recent
// user_behavior-category events; Combined = equal-weight blend of the two. =====
async function loadAiGauges() {
  const row = document.getElementById('aiGaugeRow');
  try {
    const [fraud, behavior] = await Promise.all([
      apiGet('/fraud?limit=50'),
      apiGet('/user-behavior?limit=50'),
    ]);
    const avg = (arr) => {
      const scores = arr.map(r => r.risk_score).filter(v => typeof v === 'number');
      if (!scores.length) return null;
      return scores.reduce((a, b) => a + b, 0) / scores.length;
    };
    const fraudAvg = avg(fraud.results || []);
    const behaviorAvg = avg(behavior.results || []);
    const combined = (fraudAvg !== null && behaviorAvg !== null) ? (fraudAvg + behaviorAvg) / 2
      : (fraudAvg !== null ? fraudAvg : behaviorAvg);

    row.innerHTML = `
      <div class="gauge-block">
        <div class="gauge-holder" id="gaugeFraudHolder"></div>
        <div class="gauge-label" id="gaugeFraudValue">—</div>
        <div class="gauge-label">Fraud AI</div>
        <div class="gauge-sub">avg risk, last ${(fraud.results||[]).length} flagged</div>
      </div>
      <div class="gauge-block">
        <div class="gauge-holder" id="gaugeBehaviorHolder"></div>
        <div class="gauge-label" id="gaugeBehaviorValue">—</div>
        <div class="gauge-label">Behavioral AI</div>
        <div class="gauge-sub">avg risk, last ${(behavior.results||[]).length} events</div>
      </div>
      <div class="gauge-block">
        <div class="gauge-holder" id="gaugeCombinedHolder"></div>
        <div class="gauge-label" id="gaugeCombinedValue">—</div>
        <div class="gauge-label">Combined Risk</div>
        <div class="gauge-sub">equal-weight blend</div>
      </div>`;

    // gauge-holder + gauge-value are separate elements now; reuse renderGauge with adjusted ids
    renderGaugeInline('gaugeFraudHolder', 'gaugeFraudValue', fraudAvg, '#5B7FFF');
    renderGaugeInline('gaugeBehaviorHolder', 'gaugeBehaviorValue', behaviorAvg, '#A78BFA');
    renderGaugeInline('gaugeCombinedHolder', 'gaugeCombinedValue', combined, '#34D399');
  } catch (e) {
    row.innerHTML = `<p class="lstm-hint">Couldn't load: ${e.message}</p>`;
  }
}

function renderGaugeInline(holderId, labelId, value, color) {
  const holder = document.getElementById(holderId);
  const labelEl = document.getElementById(labelId);
  if (value === null || value === undefined) {
    holder.innerHTML = `<svg width="84" height="84" viewBox="0 0 84 84"><circle cx="42" cy="42" r="34" fill="none" stroke="var(--border)" stroke-width="8"></circle></svg>`;
    labelEl.textContent = 'n/a';
    labelEl.style.color = 'var(--text-dim)';
    return;
  }
  const r = 34, c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(100, value));
  const offset = c - (pct / 100) * c;
  holder.innerHTML = `
    <svg width="84" height="84" viewBox="0 0 84 84">
      <circle cx="42" cy="42" r="${r}" fill="none" stroke="var(--border)" stroke-width="8"></circle>
      <circle cx="42" cy="42" r="${r}" fill="none" stroke="${color}" stroke-width="8"
        stroke-linecap="round" stroke-dasharray="${c}" stroke-dashoffset="${offset}"
        style="transition: stroke-dashoffset 0.4s ease;"></circle>
      <text x="42" y="47" text-anchor="middle" fill="${color}" font-family="IBM Plex Mono" font-size="15" font-weight="700" transform="rotate(90 42 42)">${Math.round(pct)}%</text>
    </svg>`;
  labelEl.style.display = 'none'; // number is drawn inside the SVG instead
}

async function loadRecentAlerts() {
  const el = document.getElementById('recentAlertsList');
  try {
    const data = await apiGet('/threats?limit=6&group_incidents=true&sort=priority&status=open');
    const results = data.results || [];
    if (!results.length) {
      el.innerHTML = '<p class="lstm-hint">No open alerts right now.</p>';
      return;
    }
    el.innerHTML = results.map(r => {
      const level = (r.risk_level || 'low').toLowerCase();
      return `
        <div class="alert-row">
          <span class="alert-dot" style="background:${SEVERITY_HEX[level] || SEVERITY_HEX.low};"></span>
          <div class="alert-main">
            <div class="alert-title">${(r.event_type || 'event').replace(/_/g,' ')} — <code class="inline">${r.ip || '—'}</code></div>
            <div class="alert-sub">${r.user_id || 'unknown user'}</div>
          </div>
          <span class="alert-badge ${level}">${level}</span>
          <span class="alert-time">${formatFeedTime(r.last_seen || r.timestamp)}</span>
        </div>`;
    }).join('');
  } catch (e) {
    el.innerHTML = `<p class="lstm-hint">Couldn't load: ${e.message}</p>`;
  }
}

async function loadLiveFeed() {
  const el = document.getElementById('liveFeedList');
  try {
    const data = await apiGet('/logs?page=1&page_size=10');
    const results = data.results || [];
    document.getElementById('liveFeedTag').textContent = results.length ? `${results.length} most recent` : '';
    if (!results.length) { el.innerHTML = '<p class="lstm-hint">No events logged yet.</p>'; return; }
    el.innerHTML = results.map(r => `
      <div class="alert-row">
        <span class="alert-time" style="width:auto; text-align:left;">${formatFeedTime(r.timestamp)}</span>
        <div class="alert-main"><code class="inline">${r.ip || '—'}</code> <span class="alert-sub" style="display:inline;">${r.event_type || '—'}</span></div>
        ${severityBadge(r.severity)}
      </div>`).join('');
  } catch (e) {
    el.innerHTML = `<p class="lstm-hint">Couldn't load: ${e.message}</p>`;
  }
}
function startLiveFeedPolling() {
  if (liveFeedTimer) clearInterval(liveFeedTimer);
  liveFeedTimer = setInterval(() => { if (activeTab === 'overview') { loadLiveFeed(); refreshNotifBadge(); } }, 8000);
}

function renderTopEntityTables(topIps, topUsers) {
  const ipEl = document.getElementById('topIpsTable');
  const userEl = document.getElementById('topUsersTable');
  const rows = (list, keyField) => !list.length
    ? emptyRow(3, 'No activity yet', '')
    : list.slice(0, 6).map(r => `<tr><td><code class="inline">${r[keyField]}</code></td><td>${r.count}×</td><td>${severityBadge(r.max_level)}</td></tr>`).join('');
  ipEl.innerHTML = rows(topIps.sort((a,b)=>b.count-a.count), 'ip');
  userEl.innerHTML = rows(topUsers.sort((a,b)=>b.count-a.count), 'user_id');
}

function renderEventTypes(cats, total) {
  const el = document.getElementById('eventTypesList');
  const labels = { threat: ['Threats', '#F5A623'], fraud: ['Fraud', '#F0465B'], user_behavior: ['User Behavior', '#5B7FFF'], admin_activity: ['Admin Activity', '#A78BFA'] };
  const entries = Object.entries(cats);
  if (!entries.length || total === 0) { el.innerHTML = '<p class="lstm-hint">No events yet.</p>'; return; }
  el.innerHTML = entries.map(([key, count]) => {
    const [label, color] = labels[key] || [key, '#5B7FFF'];
    const pct = total ? Math.round((count / total) * 100) : 0;
    return `
      <div class="event-type-row">
        <div class="event-type-top"><span class="event-type-name">${label}</span><span class="event-type-count">${count.toLocaleString()} · ${pct}%</span></div>
        <div class="event-type-bar-bg"><div class="event-type-bar-fill" style="width:${pct}%; background:${color};"></div></div>
      </div>`;
  }).join('');
}

// ===== Threats =====
async function loadThreats() {
  const tbody = document.getElementById('threatsBody');
  tbody.innerHTML = '<tr class="loading-row"><td colspan="9">Loading…</td></tr>';
  const anomalyOnly = document.getElementById('threatsAnomalyOnly').checked;
  const grouped = document.getElementById('threatsGroupIncidents').checked;
  const status = document.getElementById('threatsStatusFilter').value;
  const sortBy = document.getElementById('threatsSortBy').value;
  try {
    const params = new URLSearchParams({ limit: 100, anomaly_only: anomalyOnly, group_incidents: grouped, sort: sortBy });
    if (status) params.set('status', status);
    const data = await apiGet(`/threats?${params.toString()}`);
    const results = data.results || [];
    const isGrouped = !!data.grouped;
    if (results.length === 0) { tbody.innerHTML = emptyRow(9, 'No threats detected', 'Threat events will appear here once logged via /detect.'); return; }
    tbody.innerHTML = results.map(r => `
      <tr>
        <td>${formatFeedTime(r.last_seen || r.timestamp)}</td>
        <td>${r.count ? r.count + '×' : '1×'}</td>
        <td><code class="inline">${r.ip || '—'}</code>${renderOriginChip(r.threat_intel)}</td>
        <td>${r.event_type || '—'}${renderMitreChip(r.mitre)}</td>
        <td>${severityBadge(r.risk_level || r.severity)}</td>
        <td>${r.risk_score ?? '—'}</td>
        <td>${r.priority_score != null ? priorityBadge(r.priority_score) : '—'}</td>
        <td>${renderStatusCell(r, isGrouped)}</td>
        <td style="color:var(--text-muted);">${(r.reason || []).join(', ') || '—'}</td>
      </tr>`).join('');
  } catch (e) {
    tbody.innerHTML = errorRow(9, e.message);
  }
}
function priorityBadge(score) {
  let cls = 'low';
  if (score >= 110) cls = 'critical'; else if (score >= 80) cls = 'high'; else if (score >= 40) cls = 'medium';
  return `<span class="badge ${cls}">${score}</span>`;
}
function renderStatusCell(row, isGrouped) {
  if (!isGrouped) return '<span class="lstm-hint">—</span>';
  const statuses = ['new', 'investigating', 'resolved', 'false_positive'];
  const current = row.status || 'new';
  const options = statuses.map(s => `<option value="${s}" ${s === current ? 'selected' : ''}>${statusLabel(s)}</option>`).join('');
  return `<select class="status-select" onchange="updateAlertStatus('${row._id}', this.value)">${options}</select>
    <button class="refresh-btn" style="padding:3px 8px; font-size:9px; margin-left:4px;" title="Create a case from this alert" onclick="createCaseFromAlert('${row._id}')">+Case</button>`;
}
async function createCaseFromAlert(alertId) {
  const title = prompt('Case title:');
  if (!title || !title.trim()) return;
  try {
    const res = await authFetch(`${API_BASE}/cases`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: title.trim(), alert_ids: [alertId] }),
    });
    if (!res.ok) throw new Error(`Request failed: ${res.status}`);
    alert(`Case "${title.trim()}" created — see the Cases tab.`);
  } catch (e) {
    alert(`Couldn't create case: ${e.message}`);
  }
}
function statusLabel(s) {
  const map = { new: 'New', investigating: 'Investigating', resolved: 'Resolved', false_positive: 'False Positive' };
  return map[s] || s;
}
async function updateAlertStatus(alertId, newStatus) {
  try {
    const res = await authFetch(`${API_BASE}/alerts/${alertId}/status`, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ status: newStatus }),
    });
    if (!res.ok) throw new Error(`Request failed: ${res.status}`);
    loadThreats();
  } catch (e) {
    alert(`Couldn't update alert status: ${e.message}`);
    loadThreats();
  }
}

// ===== Cases =====
let casesState = { currentCaseId: null };

function caseStatusBadge(status) {
  const map = { open: 'high', investigating: 'medium', contained: 'low', closed: 'low' };
  const cls = map[status] || 'low';
  return `<span class="badge ${cls}">${(status || '—').replace('_', ' ')}</span>`;
}

async function loadCases() {
  const tbody = document.getElementById('casesBody');
  tbody.innerHTML = '<tr class="loading-row"><td colspan="6">Loading…</td></tr>';
  const status = document.getElementById('casesStatusFilter').value;
  const qs = status ? `?status=${status}` : '';
  try {
    const data = await apiGet(`/cases${qs}`);
    const results = data.results || [];
    if (!results.length) {
      tbody.innerHTML = emptyRow(6, 'No cases yet', 'Create one above, or use "+Case" on an alert in the Threats tab.');
      return;
    }
    tbody.innerHTML = results.map(c => `
      <tr style="cursor:pointer;" onclick="openCaseDetail('${c._id}')">
        <td>${c.title}</td>
        <td>${caseStatusBadge(c.status)}</td>
        <td>${severityBadge(c.severity)}</td>
        <td>${c.alert_count ?? 0}×</td>
        <td>${c.assigned_to || '—'}</td>
        <td>${formatFeedTime(c.updated_at)}</td>
      </tr>`).join('');
  } catch (e) {
    tbody.innerHTML = errorRow(6, e.message);
  }
}

function showNewCaseForm() { document.getElementById('newCaseFormWrap').style.display = 'block'; }
function hideNewCaseForm() {
  document.getElementById('newCaseFormWrap').style.display = 'none';
  document.getElementById('newCaseTitle').value = '';
  document.getElementById('newCaseTags').value = '';
}

async function submitNewCase() {
  const title = document.getElementById('newCaseTitle').value.trim();
  if (!title) { alert('Case title is required.'); return; }
  const tagsRaw = document.getElementById('newCaseTags').value.trim();
  const tags = tagsRaw ? tagsRaw.split(',').map(t => t.trim()).filter(Boolean) : [];
  try {
    const res = await authFetch(`${API_BASE}/cases`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, tags }),
    });
    if (!res.ok) throw new Error(`Request failed: ${res.status}`);
    hideNewCaseForm();
    loadCases();
  } catch (e) {
    alert(`Couldn't create case: ${e.message}`);
  }
}

async function openCaseDetail(caseId) {
  casesState.currentCaseId = caseId;
  document.getElementById('casesListView').style.display = 'none';
  document.getElementById('caseDetailView').style.display = 'block';
  await renderCaseDetail();
}

function closeCaseDetail() {
  casesState.currentCaseId = null;
  document.getElementById('caseDetailView').style.display = 'none';
  document.getElementById('casesListView').style.display = 'block';
  loadCases();
}

async function renderCaseDetail() {
  const panel = document.getElementById('caseDetailPanel');
  panel.innerHTML = '<p class="lstm-hint">Loading…</p>';
  try {
    const [c, suggested] = await Promise.all([
      apiGet(`/cases/${casesState.currentCaseId}`),
      apiGet(`/cases/${casesState.currentCaseId}/suggested-alerts`).catch(() => ({ results: [] })),
    ]);

    const statusOptions = ['open', 'investigating', 'contained', 'closed']
      .map(s => `<option value="${s}" ${s === c.status ? 'selected' : ''}>${s.replace('_', ' ')}</option>`).join('');

    const alertsRows = (c.alerts || []).length
      ? c.alerts.map(a => `
        <tr>
          <td>${formatFeedTime(a.last_seen || a.timestamp)}</td>
          <td><code class="inline">${a.ip || '—'}</code></td>
          <td>${a.event_type || '—'}</td>
          <td>${severityBadge(a.risk_level)}</td>
          <td><button class="refresh-btn" style="padding:3px 8px; font-size:9px;" onclick="unlinkAlertFromCase('${a._id}')">Unlink</button></td>
        </tr>`).join('')
      : `<tr><td colspan="5" class="lstm-hint" style="padding:14px;">No alerts linked yet.</td></tr>`;

    const suggestedRows = (suggested.results || []).length
      ? suggested.results.map(a => `
        <div class="alert-row">
          <div class="alert-main">
            <div class="alert-title">${(a.event_type || 'event').replace(/_/g, ' ')} — <code class="inline">${a.ip || '—'}</code></div>
            <div class="alert-sub">${a.user_id || 'unknown user'} · shares IP/user with this case</div>
          </div>
          ${severityBadge(a.risk_level)}
          <button class="refresh-btn" style="padding:3px 8px; font-size:9px;" onclick="linkAlertToCase('${a._id}')">Link</button>
        </div>`).join('')
      : `<p class="lstm-hint">No related open alerts found.</p>`;

    const timelineRows = (c.timeline || []).slice().reverse().map(t => `
      <div class="alert-row">
        <div class="alert-main">
          <div class="alert-title">${t.content}</div>
          <div class="alert-sub">${t.author} · ${(t.type || '').replace('_', ' ')}</div>
        </div>
        <span class="alert-time" style="width:auto;">${formatFeedTime(t.ts)}</span>
      </div>`).join('') || '<p class="lstm-hint">No activity yet.</p>';

    const entityChips = (c.entity_risk_snapshot || []).length
      ? c.entity_risk_snapshot.map(e => `<span class="xai-chip">${e.type}:${e.id} <b>${e.risk_score}</b></span>`).join('')
      : '<span class="lstm-hint">No entity risk history yet.</span>';

    panel.innerHTML = `
      <div class="panel-title" style="margin-bottom:12px;">
        <span>${c.title}</span>
        <span>${severityBadge(c.severity)}</span>
      </div>
      <div class="filter-row" style="margin-bottom:18px;">
        <label class="filter-checkbox">Status:
          <select class="status-select" onchange="updateCaseField('status', this.value)">${statusOptions}</select>
        </label>
        <label class="filter-checkbox">Assigned:
          <input type="text" id="caseAssignedInput" value="${c.assigned_to || ''}" placeholder="Analyst name" onblur="updateCaseField('assigned_to', this.value)">
        </label>
      </div>

      <div class="panel-title">Linked Alerts (${c.alert_count || 0})</div>
      <table style="margin-bottom:18px;">
        <thead><tr><th>Last Seen</th><th>IP</th><th>Event Type</th><th>Severity</th><th></th></tr></thead>
        <tbody>${alertsRows}</tbody>
      </table>

      <div class="panel-title">Suggested Related Alerts <span class="lstm-user-tag">shares IP/user, not yet linked</span></div>
      <div style="margin-bottom:18px;">${suggestedRows}</div>

      <div class="panel-title">Entity Risk Snapshot <span class="lstm-user-tag">UEBA</span></div>
      <div class="xai-chips" style="margin-bottom:18px;">${entityChips}</div>

      <div class="panel-title">Timeline</div>
      <div style="margin-bottom:14px;">
        <textarea id="caseNoteInput" placeholder="Add investigation note…" style="width:100%; min-height:60px; background:var(--input-bg); border:1px solid var(--border); color:var(--text); border-radius:8px; padding:8px; font-family:var(--font-sans); font-size:12.5px; resize:vertical;"></textarea>
        <button class="refresh-btn" style="margin-top:6px;" onclick="submitCaseNote()">Add Note</button>
      </div>
      <div id="caseTimelineList">${timelineRows}</div>
    `;
  } catch (e) {
    panel.innerHTML = `<p class="lstm-hint">Couldn't load case: ${e.message}</p>`;
  }
}

async function updateCaseField(field, value) {
  try {
    const res = await authFetch(`${API_BASE}/cases/${casesState.currentCaseId}`, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [field]: value }),
    });
    if (!res.ok) throw new Error(`Request failed: ${res.status}`);
    renderCaseDetail();
  } catch (e) {
    alert(`Couldn't update case: ${e.message}`);
    renderCaseDetail();
  }
}

async function linkAlertToCase(alertId) {
  try {
    const res = await authFetch(`${API_BASE}/cases/${casesState.currentCaseId}/link-alert`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ alert_id: alertId }),
    });
    if (!res.ok) throw new Error(`Request failed: ${res.status}`);
    renderCaseDetail();
  } catch (e) {
    alert(`Couldn't link alert: ${e.message}`);
  }
}

async function unlinkAlertFromCase(alertId) {
  try {
    const res = await authFetch(`${API_BASE}/cases/${casesState.currentCaseId}/unlink-alert`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ alert_id: alertId }),
    });
    if (!res.ok) throw new Error(`Request failed: ${res.status}`);
    renderCaseDetail();
  } catch (e) {
    alert(`Couldn't unlink alert: ${e.message}`);
  }
}

async function submitCaseNote() {
  const input = document.getElementById('caseNoteInput');
  const content = input.value.trim();
  if (!content) return;
  try {
    const res = await authFetch(`${API_BASE}/cases/${casesState.currentCaseId}/notes`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content }),
    });
    if (!res.ok) throw new Error(`Request failed: ${res.status}`);
    input.value = '';
    renderCaseDetail();
  } catch (e) {
    alert(`Couldn't add note: ${e.message}`);
  }
}

// ===== Fraud =====
async function loadFraud() {
  const tbody = document.getElementById('fraudBody');
  tbody.innerHTML = '<tr class="loading-row"><td colspan="6">Loading…</td></tr>';
  const minAmount = document.getElementById('fraudMinAmount').value;
  const qs = minAmount ? `?min_amount=${minAmount}` : '';
  try {
    const data = await apiGet(`/fraud${qs}`);
    const results = data.results || [];
    document.getElementById('fraudCount').textContent = data.count ?? 0;
    document.getElementById('fraudAmount').textContent = '$' + (data.total_flagged_amount ?? 0).toLocaleString();
    if (results.length === 0) { tbody.innerHTML = emptyRow(6, 'No flagged transactions', 'Transactions above the risk threshold will appear here.'); return; }
    tbody.innerHTML = results.map(r => `
      <tr>
        <td>${r.timestamp || '—'}</td>
        <td>${r.user_id || '—'}</td>
        <td><code class="inline">${r.ip || '—'}</code></td>
        <td>${r.amount != null ? '$' + Number(r.amount).toLocaleString() : '—'}</td>
        <td>${r.risk_score ?? '—'}</td>
        <td style="color:var(--text-muted);">${(r.reason || []).join(', ') || '—'}</td>
      </tr>`).join('');
  } catch (e) {
    tbody.innerHTML = errorRow(6, e.message);
  }
}

// ===== User Behavior =====
async function loadUserBehavior() {
  const tbody = document.getElementById('userBehaviorBody');
  tbody.innerHTML = '<tr class="loading-row"><td colspan="5">Loading…</td></tr>';
  const userId = document.getElementById('userBehaviorFilter').value.trim();
  loadLstmRisk(userId);
  const qs = userId ? `?user_id=${encodeURIComponent(userId)}` : '';
  try {
    const data = await apiGet(`/user-behavior${qs}`);
    const results = data.results || [];
    if (results.length === 0) { tbody.innerHTML = emptyRow(5, 'No behavior data', 'Customer/account activity events will appear here.'); return; }
    tbody.innerHTML = results.map(r => `
      <tr>
        <td>${r.timestamp || '—'}</td>
        <td>${r.user_id || '—'}</td>
        <td><code class="inline">${r.ip || '—'}</code></td>
        <td>${r.event_type || '—'}</td>
        <td>${severityBadge(r.risk_level || r.severity)}</td>
      </tr>`).join('');
  } catch (e) {
    tbody.innerHTML = errorRow(5, e.message);
  }
}

async function loadLstmRisk(userId) {
  const panel = document.getElementById('lstmRiskPanel');
  const content = document.getElementById('lstmRiskContent');
  const tag = document.getElementById('lstmUserTag');
  if (!userId) { panel.style.display = 'none'; return; }
  panel.style.display = 'block';
  tag.textContent = userId;
  content.innerHTML = '<p class="lstm-hint">Scoring recent activity…</p>';
  try {
    const data = await apiGet(`/user-behavior/${encodeURIComponent(userId)}/hybrid-risk`);
    if (!data.available) { content.innerHTML = `<p class="lstm-hint">${data.reason || 'Not enough recent activity to score this user yet.'}</p>`; return; }
    const rule = data.components.rule_based;
    const lstm = data.components.lstm_behavioral;
    const isAnomaly = !!data.hybrid_anomaly;
    const lstmRowHtml = lstm.available
      ? `${severityBadge(lstm.anomaly ? 'critical' : 'low')}<span class="hybrid-component-score">${lstm.score}</span>`
      : `<span class="lstm-hint">${lstm.reason || 'Not enough sequence history yet'}</span>`;
    const topFeaturesHtml = (lstm.top_features && lstm.top_features.length)
      ? `<div class="xai-block"><div class="event-type-name" style="margin-bottom:4px;">Top contributing signals (XAI)</div>
           <div class="xai-chips">${lstm.top_features.map(f => `<span class="xai-chip">${f.label} <b>${f.contribution_pct}%</b></span>`).join('')}</div></div>`
      : '';
    content.innerHTML = `
      <div class="lstm-score-row">
        <div class="kpi-value-v2 ${isAnomaly ? 'high' : 'accent'}">${data.unified_score}</div>
        <div class="lstm-score-meta">
          <span class="badge ${isAnomaly ? 'critical' : 'low'}">${isAnomaly ? 'Hybrid: anomalous' : 'Hybrid: normal'}</span>
          <p class="lstm-hint">Unified score blends the rule-based engine (50%) and the LSTM behavioral model (50%), and escalates if either model flags anomalous activity on its own.</p>
          <div class="hybrid-components">
            <div class="hybrid-component-row"><span class="hybrid-component-label">Rule-based</span>${severityBadge(rule.level)}<span class="hybrid-component-score">${rule.score}</span></div>
            <div class="hybrid-component-row"><span class="hybrid-component-label">LSTM behavioral</span>${lstmRowHtml}</div>
          </div>
          ${topFeaturesHtml}
        </div>
      </div>`;
  } catch (e) {
    content.innerHTML = `<p class="lstm-hint">Couldn't load hybrid risk score: ${e.message}</p>`;
  }
}

// ===== Admin Activity =====
async function loadAdminActivity() {
  const tbody = document.getElementById('adminActivityBody');
  tbody.innerHTML = '<tr class="loading-row"><td colspan="5">Loading…</td></tr>';
  try {
    const data = await apiGet('/admin-activity');
    const results = data.results || [];
    if (results.length === 0) { tbody.innerHTML = emptyRow(5, 'No admin activity', 'Internal/admin events will appear here.'); return; }
    tbody.innerHTML = results.map(r => `
      <tr>
        <td>${r.timestamp || '—'}</td>
        <td>${r.user_id || '—'}</td>
        <td><code class="inline">${r.ip || '—'}</code>${renderOriginChip(r.threat_intel)}</td>
        <td>${r.event_type || '—'}${renderMitreChip(r.mitre)}</td>
        <td>${severityBadge(r.risk_level || r.severity)}</td>
      </tr>`).join('');
  } catch (e) {
    tbody.innerHTML = errorRow(5, e.message);
  }
}

// ===== AI Insights =====
async function loadAiInsights() {
  const summaryEl = document.getElementById('aiInsightsSummary');
  summaryEl.textContent = 'Loading…';
  try {
    const data = await apiGet('/ai-insights');
    summaryEl.textContent = data.summary || 'No summary available.';
    const risks = data.top_risks || [];
    const panel = document.getElementById('aiTopRisksPanel');
    if (risks.length > 0) {
      panel.style.display = 'block';
      document.getElementById('aiTopRisksList').innerHTML = risks.map(r => `<div style="padding:7px 0; border-bottom:1px solid var(--border); font-size:12.5px; color:var(--text-muted);">${r}</div>`).join('');
    } else { panel.style.display = 'none'; }
  } catch (e) {
    summaryEl.textContent = `Couldn't load AI insights: ${e.message}`;
  }
}

// ===== Logs =====
function debouncedLogsSearch() {
  clearTimeout(logsDebounceTimer);
  logsDebounceTimer = setTimeout(() => { logsState.page = 1; loadLogs(); }, 350);
}
async function loadLogs() {
  const tbody = document.getElementById('logsBody');
  tbody.innerHTML = '<tr class="loading-row"><td colspan="4">Loading…</td></tr>';
  const search = document.getElementById('logsSearch').value.trim();
  const eventType = document.getElementById('logsEventType').value.trim();
  const severity = document.getElementById('logsSeverity').value;
  const params = new URLSearchParams({ page: logsState.page, page_size: logsState.page_size });
  if (search) params.set('search', search);
  if (eventType) params.set('event_type', eventType);
  if (severity) params.set('severity', severity);
  try {
    const data = await apiGet(`/logs?${params.toString()}`);
    const results = data.results || [];
    logsState.total_pages = data.total_pages || 1;
    if (results.length === 0) {
      tbody.innerHTML = emptyRow(4, 'No logs found', 'Try adjusting filters, or check that events have been ingested.');
    } else {
      tbody.innerHTML = results.map(r => `
        <tr>
          <td>${r.timestamp || '—'}</td>
          <td><code class="inline">${r.ip || '—'}</code></td>
          <td>${r.event_type || '—'}</td>
          <td>${severityBadge(r.severity)}</td>
        </tr>`).join('');
    }
    document.getElementById('logsPageInfo').textContent = `Page ${data.page || 1} of ${data.total_pages || 1} (${data.total ?? 0} total)`;
    document.getElementById('logsPrevBtn').disabled = (data.page || 1) <= 1;
    document.getElementById('logsNextBtn').disabled = (data.page || 1) >= (data.total_pages || 1);
  } catch (e) {
    tbody.innerHTML = errorRow(4, e.message);
  }
}
function changeLogsPage(delta) {
  const newPage = logsState.page + delta;
  if (newPage < 1 || newPage > logsState.total_pages) return;
  logsState.page = newPage;
  loadLogs();
}

// ===== Webhooks =====
const WEBHOOK_EVENT_LABELS = {
  'alert.created': 'Alert created',
  'alert.status_changed': 'Alert status changed',
  'case.created': 'Case created',
  'case.status_changed': 'Case status changed',
  'case.note_added': 'Case note added',
};
let webhooksState = { availableEvents: Object.keys(WEBHOOK_EVENT_LABELS) };

function eventCheckboxesHtml(idPrefix, checkedEvents) {
  checkedEvents = checkedEvents || [];
  return webhooksState.availableEvents.map(ev => `
    <label class="filter-checkbox" style="margin-right:4px;">
      <input type="checkbox" id="${idPrefix}_${ev}" value="${ev}" ${checkedEvents.includes(ev) ? 'checked' : ''}>
      ${WEBHOOK_EVENT_LABELS[ev] || ev}
    </label>`).join('');
}
function readCheckedEvents(idPrefix) {
  return webhooksState.availableEvents.filter(ev => document.getElementById(`${idPrefix}_${ev}`)?.checked);
}

async function loadWebhooks() {
  const tbody = document.getElementById('webhooksBody');
  tbody.innerHTML = '<tr class="loading-row"><td colspan="5">Loading…</td></tr>';
  try {
    const data = await apiGet('/webhooks');
    webhooksState.availableEvents = data.available_events && data.available_events.length ? data.available_events : webhooksState.availableEvents;
    document.getElementById('newWebhookEventsWrap').innerHTML = eventCheckboxesHtml('newWh', []);
    const results = data.results || [];
    if (!results.length) {
      tbody.innerHTML = emptyRow(5, 'No webhooks configured', 'Add one above to get notified when alerts or cases change.');
      return;
    }
    tbody.innerHTML = results.map(w => `
      <tr>
        <td style="max-width:260px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${w.url}">${w.description ? `<div>${w.description}</div>` : ''}<code class="inline">${w.url}</code></td>
        <td style="color:var(--text-muted); font-size:11.5px;">${(w.events || []).map(e => WEBHOOK_EVENT_LABELS[e] || e).join(', ')}</td>
        <td>${w.active ? '<span class="badge low">active</span>' : '<span class="badge medium">paused</span>'}</td>
        <td>${w.has_secret ? '<span class="badge low">yes</span>' : '<span class="lstm-hint">no</span>'}</td>
        <td style="white-space:nowrap;">
          <button class="refresh-btn" style="padding:3px 8px; font-size:9px;" onclick="toggleWebhookActive('${w._id}', ${!w.active})">${w.active ? 'Pause' : 'Resume'}</button>
          <button class="refresh-btn" style="padding:3px 8px; font-size:9px;" onclick="testWebhook('${w._id}')">Test</button>
          <button class="refresh-btn" style="padding:3px 8px; font-size:9px;" onclick="viewWebhookDeliveries('${w._id}', '${w.url.replace(/'/g, "\\'")}')">Deliveries</button>
          <button class="refresh-btn" style="padding:3px 8px; font-size:9px;" onclick="deleteWebhook('${w._id}')">Delete</button>
        </td>
      </tr>`).join('');
  } catch (e) {
    tbody.innerHTML = errorRow(5, e.message);
  }
}

function showNewWebhookForm() { document.getElementById('newWebhookFormWrap').style.display = 'block'; }
function hideNewWebhookForm() {
  document.getElementById('newWebhookFormWrap').style.display = 'none';
  document.getElementById('newWebhookUrl').value = '';
  document.getElementById('newWebhookSecret').value = '';
  document.getElementById('newWebhookDescription').value = '';
}

async function submitNewWebhook() {
  const url = document.getElementById('newWebhookUrl').value.trim();
  if (!url) { alert('A URL is required.'); return; }
  const events = readCheckedEvents('newWh');
  if (!events.length) { alert('Pick at least one event to subscribe to.'); return; }
  const secret = document.getElementById('newWebhookSecret').value.trim() || null;
  const description = document.getElementById('newWebhookDescription').value.trim() || null;
  try {
    const res = await authFetch(`${API_BASE}/webhooks`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, events, secret, description }),
    });
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `Request failed: ${res.status}`);
    }
    hideNewWebhookForm();
    loadWebhooks();
  } catch (e) {
    alert(`Couldn't create webhook: ${e.message}`);
  }
}

async function toggleWebhookActive(webhookId, newActive) {
  try {
    const res = await authFetch(`${API_BASE}/webhooks/${webhookId}`, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ active: newActive }),
    });
    if (!res.ok) throw new Error(`Request failed: ${res.status}`);
    loadWebhooks();
  } catch (e) {
    alert(`Couldn't update webhook: ${e.message}`);
  }
}

async function testWebhook(webhookId) {
  try {
    const res = await authFetch(`${API_BASE}/webhooks/${webhookId}/test`, { method: 'POST' });
    if (!res.ok) throw new Error(`Request failed: ${res.status}`);
    const result = await res.json();
    if (result.success) {
      alert(`Test delivered successfully (HTTP ${result.status_code}).`);
    } else {
      alert(`Test delivery failed: ${result.error || 'unknown error'}`);
    }
  } catch (e) {
    alert(`Couldn't send test: ${e.message}`);
  }
}

async function deleteWebhook(webhookId) {
  if (!confirm('Delete this webhook? This cannot be undone.')) return;
  try {
    const res = await authFetch(`${API_BASE}/webhooks/${webhookId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error(`Request failed: ${res.status}`);
    loadWebhooks();
  } catch (e) {
    alert(`Couldn't delete webhook: ${e.message}`);
  }
}

async function viewWebhookDeliveries(webhookId, url) {
  document.getElementById('webhooksListPanel').style.display = 'none';
  const panel = document.getElementById('webhookDeliveriesPanel');
  panel.style.display = 'block';
  panel.dataset.webhookId = webhookId;
  document.getElementById('webhookDeliveriesUrlTag').textContent = url;
  const tbody = document.getElementById('webhookDeliveriesBody');
  tbody.innerHTML = '<tr class="loading-row"><td colspan="4">Loading…</td></tr>';
  try {
    const data = await apiGet(`/webhooks/${webhookId}/deliveries?limit=30`);
    const results = data.results || [];
    if (!results.length) {
      tbody.innerHTML = emptyRow(4, 'No deliveries yet', 'Nothing has fired for this webhook, or try the Test button.');
      return;
    }
    tbody.innerHTML = results.map(d => `
      <tr>
        <td>${formatFeedTime(d.sent_at)}</td>
        <td>${WEBHOOK_EVENT_LABELS[d.event_type] || d.event_type}</td>
        <td>${d.success ? '<span class="badge low">delivered</span>' : '<span class="badge critical">failed</span>'}</td>
        <td style="color:var(--text-muted); font-size:11.5px;">${d.status_code ? `HTTP ${d.status_code}` : ''}${d.error ? ` — ${d.error}` : ''}</td>
      </tr>`).join('');
  } catch (e) {
    tbody.innerHTML = errorRow(4, e.message);
  }
}
function closeWebhookDeliveries() {
  document.getElementById('webhookDeliveriesPanel').style.display = 'none';
  document.getElementById('webhooksListPanel').style.display = 'block';
}

// ===== Admin Users (Settings tab, admin-only) =====
async function loadAdminUsersIfAdmin() {
  if (!currentUser || currentUser.role !== 'admin') return;
  const tbody = document.getElementById('adminUsersBody');
  tbody.innerHTML = '<tr class="loading-row"><td colspan="5">Loading…</td></tr>';
  try {
    const data = await apiGet('/auth/users');
    const results = data.results || [];
    if (!results.length) { tbody.innerHTML = emptyRow(5, 'No accounts', ''); return; }
    tbody.innerHTML = results.map(u => `
      <tr>
        <td>${u.username}${u.display_name && u.display_name !== u.username ? ` <span class="lstm-hint">(${u.display_name})</span>` : ''}</td>
        <td style="text-transform:capitalize;">${u.role}</td>
        <td>${u.active ? '<span class="badge low">active</span>' : '<span class="badge medium">disabled</span>'}</td>
        <td>${u.last_login ? formatFeedTime(u.last_login) : '<span class="lstm-hint">never</span>'}</td>
        <td>
          ${u.username === currentUser.username ? '<span class="lstm-hint">you</span>' :
            `<button class="refresh-btn" style="padding:3px 8px; font-size:9px;" onclick="toggleUserActive('${u._id}', ${!u.active})">${u.active ? 'Disable' : 'Enable'}</button>`}
        </td>
      </tr>`).join('');
  } catch (e) {
    tbody.innerHTML = errorRow(5, e.message);
  }
}
function showNewUserForm() { document.getElementById('newUserFormWrap').style.display = 'block'; }
function hideNewUserForm() {
  document.getElementById('newUserFormWrap').style.display = 'none';
  document.getElementById('newUserUsername').value = '';
  document.getElementById('newUserPassword').value = '';
  document.getElementById('newUserDisplayName').value = '';
}
async function submitNewUser() {
  const username = document.getElementById('newUserUsername').value.trim();
  const password = document.getElementById('newUserPassword').value;
  const display_name = document.getElementById('newUserDisplayName').value.trim() || null;
  const role = document.getElementById('newUserRole').value;
  if (!username || !password) { alert('Username and password are required.'); return; }
  if (password.length < 8) { alert('Password must be at least 8 characters.'); return; }
  try {
    const res = await authFetch(`${API_BASE}/auth/users`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password, role, display_name }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed: ${res.status}`);
    }
    hideNewUserForm();
    loadAdminUsersIfAdmin();
  } catch (e) {
    alert(`Couldn't create account: ${e.message}`);
  }
}
async function toggleUserActive(userId, newActive) {
  try {
    const res = await authFetch(`${API_BASE}/auth/users/${userId}/active`, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ active: newActive }),
    });
    if (!res.ok) throw new Error(`Request failed: ${res.status}`);
    loadAdminUsersIfAdmin();
  } catch (e) {
    alert(`Couldn't update account: ${e.message}`);
  }
}

// ===== Theme =====
(function () {
  const savedTheme = localStorage.getItem('siem-dashboard-theme');
  if (savedTheme === 'light') document.body.classList.add('light-mode');
})();
function updateThemeButton() {
  const isLight = document.body.classList.contains('light-mode');
  document.getElementById('themeToggle').textContent = isLight ? 'LIGHT THEME' : 'DARK THEME';
}
function toggleTheme() {
  const isLight = document.body.classList.toggle('light-mode');
  localStorage.setItem('siem-dashboard-theme', isLight ? 'light' : 'dark');
  updateThemeButton();
  if (activeTab === 'overview') loadOverview();
  window.dispatchEvent(new Event('resize'));
}
document.addEventListener('DOMContentLoaded', updateThemeButton);
updateThemeButton();

// ===== Init =====
applyAuthUI();
checkStatus();
setInterval(checkStatus, 15000);
// Only load data once an analyst is actually signed in -- most of these
// calls hit endpoints that now require a session token anyway, and
// firing them before login would just produce a wall of 401s.
if (authToken && currentUser) {
  loadOverview();
  loadAdminUsersIfAdmin();
}