/**
 * FinSight AI - Financial Risk Assessment Matrix Controller
 */

let currentReviewData = null;
let agentChartInstance = null;

document.addEventListener('DOMContentLoaded', async () => {
  if (typeof requireAuth === "function") {
    const authed = await requireAuth();
    if (!authed) return;
  }
  if (typeof updateSidebarUserInfo === "function") {
    updateSidebarUserInfo();
  }

  const container = document.getElementById('riskContainer');
  if (!container) return;

  showLoadingState(container, "Calculating risk score & multi-agent breakdown...");

  const review = await loadCurrentReview();
  if (!review) {
    showEmptyReviewState(container);
    return;
  }
  currentReviewData = review;

  const subhead = document.getElementById('riskSubhead');
  if (subhead && review.filename) {
    subhead.textContent = `Composite risk evaluation for statement: ${review.filename} (ID: ${review.review_id})`;
  }

  renderRiskOverview(review);
  renderAgentChart(review);
  renderContributorsTable(review);

  window.addEventListener('themeChanged', () => {
    if (currentReviewData) {
      renderAgentChart(currentReviewData);
    }
  });
});

function renderRiskOverview(review) {
  const riskScore = review.risk_score || { total_points: 0, risk_level: 'LOW', breakdown: [] };
  const totalPoints = riskScore.total_points || 0;
  const level = (riskScore.risk_level || 'LOW').toUpperCase();

  const ptsEl = document.getElementById('riskScorePoints');
  const badgeEl = document.getElementById('riskLevelBadge');
  const summaryEl = document.getElementById('riskScoreSummary');

  if (ptsEl) ptsEl.textContent = `${totalPoints} / 100`;
  
  if (badgeEl) {
    badgeEl.textContent = `${level} RISK`;
    const sevClass = level === 'CRITICAL' ? 'badge-critical' : (level === 'HIGH' ? 'badge-high' : (level === 'MEDIUM' ? 'badge-medium' : 'badge-low'));
    badgeEl.className = `badge ${sevClass}`;
  }

  if (summaryEl) {
    const failedVals = (review.validation_results || []).filter(v => !v.passed).length;
    const anomalyCount = (review.anomalies || []).length;
    const devCount = (review.trend_deviations || []).length;

    summaryEl.textContent = `Statement risk profile evaluated across ${failedVals} failed rule checks, ${anomalyCount} anomaly flags, and ${devCount} material trend deviations.`;
  }
}

function getNormalizedContributors(review) {
  const riskScore = review.risk_score || {};
  if (Array.isArray(riskScore.breakdown) && riskScore.breakdown.length > 0) {
    return riskScore.breakdown.map(b => ({
      source_agent: formatAgentName(b.source_agent),
      points: b.points || 0,
      severity: b.points >= 20 ? 'CRITICAL' : (b.points >= 10 ? 'HIGH' : 'MEDIUM'),
      reasoning: b.reasoning || 'Risk factor identified by agent.'
    })).sort((a, b) => b.points - a.points);
  }

  // Fallback: build from findings
  const list = [];
  (review.validation_results || []).filter(v => !v.passed).forEach(v => {
    list.push({
      source_agent: 'Validation Agent',
      points: v.severity === 'CRITICAL' ? 25 : (v.severity === 'HIGH' ? 15 : 10),
      severity: (v.severity || 'HIGH').toUpperCase(),
      reasoning: `Failed Rule: ${v.rule_id} (${v.rule_name || v.description})`
    });
  });

  (review.anomalies || []).forEach(a => {
    list.push({
      source_agent: 'Anomaly Agent',
      points: a.severity === 'CRITICAL' ? 20 : (a.severity === 'HIGH' ? 12 : 8),
      severity: (a.severity || 'MEDIUM').toUpperCase(),
      reasoning: `Flagged Metric: ${a.metric} (${a.description || a.anomaly_type})`
    });
  });

  (review.trend_deviations || []).forEach(d => {
    list.push({
      source_agent: 'Ratio Trend Agent',
      points: d.severity === 'CRITICAL' ? 15 : (d.severity === 'HIGH' ? 10 : 5),
      severity: (d.severity || 'HIGH').toUpperCase(),
      reasoning: `Deviated Metric: ${d.metric} (${formatPercent(d.deviation_percent)} deviation)`
    });
  });

  return list.sort((a, b) => b.points - a.points);
}

function formatAgentName(raw) {
  if (!raw) return 'Audit Agent';
  return raw.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function renderAgentChart(review) {
  const canvas = document.getElementById('agentRiskChart');
  if (!canvas) return;

  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  const textColor = isLight ? '#475569' : '#9ca3af';
  const gridColor = isLight ? '#e2e8f0' : '#374151';

  const contributors = getNormalizedContributors(review);
  const agentPointsMap = {};

  contributors.forEach(c => {
    agentPointsMap[c.source_agent] = (agentPointsMap[c.source_agent] || 0) + c.points;
  });

  const labels = Object.keys(agentPointsMap).length > 0 ? Object.keys(agentPointsMap) : ['Validation Agent', 'Anomaly Agent', 'Ratio Agent'];
  const points = Object.keys(agentPointsMap).length > 0 ? Object.values(agentPointsMap) : [0, 0, 0];

  if (agentChartInstance) agentChartInstance.destroy();

  agentChartInstance = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Risk Points Contributed',
        data: points,
        backgroundColor: ['#ef4444', '#f59e0b', '#3b82f6', '#06b6d4', '#6366f1'],
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: textColor, font: { family: 'Inter' } } }
      },
      scales: {
        x: { grid: { color: gridColor }, ticks: { color: textColor } },
        y: { grid: { color: gridColor }, ticks: { color: textColor, stepSize: 5 } }
      }
    }
  });
}

function renderContributorsTable(review) {
  const tbody = document.getElementById('riskContributorsTableBody');
  if (!tbody) return;

  tbody.textContent = '';

  const contributors = getNormalizedContributors(review);
  if (contributors.length === 0) {
    const tr = createElement('tr');
    const td = createElement('td', null, 'No active risk factor contributors recorded.');
    td.colSpan = 4;
    td.style.textAlign = 'center';
    td.style.padding = '24px';
    td.style.color = 'var(--text-muted)';
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  contributors.forEach(item => {
    const tr = createElement('tr');

    // Agent
    const tdAgent = createElement('td');
    const badgeAgent = createElement('span', 'badge badge-low', item.source_agent);
    tdAgent.appendChild(badgeAgent);

    // Points
    const tdPoints = createElement('td', null, `+${item.points} pts`);
    tdPoints.style.fontWeight = '700';
    tdPoints.style.color = item.points >= 15 ? 'var(--status-danger)' : 'var(--text-main)';

    // Severity
    const tdSev = createElement('td');
    const sevClass = item.severity === 'CRITICAL' ? 'badge-critical' : (item.severity === 'HIGH' ? 'badge-high' : 'badge-medium');
    const badgeSev = createElement('span', `badge ${sevClass}`, item.severity);
    tdSev.appendChild(badgeSev);

    // Reasoning
    const tdReason = createElement('td', null, item.reasoning);
    tdReason.style.fontSize = '0.88rem';

    tr.appendChild(tdAgent);
    tr.appendChild(tdPoints);
    tr.appendChild(tdSev);
    tr.appendChild(tdReason);

    tbody.appendChild(tr);
  });
}
