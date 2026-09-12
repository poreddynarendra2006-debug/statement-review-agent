/**
 * AuditLens - Executive Dashboard Controller
 */

let failedRulesChartInstance = null;
let anomaliesSeverityChartInstance = null;

document.addEventListener('DOMContentLoaded', async () => {
  const user = typeof getCurrentUser === "function" ? getCurrentUser() : null;
  if (typeof requireAuth === "function") {
    const authed = await requireAuth();
    if (!authed) return;
  }
  if (typeof updateSidebarUserInfo === "function") {
    updateSidebarUserInfo();
  }

  const container = document.getElementById('dashboardContainer');
  if (!container) return;

  showLoadingState(container, "Loading executive dashboard & AI findings...");

  const review = await loadCurrentReview();
  if (!review) {
    showEmptyReviewState(container);
    return;
  }

  // Update PDF Download Link / Handler
  const pdfBtn = document.getElementById('downloadPdfBtn');
  if (pdfBtn) {
    pdfBtn.addEventListener('click', (e) => {
      e.preventDefault();
      downloadPDFReport(review.review_id);
    });
  }

  // Update Subhead with filename & date
  const subhead = document.getElementById('dashboardSubhead');
  if (subhead && review.filename) {
    const createdStr = review.created_at ? new Date(review.created_at).toLocaleDateString() : '';
    subhead.textContent = `Reviewing statement: ${review.filename} (ID: ${review.review_id}) ${createdStr ? '• ' + createdStr : ''}`;
  }

  renderDashboardView(container, review);

  window.addEventListener('themeChanged', () => {
    if (review) {
      renderDashboardCharts(review);
    }
  });
});

function renderDashboardView(container, review) {
  container.textContent = '';

  const validationResults = review.validation_results || [];
  const anomalies = review.anomalies || [];
  const deviations = review.material_deviations || [];
  const riskResult = review.risk_result || {};
  const riskPoints = review.risk_score === null || review.risk_score === undefined ? 0 : review.risk_score;
  const riskLevel = riskResult.risk_level || 'LOW';

  const failedValidation = review.failed_validations
    || validationResults.filter(r => r.status === 'FAIL');
  const totalRuleChecks = validationResults.length;

  // --- 1. KPI Grid ---
  const kpiGrid = createElement('div', 'kpi-grid');

  // KPI 1: Rule Checks
  const kpi1 = createElement('div', 'kpi-card cyan');
  kpi1.appendChild(createKpiHeader('Total Statement Checks', 'fa-solid fa-list-check'));
  kpi1.appendChild(createElement('div', 'kpi-value', formatNumber(totalRuleChecks)));
  kpi1.appendChild(createKpiFooter('fa-solid fa-shield-halved', `${validationResults.length - failedValidation.length} checks passed`));

  // KPI 2: Failed Checks
  const kpi2 = createElement('div', 'kpi-card rose');
  kpi2.appendChild(createKpiHeader('Failed Validation Rules', 'fa-solid fa-circle-xmark'));
  kpi2.appendChild(createElement('div', 'kpi-value', formatNumber(failedValidation.length)));
  kpi2.appendChild(createKpiFooter('fa-solid fa-triangle-exclamation', failedValidation.length > 0 ? 'Requires reviewer action' : 'No rule failures', failedValidation.length > 0));

  // KPI 3: Trend Anomalies
  const kpi3 = createElement('div', 'kpi-card amber');
  kpi3.appendChild(createKpiHeader('Anomaly Flags', 'fa-solid fa-flag'));
  kpi3.appendChild(createElement('div', 'kpi-value', formatNumber(anomalies.length)));
  kpi3.appendChild(createKpiFooter('fa-solid fa-chart-line', `${deviations.length} material deviations`));

  // KPI 4: Composite Risk
  const kpi4Class = riskLevel === 'CRITICAL' || riskLevel === 'HIGH' ? 'kpi-card rose' : 'kpi-card emerald';
  const kpi4 = createElement('div', kpi4Class);
  kpi4.appendChild(createKpiHeader('Composite Risk Score', 'fa-solid fa-shield-virus'));
  kpi4.appendChild(createElement('div', 'kpi-value', `${riskPoints} / 100`));
  kpi4.appendChild(createKpiFooter('fa-solid fa-shield', `Risk Level: ${riskLevel}`));

  kpiGrid.appendChild(kpi1);
  kpiGrid.appendChild(kpi2);
  kpiGrid.appendChild(kpi3);
  kpiGrid.appendChild(kpi4);
  container.appendChild(kpiGrid);

  // --- 2. Charts Section ---
  const grid2 = createElement('div', 'grid-2');

  // Chart Card 1: Failed Checks by Rule
  const cardChart1 = createElement('div', 'card');
  const headerChart1 = createElement('div', 'card-header');
  const titleChart1 = createElement('div', 'card-title');
  titleChart1.appendChild(createElement('i', 'fa-solid fa-chart-bar'));
  titleChart1.appendChild(document.createTextNode(' Failed Rule Checks'));
  headerChart1.appendChild(titleChart1);
  cardChart1.appendChild(headerChart1);

  const canvasContainer1 = createElement('div');
  canvasContainer1.style.height = '280px';
  canvasContainer1.style.position = 'relative';
  const canvas1 = createElement('canvas');
  canvas1.id = 'failedRulesChart';
  canvasContainer1.appendChild(canvas1);
  cardChart1.appendChild(canvasContainer1);

  // Chart Card 2: Anomalies by Severity
  const cardChart2 = createElement('div', 'card');
  const headerChart2 = createElement('div', 'card-header');
  const titleChart2 = createElement('div', 'card-title');
  titleChart2.appendChild(createElement('i', 'fa-solid fa-pie-chart'));
  titleChart2.appendChild(document.createTextNode(' Findings Severity Breakdown'));
  headerChart2.appendChild(titleChart2);
  cardChart2.appendChild(headerChart2);

  const canvasContainer2 = createElement('div');
  canvasContainer2.style.height = '280px';
  canvasContainer2.style.position = 'relative';
  canvasContainer2.style.display = 'flex';
  canvasContainer2.style.alignItems = 'center';
  canvasContainer2.style.justifyContent = 'center';
  const canvas2 = createElement('canvas');
  canvas2.id = 'anomaliesSeverityChart';
  canvasContainer2.appendChild(canvas2);
  cardChart2.appendChild(canvasContainer2);

  grid2.appendChild(cardChart1);
  grid2.appendChild(cardChart2);
  container.appendChild(grid2);

  // --- 3. Executive AI Synthesis & Table Grid ---
  const gridEqual = createElement('div', 'grid-equal');

  // AI Synthesis Card
  const aiCard = createElement('div', 'card');
  aiCard.style.borderLeft = '4px solid var(--accent-cyan)';

  const aiHeader = createElement('div', 'card-header');
  const aiTitle = createElement('div', 'card-title');
  const brainIcon = createElement('i', 'fa-solid fa-brain');
  brainIcon.style.color = 'var(--accent-cyan)';
  aiTitle.appendChild(brainIcon);
  aiTitle.appendChild(document.createTextNode(' Executive Summary & AI Synthesis'));
  
  const modeBadge = createElement('span', 'badge badge-low', (review.ai_summary && review.ai_summary.review_mode) ? review.ai_summary.review_mode : 'AI Review');
  aiHeader.appendChild(aiTitle);
  aiHeader.appendChild(modeBadge);
  aiCard.appendChild(aiHeader);

  const aiBody = createElement('div');
  aiBody.style.display = 'flex';
  aiBody.style.flexDirection = 'column';
  aiBody.style.gap = '12px';
  aiBody.style.fontSize = '0.9rem';

  const summaryText = typeof review.summary === 'string' ? review.summary : (review.ai_summary?.summary || 'Statement processed and validated by AuditLens agent.');
  const paragraphs = summaryText.split('\n\n').filter(p => p.trim());

  paragraphs.forEach(p => {
    const pEl = createElement('p', null, p);
    pEl.style.lineHeight = '1.6';
    pEl.style.color = 'var(--text-main)';
    aiBody.appendChild(pEl);
  });

  // Skipped coverage or warnings. The API reports these as coverage.skipped,
  // an object of agent name -> the reason it did not run.
  const skipped = Object.entries((review.coverage && review.coverage.skipped) || {})
    .map(([agent, reason]) => `${agent}: ${reason}`);
  if (skipped.length > 0) {
    const warnBox = createElement('div');
    warnBox.style.padding = '10px 14px';
    warnBox.style.borderRadius = 'var(--radius-md)';
    warnBox.style.background = 'var(--bg-input)';
    warnBox.style.border = '1px solid var(--border-color)';
    warnBox.style.marginTop = '8px';

    const warnTitle = createElement('strong', null, 'Skipped Coverage / Scope Limitations:');
    warnTitle.style.color = 'var(--status-warning)';
    warnTitle.style.display = 'block';
    warnTitle.style.marginBottom = '4px';
    warnBox.appendChild(warnTitle);

    skipped.forEach(item => {
      const itemEl = createElement('div', null, `• ${item}`);
      itemEl.style.fontSize = '0.82rem';
      itemEl.style.color = 'var(--text-muted)';
      warnBox.appendChild(itemEl);
    });
    aiBody.appendChild(warnBox);
  }

  const btnGroup = createElement('div');
  btnGroup.style.marginTop = '16px';
  btnGroup.style.display = 'flex';
  btnGroup.style.gap = '12px';

  const findingsBtn = createElement('a', 'btn btn-primary btn-sm', 'Review Audit Findings');
  findingsBtn.href = 'findings.html';
  const fIcon = createElement('i', 'fa-solid fa-arrow-right');
  fIcon.style.marginRight = '6px';
  findingsBtn.prepend(fIcon);

  const aiBtn = createElement('a', 'btn btn-secondary btn-sm', 'Full AI Summary');
  aiBtn.href = 'chatbot.html';
  const aiIcon = createElement('i', 'fa-solid fa-robot');
  aiIcon.style.marginRight = '6px';
  aiBtn.prepend(aiIcon);

  btnGroup.appendChild(findingsBtn);
  btnGroup.appendChild(aiBtn);
  aiBody.appendChild(btnGroup);

  aiCard.appendChild(aiBody);

  // High Priority Flags Card Table
  const flagsCard = createElement('div', 'card');
  const flagsHeader = createElement('div', 'card-header');
  const flagsTitle = createElement('div', 'card-title');
  flagsTitle.appendChild(createElement('i', 'fa-solid fa-triangle-exclamation'));
  flagsTitle.appendChild(document.createTextNode(' Priority Potential Inconsistencies'));

  const viewAllLink = createElement('a', null, `View All (${failedValidation.length + anomalies.length})`);
  viewAllLink.href = 'findings.html';
  viewAllLink.style.fontSize = '0.8rem';
  
  flagsHeader.appendChild(flagsTitle);
  flagsHeader.appendChild(viewAllLink);
  flagsCard.appendChild(flagsHeader);

  const tableWrapper = createElement('div', 'table-responsive');
  const table = createElement('table', 'custom-table');

  const thead = createElement('thead');
  const trHead = createElement('tr');
  ['Source', 'Metric / Rule', 'Severity', 'Company / Year'].forEach(h => {
    trHead.appendChild(createElement('th', null, h));
  });
  thead.appendChild(trHead);
  table.appendChild(thead);

  const tbody = createElement('tbody');
  
  // Collect top findings
  const priorityItems = [];
  failedValidation.forEach(v => {
    priorityItems.push({
      source: 'Validation Rule',
      name: `${v.rule_id}: ${v.rule_name || v.description}`,
      severity: v.severity || 'HIGH',
      info: `${v.company || '-'} (${v.year || '-'})`
    });
  });
  anomalies.forEach(a => {
    priorityItems.push({
      source: 'Anomaly',
      name: a.anomaly_type || a.description || 'Anomaly',
      severity: a.severity || 'MEDIUM',
      info: `${a.company || '-'} (${a.year || '-'})`
    });
  });

  if (priorityItems.length === 0) {
    const emptyTr = createElement('tr');
    const emptyTd = createElement('td', null, 'No priority inconsistencies found in this review.');
    emptyTd.colSpan = 4;
    emptyTd.style.textAlign = 'center';
    emptyTd.style.padding = '24px';
    emptyTd.style.color = 'var(--text-muted)';
    emptyTr.appendChild(emptyTd);
    tbody.appendChild(emptyTr);
  } else {
    priorityItems.slice(0, 5).forEach(item => {
      const tr = createElement('tr');
      
      const tdSource = createElement('td');
      const badgeSource = createElement('span', 'badge badge-low', item.source);
      tdSource.appendChild(badgeSource);

      const tdName = createElement('td', null, item.name);
      
      const tdSev = createElement('td');
      const sevClass = item.severity === 'CRITICAL' ? 'badge-critical' : (item.severity === 'HIGH' ? 'badge-high' : 'badge-medium');
      const badgeSev = createElement('span', `badge ${sevClass}`, item.severity);
      tdSev.appendChild(badgeSev);

      const tdInfo = createElement('td', null, item.info);

      tr.appendChild(tdSource);
      tr.appendChild(tdName);
      tr.appendChild(tdSev);
      tr.appendChild(tdInfo);
      tbody.appendChild(tr);
    });
  }

  table.appendChild(tbody);
  tableWrapper.appendChild(table);
  flagsCard.appendChild(tableWrapper);

  gridEqual.appendChild(aiCard);
  gridEqual.appendChild(flagsCard);
  container.appendChild(gridEqual);

  // Mandatory Footer
  const footer = createElement('footer', 'page-footer', 'Potential issues for human review. Not an audit opinion.');
  container.appendChild(footer);

  // Initialize Charts
  renderDashboardCharts(review);
}

function createKpiHeader(titleText, iconClass) {
  const header = createElement('div', 'kpi-header');
  const span = createElement('span', null, titleText);
  const iconBox = createElement('div', 'kpi-icon');
  iconBox.appendChild(createElement('i', iconClass));
  header.appendChild(span);
  header.appendChild(iconBox);
  return header;
}

function createKpiFooter(iconClass, text, isWarn = false) {
  const footer = createElement('div', 'kpi-footer');
  const span = createElement('span', isWarn ? 'trend-down' : 'trend-up');
  const icon = createElement('i', iconClass);
  span.appendChild(icon);
  span.appendChild(document.createTextNode(` ${text}`));
  footer.appendChild(span);
  return footer;
}

function renderDashboardCharts(review) {
  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  const textColor = isLight ? '#475569' : '#9ca3af';
  const gridColor = isLight ? '#e2e8f0' : '#374151';
  const borderColor = isLight ? '#ffffff' : '#1f2937';

  const validationResults = review.validation_results || [];
  const anomalies = review.anomalies || [];
  const failedValidation = review.failed_validations
    || validationResults.filter(r => r.status === 'FAIL');

  // Chart 1: Failed Checks by Rule ID
  const failedRules = failedValidation;
  const ruleCounts = {};
  failedRules.forEach(r => {
    const id = r.rule_id || 'Other';
    ruleCounts[id] = (ruleCounts[id] || 0) + 1;
  });

  const ruleLabels = Object.keys(ruleCounts).length > 0 ? Object.keys(ruleCounts) : ['No Failures'];
  const ruleData = Object.keys(ruleCounts).length > 0 ? Object.values(ruleCounts) : [0];

  const canvas1 = document.getElementById('failedRulesChart');
  if (canvas1) {
    if (failedRulesChartInstance) failedRulesChartInstance.destroy();
    failedRulesChartInstance = new Chart(canvas1, {
      type: 'bar',
      data: {
        labels: ruleLabels,
        datasets: [{
          label: 'Failed Checks Count',
          data: ruleData,
          backgroundColor: '#ef4444',
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
          y: { grid: { color: gridColor }, ticks: { color: textColor, stepSize: 1 } }
        }
      }
    });
  }

  // Chart 2: Findings by Severity
  const sevCounts = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 };
  failedValidation.forEach(r => {
    const s = (r.severity || 'HIGH').toUpperCase();
    if (sevCounts[s] !== undefined) sevCounts[s]++;
    else sevCounts.HIGH++;
  });
  anomalies.forEach(a => {
    const s = (a.severity || 'MEDIUM').toUpperCase();
    if (sevCounts[s] !== undefined) sevCounts[s]++;
    else sevCounts.MEDIUM++;
  });

  const canvas2 = document.getElementById('anomaliesSeverityChart');
  if (canvas2) {
    if (anomaliesSeverityChartInstance) anomaliesSeverityChartInstance.destroy();
    anomaliesSeverityChartInstance = new Chart(canvas2, {
      type: 'doughnut',
      data: {
        labels: ['Critical', 'High', 'Medium', 'Low'],
        datasets: [{
          data: [sevCounts.CRITICAL, sevCounts.HIGH, sevCounts.MEDIUM, sevCounts.LOW],
          backgroundColor: ['#ef4444', '#f59e0b', '#3b82f6', '#10b981'],
          borderWidth: 2,
          borderColor: borderColor
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'right',
            labels: { color: textColor, font: { size: 11, family: 'Inter' } }
          }
        }
      }
    });
  }
}
