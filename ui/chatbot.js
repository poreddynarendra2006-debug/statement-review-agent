/**
 * FinSight AI - AI Summary Controller
 */

document.addEventListener('DOMContentLoaded', async () => {
  if (typeof requireAuth === "function") {
    const authed = await requireAuth();
    if (!authed) return;
  }
  if (typeof updateSidebarUserInfo === "function") {
    updateSidebarUserInfo();
  }

  const container = document.getElementById('summaryContainer');
  if (!container) return;

  showLoadingState(container, "Loading AI executive summary & security flags...");

  const review = await loadCurrentReview();
  if (!review) {
    showEmptyReviewState(container);
    return;
  }

  const subhead = document.getElementById('summarySubhead');
  if (subhead && review.filename) {
    subhead.textContent = `Autonomous AI synthesis for statement: ${review.filename} (ID: ${review.review_id})`;
  }

  renderSummaryPage(container, review);
});

function renderSummaryPage(container, review) {
  container.textContent = '';

  const aiSummaryObj = review.ai_summary || {};
  const mode = aiSummaryObj.review_mode || (review.companies && review.companies.length > 1 ? 'Multi-Company' : 'Single-Company');
  const summaryText = typeof review.summary === 'string' ? review.summary : (aiSummaryObj.summary || 'Statement reviewed and verified by FinSight AI.');
  const securityFlags = aiSummaryObj.security_flags || review.security_flags || [];

  // --- Card 1: Executive AI Synthesis ---
  const card1 = createElement('div', 'card');
  card1.style.borderLeft = '4px solid var(--accent-cyan)';

  const header1 = createElement('div', 'card-header');
  const title1 = createElement('div', 'card-title');
  const icon1 = createElement('i', 'fa-solid fa-robot');
  icon1.style.color = 'var(--accent-cyan)';
  title1.appendChild(icon1);
  title1.appendChild(document.createTextNode(' Executive Summary & Findings'));
  
  const modeBadge = createElement('span', 'badge badge-low', mode.replace(/_/g, ' ').toUpperCase());

  header1.appendChild(title1);
  header1.appendChild(modeBadge);
  card1.appendChild(header1);

  const body1 = createElement('div');
  body1.style.display = 'flex';
  body1.style.flexDirection = 'column';
  body1.style.gap = '14px';

  const paragraphs = summaryText.split('\n\n').filter(p => p.trim());
  paragraphs.forEach(p => {
    const pEl = createElement('p', null, p);
    pEl.style.lineHeight = '1.6';
    pEl.style.fontSize = '0.95rem';
    pEl.style.color = 'var(--text-main)';
    body1.appendChild(pEl);
  });

  // Security Flags if any
  if (Array.isArray(securityFlags) && securityFlags.length > 0) {
    const secBox = createElement('div');
    secBox.style.padding = '14px 18px';
    secBox.style.borderRadius = 'var(--radius-md)';
    secBox.style.background = 'var(--status-danger-bg)';
    secBox.style.border = '1px solid rgba(239, 68, 68, 0.3)';
    secBox.style.marginTop = '8px';

    const secTitle = createElement('strong', null, 'Security & Compliance Flags:');
    secTitle.style.color = 'var(--status-danger)';
    secTitle.style.display = 'block';
    secTitle.style.marginBottom = '6px';
    secBox.appendChild(secTitle);

    securityFlags.forEach(flag => {
      const flagEl = createElement('div', null, `• ${flag}`);
      flagEl.style.fontSize = '0.88rem';
      flagEl.style.color = 'var(--status-danger)';
      secBox.appendChild(flagEl);
    });

    body1.appendChild(secBox);
  }

  card1.appendChild(body1);
  container.appendChild(card1);

  // --- Card 2: Statement Review Metadata Overview ---
  const card2 = createElement('div', 'card');
  const header2 = createElement('div', 'card-header');
  const title2 = createElement('div', 'card-title');
  title2.appendChild(createElement('i', 'fa-solid fa-file-contract'));
  title2.appendChild(document.createTextNode(' Review Execution Metadata'));
  header2.appendChild(title2);
  card2.appendChild(header2);

  const grid2 = createElement('div');
  grid2.style.display = 'grid';
  grid2.style.gridTemplateColumns = 'repeat(auto-fit, minmax(200px, 1fr))';
  grid2.style.gap = '16px';
  grid2.style.fontSize = '0.88rem';

  const failedCount = (review.validation_results || []).filter(v => !v.passed).length;
  const anomalyCount = (review.anomalies || []).length;

  addMetaPill(grid2, 'Review ID:', review.review_id || '-');
  addMetaPill(grid2, 'File Name:', review.filename || '-');
  addMetaPill(grid2, 'Materiality Threshold:', formatPercent(review.materiality * 100));
  addMetaPill(grid2, 'Companies Detected:', (review.companies || []).join(', ') || 'Default');
  addMetaPill(grid2, 'Validation Failures:', formatNumber(failedCount));
  addMetaPill(grid2, 'Anomaly Flags:', formatNumber(anomalyCount));

  card2.appendChild(grid2);
  container.appendChild(card2);

  // Mandatory Footer
  const footer = createElement('footer', 'page-footer', 'Potential issues for human review. Not an audit opinion.');
  container.appendChild(footer);
}

function addMetaPill(container, labelText, valueText) {
  const div = createElement('div');
  div.style.background = 'var(--bg-input)';
  div.style.padding = '12px 16px';
  div.style.borderRadius = 'var(--radius-md)';
  div.style.border = '1px solid var(--border-color)';

  const label = createElement('span', null, labelText);
  label.style.color = 'var(--text-muted)';
  label.style.display = 'block';
  label.style.fontSize = '0.8rem';
  label.style.marginBottom = '4px';

  const val = createElement('strong', null, valueText);
  val.style.color = 'var(--text-main)';

  div.appendChild(label);
  div.appendChild(val);
  container.appendChild(div);
}
