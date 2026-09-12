/**
 * AuditLens - Financial & Ratio Trends Controller
 */

let currentReviewData = null;
let ratioChartInstance = null;

document.addEventListener('DOMContentLoaded', async () => {
  if (typeof requireAuth === "function") {
    const authed = await requireAuth();
    if (!authed) return;
  }
  if (typeof updateSidebarUserInfo === "function") {
    updateSidebarUserInfo();
  }

  const container = document.getElementById('trendsContainer');
  if (!container) return;

  showLoadingState(container, "Loading financial ratios & trend analysis...");

  const review = await loadCurrentReview();
  if (!review) {
    showEmptyReviewState(container);
    return;
  }
  currentReviewData = review;
  // The page's own filters, table and canvases live inside the container,
  // so put them back before anything looks them up.
  restorePageMarkup(container);

  const subhead = document.getElementById('trendsSubhead');
  if (subhead && review.filename) {
    subhead.textContent = `Ratio trajectory & YoY variance analysis for: ${review.filename} (ID: ${review.review_id})`;
  }

  setupCompanyDropdown(review);
  setupEventListeners();

  renderRatioChart();
  renderDeviationsTable(review);

  window.addEventListener('themeChanged', () => {
    if (currentReviewData) {
      renderRatioChart();
    }
  });
});

function setupCompanyDropdown(review) {
  const selectCompany = document.getElementById('selectCompany');
  if (!selectCompany) return;

  selectCompany.textContent = '';
  const companies = review.companies && review.companies.length > 0 
    ? review.companies 
    : Array.from(new Set((review.ratio_results || []).map(r => r.company).filter(Boolean)));

  if (companies.length === 0) companies.push('Default');

  companies.forEach((comp, idx) => {
    const opt = createElement('option', null, comp);
    opt.value = comp;
    if (idx === 0) opt.selected = true;
    selectCompany.appendChild(opt);
  });
}

function setupEventListeners() {
  const selectCompany = document.getElementById('selectCompany');
  const selectRatio = document.getElementById('selectRatio');

  if (selectCompany) {
    selectCompany.addEventListener('change', () => renderRatioChart());
  }
  if (selectRatio) {
    selectRatio.addEventListener('change', () => renderRatioChart());
  }
}

function renderRatioChart() {
  const canvas = document.getElementById('ratioTrendChart');
  if (!canvas || !currentReviewData) return;

  const selectCompany = document.getElementById('selectCompany');
  const selectRatio = document.getElementById('selectRatio');

  const selectedCompany = selectCompany ? selectCompany.value : '';
  const selectedRatioKey = selectRatio ? selectRatio.value : 'current_ratio';

  const chartTitle = document.getElementById('chartTitle');
  if (chartTitle && selectRatio) {
    const ratioText = selectRatio.options[selectRatio.selectedIndex]?.text || selectedRatioKey;
    chartTitle.textContent = `${ratioText} (${selectedCompany})`;
  }

  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  const textColor = isLight ? '#475569' : '#9ca3af';
  const gridColor = isLight ? '#e2e8f0' : '#374151';

  // Search ratios array
  const ratiosList = currentReviewData.ratios || [];
  const foundRatio = ratiosList.find(r => 
    (r.company === selectedCompany || !selectedCompany) && 
    (r.ratio_name === selectedRatioKey || r.ratio_name.toLowerCase() === selectedRatioKey.toLowerCase())
  ) || ratiosList.find(r => r.ratio_name === selectedRatioKey);

  let labels = [];
  let dataPoints = [];

  if (foundRatio && foundRatio.values && typeof foundRatio.values === 'object') {
    const sortedYears = Object.keys(foundRatio.values).sort();
    labels = sortedYears;
    dataPoints = sortedYears.map(y => foundRatio.values[y]);
  } else {
    labels = ['No Data Available'];
    dataPoints = [0];
  }

  if (ratioChartInstance) ratioChartInstance.destroy();

  ratioChartInstance = new Chart(canvas, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: selectRatio ? selectRatio.options[selectRatio.selectedIndex]?.text : selectedRatioKey,
        data: dataPoints,
        borderColor: '#06b6d4',
        backgroundColor: 'rgba(6, 182, 212, 0.15)',
        fill: true,
        tension: 0.3,
        pointRadius: 6,
        pointHoverRadius: 8
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
        y: { grid: { color: gridColor }, ticks: { color: textColor } }
      }
    }
  });
}

function renderDeviationsTable(review) {
  const tbody = document.getElementById('deviationsTableBody');
  if (!tbody) return;

  tbody.textContent = '';

  const deviations = review.material_deviations || [];
  if (deviations.length === 0) {
    const tr = createElement('tr');
    const td = createElement('td', null, 'No material trend deviations recorded.');
    td.colSpan = 7;
    td.style.textAlign = 'center';
    td.style.padding = '24px';
    td.style.color = 'var(--text-muted)';
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  deviations.forEach(dev => {
    const tr = createElement('tr');

    // Company
    const tdComp = createElement('td', null, dev.company || '-');

    // Metric / Ratio
    const tdMetric = createElement('td', null, dev.metric || '-');
    tdMetric.style.fontWeight = '600';

    // Prior Period
    const tdPrior = createElement('td', null, formatNumber(dev.prior_value));

    // Current Period
    const tdCurr = createElement('td', null, formatNumber(dev.current_value));

    // Deviation %
    const tdDev = createElement('td', null, formatPercent(dev.deviation_percent));
    tdDev.style.fontWeight = '700';
    tdDev.style.color = dev.deviation_percent < 0 ? 'var(--status-danger)' : 'var(--status-success)';

    // Materiality Cutoff
    const tdMat = createElement('td', null, formatPercent(dev.materiality_threshold));

    // Severity
    const tdSev = createElement('td');
    const sevClass = (dev.severity || 'HIGH').toUpperCase() === 'CRITICAL' ? 'badge-critical' : ((dev.severity || 'HIGH').toUpperCase() === 'HIGH' ? 'badge-high' : 'badge-medium');
    const badge = createElement('span', `badge ${sevClass}`, (dev.severity || 'HIGH').toUpperCase());
    tdSev.appendChild(badge);

    tr.appendChild(tdComp);
    tr.appendChild(tdMetric);
    tr.appendChild(tdPrior);
    tr.appendChild(tdCurr);
    tr.appendChild(tdDev);
    tr.appendChild(tdMat);
    tr.appendChild(tdSev);

    tbody.appendChild(tr);
  });
}
