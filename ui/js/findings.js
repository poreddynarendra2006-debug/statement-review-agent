/**
 * AuditLens - Audit Findings Controller
 */

let currentReviewData = null;
let currentActionsMap = {};
let allFindings = [];

// What each button on the finding means to the API. The API accepts these
// three values and nothing else, and stores the choice as `status`.
const ACTION_STATUS = {
  acknowledged: 'VERIFIED',
  flagged_for_followup: 'NEEDS_INVESTIGATION',
  dismissed: 'DISMISSED'
};

// How a recorded decision reads on screen.
const ACTION_LABEL = {
  VERIFIED: 'VERIFIED',
  NEEDS_INVESTIGATION: 'FLAGGED FOR FOLLOW-UP',
  DISMISSED: 'DISMISSED'
};

function actionLabel(action) {
  if (!action || !action.status) return 'REVIEWED';
  return ACTION_LABEL[action.status] || String(action.status).replace(/_/g, ' ');
}

document.addEventListener('DOMContentLoaded', async () => {
  if (typeof requireAuth === "function") {
    const authed = await requireAuth();
    if (!authed) return;
  }
  if (typeof updateSidebarUserInfo === "function") {
    updateSidebarUserInfo();
  }

  const container = document.getElementById('findingsContainer');
  if (!container) return;

  showLoadingState(container, "Loading audit findings & verification actions...");

  const review = await loadCurrentReview();
  if (!review) {
    showEmptyReviewState(container);
    return;
  }
  currentReviewData = review;
  // The page's own filters, table and canvases live inside the container,
  // so put them back before anything looks them up.
  restorePageMarkup(container);

  // Subhead update
  const subhead = document.getElementById('findingsSubhead');
  if (subhead && review.filename) {
    subhead.textContent = `Potential inconsistencies requiring review for statement: ${review.filename} (ID: ${review.review_id})`;
  }

  // Load actions history
  try {
    const actionsList = await listActions(review.review_id);
    if (Array.isArray(actionsList)) {
      actionsList.forEach(act => {
        if (act.finding_ref) {
          currentActionsMap[act.finding_ref] = act;
        }
      });
    }
  } catch (e) {
    console.warn("Could not fetch actions list:", e);
  }

  // Normalize findings
  allFindings = extractNormalizedFindings(review);

  // Populate company dropdown
  populateCompanyFilter(allFindings);

  // Setup listeners
  setupFiltersAndModal();

  // Initial render
  filterAndRenderFindings();
});

function extractNormalizedFindings(review) {
  const list = [];

  // 1. Validation Failures
  const validationResults = review.validation_results || [];
  validationResults.forEach((val, idx) => {
    if (val.status === 'FAIL') {
      const ruleId = val.rule_id || `VAL_${idx + 1}`;
      const company = val.company || (review.companies ? review.companies[0] : 'Company');
      const year = val.year || '-';
      const ref = `${ruleId}:${company}:${year}`;

      list.push({
        finding_ref: ref,
        source: 'Validation',
        topic: ruleId,
        description: val.rule_name || val.description || 'Validation rule check failed',
        severity: (val.severity || 'HIGH').toUpperCase(),
        company: company,
        year: year,
        details: val.details || {},
        original: val
      });
    }
  });

  // 2. Anomalies
  const anomalies = review.anomalies || [];
  anomalies.forEach((ano, idx) => {
    const company = ano.company || (review.companies ? review.companies[0] : 'Company');
    const year = ano.year || '-';
    const ref = `anomaly:${company}:${year}`;

    list.push({
      finding_ref: ref,
      source: 'Anomaly',
      topic: ano.anomaly_type || `Anomaly_${idx + 1}`,
      description: `${ano.anomaly_type || 'Anomaly'}: ${ano.description || ''}`,
      severity: (ano.severity || 'MEDIUM').toUpperCase(),
      confidence: ano.confidence,
      company: company,
      year: year,
      details: ano.details || {},
      original: ano
    });
  });

  // 3. Trend Deviations
  const deviations = review.material_deviations || [];
  deviations.forEach((dev, idx) => {
    const company = dev.company || (review.companies ? review.companies[0] : 'Company');
    const year = dev.year || '-';
    const metric = dev.metric || `Metric_${idx + 1}`;
    const ref = `deviation:${company}:${year}:${metric}`;

    const desc = `Variance of ${formatPercent(dev.deviation_percent)} detected vs prior value (${formatNumber(dev.prior_value)} -> ${formatNumber(dev.current_value)})`;

    list.push({
      finding_ref: ref,
      source: 'Deviation',
      topic: metric,
      description: desc,
      severity: (dev.severity || 'HIGH').toUpperCase(),
      company: company,
      year: year,
      details: dev.details || {},
      original: dev
    });
  });

  // 4. Recurring issues - the same problem in three or more years, which
  // matters more to a reviewer than any single year's finding.
  const recurring = review.recurring_issues || [];
  recurring.forEach((rec, idx) => {
    const company = rec.company || 'Company';
    const years = Array.isArray(rec.years) ? rec.years : [];
    const ref = `recurring:${company}:${rec.key || idx}`;

    list.push({
      finding_ref: ref,
      source: 'Recurring',
      topic: rec.key || `Recurring_${idx + 1}`,
      description: rec.issue || `Repeated in ${years.length} years`,
      severity: (rec.severity || 'MEDIUM').toUpperCase(),
      company: company,
      year: years.length ? `${years[0]}-${years[years.length - 1]}` : '-',
      details: { Years: years.join(', '), Occurrences: rec.occurrences, Evidence: rec.evidence },
      original: rec
    });
  });

  // 5. Peer comparison - how this company sits against others in its industry.
  const peers = review.peer_findings || [];
  peers.forEach((peer, idx) => {
    const company = peer.company || 'Company';
    const year = peer.year || '-';

    list.push({
      finding_ref: `peer:${company}:${year}:${peer.metric || idx}`,
      source: 'Peer',
      topic: peer.metric || `Peer_${idx + 1}`,
      description: peer.issue || 'Stands apart from peers',
      severity: (peer.severity || 'MEDIUM').toUpperCase(),
      company: company,
      year: year,
      details: {
        'Peer group': peer.peer_group,
        'Peers compared': peer.peer_count,
        'Peer median': peer.peer_median,
        'Middle half': `${peer.peer_low} to ${peer.peer_high}`
      },
      original: peer
    });
  });

  return list;
}

function populateCompanyFilter(findings) {
  const companySelect = document.getElementById('filterCompany');
  if (!companySelect) return;

  const companies = new Set();
  findings.forEach(f => {
    if (f.company && f.company !== '-') companies.add(f.company);
  });

  companySelect.textContent = '';
  const optAll = createElement('option', null, 'All Companies');
  optAll.value = 'ALL';
  companySelect.appendChild(optAll);

  companies.forEach(c => {
    const opt = createElement('option', null, c);
    opt.value = c;
    companySelect.appendChild(opt);
  });
}

function setupFiltersAndModal() {
  ['filterKeyword', 'filterSource', 'filterSeverity', 'filterCompany'].forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener(el.tagName === 'INPUT' ? 'input' : 'change', () => filterAndRenderFindings());
    }
  });

  const exportBtn = document.getElementById('exportCsvBtn');
  if (exportBtn) {
    exportBtn.addEventListener('click', () => exportFindingsCSV());
  }

  const modalClose = document.getElementById('modalCloseBtn');
  if (modalClose) {
    modalClose.addEventListener('click', () => closeModal());
  }
}

function filterAndRenderFindings() {
  const keyword = (document.getElementById('filterKeyword')?.value || '').toLowerCase();
  const source = document.getElementById('filterSource')?.value || 'ALL';
  const severity = document.getElementById('filterSeverity')?.value || 'ALL';
  const company = document.getElementById('filterCompany')?.value || 'ALL';

  const filtered = allFindings.filter(f => {
    const matchKw = !keyword || 
      f.finding_ref.toLowerCase().includes(keyword) || 
      f.topic.toLowerCase().includes(keyword) || 
      f.description.toLowerCase().includes(keyword);

    const matchSource = source === 'ALL' || f.source === source;
    const matchSev = severity === 'ALL' || f.severity === severity;
    const matchCompany = company === 'ALL' || f.company === company;

    return matchKw && matchSource && matchSev && matchCompany;
  });

  renderFindingsTable(filtered);
}

function renderFindingsTable(findings) {
  const tbody = document.getElementById('findingsTableBody');
  const countEl = document.getElementById('findingsCount');

  if (countEl) countEl.textContent = formatNumber(findings.length);
  if (!tbody) return;

  tbody.textContent = '';

  if (findings.length === 0) {
    const tr = createElement('tr');
    const td = createElement('td', null, 'No potential issues found matching criteria.');
    td.colSpan = 7;
    td.style.textAlign = 'center';
    td.style.padding = '24px';
    td.style.color = 'var(--text-muted)';
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  findings.forEach(f => {
    const tr = createElement('tr');

    // Finding Ref
    const tdRef = createElement('td');
    const codeRef = createElement('code', null, f.finding_ref);
    tdRef.appendChild(codeRef);

    // Source
    const tdSource = createElement('td');
    const badgeSource = createElement('span', 'badge badge-low', f.source);
    tdSource.appendChild(badgeSource);

    // Topic / Rule
    const tdTopic = createElement('td', null, f.topic);
    tdTopic.style.fontWeight = '600';

    // Description
    const tdDesc = createElement('td', null, f.description);
    tdDesc.style.fontSize = '0.85rem';

    // Severity
    const tdSev = createElement('td');
    const sevClass = f.severity === 'CRITICAL' ? 'badge-critical' : (f.severity === 'HIGH' ? 'badge-high' : 'badge-medium');
    const badgeSev = createElement('span', `badge ${sevClass}`, f.severity);
    tdSev.appendChild(badgeSev);

    // Status (Action recorded)
    const tdStatus = createElement('td');
    const action = currentActionsMap[f.finding_ref];
    if (action) {
      const badgeAction = createElement('span', 'badge badge-low', actionLabel(action));
      tdStatus.appendChild(badgeAction);
    } else {
      const badgeAction = createElement('span', 'badge badge-high', 'UNREVIEWED');
      badgeAction.style.background = 'rgba(100, 116, 139, 0.2)';
      badgeAction.style.color = 'var(--text-muted)';
      tdStatus.appendChild(badgeAction);
    }

    // Action button
    const tdAction = createElement('td');
    const btnInspect = createElement('button', 'btn btn-secondary btn-sm');
    const icon = createElement('i', 'fa-solid fa-expand');
    icon.style.marginRight = '4px';
    btnInspect.appendChild(icon);
    btnInspect.appendChild(document.createTextNode(' Inspect'));
    btnInspect.addEventListener('click', () => openModal(f));
    tdAction.appendChild(btnInspect);

    tr.appendChild(tdRef);
    tr.appendChild(tdSource);
    tr.appendChild(tdTopic);
    tr.appendChild(tdDesc);
    tr.appendChild(tdSev);
    tr.appendChild(tdStatus);
    tr.appendChild(tdAction);

    tbody.appendChild(tr);
  });
}

function openModal(finding) {
  const modal = document.getElementById('findingModal');
  if (!modal) return;

  document.getElementById('modalRefId').textContent = finding.finding_ref;
  document.getElementById('modalTitle').textContent = `${finding.source}: ${finding.topic}`;
  document.getElementById('modalReasoning').textContent = finding.description;

  const sevBadge = document.getElementById('modalSeverityBadge');
  if (sevBadge) {
    sevBadge.textContent = finding.severity;
    sevBadge.className = `badge badge-${finding.severity.toLowerCase()}`;
  }

  const srcBadge = document.getElementById('modalSourceBadge');
  if (srcBadge) srcBadge.textContent = finding.source;

  // Grid details
  const grid = document.getElementById('modalDetailsGrid');
  if (grid) {
    grid.textContent = '';

    addDetailItem(grid, 'Company:', finding.company);
    addDetailItem(grid, 'Year / Period:', finding.year);
    if (finding.confidence !== undefined) {
      addDetailItem(grid, 'Model Confidence:', formatConfidence(finding.confidence));
    }

    if (finding.details && typeof finding.details === 'object') {
      Object.keys(finding.details).forEach(k => {
        const val = finding.details[k];
        const displayVal = typeof val === 'object' ? JSON.stringify(val) : String(val);
        addDetailItem(grid, `${k.replace(/_/g, ' ')}:`, displayVal);
      });
    }
  }

  // Check action status
  updateModalActionButtons(finding);

  modal.classList.add('show');
}

function addDetailItem(container, labelText, valueText) {
  const div = createElement('div');
  const label = createElement('span', null, labelText);
  label.style.color = 'var(--text-muted)';
  label.style.display = 'block';
  label.style.fontSize = '0.8rem';
  
  const val = createElement('strong', null, valueText || '-');
  val.style.color = 'var(--text-main)';

  div.appendChild(label);
  div.appendChild(val);
  container.appendChild(div);
}

function updateModalActionButtons(finding) {
  const statusArea = document.getElementById('modalActionStatusArea');
  const action = currentActionsMap[finding.finding_ref];

  if (statusArea) {
    if (action) {
      statusArea.textContent = `Action recorded by ${action.reviewer || 'Reviewer'}: ${actionLabel(action)}`;
      statusArea.style.display = 'block';
    } else {
      statusArea.style.display = 'none';
    }
  }

  const ackBtn = document.getElementById('modalAcknowledgeBtn');
  const flagBtn = document.getElementById('modalFlagBtn');
  const disBtn = document.getElementById('modalDismissBtn');

  const handleAction = async (actionType) => {
    const user = typeof getCurrentUser === "function" ? getCurrentUser() : null;
    const reviewerName = user ? user.name : "Reviewer";
    if (!currentReviewData) return;

    const status = ACTION_STATUS[actionType];
    if (!status) return;

    try {
      // The API records a decision as `status` with `note`; these three are the
      // only values it accepts. Sending the button's own name was rejected as a
      // missing field, so nothing was ever recorded.
      const res = await recordAction(currentReviewData.review_id, {
        finding_ref: finding.finding_ref,
        status: status,
        reviewer: reviewerName,
        note: `Recorded on ${new Date().toISOString()}`
      });

      // The reply carries only the new action's id, so the row we keep for the
      // table is built here rather than from the reply.
      currentActionsMap[finding.finding_ref] = {
        action_id: res && res.action_id,
        status: status,
        reviewer: reviewerName
      };
      updateModalActionButtons(finding);
      filterAndRenderFindings();
    } catch (err) {
      alert(`Failed to record action: ${err.message}`);
    }
  };

  if (ackBtn) {
    ackBtn.onclick = () => handleAction('acknowledged');
  }
  if (flagBtn) {
    flagBtn.onclick = () => handleAction('flagged_for_followup');
  }
  if (disBtn) {
    disBtn.onclick = () => handleAction('dismissed');
  }
}

function closeModal() {
  const modal = document.getElementById('findingModal');
  if (modal) modal.classList.remove('show');
}

function exportFindingsCSV() {
  let csv = 'Finding Ref,Source,Topic,Description,Severity,Company,Year,Status\n';
  allFindings.forEach(f => {
    const act = currentActionsMap[f.finding_ref];
    const statusStr = act ? actionLabel(act) : 'unreviewed';
    const cleanDesc = f.description.replace(/"/g, '""');
    csv += `"${f.finding_ref}","${f.source}","${f.topic}","${cleanDesc}","${f.severity}","${f.company}","${f.year}","${statusStr}"\n`;
  });

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const a = createElement('a');
  a.href = url;
  a.download = `AuditLens_Findings_${currentReviewData?.review_id || 'Report'}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(url);
}
