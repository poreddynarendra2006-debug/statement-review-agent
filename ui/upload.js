/**
 * FinSight AI - Document Upload & SheetJS Preview Controller
 */

let currentUploadedFile = null;
let currentWorkbook = null;

document.addEventListener('DOMContentLoaded', () => {
  const user = typeof getCurrentUser === "function" ? getCurrentUser() : null;
  if (typeof requireAuth === "function") {
    requireAuth();
  }
  if (typeof updateSidebarUserInfo === "function") {
    updateSidebarUserInfo();
  }

  setupMaterialitySlider();
  setupDropzone();
  setupSheetSelect();
  setupRefreshBtn();

  loadRecentReviews();
});

function setupMaterialitySlider() {
  const slider = document.getElementById('materialitySlider');
  const display = document.getElementById('materialityDisplay');
  if (!slider || !display) return;

  display.textContent = `${(parseFloat(slider.value) * 100).toFixed(1)}%`;
  slider.addEventListener('input', () => {
    const pct = (parseFloat(slider.value) * 100).toFixed(1);
    display.textContent = `${pct}%`;
  });
}

function setupDropzone() {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const browseBtn = document.getElementById('browseBtn');

  if (!dropzone || !fileInput) return;

  if (browseBtn) {
    browseBtn.addEventListener('click', () => fileInput.click());
  }

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      handleSelectedFile(fileInput.files[0]);
    }
  });
}

function setupSheetSelect() {
  const sheetSelect = document.getElementById('sheetSelect');
  if (sheetSelect) {
    sheetSelect.addEventListener('change', (e) => {
      if (currentWorkbook) {
        renderExcelSheet(currentWorkbook, e.target.value);
      }
    });
  }
}

function setupRefreshBtn() {
  const refreshBtn = document.getElementById('refreshReviewsBtn');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', () => loadRecentReviews());
  }
}

async function handleSelectedFile(file) {
  if (!file) return;

  const validExts = ['.csv', '.xlsx'];
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!validExts.includes(ext)) {
    showUploadError('Please select a valid CSV (.csv) or Excel (.xlsx) file.');
    return;
  }

  currentUploadedFile = file;
  hideUploadError();
  parseFilePreview(file);

  const processingSection = document.getElementById('processingSection');
  const processingStatus = document.getElementById('processingStatus');

  if (processingSection) processingSection.style.display = 'block';
  if (processingStatus) {
    processingStatus.textContent = '';
    const spinner = createElement('i', 'fa-solid fa-spinner fa-spin');
    const text = document.createTextNode(` Uploading & processing ${file.name}...`);
    processingStatus.appendChild(spinner);
    processingStatus.appendChild(text);
  }

  const slider = document.getElementById('materialitySlider');
  const materiality = slider ? parseFloat(slider.value) : 0.10;

  try {
    const res = await uploadStatement(file, materiality);

    if (processingSection) processingSection.style.display = 'none';

    setCurrentReviewId(res.review_id);

    showUploadSuccess(res);
    loadRecentReviews();
  } catch (err) {
    if (processingSection) processingSection.style.display = 'none';
    showUploadError(err.message || 'Failed to upload financial statement.');
  }
}

function showUploadError(msg) {
  const errArea = document.getElementById('uploadErrorArea');
  if (!errArea) return;
  errArea.textContent = '';
  const icon = createElement('i', 'fa-solid fa-circle-exclamation');
  icon.style.marginRight = '8px';
  errArea.appendChild(icon);
  errArea.appendChild(document.createTextNode(msg));
  errArea.style.display = 'inline-block';
}

function hideUploadError() {
  const errArea = document.getElementById('uploadErrorArea');
  if (errArea) errArea.style.display = 'none';
}

function showUploadSuccess(res) {
  const processingSection = document.getElementById('processingSection');
  const processingStatus = document.getElementById('processingStatus');

  if (!processingSection || !processingStatus) return;

  processingSection.style.display = 'block';
  processingStatus.textContent = '';

  const checkIcon = createElement('i', 'fa-solid fa-circle-check');
  checkIcon.style.color = 'var(--status-success)';
  checkIcon.style.marginRight = '8px';

  const strong = createElement('strong', null, res.filename || 'Statement');
  const msg = document.createTextNode(' processed successfully! Review ID: ');
  const code = createElement('code', null, res.review_id);

  const openBtn = createElement('a', 'btn btn-primary btn-sm', 'Open Dashboard');
  openBtn.href = 'dashboard.html';
  openBtn.style.marginLeft = '16px';
  openBtn.style.display = 'inline-flex';
  openBtn.style.alignItems = 'center';
  openBtn.style.gap = '6px';

  const btnIcon = createElement('i', 'fa-solid fa-gauge-high');
  openBtn.prepend(btnIcon);

  processingStatus.appendChild(checkIcon);
  processingStatus.appendChild(strong);
  processingStatus.appendChild(msg);
  processingStatus.appendChild(code);
  processingStatus.appendChild(openBtn);
}

function parseFilePreview(file) {
  const previewSection = document.getElementById('filePreviewSection');
  if (previewSection) previewSection.style.display = 'block';

  const nameEl = document.getElementById('previewFileName');
  const sizeEl = document.getElementById('previewFileSize');

  if (nameEl) nameEl.textContent = file.name;
  if (sizeEl) sizeEl.textContent = `${(file.size / 1024).toFixed(1)} KB`;

  const sheetWrapper = document.getElementById('sheetSelectorWrapper');
  if (sheetWrapper) sheetWrapper.style.display = 'none';

  const ext = file.name.split('.').pop().toLowerCase();
  if (ext === 'csv') {
    readCSVFile(file);
  } else if (ext === 'xlsx' || ext === 'xls') {
    readExcelFile(file);
  }
}

function readCSVFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const text = e.target.result;
    const lines = text.split(/\r\n|\n/).filter(l => l.trim().length > 0);
    if (lines.length === 0) return;

    const rows = lines.map(line => line.split(',').map(cell => cell.replace(/^"(.*)"$/, '$1').trim()));
    const headers = rows[0] || [];
    const bodyRows = rows.slice(1);

    const rowCountEl = document.getElementById('previewRowCount');
    const colCountEl = document.getElementById('previewColCount');
    if (rowCountEl) rowCountEl.textContent = formatNumber(bodyRows.length);
    if (colCountEl) colCountEl.textContent = formatNumber(headers.length);

    renderPreviewTable(headers, bodyRows);
  };
  reader.readAsText(file);
}

function readExcelFile(file) {
  if (!window.XLSX) {
    console.warn("SheetJS not loaded");
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    const data = new Uint8Array(e.target.result);
    const workbook = XLSX.read(data, { type: 'array' });
    currentWorkbook = workbook;

    const sheetNames = workbook.SheetNames;
    const sheetSelect = document.getElementById('sheetSelect');
    const wrapper = document.getElementById('sheetSelectorWrapper');

    if (sheetNames.length > 0 && wrapper && sheetSelect) {
      wrapper.style.display = 'flex';
      sheetSelect.textContent = '';
      sheetNames.forEach(name => {
        const opt = createElement('option', null, name);
        opt.value = name;
        sheetSelect.appendChild(opt);
      });
    }

    if (sheetNames.length > 0) {
      renderExcelSheet(workbook, sheetNames[0]);
    }
  };
  reader.readAsArrayBuffer(file);
}

function renderExcelSheet(workbook, sheetName) {
  const sheet = workbook.Sheets[sheetName];
  if (!sheet) return;

  const json = XLSX.utils.sheet_to_json(sheet, { header: 1 });
  if (json.length === 0) return;

  const headers = json[0] || [];
  const bodyRows = json.slice(1).filter(r => r.length > 0);

  const rowCountEl = document.getElementById('previewRowCount');
  const colCountEl = document.getElementById('previewColCount');
  if (rowCountEl) rowCountEl.textContent = formatNumber(bodyRows.length);
  if (colCountEl) colCountEl.textContent = formatNumber(headers.length);

  renderPreviewTable(headers, bodyRows);
}

function renderPreviewTable(headers, bodyRows) {
  const headEl = document.getElementById('previewTableHead');
  const bodyEl = document.getElementById('previewTableBody');

  if (!headEl || !bodyEl) return;

  headEl.textContent = '';
  bodyEl.textContent = '';

  const trHead = createElement('tr');
  headers.forEach(h => {
    const th = createElement('th', null, h || 'Column');
    trHead.appendChild(th);
  });
  headEl.appendChild(trHead);

  bodyRows.slice(0, 50).forEach(row => {
    const tr = createElement('tr');
    headers.forEach((_, idx) => {
      const val = row[idx] !== undefined && row[idx] !== null ? String(row[idx]) : '';
      const td = createElement('td', null, val);
      tr.appendChild(td);
    });
    bodyEl.appendChild(tr);
  });
}

async function loadRecentReviews() {
  const tbody = document.getElementById('documentsTableBody');
  if (!tbody) return;

  tbody.textContent = '';
  const loadingTr = createElement('tr');
  const loadingTd = createElement('td');
  loadingTd.colSpan = 5;
  loadingTd.style.textAlign = 'center';
  loadingTd.style.padding = '24px';
  loadingTd.style.color = 'var(--text-muted)';
  
  const spinner = createElement('i', 'fa-solid fa-circle-notch fa-spin');
  loadingTd.appendChild(spinner);
  loadingTd.appendChild(document.createTextNode(' Loading recent reviews...'));
  loadingTr.appendChild(loadingTd);
  tbody.appendChild(loadingTr);

  try {
    const reviews = await listReviews(20);
    tbody.textContent = '';

    if (!reviews || reviews.length === 0) {
      const emptyTr = createElement('tr');
      const emptyTd = createElement('td', null, 'No recent reviews found.');
      emptyTd.colSpan = 5;
      emptyTd.style.textAlign = 'center';
      emptyTd.style.padding = '24px';
      emptyTd.style.color = 'var(--text-muted)';
      emptyTr.appendChild(emptyTd);
      tbody.appendChild(emptyTr);
      return;
    }

    reviews.forEach(r => {
      const tr = createElement('tr');

      // Review ID
      const tdId = createElement('td');
      const idCode = createElement('code', null, r.review_id);
      tdId.appendChild(idCode);

      // File Name
      const tdFile = createElement('td');
      const ext = (r.filename || '').split('.').pop().toLowerCase();
      const iconClass = ext === 'xlsx' || ext === 'xls' ? 'fa-file-excel' : 'fa-file-csv';
      const iconColor = ext === 'xlsx' || ext === 'xls' ? 'var(--status-success)' : 'var(--accent-cyan)';
      const fileIcon = createElement('i', `fa-solid ${iconClass}`);
      fileIcon.style.color = iconColor;
      fileIcon.style.marginRight = '8px';
      tdFile.appendChild(fileIcon);
      tdFile.appendChild(document.createTextNode(r.filename || 'Financial_Statement'));

      // Materiality
      const tdMat = createElement('td', null, formatPercent(r.materiality * 100));

      // Created Date
      const createdDate = r.created_at ? new Date(r.created_at).toLocaleString() : '-';
      const tdDate = createElement('td', null, createdDate);

      // Actions
      const tdActions = createElement('td');
      tdActions.style.display = 'flex';
      tdActions.style.gap = '8px';

      const openBtn = createElement('button', 'btn btn-primary btn-sm');
      const openIcon = createElement('i', 'fa-solid fa-folder-open');
      openIcon.style.marginRight = '4px';
      openBtn.appendChild(openIcon);
      openBtn.appendChild(document.createTextNode(' Open'));
      openBtn.addEventListener('click', () => {
        setCurrentReviewId(r.review_id);
        window.location.href = 'dashboard.html';
      });

      const pdfBtn = createElement('button', 'btn btn-secondary btn-sm');
      const pdfIcon = createElement('i', 'fa-solid fa-file-pdf');
      pdfIcon.style.color = 'var(--status-danger)';
      pdfIcon.style.marginRight = '4px';
      pdfBtn.appendChild(pdfIcon);
      pdfBtn.appendChild(document.createTextNode(' PDF'));
      pdfBtn.addEventListener('click', () => {
        downloadPDFReport(r.review_id);
      });

      tdActions.appendChild(openBtn);
      tdActions.appendChild(pdfBtn);

      tr.appendChild(tdId);
      tr.appendChild(tdFile);
      tr.appendChild(tdMat);
      tr.appendChild(tdDate);
      tr.appendChild(tdActions);

      tbody.appendChild(tr);
    });
  } catch (err) {
    tbody.textContent = '';
    const errTr = createElement('tr');
    const errTd = createElement('td', null, err.message || 'Failed to load reviews.');
    errTd.colSpan = 5;
    errTd.style.textAlign = 'center';
    errTd.style.padding = '24px';
    errTd.style.color = 'var(--status-danger)';
    errTr.appendChild(errTd);
    tbody.appendChild(errTr);
  }
}
