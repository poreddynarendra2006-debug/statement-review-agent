/**
 * AuditLens - Unified API & Endpoint Helper
 */

const API_BASE_URL = "";

async function apiFetch(path, options = {}) {
  const token = typeof getToken === "function" ? getToken() : null;
  const headers = options.headers ? { ...options.headers } : {};

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  // If body is FormData, ensure Content-Type is not manually set to application/json
  if (options.body && options.body instanceof FormData) {
    delete headers["Content-Type"];
  }

  let res;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers
    });
  } catch (err) {
    throw new Error("Can't reach the review service. Is the backend running?");
  }

  if (res.status === 401) {
    if (typeof clearSession === "function") clearSession();
    const currentPage = window.location.pathname.split("/").pop() || "dashboard.html";
    if (currentPage !== "index.html" && currentPage !== "signup.html") {
      window.location.href = `index.html?next=${encodeURIComponent(currentPage)}`;
    }
    let errDetail = "Sign in to continue.";
    try {
      const errJson = await res.json();
      if (errJson && errJson.detail) {
        errDetail = typeof errJson.detail === "string" ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch (e) {}
    throw new Error(errDetail);
  }

  if (res.status === 204) {
    return null;
  }

  if (!res.ok) {
    let errDetail = `Request failed (${res.status})`;
    try {
      const errJson = await res.json();
      if (errJson && errJson.detail) {
        errDetail = typeof errJson.detail === "string" ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch (e) {}
    throw new Error(errDetail);
  }

  return await res.json();
}

async function uploadStatement(file, materiality = 0.10) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("materiality", materiality.toString());
  return await apiFetch("/review/upload", {
    method: "POST",
    body: formData
  });
}

async function getReview(id) {
  return await apiFetch(`/reviews/${encodeURIComponent(id)}`);
}

async function listReviews(limit = 20) {
  return await apiFetch(`/reviews?limit=${limit}`);
}

async function listActions(id) {
  return await apiFetch(`/reviews/${encodeURIComponent(id)}/actions`);
}

async function recordAction(id, body) {
  return await apiFetch(`/reviews/${encodeURIComponent(id)}/actions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
}

async function downloadPDFReport(reviewId, customFilename) {
  const token = typeof getToken === "function" ? getToken() : null;
  const user = typeof getCurrentUser === "function" ? getCurrentUser() : null;
  const reviewerName = user ? user.name : "Reviewer";

  const url = `${API_BASE_URL}/reviews/${encodeURIComponent(reviewId)}/report.pdf?reviewer_name=${encodeURIComponent(reviewerName)}`;
  
  const headers = {};
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  let res;
  try {
    res = await fetch(url, { headers });
  } catch (err) {
    alert("Could not reach backend to download PDF report.");
    return;
  }

  if (res.status === 401) {
    if (typeof clearSession === "function") clearSession();
    const currentPage = window.location.pathname.split("/").pop() || "dashboard.html";
    window.location.href = `index.html?next=${encodeURIComponent(currentPage)}`;
    return;
  }

  if (!res.ok) {
    alert(`Failed to download PDF report (${res.status}).`);
    return;
  }

  const blob = await res.blob();
  const blobUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = blobUrl;
  a.download = customFilename || `AuditLens_Report_${reviewId}.pdf`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(blobUrl);
}

function setCurrentReviewId(id) {
  if (id) {
    sessionStorage.setItem("currentReviewId", id);
  } else {
    sessionStorage.removeItem("currentReviewId");
  }
}

function getCurrentReviewId() {
  return sessionStorage.getItem("currentReviewId") || null;
}

let cachedReviewData = null;
let cachedReviewId = null;

async function loadCurrentReview() {
  const id = getCurrentReviewId();
  if (!id) return null;
  if (cachedReviewData && cachedReviewId === id) {
    return cachedReviewData;
  }
  try {
    const review = await getReview(id);
    cachedReviewData = review;
    cachedReviewId = id;
    return review;
  } catch (err) {
    return null;
  }
}

function createElement(tag, className, textContent) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (textContent !== undefined && textContent !== null) el.textContent = textContent;
  return el;
}

function restorePageMarkup(containerEl) {
  // Puts back the markup showLoadingState() set aside, so the page's own
  // filters, tables and chart canvases exist again before anything renders.
  if (containerEl && containerEl.dataset.pageMarkup !== undefined) {
    containerEl.innerHTML = containerEl.dataset.pageMarkup;
  }
}


function showEmptyReviewState(containerEl) {
  containerEl.innerHTML = "";
  const card = createElement("div", "card");
  card.style.textAlign = "center";
  card.style.padding = "48px 24px";

  const icon = createElement("i", "fa-solid fa-file-circle-exclamation");
  icon.style.fontSize = "3rem";
  icon.style.color = "var(--accent-cyan)";
  icon.style.marginBottom = "16px";

  const msg = createElement("h3", null, "No review yet — upload a statement file to start");
  msg.style.fontSize = "1.2rem";
  msg.style.marginBottom = "16px";

  const link = createElement("a", "btn btn-primary", "Go to Upload Page");
  link.href = "upload.html";
  link.style.display = "inline-flex";
  link.style.alignItems = "center";
  link.style.gap = "8px";

  card.appendChild(icon);
  card.appendChild(msg);
  card.appendChild(link);
  containerEl.appendChild(card);
}

function showLoadingState(containerEl, message = "Loading review data...") {
  // Some pages keep their filters, table and canvases inside this container,
  // so the markup is kept and put back by restorePageMarkup() once the data
  // arrives. Without that, the spinner would delete the page it loads into.
  if (containerEl.dataset.pageMarkup === undefined) {
    containerEl.dataset.pageMarkup = containerEl.innerHTML;
  }
  containerEl.innerHTML = "";
  const card = createElement("div", "card");
  card.style.textAlign = "center";
  card.style.padding = "48px 24px";

  const spinner = createElement("i", "fa-solid fa-circle-notch fa-spin");
  spinner.style.fontSize = "2.5rem";
  spinner.style.color = "var(--accent-cyan)";
  spinner.style.marginBottom = "16px";

  const text = createElement("p", null, message);
  text.style.fontSize = "1rem";
  text.style.color = "var(--text-muted)";

  card.appendChild(spinner);
  card.appendChild(text);
  containerEl.appendChild(card);
}

function formatNumber(num) {
  if (num === null || num === undefined || isNaN(num)) return "-";
  return Number(num).toLocaleString();
}

function formatPercent(val) {
  if (val === null || val === undefined || isNaN(val)) return "-";
  return `${Number(val).toFixed(1)}%`;
}

function formatConfidence(val) {
  if (val === null || val === undefined || isNaN(val)) return "-";
  return `${(Number(val) * 100).toFixed(0)}%`;
}
