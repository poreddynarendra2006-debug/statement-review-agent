/**
 * AuditLens - Authentication & Session Module
 */

const AUTH_SESSION_KEY = "authSession";

const ALLOWED_APP_PAGES = [
  "dashboard.html",
  "upload.html",
  "findings.html",
  "trends.html",
  "risk.html",
  "chatbot.html"
];

function getSession() {
  let sessionStr = localStorage.getItem(AUTH_SESSION_KEY) || sessionStorage.getItem(AUTH_SESSION_KEY);
  if (!sessionStr) return null;

  try {
    const session = JSON.parse(sessionStr);
    if (!session || !session.token || !session.expires_at) {
      clearSession();
      return null;
    }
    const expiresDate = new Date(session.expires_at);
    if (isNaN(expiresDate.getTime()) || expiresDate <= new Date()) {
      clearSession();
      return null;
    }
    return session;
  } catch (e) {
    clearSession();
    return null;
  }
}

function saveSession(sessionData, rememberMe = false) {
  clearSession();
  const dataStr = JSON.stringify(sessionData);
  if (rememberMe) {
    localStorage.setItem(AUTH_SESSION_KEY, dataStr);
  } else {
    sessionStorage.setItem(AUTH_SESSION_KEY, dataStr);
  }
}

function clearSession() {
  localStorage.removeItem(AUTH_SESSION_KEY);
  sessionStorage.removeItem(AUTH_SESSION_KEY);
}

function getToken() {
  const session = getSession();
  return session ? session.token : null;
}

function getCurrentUser() {
  const session = getSession();
  return session ? session.user : null;
}

function getValidNextPage(urlParam) {
  if (!urlParam) return "dashboard.html";
  const clean = urlParam.trim().split("?")[0].split("/").pop();
  if (ALLOWED_APP_PAGES.includes(clean)) {
    return clean;
  }
  return "dashboard.html";
}

async function signOut() {
  const token = getToken();
  if (token) {
    try {
      await fetch("/auth/logout", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`
        }
      });
    } catch (e) {
      // Ignore network errors on logout
    }
  }
  clearSession();
  window.location.href = "index.html";
}

async function requireAuth() {
  const currentFile = window.location.pathname.split("/").pop() || "dashboard.html";
  
  if (currentFile === "index.html" || currentFile === "signup.html" || currentFile === "") {
    return true;
  }

  const session = getSession();
  if (!session || !session.token) {
    window.location.href = `index.html?next=${encodeURIComponent(currentFile)}`;
    return false;
  }

  try {
    const res = await fetch("/auth/me", {
      headers: {
        "Authorization": `Bearer ${session.token}`
      }
    });

    if (!res.ok) {
      clearSession();
      window.location.href = `index.html?next=${encodeURIComponent(currentFile)}`;
      return false;
    }

    const user = await res.json();
    session.user = user;
    const isLocal = !!localStorage.getItem(AUTH_SESSION_KEY);
    saveSession(session, isLocal);

    updateSidebarUserInfo();
    return true;
  } catch (err) {
    updateSidebarUserInfo();
    return true;
  }
}

function updateSidebarUserInfo() {
  const user = getCurrentUser();
  if (!user) return;

  const nameEl = document.getElementById("userNameDisplay");
  const roleEl = document.getElementById("userRoleDisplay");
  const avatarEl = document.getElementById("userAvatar");

  if (nameEl) nameEl.textContent = user.name || "Reviewer";
  if (roleEl) roleEl.textContent = user.role || "Financial Auditor";
  if (avatarEl && user.name) {
    const initials = user.name.trim().split(" ").map(n => n[0]).join("").substring(0, 2).toUpperCase();
    avatarEl.textContent = initials || "FA";
  }
}
