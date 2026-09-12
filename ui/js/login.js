/**
 * AuditLens - Sign In Controller
 */

document.addEventListener("DOMContentLoaded", async () => {
  // Check URL query parameters
  const urlParams = new URLSearchParams(window.location.search);
  const emailParam = urlParams.get("email");
  const emailInput = document.getElementById("emailInput");

  if (emailParam && emailInput) {
    emailInput.value = emailParam.trim();
  }

  // Check if valid session already exists
  const session = typeof getSession === "function" ? getSession() : null;
  if (session && session.token) {
    try {
      const res = await fetch("/auth/me", {
        headers: { "Authorization": `Bearer ${session.token}` }
      });
      if (res.ok) {
        const nextPage = typeof getValidNextPage === "function" ? getValidNextPage(urlParams.get("next")) : "dashboard.html";
        window.location.href = nextPage;
        return;
      } else {
        if (typeof clearSession === "function") clearSession();
      }
    } catch (e) {
      // Ignore network error on auto-login check
    }
  }

  setupPasswordToggle();
  setupLoginForm();
});

function setupPasswordToggle() {
  const toggleBtn = document.getElementById("togglePasswordBtn");
  const passwordInput = document.getElementById("passwordInput");
  if (!toggleBtn || !passwordInput) return;

  toggleBtn.addEventListener("click", () => {
    if (passwordInput.type === "password") {
      passwordInput.type = "text";
      toggleBtn.innerHTML = `<i class="fa-regular fa-eye-slash"></i>`;
    } else {
      passwordInput.type = "password";
      toggleBtn.innerHTML = `<i class="fa-regular fa-eye"></i>`;
    }
  });
}

function setupLoginForm() {
  const loginForm = document.getElementById("loginForm");
  if (!loginForm) return;

  loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideAlert();

    const emailInput = document.getElementById("emailInput");
    const passwordInput = document.getElementById("passwordInput");
    const rememberMeCheck = document.getElementById("rememberMeCheck");
    const submitBtn = document.getElementById("loginSubmitBtn");

    const email = emailInput ? emailInput.value.trim() : "";
    const password = passwordInput ? passwordInput.value : "";
    const rememberMe = rememberMeCheck ? rememberMeCheck.checked : false;

    if (!email || !password) {
      showAlert("Email and password are required.");
      return;
    }

    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Signing in...`;
    }

    try {
      const res = await fetch("/auth/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ email, password })
      });

      if (res.ok) {
        const data = await res.json();
        // Never store password!
        saveSession({
          token: data.token,
          expires_at: data.expires_at,
          user: data.user
        }, rememberMe);

        const urlParams = new URLSearchParams(window.location.search);
        const nextPage = getValidNextPage(urlParams.get("next"));
        window.location.href = nextPage;
      } else {
        let errorMsg = "Invalid email or password.";
        try {
          const errData = await res.json();
          if (errData && errData.detail) {
            errorMsg = typeof errData.detail === "string" ? errData.detail : JSON.stringify(errData.detail);
          }
        } catch (err) {}

        showAlert(errorMsg);
        if (submitBtn) {
          submitBtn.disabled = false;
          submitBtn.innerHTML = `<i class="fa-solid fa-right-to-bracket"></i> Sign In`;
        }
      }
    } catch (err) {
      showAlert("Can't reach the review service. Is the backend running?");
      if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.innerHTML = `<i class="fa-solid fa-right-to-bracket"></i> Sign In`;
      }
    }
  });
}

function showAlert(message) {
  const alertArea = document.getElementById("authAlertArea");
  if (!alertArea) return;
  alertArea.textContent = "";

  const icon = document.createElement("i");
  icon.className = "fa-solid fa-circle-exclamation";

  const text = document.createTextNode(` ${message}`);

  alertArea.appendChild(icon);
  alertArea.appendChild(text);
  alertArea.style.display = "flex";
}

function hideAlert() {
  const alertArea = document.getElementById("authAlertArea");
  if (alertArea) alertArea.style.display = "none";
}
