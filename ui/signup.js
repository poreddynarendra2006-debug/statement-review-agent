/**
 * FinSight AI - Create Account Controller
 */

document.addEventListener("DOMContentLoaded", () => {
  setupPasswordToggles();
  setupSignupForm();
});

function setupPasswordToggles() {
  const togglePassBtn = document.getElementById("toggleRegPasswordBtn");
  const passInput = document.getElementById("regPasswordInput");
  if (togglePassBtn && passInput) {
    togglePassBtn.addEventListener("click", () => {
      if (passInput.type === "password") {
        passInput.type = "text";
        togglePassBtn.innerHTML = `<i class="fa-regular fa-eye-slash"></i>`;
      } else {
        passInput.type = "password";
        togglePassBtn.innerHTML = `<i class="fa-regular fa-eye"></i>`;
      }
    });
  }

  const toggleConfBtn = document.getElementById("toggleConfirmPasswordBtn");
  const confInput = document.getElementById("regConfirmPasswordInput");
  if (toggleConfBtn && confInput) {
    toggleConfBtn.addEventListener("click", () => {
      if (confInput.type === "password") {
        confInput.type = "text";
        toggleConfBtn.innerHTML = `<i class="fa-regular fa-eye-slash"></i>`;
      } else {
        confInput.type = "password";
        toggleConfBtn.innerHTML = `<i class="fa-regular fa-eye"></i>`;
      }
    });
  }
}

function setupSignupForm() {
  const signupForm = document.getElementById("signupForm");
  if (!signupForm) return;

  signupForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearErrors();

    const nameInput = document.getElementById("regNameInput");
    const emailInput = document.getElementById("regEmailInput");
    const roleSelect = document.getElementById("regRoleSelect");
    const passwordInput = document.getElementById("regPasswordInput");
    const confirmPasswordInput = document.getElementById("regConfirmPasswordInput");
    const submitBtn = document.getElementById("signupSubmitBtn");

    const name = nameInput ? nameInput.value.trim() : "";
    const email = emailInput ? emailInput.value.trim() : "";
    const role = roleSelect ? roleSelect.value : "";
    const password = passwordInput ? passwordInput.value : "";
    const confirmPassword = confirmPasswordInput ? confirmPasswordInput.value : "";

    let isValid = true;

    // 1. Check all fields filled
    if (!name) {
      showFieldError("nameError", "Full name is required.");
      isValid = false;
    }

    // 2. Email format check
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!email) {
      showFieldError("emailError", "Email address is required.");
      isValid = false;
    } else if (!emailRegex.test(email)) {
      showFieldError("emailError", "Please enter a valid email address.");
      isValid = false;
    }

    if (!role) {
      showFieldError("roleError", "Please select a role.");
      isValid = false;
    }

    // 3. Password requirements (at least 8 chars, letter & number)
    const hasLetter = /[a-zA-Z]/.test(password);
    const hasNumber = /[0-9]/.test(password);

    if (!password) {
      showFieldError("passwordError", "Password is required.");
      isValid = false;
    } else if (password.length < 8) {
      showFieldError("passwordError", "Password must be at least 8 characters long.");
      isValid = false;
    } else if (!hasLetter || !hasNumber) {
      showFieldError("passwordError", "Password must contain at least one letter and one number.");
      isValid = false;
    }

    // 4. Passwords match check
    if (!confirmPassword) {
      showFieldError("confirmPasswordError", "Please confirm your password.");
      isValid = false;
    } else if (password !== confirmPassword) {
      showFieldError("confirmPasswordError", "Passwords do not match.");
      isValid = false;
    }

    if (!isValid) return;

    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Creating account...`;
    }

    try {
      const res = await fetch("/auth/register", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ name, email, password, role })
      });

      if (res.status === 201) {
        showGlobalAlert("Account created. Please sign in.", "success");
        setTimeout(() => {
          window.location.href = `index.html?email=${encodeURIComponent(email)}`;
        }, 600);
      } else {
        let errorMsg = "Registration failed.";
        try {
          const errData = await res.json();
          if (errData && errData.detail) {
            errorMsg = typeof errData.detail === "string" ? errData.detail : JSON.stringify(errData.detail);
          }
        } catch (err) {}

        showGlobalAlert(errorMsg, "error");
        if (submitBtn) {
          submitBtn.disabled = false;
          submitBtn.innerHTML = `<i class="fa-solid fa-user-plus"></i> Create Account`;
        }
      }
    } catch (err) {
      showGlobalAlert("Can't reach the review service. Is the backend running?", "error");
      if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.innerHTML = `<i class="fa-solid fa-user-plus"></i> Create Account`;
      }
    }
  });
}

function showFieldError(elementId, message) {
  const el = document.getElementById(elementId);
  if (!el) return;
  el.textContent = message;
  el.style.display = "block";
}

function showGlobalAlert(message, type = "error") {
  const alertArea = document.getElementById("authAlertArea");
  if (!alertArea) return;
  alertArea.textContent = "";
  alertArea.className = `auth-alert ${type}`;

  const icon = document.createElement("i");
  icon.className = type === "error" ? "fa-solid fa-circle-exclamation" : "fa-solid fa-circle-check";

  const text = document.createTextNode(` ${message}`);

  alertArea.appendChild(icon);
  alertArea.appendChild(text);
  alertArea.style.display = "flex";
}

function clearErrors() {
  const fieldErrors = document.querySelectorAll(".field-error");
  fieldErrors.forEach(el => {
    el.textContent = "";
    el.style.display = "none";
  });

  const alertArea = document.getElementById("authAlertArea");
  if (alertArea) {
    alertArea.textContent = "";
    alertArea.style.display = "none";
  }
}
