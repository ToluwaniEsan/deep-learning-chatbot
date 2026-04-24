(function () {
  const chatPanel = document.getElementById("chatPanel");
  const welcome = document.getElementById("welcome");
  const form = document.getElementById("chatForm");
  const input = document.getElementById("message");
  const sendBtn = document.getElementById("sendBtn");
  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");
  const modelAlert = document.getElementById("modelAlert");
  const modelAlertMsg = document.getElementById("modelAlertMsg");
  const suggestions = document.getElementById("suggestions");
  const themeToggle = document.getElementById("themeToggle");
  const themeToggleText = document.getElementById("themeToggleText");
  const THEME_KEY = "academicTutorTheme";
  let themeTransitionTimer = null;

  function applyTheme(theme) {
    const nextTheme = theme === "dark" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", nextTheme);
    if (themeToggle) {
      const darkEnabled = nextTheme === "dark";
      themeToggle.setAttribute("aria-pressed", String(darkEnabled));
      themeToggleText.textContent = darkEnabled ? "Light mode" : "Dark mode";
    }
  }

  function animateThemeSwitch() {
    const prefersReducedMotion =
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReducedMotion) return;

    document.documentElement.classList.add("theme-transitioning");
    if (themeTransitionTimer) {
      window.clearTimeout(themeTransitionTimer);
    }
    themeTransitionTimer = window.setTimeout(function () {
      document.documentElement.classList.remove("theme-transitioning");
      themeTransitionTimer = null;
    }, 380);
  }

  function initTheme() {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored === "dark" || stored === "light") {
      applyTheme(stored);
      return;
    }
    const prefersDark =
      window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches;
    applyTheme(prefersDark ? "dark" : "light");
  }

  function setReady(ok) {
    statusDot.classList.toggle("ok", ok);
    statusDot.classList.toggle("warn", !ok);
    statusText.textContent = ok ? "Model ready" : "Model offline";
  }

  async function checkHealth() {
    try {
      const r = await fetch("/api/health");
      const data = await r.json();
      setReady(!!data.ok);
      if (!data.ok) {
        modelAlert.hidden = false;
        modelAlertMsg.textContent =
          "Run `python train.py` in the project folder, then restart the server (`python app.py`).";
      } else {
        modelAlert.hidden = true;
      }
    } catch (e) {
      setReady(false);
      modelAlert.hidden = false;
      modelAlertMsg.textContent = "Could not reach the server.";
    }
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  /**
   * Light markdown for assistant replies: **bold**, *italic* (after bold pass).
   */
  function formatAssistantRichText(text) {
    var t = escapeHtml(text);
    t = t.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    t = t.replace(/\*([^*\n]+?)\*/g, "<em>$1</em>");
    return t;
  }

  function appendMessage(role, text, meta) {
    if (welcome) welcome.style.display = "none";

    const wrap = document.createElement("div");
    wrap.className = "msg " + (role === "user" ? "user" : "assistant");

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    if (role === "assistant") {
      bubble.innerHTML = formatAssistantRichText(text);
    } else {
      bubble.textContent = text;
    }
    wrap.appendChild(bubble);

    if (meta) {
      const m = document.createElement("div");
      m.className = "meta";
      if (meta.subject) {
        const p = document.createElement("span");
        p.className = "pill";
        p.textContent = meta.subject;
        m.appendChild(p);
      }
      if (typeof meta.confidence === "number") {
        const c = document.createElement("span");
        c.textContent = "Confidence " + (meta.confidence * 100).toFixed(1) + "%";
        m.appendChild(c);
      }
      wrap.appendChild(m);

      if (
        role === "assistant" &&
        Array.isArray(meta.followUpOptions) &&
        Array.isArray(meta.followUpPrompts) &&
        meta.followUpOptions.length > 0 &&
        meta.followUpPrompts.length === meta.followUpOptions.length
      ) {
        const actions = document.createElement("div");
        actions.className = "followup-actions";
        meta.followUpOptions.forEach(function (label, idx) {
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "followup-chip";
          btn.textContent = label;
          btn.setAttribute("data-followup-prompt", meta.followUpPrompts[idx]);
          actions.appendChild(btn);
        });
        wrap.appendChild(actions);
      }
    }

    chatPanel.appendChild(wrap);
    chatPanel.scrollTop = chatPanel.scrollHeight;
  }

  function resizeInput() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
  }

  input.addEventListener("input", resizeInput);

  form.addEventListener("submit", async function (e) {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;

    appendMessage("user", text, null);
    input.value = "";
    resizeInput();
    sendBtn.disabled = true;

    try {
      const r = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      const data = await r.json();
      if (!r.ok) {
        appendMessage("assistant", data.error || "Something went wrong.", null);
        return;
      }
      appendMessage("assistant", data.reply, {
        subject: data.subject,
        confidence: data.confidence,
        followUpOptions: data.follow_up_options || null,
        followUpPrompts: data.follow_up_prompts || null,
      });
    } catch (err) {
      appendMessage("assistant", "Network error. Is the server running?", null);
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  });

  suggestions.addEventListener("click", function (e) {
    const t = e.target;
    if (t.classList.contains("chip") && t.dataset.q) {
      input.value = t.dataset.q;
      input.focus();
      resizeInput();
    }
  });

  chatPanel.addEventListener("click", function (e) {
    const t = e.target;
    if (t.classList.contains("followup-chip")) {
      const prompt = t.getAttribute("data-followup-prompt");
      if (!prompt) return;
      input.value = prompt;
      resizeInput();
      form.requestSubmit();
    }
  });

  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  if (themeToggle) {
    themeToggle.addEventListener("click", function () {
      const current =
        document.documentElement.getAttribute("data-theme") === "dark"
          ? "dark"
          : "light";
      const next = current === "dark" ? "light" : "dark";
      animateThemeSwitch();
      applyTheme(next);
      localStorage.setItem(THEME_KEY, next);
    });
  }

  initTheme();
  checkHealth();
  input.focus();
})();
