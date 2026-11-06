const form = document.querySelector("#chat-form");
const promptInput = document.querySelector("#prompt");
const submitButton = document.querySelector("#submit");
const statusMessage = document.querySelector("#status");
const serviceStatus = document.querySelector("#service-status");
const captchaWrap = document.querySelector("#captcha-wrap");
const charCount = document.querySelector("#char-count");
const welcome = document.querySelector("#welcome");
const messageList = document.querySelector("#message-list");
const newChatButton = document.querySelector("#new-chat");

let captchaToken = "";
let captchaWidget;
let serviceReady = false;
let pending = false;

function updateSendButton() {
  submitButton.disabled =
    !serviceReady || !captchaToken || !promptInput.value.trim() || pending;
}

function setServiceStatus(ready, text) {
  serviceStatus.classList.toggle("is-ready", ready);
  serviceStatus.lastChild.textContent = ` ${text}`;
}

function addMessage(role, text, { error = false, copy = false } = {}) {
  const message = document.createElement("article");
  message.className = `message ${role}${error ? " message-error" : ""}`;

  const avatar = document.createElement("span");
  avatar.className = "message-avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = role === "user" ? "YOU" : "AI";

  const body = document.createElement("div");
  body.className = "message-body";
  const name = document.createElement("p");
  name.className = "message-name";
  name.textContent = role === "user" ? "You" : "AI Data Agent";

  const content = document.createElement("p");
  content.className = "message-text";
  content.textContent = text;

  body.append(name, content);
  if (copy && !error) {
    const actions = document.createElement("div");
    actions.className = "message-actions";
    const copyButton = document.createElement("button");
    copyButton.className = "copy-button";
    copyButton.type = "button";
    copyButton.textContent = "Copy response";
    copyButton.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(text);
        copyButton.textContent = "Copied";
        window.setTimeout(() => {
          copyButton.textContent = "Copy response";
        }, 1500);
      } catch {
        statusMessage.textContent = "Clipboard access is unavailable in this browser.";
      }
    });
    actions.append(copyButton);
    body.append(actions);
  }

  message.append(avatar, body);
  messageList.append(message);
  welcome.hidden = true;
  message.scrollIntoView({ block: "nearest", behavior: "smooth" });
  return message;
}

function addTypingIndicator() {
  const message = document.createElement("article");
  message.className = "message assistant";
  message.setAttribute("aria-label", "AI Data Agent is responding");

  const avatar = document.createElement("span");
  avatar.className = "message-avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = "AI";

  const body = document.createElement("div");
  body.className = "message-body";
  const name = document.createElement("p");
  name.className = "message-name";
  name.textContent = "AI Data Agent";
  const dots = document.createElement("span");
  dots.className = "typing-dots";
  dots.setAttribute("aria-hidden", "true");
  for (let index = 0; index < 3; index += 1) {
    dots.append(document.createElement("i"));
  }
  body.append(name, dots);
  message.append(avatar, body);
  messageList.append(message);
  message.scrollIntoView({ block: "nearest", behavior: "smooth" });
  return message;
}

async function initialize() {
  try {
    const response = await fetch("/api/config");
    if (!response.ok) {
      throw new Error("Could not check server configuration.");
    }
    const config = await response.json();
    serviceReady = config.ready && Boolean(config.captchaSiteKey);

    if (!serviceReady) {
      setServiceStatus(false, "Setup required");
      const issues = [
        ...(config.missingSettings || []),
        ...(config.invalidSettings || []),
      ];
      statusMessage.textContent = issues.length
        ? `Server setup required: ${issues.join(", ")}. See the project README for configuration steps.`
        : "The service is not configured. Contact the site administrator.";
      updateSendButton();
      return;
    }

    setServiceStatus(true, "Ready");
    statusMessage.textContent = "Complete the CAPTCHA to send a prompt.";
    captchaWrap.hidden = false;

    const captchaScript = document.createElement("script");
    captchaScript.src =
      "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";
    captchaScript.async = true;
    captchaScript.defer = true;
    captchaScript.onload = () => {
      if (!window.turnstile) {
        setServiceStatus(false, "CAPTCHA unavailable");
        statusMessage.textContent = "CAPTCHA could not be initialized. Please reload.";
        return;
      }
      captchaWidget = window.turnstile.render("#captcha", {
        sitekey: config.captchaSiteKey,
        callback: (token) => {
          captchaToken = token;
          statusMessage.textContent = "Ready. Your prompt is sent securely to the AI service.";
          updateSendButton();
        },
        "expired-callback": () => {
          captchaToken = "";
          statusMessage.textContent = "CAPTCHA expired. Complete it again to send.";
          updateSendButton();
        },
        "error-callback": () => {
          captchaToken = "";
          setServiceStatus(false, "CAPTCHA unavailable");
          statusMessage.textContent = "CAPTCHA verification failed to load. Please retry.";
          updateSendButton();
        },
      });
    };
    captchaScript.onerror = () => {
      setServiceStatus(false, "CAPTCHA unavailable");
      statusMessage.textContent = "CAPTCHA could not be loaded. Check your connection and retry.";
    };
    document.head.append(captchaScript);
  } catch (error) {
    setServiceStatus(false, "Unavailable");
    statusMessage.textContent =
      error.message || "Could not check the service. Please reload.";
  }
  updateSendButton();
}

async function sendPrompt(prompt) {
  if (!serviceReady || !captchaToken || pending) {
    statusMessage.textContent = !serviceReady
      ? "The service is not configured yet."
      : "Complete the CAPTCHA before sending a prompt.";
    return;
  }

  pending = true;
  updateSendButton();
  promptInput.value = "";
  charCount.textContent = "0 / 2000";
  addMessage("user", prompt);
  const typing = addTypingIndicator();
  statusMessage.textContent = "Preparing a response…";

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        captcha_token: captchaToken,
      }),
    });
    const result = await response.json();
    if (!response.ok) {
      const detail = result.error || result.detail;
      const message = Array.isArray(detail)
        ? detail.map((item) => item.msg).join("; ")
        : detail;
      throw new Error(message || "The request could not be completed.");
    }
    addMessage("assistant", result.reply, { copy: true });
    statusMessage.textContent = "Response complete.";
  } catch (error) {
    addMessage(
      "assistant",
      error.message || "The request failed. Please try again.",
      { error: true },
    );
    statusMessage.textContent = "The response could not be generated.";
  } finally {
    typing.remove();
    pending = false;
    captchaToken = "";
    if (window.turnstile && captchaWidget !== undefined) {
      window.turnstile.reset(captchaWidget);
    }
    updateSendButton();
    promptInput.focus();
  }
}

function startNewConversation() {
  if (pending) {
    statusMessage.textContent = "Please wait for the current response to finish.";
    return;
  }
  messageList.replaceChildren();
  welcome.hidden = false;
  promptInput.value = "";
  charCount.textContent = "0 / 2000";
  statusMessage.textContent = serviceReady
    ? "Complete the CAPTCHA to send a prompt."
    : "The service is not configured yet.";
  promptInput.focus();
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const prompt = promptInput.value.trim();
  if (prompt) {
    sendPrompt(prompt);
  }
});

promptInput.addEventListener("input", () => {
  charCount.textContent = `${promptInput.value.length} / 2000`;
  updateSendButton();
});

promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!submitButton.disabled) {
      form.requestSubmit();
    }
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    promptInput.value = button.dataset.prompt || "";
    charCount.textContent = `${promptInput.value.length} / 2000`;
    updateSendButton();
    promptInput.focus();
  });
});

newChatButton.addEventListener("click", startNewConversation);
document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    startNewConversation();
  }
});

initialize();
