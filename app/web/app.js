const form = document.querySelector("#chat-form");
const input = document.querySelector("#question");
const messages = document.querySelector("#messages");
const exampleButtons = document.querySelectorAll(".example-button");
let loadingMessage = null;

exampleButtons.forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.textContent.trim();
    input.focus();
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  addMessage(question, "user");
  input.value = "";
  setLoading(true);

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Request failed");
    }

    addMessage(data.answer, "assistant", data.sources || []);
  } catch (error) {
    addMessage(`Blad: ${error.message}`, "assistant");
  } finally {
    setLoading(false);
  }
});

function addMessage(text, role, sources = []) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const answer = document.createElement("div");
  answer.className = "answer";
  answer.textContent = text;
  article.appendChild(answer);

  if (sources.length > 0) {
    const wrapper = document.createElement("div");
    wrapper.className = "sources";
    const title = document.createElement("div");
    title.className = "sources-title";
    title.textContent = "Źródła";
    wrapper.appendChild(title);

    sources.slice(0, 3).forEach((source) => {
      wrapper.appendChild(createSourceCard(source));
    });
    article.appendChild(wrapper);
  }

  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
}

function setLoading(isLoading) {
  const button = form.querySelector("button");
  button.disabled = isLoading;
  button.textContent = isLoading ? "Szukam..." : "Wyślij";
  input.disabled = isLoading;

  if (isLoading) {
    loadingMessage = document.createElement("article");
    loadingMessage.className = "message assistant loading-message";
    loadingMessage.innerHTML = `
      <div class="loading-indicator" aria-live="polite">
        <span></span>
        <span></span>
        <span></span>
        <p>Wyszukuję kontekst w FAISS i przygotowuję odpowiedź...</p>
      </div>
    `;
    messages.appendChild(loadingMessage);
    messages.scrollTop = messages.scrollHeight;
    return;
  }

  if (loadingMessage) {
    loadingMessage.remove();
    loadingMessage = null;
  }
}

function createSourceCard(source) {
  const item = document.createElement("section");
  item.className = "source-card";

  const header = document.createElement("div");
  header.className = "source-header";

  const fileName = document.createElement("strong");
  fileName.textContent = source.file_name || "unknown";
  header.appendChild(fileName);

  if (source.document_type) {
    const type = document.createElement("span");
    type.className = "source-type";
    type.textContent = source.document_type;
    header.appendChild(type);
  }
  item.appendChild(header);

  const meta = document.createElement("dl");
  meta.className = "source-meta";
  [
    ["Przedmiot", source.subject],
    ["Semestr", source.semester],
    ["Prowadzący", source.lecturer],
    ["ECTS", source.ects],
  ]
    .filter(([, value]) => value !== null && value !== undefined && value !== "")
    .forEach(([label, value]) => {
      const term = document.createElement("dt");
      term.textContent = label;
      const description = document.createElement("dd");
      description.textContent = value;
      meta.append(term, description);
    });
  if (meta.children.length > 0) item.appendChild(meta);

  if (source.preview) {
    const preview = document.createElement("p");
    preview.className = "source-preview";
    preview.textContent = source.preview;
    item.appendChild(preview);
  }

  const technical = [];
  if (source.score != null) technical.push(`score ${source.score.toFixed(3)}`);
  if (source.chunk_number != null) technical.push(`chunk ${source.chunk_number}`);
  if (technical.length > 0) {
    const details = document.createElement("details");
    details.className = "technical-details";
    const summary = document.createElement("summary");
    summary.textContent = "Szczegóły techniczne";
    const muted = document.createElement("small");
    muted.textContent = technical.join(" · ");
    details.append(summary, muted);
    item.appendChild(details);
  }

  return item;
}
