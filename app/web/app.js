const form = document.querySelector("#chat-form");
const input = document.querySelector("#question");
const messages = document.querySelector("#messages");

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

  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  article.appendChild(paragraph);

  if (sources.length > 0) {
    const wrapper = document.createElement("div");
    wrapper.className = "sources";
    sources.forEach((source) => {
      const item = document.createElement("div");
      item.className = "source";
      item.textContent = `${source.source} | score: ${source.score.toFixed(3)} | ${source.preview}`;
      wrapper.appendChild(item);
    });
    article.appendChild(wrapper);
  }

  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
}

function setLoading(isLoading) {
  const button = form.querySelector("button");
  button.disabled = isLoading;
  button.textContent = isLoading ? "Szukam..." : "Wyslij";
}
