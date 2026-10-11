// Renders a chat answer as Markdown, then sanitizes it.
//
// Answers are untrusted: the model can echo text from an uploaded document.
// So raw HTML in the answer is shown as text, images are never loaded, and the
// rendered HTML is passed through DOMPurify with a small allowlist. Only
// http(s) and mailto links survive, and they open in a new tab.
(function () {
  "use strict";

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  const md = new marked.Marked({
    gfm: true,
    breaks: true,
    async: false,
    renderer: {
      // Raw HTML is displayed literally instead of being parsed as HTML.
      html({ text }) {
        return escapeHtml(text);
      },
      // No requests to URLs taken from document content: show the alt text only.
      image({ text }) {
        return escapeHtml(text);
      },
    },
  });

  // A private DOMPurify instance, so this hook and config affect nothing else.
  const purify = DOMPurify(window);
  purify.addHook("afterSanitizeAttributes", (node) => {
    if (node.tagName === "A") {
      node.setAttribute("target", "_blank");
      node.setAttribute("rel", "noopener noreferrer");
    }
  });

  const PURIFY_CONFIG = {
    ALLOWED_TAGS: [
      "p", "br", "strong", "em", "del", "code", "pre", "blockquote", "hr",
      "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li",
      "table", "thead", "tbody", "tr", "th", "td", "a",
    ],
    ALLOWED_ATTR: ["href", "title", "align", "start"],
    ALLOWED_URI_REGEXP: /^(?:https?:|mailto:)/i,
    RETURN_DOM_FRAGMENT: true,
  };

  // Returns a sanitized DocumentFragment for the given Markdown text.
  function render(markdown) {
    const html = md.parse(String(markdown ?? ""));
    return purify.sanitize(html, PURIFY_CONFIG);
  }

  window.RiskBotMarkdown = { render };
})();
