/* =====================================================
   Ferhat Erata — website-v2
   Vanilla JS: theme toggle, news, publications (filterable), patents
   ===================================================== */

(() => {
  // ---------- theme ----------
  const THEME_KEY = "fe-theme";
  const root = document.documentElement;
  const btn = document.getElementById("theme-toggle");

  const mediaPref = () =>
    window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";

  const applyTheme = (t) => {
    root.dataset.theme = t;
    if (btn) {
      btn.setAttribute("aria-label", t === "dark" ? "Switch to light" : "Switch to dark");
    }
  };

  const saved = localStorage.getItem(THEME_KEY);
  applyTheme(saved || mediaPref());

  if (btn) {
    btn.addEventListener("click", () => {
      const next = root.dataset.theme === "dark" ? "light" : "dark";
      applyTheme(next);
      localStorage.setItem(THEME_KEY, next);
    });
  }

  // ---------- year in footer ----------
  const y = document.getElementById("year");
  if (y) y.textContent = new Date().getFullYear();

  // ---------- fetch helper (works from filesystem too) ----------
  // Append a version query so the browser doesn't serve a stale
  // JSON body when the file has changed.
  const JSON_VERSION = "2026-07-04";
  const fetchJSON = async (path) => {
    try {
      const sep = path.includes("?") ? "&" : "?";
      const r = await fetch(`${path}${sep}v=${JSON_VERSION}`);
      if (!r.ok) throw new Error(r.statusText);
      return await r.json();
    } catch (e) {
      console.error(`[${path}]`, e);
      return null;
    }
  };

  // ---------- news ----------
  const trophy = `<svg class="pill-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/><path d="M4 22h16"/><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/><path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/><path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"/></svg>`;

  // Render one news item to HTML. `pathPrefix` is prepended to any relative
  // image or href (so the archive page at /news/ can point to detail pages
  // at /news/<slug>/ without needing the feed to be rewritten).
  const renderNewsItem = (n, pathPrefix = "") => {
    const rewrite = (u) => /^([a-z]+:|\/|#)/i.test(u) ? u : pathPrefix + u;
    const imgTag = n.image
      ? `<img src="${escapeAttr(rewrite(n.image))}" alt="" loading="lazy" onerror="this.closest('.news-image').remove()"/>`
      : "";
    const figure = imgTag
      ? `<figure class="news-image">${imgTag}</figure>`
      : "";
    // Rewrite relative hrefs inside the item's HTML body.
    const body = pathPrefix
      ? n.html.replace(/href="(?!([a-z]+:|\/|#))/gi, `href="${pathPrefix}`)
      : n.html;
    const rowHref = n.href ? rewrite(n.href) : "";
    const dataHref = rowHref ? ` data-href="${escapeAttr(rowHref)}"` : "";
    return `
      <li class="news-item${rowHref ? " news-item-clickable" : ""}"${dataHref}>
        <div class="news-body">
          <span class="news-date">${formatDate(n.date)}</span>
          <div class="news-content"><span class="tag ${n.award ? "tag-award" : ""}">${n.award ? trophy : ""}${escapeHTML(n.tag)}</span>${body}</div>
        </div>
        ${figure}
      </li>`;
  };

  const renderNews = async () => {
    const el = document.getElementById("news-list");
    if (!el) return;
    const data = await fetchJSON("news.json");
    if (!data) {
      el.innerHTML = `<li class="empty-state">News feed could not be loaded. If opening this file directly, serve with a local HTTP server.</li>`;
      return;
    }
    // Main-page list: show items flagged featured:true first, then fall back
    // to newest 10 if fewer than 10 featured items exist.
    const featured = data.news.filter((n) => n.featured === true);
    let visible;
    if (featured.length >= 10) {
      visible = featured.slice(0, 10);
    } else if (featured.length > 0) {
      const rest = data.news.filter((n) => n.featured !== true);
      visible = featured.concat(rest).slice(0, 10);
    } else {
      visible = data.news.slice(0, 10);
    }

    el.innerHTML = visible.map((n) => renderNewsItem(n)).join("");

    // "All news →" link when there are more items than shown
    if (data.news.length > visible.length) {
      const more = document.createElement("p");
      more.className = "news-more";
      more.innerHTML = `<a href="news/">All news (${data.news.length})&nbsp;→</a>`;
      el.after(more);
    }
  };

  // Archive page: full news list under /news/index.html.
  // Renders post-card-style entries (matches the /posts/ writing listing):
  // clickable card per item, hairline separators, subtle left-nudge on hover.
  const renderNewsArchive = async () => {
    const cards = document.getElementById("news-cards");
    if (!cards) return;
    const data = await fetchJSON("../news.json");
    if (!data) {
      cards.innerHTML = `<p class="empty-state">News feed could not be loaded.</p>`;
      return;
    }
    // Pull a short headline and excerpt out of the item's html.
    const tmp = document.createElement("div");
    const extract = (html) => {
      tmp.innerHTML = html;
      // Headline: first <em> (usually the paper/product title), or the
      // first sentence of plain text.
      const firstEm = tmp.querySelector("em");
      let headline;
      if (firstEm && firstEm.textContent.trim()) {
        headline = firstEm.textContent.trim();
      } else {
        const txt = tmp.textContent.trim();
        headline = txt.split(/(?<=[.?!])\s/)[0] || txt.slice(0, 80);
      }
      // Excerpt: whole plain-text body (the card CSS clamps it).
      const excerpt = tmp.textContent.trim();
      return { headline, excerpt };
    };
    const rewrite = (u) => /^([a-z]+:|\/|#)/i.test(u) ? u : "../" + u;
    cards.innerHTML = data.news.map((n) => {
      const { headline, excerpt } = extract(n.html);
      const href = n.href ? rewrite(n.href) : null;
      const dateLabel = `${formatDate(n.date)} · ${n.tag}`;
      const body = `
          <span class="post-card-date">${escapeHTML(dateLabel)}</span>
          <h3 class="post-card-title">${escapeHTML(headline)}</h3>
          <p class="post-card-excerpt">${escapeHTML(excerpt)}</p>`;
      return href
        ? `<a class="post-card" href="${escapeAttr(href)}">${body}</a>`
        : `<div class="post-card">${body}</div>`;
    }).join("");
  };

  // Make the whole news row behave like a link when it has data-href,
  // while still letting inline <a> inside the body click through normally.
  document.addEventListener("click", (e) => {
    const row = e.target.closest(".news-item-clickable");
    if (!row) return;
    // Don't hijack clicks that already landed on a real link inside the row.
    if (e.target.closest("a")) return;
    const href = row.dataset.href;
    if (!href) return;
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button === 1) {
      window.open(href, "_blank");
    } else {
      window.location.href = href;
    }
  });

  // ---------- publications ----------
  let PUBS = [];
  let activeFilter = { type: "all", value: null };

  const renderPubs = () => {
    const container = document.getElementById("publications-list");
    if (!container) return;

    const visible = PUBS
      // papers explicitly marked selected: false are hidden from the list
      .filter((p) => p.selected !== false)
      .filter((p) => {
        if (activeFilter.type === "all") return true;
        if (activeFilter.type === "year") return p.year === activeFilter.value;
        if (activeFilter.type === "tag") return (p.tags || []).includes(activeFilter.value);
        return true;
      });

    if (visible.length === 0) {
      container.innerHTML = `<p class="empty-state">No publications match this filter.</p>`;
      return;
    }

    // group by year desc
    const byYear = {};
    visible.forEach((p) => {
      (byYear[p.year] ||= []).push(p);
    });
    const years = Object.keys(byYear).sort((a, b) => b - a);

    container.innerHTML = years
      .map(
        (yr) => `
      <div class="pub-group">
        <h3 class="pub-year">${yr}</h3>
        <ol class="pub-list">
          ${byYear[yr].map(pubRow).join("")}
        </ol>
      </div>`
      )
      .join("");
  };

  const pubRow = (p) => {
    // If `doubleBlind: true`, we hide venue name and long venue line to respect
    // anonymous-submission policies. `status` is informational only.
    const hideVenue = p.doubleBlind === true;

    const authors = (p.authors || [])
      .map((a) => (a === "Ferhat Erata" ? `<span class="me">Ferhat Erata</span>` : escapeHTML(a)))
      .join(", ");

    // GitHub mark for links flagged `icon: "github"`; everything else gets
    // the trailing ↗ arrow.
    const ghIcon = `<svg class="pill-icon" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .5C5.37.5 0 5.87 0 12.5c0 5.31 3.44 9.81 8.21 11.4.6.11.82-.26.82-.58 0-.28-.01-1.04-.02-2.04-3.34.72-4.04-1.61-4.04-1.61-.55-1.39-1.34-1.76-1.34-1.76-1.09-.75.08-.73.08-.73 1.21.08 1.84 1.24 1.84 1.24 1.07 1.84 2.81 1.31 3.5 1 .11-.78.42-1.31.76-1.61-2.67-.3-5.47-1.33-5.47-5.94 0-1.31.47-2.38 1.24-3.22-.13-.3-.54-1.53.11-3.19 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 6.01 0c2.29-1.55 3.29-1.23 3.29-1.23.66 1.66.25 2.89.12 3.19.77.84 1.24 1.91 1.24 3.22 0 4.62-2.8 5.63-5.48 5.93.43.37.82 1.1.82 2.22 0 1.61-.01 2.9-.01 3.3 0 .32.22.7.83.58A12.01 12.01 0 0 0 24 12.5C24 5.87 18.63.5 12 .5z"/></svg>`;
    // In-preparation papers don't have links yet anyway; but if they did, we'd show them.
    const linkButtons = (p.links || [])
      .map((l) => l.icon === "github"
        ? `<a class="pub-link" href="${escapeAttr(l.url)}">${ghIcon}${escapeHTML(l.label)}</a>`
        : `<a class="pub-link" href="${escapeAttr(l.url)}">${escapeHTML(l.label)} <span aria-hidden="true">↗</span></a>`)
      .join("");

    const venuePillText = hideVenue ? `Preprint ${p.year}` : `${escapeHTML(p.venue || "")} ${p.year}`.trim();
    const showStatus = p.status && p.status !== "accepted";
    const trophy = `<svg class="pill-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/><path d="M4 22h16"/><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/><path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/><path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"/></svg>`;
    const pills = [
      `<span class="pill pill-venue">${venuePillText}</span>`,
      p.award ? `<span class="pill pill-award">${trophy}${escapeHTML(p.award)}</span>` : "",
      showStatus ? `<span class="pill pill-status">${escapeHTML(p.status)}</span>` : "",
    ].filter(Boolean).join("");

    const meta = [
      p.awardDetail ? escapeHTML(p.awardDetail) : "",
      !hideVenue && p.venueFull ? escapeHTML(p.venueFull) : "",
      (p.tags || []).map((t) => `<a href="#publications" class="tag" data-tag="${escapeAttr(t)}">#${escapeHTML(t)}</a>`).join(" "),
    ]
      .filter(Boolean)
      .join(" · ");

    return `
      <li class="pub">
        <div class="pub-pills">${pills}${linkButtons}</div>
        <h4 class="pub-title">${escapeHTML(p.title)}</h4>
        <p class="pub-authors">${authors}</p>
        <div class="pub-meta">${meta}</div>
      </li>`;
  };

  const renderPubControls = () => {
    const el = document.getElementById("pub-controls");
    if (!el) return;

    const years = [...new Set(PUBS.map((p) => p.year))].sort((a, b) => b - a);
    const tags = [...new Set(PUBS.flatMap((p) => p.tags || []))].sort();

    const btn = (label, type, value, pressed) =>
      `<button class="pub-filter" data-type="${type}" ${value != null ? `data-value="${escapeAttr(value)}"` : ""} aria-pressed="${pressed ? "true" : "false"}">${escapeHTML(label)}</button>`;

    el.innerHTML =
      `<span class="pub-controls-label">All</span>` +
      btn("Everything", "all", null, activeFilter.type === "all") +
      `<span class="pub-controls-label" style="margin-left:1.5rem">Year</span>` +
      years.map((y) => btn(String(y), "year", y, activeFilter.type === "year" && activeFilter.value === y)).join("") +
      `<span class="pub-controls-label" style="margin-left:1.5rem">Topic</span>` +
      tags.map((t) => btn(t, "tag", t, activeFilter.type === "tag" && activeFilter.value === t)).join("");

    el.querySelectorAll(".pub-filter").forEach((b) => {
      b.addEventListener("click", () => {
        const type = b.dataset.type;
        const raw = b.dataset.value;
        const value = type === "year" ? Number(raw) : raw || null;
        activeFilter = { type, value };
        renderPubControls();
        renderPubs();
      });
    });
  };

  const loadPubs = async () => {
    const data = await fetchJSON("publications.json");
    if (!data) {
      const c = document.getElementById("publications-list");
      if (c)
        c.innerHTML = `<p class="empty-state">Publications could not be loaded. If opening this file directly, serve with <code>python3 -m http.server</code>.</p>`;
      return;
    }
    PUBS = data.publications;
    renderPubControls();
    renderPubs();

    // click an inline #tag in a publication row to activate that tag filter
    const container = document.getElementById("publications-list");
    if (container && !container.__tagWired) {
      container.addEventListener("click", (e) => {
        const a = e.target.closest("a.tag[data-tag]");
        if (!a) return;
        e.preventDefault();
        activeFilter = { type: "tag", value: a.dataset.tag };
        renderPubControls();
        renderPubs();
        document.getElementById("publications").scrollIntoView({ block: "start" });
      });
      container.__tagWired = true;
    }
  };

  // ---------- patents ----------
  const renderPatents = async () => {
    const el = document.getElementById("patent-list");
    if (!el) return;
    const data = await fetchJSON("patents.json");
    if (!data) return;

    el.innerHTML = data.patents
      .map(
        (p) => `
      <li class="patent">
        <span class="patent-role ${p.role}">${p.role === "primary" ? "Primary" : "Secondary"}</span>
        <span class="patent-title">${escapeHTML(p.title)}</span>
        <span class="patent-date">${formatDate(p.date)}</span>
      </li>`
      )
      .join("");
  };

  // ---------- helpers ----------
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const formatDate = (iso) => {
    if (!iso) return "";
    const [y, m, d] = iso.split("-");
    if (!m) return y;
    if (!d) return `${MONTHS[Number(m) - 1]} ${y}`;
    return `${MONTHS[Number(m) - 1]} ${Number(d)}, ${y}`;
  };

  const escapeHTML = (s) =>
    String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const escapeAttr = (s) => escapeHTML(s);

  // ---------- init ----------
  renderNews();
  renderNewsArchive();
  loadPubs();
  renderPatents();

  // ---------- homepage writing section (latest 2 essays) ----------
  const writingList = document.getElementById("writing-list");
  if (writingList) {
    fetchJSON("posts.json").then((data) => {
      const posts = (data && data.posts) || [];
      if (!posts.length) {
        writingList.innerHTML = `<p class="empty-state">No essays yet.</p>`;
        return;
      }
      // Entries with `url` link straight to an external artifact
      // (slide deck, demo); otherwise they route into the essay view.
      const postHref = (p) => p.url
        ? (/^([a-z]+:|\/)/i.test(p.url) ? p.url : `posts/${p.url}`)
        : `posts/#${p.slug}`;
      writingList.innerHTML = posts
        .slice(0, 3)
        .map(
          (p) => `
        <a class="post-card" href="${escapeAttr(postHref(p))}">
          <span class="post-card-date">${formatDate(p.date)}${p.kicker ? ` · ${escapeHTML(p.kicker)}` : ""}</span>
          <h3 class="post-card-title">${escapeHTML(p.title)}</h3>
          <p class="post-card-excerpt">${escapeHTML(p.excerpt || "")}</p>
        </a>`
        )
        .join("");
    });
  }
})();
