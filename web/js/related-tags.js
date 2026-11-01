// Related Tags panel for Autocomplete++
//
// Shows tags that frequently appear together with the tag under the cursor
// (data comes from the Danbooru related-tags endpoint, fetched and cached by the
// ComfyUI server - see modules/related_tags.py). Clicking a suggestion inserts it
// next to the source tag using the same formatting rules as normal autocomplete.
//
// Open it with the configured trigger (Settings -> Autocomplete++ -> Related Tags,
// default Ctrl+Click on a tag) or with Ctrl+Shift+Space at the cursor.

import { settingValues, persistSetting, isComfyThemeLight, logDebug } from "./settings.js";
import { findTagBoundariesForWeight } from "./hotkeys.js";
import { getFixedCaretCoordinates } from "./caret-position.js";
import {
    getEffectiveFormattingSettings,
    buildUnderscoreExclusionSet,
    formatTagUnderscores,
    isKeepUnderscoresTextarea
} from "./auto-formatter.js";

const API_URL = "/autocomplete-plus-plus/related-tags";
const SORT_ORDERS = ["Cosine", "Jaccard", "Overlap", "Frequency"];
const SORT_LABELS = { Cosine: "Cos", Jaccard: "Jac", Overlap: "Ovl", Frequency: "Frq" };
const SORT_TITLES = {
    Cosine: "Sorted by cosine similarity (click to change)",
    Jaccard: "Sorted by Jaccard similarity (click to change)",
    Overlap: "Sorted by overlap coefficient (click to change)",
    Frequency: "Sorted by frequency (click to change)"
};
const SORT_FIELDS = { Cosine: "cosine", Jaccard: "jaccard", Overlap: "overlap", Frequency: "frequency" };
const CATEGORY_NAMES = { 0: "general", 1: "artist", 3: "copyright", 4: "character", 5: "meta" };
const MEMO_TTL_MS = 10 * 60 * 1000;

// ---------------------------------------------------------------------------
// Pure helpers (no DOM) - exported for testing
// ---------------------------------------------------------------------------

function stripEmphasis(text) {
    let t = text.replace(/^[(\[]+/, "");                 // leading unescaped "(" / "[" (an escaped "\(" starts with a backslash)
    t = t.replace(/:-?\d*\.?\d+[)\]]*$/, "");            // trailing weight, e.g. ":1.2)"
    t = t.replace(/(?<!\\)[)\]]+$/, "");                 // trailing unescaped ")" / "]"
    return t.trim();
}

/**
 * Convert text found in a prompt into the tag name the API expects
 * (lowercase, underscores) or null when it is not a plain tag
 * (LoRA, embedding, wildcard, weight, emoticon with brackets ...).
 */
export function extractApiTagName(raw) {
    let t = String(raw ?? "").trim();
    if (!t) return null;
    const NOT_A_TAG = /^<|^__|^(?:embedding|emb):/i;     // LoRA / wildcard / embedding syntax
    if (NOT_A_TAG.test(t)) return null;
    t = stripEmphasis(t);
    if (NOT_A_TAG.test(t)) return null;                  // e.g. "(embedding:name:1.0)" once the weight wrapper is gone
    t = t.replace(/^@/, "");                             // Anima artist prefix
    t = t.replace(/\\([()\[\]])/g, "$1");                // \( \) -> ( )
    t = t.trim().toLowerCase().replace(/\s+/g, "_").replace(/_+/g, "_");
    if (!t || t.length > 100) return null;
    if (/^[\d.:_-]+$/.test(t)) return null;              // bare numbers / weights
    if (/[{}|<>]/.test(t)) return null;
    return t;
}

/** Split a prompt into tag segments with their positions. */
export function splitPromptTags(text) {
    const result = [];
    if (!text) return result;
    const re = /[^,\n\r|{}]+/g;
    let m;
    while ((m = re.exec(text)) !== null) {
        const raw = m[0];
        const lead = raw.length - raw.trimStart().length;
        const trimmed = raw.trim();
        if (!trimmed) continue;
        const start = m.index + lead;
        const end = start + trimmed.length;
        result.push({ start, end, raw: trimmed, name: extractApiTagName(trimmed) });
    }
    return result;
}

/**
 * Work out the new prompt text after inserting `insertText` right after the tag
 * that ends at `anchorEnd`. Returns the new value, where the caret should go,
 * and where the inserted tag ends (anchor for the next insertion).
 */
export function computeInsertion(text, anchorEnd, insertText, autoComma = true) {
    const rest = text.slice(anchorEnd);
    const comma = rest.match(/^[ \t]*,[ \t]*/);
    let at;
    let inserted;
    if (comma) {
        at = anchorEnd + comma[0].length;
        const lead = /[ \t]$/.test(comma[0]) ? "" : " ";
        inserted = lead + insertText + (autoComma ? ", " : " ");
    } else {
        at = anchorEnd;
        const endsLine = /^[ \t]*\r?\n/.test(rest);
        inserted = ", " + insertText + (autoComma ? (endsLine ? "," : ", ") : "");
    }
    const tagEnd = at + inserted.indexOf(insertText) + insertText.length;
    return {
        value: text.slice(0, at) + inserted + text.slice(at),
        caret: at + inserted.length,
        tagEnd
    };
}

export function formatCount(n) {
    const v = Number(n) || 0;
    if (v >= 1e6) return (v / 1e6).toFixed(v >= 1e7 ? 0 : 1) + "M";
    if (v >= 1e3) return (v / 1e3).toFixed(v >= 1e4 ? 0 : 1) + "k";
    return String(v);
}

function clampMax(value) {
    const n = Math.round(Number(value));
    if (!Number.isFinite(n)) return 30;
    return Math.max(5, Math.min(100, n));
}

// ---------------------------------------------------------------------------
// Panel
// ---------------------------------------------------------------------------

export class RelatedTagsPanel {
    constructor(getEngine = () => null) {
        this.getEngine = getEngine;
        this.visible = false;
        this.pinned = false;
        this.textarea = null;
        this.tag = null;
        this.anchorEnd = 0;
        this.lastValue = "";
        this.items = [];
        this.selected = -1;
        this.requestId = 0;
        this.inserting = false;
        this.memo = new Map();
        this.buildDOM();
    }

    isEnabled() {
        return settingValues.enableRelatedTags !== false;
    }

    currentOrder() {
        return SORT_ORDERS.includes(settingValues.relatedTagsSort) ? settingValues.relatedTagsSort : "Cosine";
    }

    // ----- DOM -------------------------------------------------------------
    buildDOM() {
        const c = document.createElement("div");
        c.id = "tagcomplete-related-container";
        c.className = "acRelatedParent";

        const header = document.createElement("div");
        header.className = "acRelatedHeader";

        const title = document.createElement("span");
        title.className = "acRelatedTitle";
        const label = document.createElement("span");
        label.className = "acRelatedLabel";
        label.textContent = "Related to";
        this.tagLabel = document.createElement("span");
        this.tagLabel.className = "acRelatedTag";
        title.append(label, " ", this.tagLabel);

        const actions = document.createElement("span");
        actions.className = "acRelatedActions";
        this.sortBtn = this.makeButton("sort", "Cos", SORT_TITLES.Cosine);
        this.pinBtn = this.makeButton("pin", "📌", "Keep open while you edit (Esc closes)");
        this.closeBtn = this.makeButton("close", "✕", "Close (Esc)");
        actions.append(this.sortBtn, this.pinBtn, this.closeBtn);

        header.append(title, actions);

        this.statusEl = document.createElement("div");
        this.statusEl.className = "acRelatedStatus";

        this.body = document.createElement("div");
        this.body.className = "acRelatedBody";
        this.list = document.createElement("ul");
        this.list.className = "acRelatedList";
        this.body.appendChild(this.list);

        c.append(header, this.statusEl, this.body);
        document.body.appendChild(c);
        this.container = c;

        // Keep keyboard focus (and the caret) in the textarea while using the panel.
        const keepFocus = (e) => { e.preventDefault(); e.stopPropagation(); };
        c.addEventListener("mousedown", keepFocus);
        c.addEventListener("pointerdown", (e) => e.stopPropagation());
        c.addEventListener("mouseup", (e) => e.stopPropagation());
        c.addEventListener("wheel", (e) => e.stopPropagation(), { passive: true });

        c.addEventListener("click", (e) => {
            e.stopPropagation();
            const btn = e.target.closest("button[data-act]");
            if (btn) {
                this.onAction(btn.dataset.act);
                return;
            }
            const li = e.target.closest("li[data-index]");
            if (li) {
                const item = this.visibleItems[Number(li.dataset.index)];
                if (item) this.insertItem(item);
            }
        });

        document.addEventListener("mousedown", (e) => {
            if (!this.visible || this.pinned) return;
            if (this.container.contains(e.target) || e.target === this.textarea) return;
            this.hide();
        });
        window.addEventListener("resize", () => { if (this.visible) this.position(); });
    }

    makeButton(action, text, title) {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "acRelatedBtn";
        b.dataset.act = action;
        b.textContent = text;
        b.title = title;
        return b;
    }

    onAction(action) {
        if (action === "close") {
            this.pinned = false;
            this.hide();
        } else if (action === "pin") {
            this.pinned = !this.pinned;
            this.updatePinButton();
            this.position();
        } else if (action === "sort") {
            const next = SORT_ORDERS[(SORT_ORDERS.indexOf(this.currentOrder()) + 1) % SORT_ORDERS.length];
            settingValues.relatedTagsSort = next;
            persistSetting("RelatedTagsSort", next);
            this.updateSortButton();
            this.load();
        }
    }

    updatePinButton() {
        this.pinBtn.classList.toggle("active", this.pinned);
        this.pinBtn.setAttribute("aria-pressed", String(this.pinned));
    }

    updateSortButton() {
        const order = this.currentOrder();
        this.sortBtn.textContent = SORT_LABELS[order];
        this.sortBtn.title = SORT_TITLES[order];
    }

    // ----- Triggers --------------------------------------------------------
    /** Called from the textarea "click" listener. Returns true when the panel was opened. */
    handleClick(e, textarea) {
        if (!this.isEnabled()) return false;
        const mode = settingValues.relatedTagsTrigger;
        const plain = !e.ctrlKey && !e.metaKey && !e.shiftKey && !e.altKey;
        const hit = (mode === "Click" && plain) || (mode === "Ctrl+Click" && (e.ctrlKey || e.metaKey));
        if (hit && textarea.selectionStart === textarea.selectionEnd) {
            return this.showFor(textarea);
        }
        if (this.visible && !this.pinned && !hit) this.hide();
        return false;
    }

    handleDoubleClick(e, textarea) {
        if (!this.isEnabled() || settingValues.relatedTagsTrigger !== "Double-click") return false;
        return this.showFor(textarea);
    }

    /** Called from the capture-phase keydown listener. Returns true when the key was consumed. */
    handleKeyDown(e, textarea) {
        if (!this.isEnabled()) return false;

        if ((e.ctrlKey || e.metaKey) && e.shiftKey && !e.altKey && e.code === "Space") {
            if (this.visible && this.textarea === textarea) {
                this.hide();
            } else {
                this.showFor(textarea);
            }
            return true;
        }

        if (!this.visible) return false;
        switch (e.key) {
            case "Escape":
                this.pinned = false;
                this.hide();
                return true;
            case "ArrowDown":
            case "ArrowUp":
                if (this.pinned && this.selected < 0) return false;   // do not hijack caret movement while pinned
                this.moveSelection(e.key === "ArrowDown" ? 1 : -1);
                return true;
            case "Enter":
            case "Tab":
                if (this.selected >= 0 && !e.shiftKey && !e.ctrlKey && !e.altKey && this.visibleItems[this.selected]) {
                    this.insertItem(this.visibleItems[this.selected]);
                    return true;
                }
                return false;
            default:
                return false;
        }
    }

    handleInput() {
        if (!this.visible || this.inserting) return;
        if (this.pinned) {
            this.refreshUsedState();
        } else {
            this.hide();
        }
    }

    handleBlur() {
        setTimeout(() => {
            if (this.visible && !this.pinned && document.activeElement !== this.textarea) this.hide();
        }, 150);
    }

    // ----- Show / hide -----------------------------------------------------
    showFor(textarea) {
        const bounds = findTagBoundariesForWeight(textarea.value, textarea.selectionStart);
        if (!bounds || bounds.isLora || bounds.isWildcard) {
            this.hide();
            return false;
        }
        const name = extractApiTagName(bounds.text);
        if (!name) {
            this.hide();
            return false;
        }

        this.textarea = textarea;
        this.tag = name;
        this.anchorEnd = bounds.end;
        this.lastValue = textarea.value;
        this.selected = -1;

        const engine = this.getEngine();
        if (engine && engine.isVisible) engine.hide();

        this.tagLabel.textContent = name.replace(/_/g, " ");
        this.tagLabel.className = "acRelatedTag";
        this.updateSortButton();
        this.updatePinButton();
        if (isComfyThemeLight()) this.container.setAttribute("data-theme", "light");
        else this.container.removeAttribute("data-theme");

        this.visible = true;
        this.container.style.display = "block";
        this.position();
        this.load();
        return true;
    }

    hide() {
        this.requestId++;
        this.visible = false;
        this.container.style.display = "none";
    }

    position() {
        const ta = this.textarea;
        if (!ta || !this.visible) return;
        const c = this.container;
        c.style.display = "block";
        const rect = c.getBoundingClientRect();
        let left;
        let top;

        if (this.pinned) {
            const r = ta.getBoundingClientRect();
            left = r.right + 8;
            top = r.top;
            if (left + rect.width > window.innerWidth - 10) left = r.left - rect.width - 8;
        } else {
            const coords = getFixedCaretCoordinates(ta, ta.selectionStart);
            left = coords.left;
            top = coords.bottom + 6;
            if (top + rect.height > window.innerHeight - 10 && coords.top - rect.height - 6 > 10) {
                top = coords.top - rect.height - 6;
            }
        }
        left = Math.max(10, Math.min(left, window.innerWidth - rect.width - 10));
        top = Math.max(10, Math.min(top, window.innerHeight - rect.height - 10));
        c.style.left = `${left}px`;
        c.style.top = `${top}px`;
    }

    // ----- Data ------------------------------------------------------------
    async fetchRelated(tag, order) {
        const key = `${order}|${tag}`;
        const hit = this.memo.get(key);
        if (hit && Date.now() - hit.t < MEMO_TTL_MS) return hit.data;

        const ttl = Number.isFinite(Number(settingValues.relatedTagsCacheDays)) ? Number(settingValues.relatedTagsCacheDays) : 7;
        const source = ["Auto", "Local", "Danbooru"].includes(settingValues.relatedTagsSource) ? settingValues.relatedTagsSource : "Auto";
        const url = `${API_URL}?tag=${encodeURIComponent(tag)}&order=${encodeURIComponent(order)}&ttl_days=${encodeURIComponent(ttl)}&source=${source}`;
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (!data || !data.success) throw new Error((data && data.error) || "unknown error");
        if (data.pending) return data;
        this.memo.set(key, { t: Date.now(), data });
        if (this.memo.size > 100) this.memo.delete(this.memo.keys().next().value);
        return data;
    }

    async load() {
        const reqId = ++this.requestId;
        const tag = this.tag;
        const order = this.currentOrder();
        this.items = [];
        this.visibleItems = [];
        this.selected = -1;
        this.list.textContent = "";
        this.setStatus("Loading…");
        this.position();

        try {
            const data = await this.fetchRelated(tag, order);
            if (reqId !== this.requestId) return;
            if (data.pending) {
                this.setStatus(data.pending === "downloading"
                    ? "Downloading related tags data (one-time, runs in the background)…"
                    : "Preparing related tags data…");
                this.position();
                clearTimeout(this.pendingTimer);
                this.pendingTimer = setTimeout(() => {
                    if (reqId === this.requestId && this.visible) this.load();
                }, 3000);
                return;
            }
            this.items = Array.isArray(data.items) ? data.items : [];
            this.render(data);
        } catch (err) {
            if (reqId !== this.requestId) return;
            logDebug("[Autocomplete++] Related tags failed:", err);
            this.setStatus(`Could not load related tags: ${err.message}`, true);
            this.position();
        }
    }

    setStatus(text, isError = false) {
        this.statusEl.textContent = text;
        this.statusEl.classList.toggle("error", !!isError);
        this.statusEl.style.display = text ? "block" : "none";
    }

    // ----- Rendering -------------------------------------------------------
    usedNames() {
        const set = new Set();
        if (this.textarea) {
            for (const t of splitPromptTags(this.textarea.value)) if (t.name) set.add(t.name);
        }
        return set;
    }

    render(data) {
        const max = clampMax(settingValues.maxRelatedTags);
        const order = this.currentOrder();
        this.visibleItems = this.items.slice(0, max);
        this.list.textContent = "";

        const notes = [];
        if (data && data.stale) notes.push("Showing cached results (Danbooru could not be reached).");
        if (!this.visibleItems.length) notes.push(`No related tags found for "${this.tag.replace(/_/g, " ")}".`);
        this.setStatus(notes.join(" "), !!(data && data.stale) && !this.visibleItems.length);

        const used = this.usedNames();
        const field = SORT_FIELDS[order];
        const fmt = getEffectiveFormattingSettings();
        const showSpaces = !!fmt.replaceUnderscore;

        this.visibleItems.forEach((item, index) => {
            const li = document.createElement("li");
            li.className = "acRelatedItem";
            li.dataset.index = String(index);
            li.dataset.name = item.name;
            if (used.has(item.name)) li.classList.add("used");

            const name = document.createElement("span");
            name.className = "acRelatedName ac-rel-cat-" + (CATEGORY_NAMES[item.category] || "unknown");
            name.textContent = showSpaces ? item.name.replace(/_/g, " ") : item.name;
            li.title = item.name;

            const meta = document.createElement("span");
            meta.className = "acRelatedMeta";
            const score = Number(item[field]) || 0;
            if (score > 0) {
                const s = document.createElement("span");
                s.className = "acRelatedScore";
                s.textContent = `${Math.round(score * 100)}%`;
                meta.appendChild(s);
            }
            const count = document.createElement("span");
            count.className = "acRelatedCount";
            count.textContent = formatCount(item.post_count);
            meta.appendChild(count);

            li.append(name, meta);
            this.list.appendChild(li);
        });
        this.position();
    }

    refreshUsedState() {
        const used = this.usedNames();
        this.list.querySelectorAll("li[data-name]").forEach((li) => {
            li.classList.toggle("used", used.has(li.dataset.name));
        });
    }

    moveSelection(delta) {
        const n = this.visibleItems ? this.visibleItems.length : 0;
        if (!n) return;
        this.selected = this.selected < 0 ? (delta > 0 ? 0 : n - 1) : (this.selected + delta + n) % n;
        this.list.querySelectorAll("li").forEach((li, i) => li.classList.toggle("selected", i === this.selected));
        const el = this.list.children[this.selected];
        if (el && el.scrollIntoView) el.scrollIntoView({ block: "nearest" });
    }

    // ----- Insertion -------------------------------------------------------
    resolveAnchor(ta) {
        // The prompt changed since we last looked (pinned panel + typing): re-find the source tag.
        if (ta.value === this.lastValue) return this.anchorEnd;
        const own = splitPromptTags(ta.value).find((t) => t.name === this.tag);
        return own ? own.end : Math.min(ta.selectionStart, ta.value.length);
    }

    insertItem(item) {
        const ta = this.textarea;
        if (!ta || !item) return;
        const engine = this.getEngine();

        // Already in the prompt: select it instead of adding a duplicate.
        const existing = splitPromptTags(ta.value).find((t) => t.name === item.name);
        if (existing) {
            ta.focus();
            ta.setSelectionRange(existing.start, existing.end);
            return;
        }

        const fmt = { ...getEffectiveFormattingSettings() };
        if (isKeepUnderscoresTextarea(ta)) fmt.replaceUnderscore = false;

        let text = item.name;
        if (fmt.replaceUnderscore) {
            text = formatTagUnderscores(text, buildUnderscoreExclusionSet(fmt.formatKeepUnderscoresList));
        }
        if (fmt.escapeParentheses) {
            text = text
                .replace(/\\?\(/g, "\\(")
                .replace(/\\?\)/g, "\\)")
                .replace(/\\?\[/g, "\\[")
                .replace(/\\?\]/g, "\\]");
        }
        if (item.category === 1 && engine && typeof engine.isAnimaWorkflowActive === "function" && engine.isAnimaWorkflowActive()) {
            text = "@" + text;
        }

        const anchor = this.resolveAnchor(ta);
        const result = computeInsertion(ta.value, anchor, text, fmt.autoInsertComma !== false);

        this.inserting = true;
        if (engine) engine.isInserting = true;
        try {
            ta.value = result.value;
            ta.setSelectionRange(result.caret, result.caret);
            ta.dispatchEvent(new Event("input", { bubbles: true }));
            ta.dispatchEvent(new Event("change", { bubbles: true }));
        } finally {
            if (engine) engine.isInserting = false;
            this.inserting = false;
        }

        this.anchorEnd = result.tagEnd;
        this.lastValue = ta.value;
        if (engine && typeof engine.recordTagUsage === "function") engine.recordTagUsage(item.name);
        ta.focus();
        this.refreshUsedState();
        this.position();
    }
}
