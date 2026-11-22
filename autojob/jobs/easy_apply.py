"""
The form engine behind continuous applying on job boards (LinkedIn Easy
Apply, Indeed Apply): it fills an apply flow that is already open, one step at
a time, until the site confirms the application.

Each step:
    1. popups   anything covering the flow is classified by its text and closed
                the way a person would ("Not now", "Continue applying", X);
                a CAPTCHA stays on screen and the run waits for Boss to complete
                it in the Edge window, then carries on with the same job; a
                login wall or the site's daily limit pauses the run;
    2. read     every question in the apply window (or page): its label, kind
                (text, number, dropdown, radio, checkbox, typeahead, file...),
                options, current value and the site's error message;
    3. answer   screening_answers.BoardAnswerer: profile rules, then one
                evidence-checked model call; a required question it cannot
                answer truthfully ends the attempt (needs_user) and nothing is
                guessed;
    4. act      fill, then Next / Review / Submit. If the step does not move,
                the site's errors are read and fixed once, else the job fails.

Boss approves the whole run once (continuous_apply), so Submit is clicked here
without a per-job question - but only inside the board's own apply flow, on
the board's own hosts. Account creation, passwords, payments and CAPTCHAs are
never handled here.

All page access runs on the browser owner thread through sess.submit(); the
model calls run on the caller's thread so a slow answer never blocks Edge.
"""

from __future__ import annotations

import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from urllib.parse import urlparse

from autojob.integrations import browser_service as bs
from autojob.jobs.screening_answers import UPLOAD_CV, BoardAnswerer, real_options

logger = logging.getLogger(__name__)

# How long a run waits on screen for Boss to finish a CAPTCHA before it pauses.
HUMAN_CHECK_WAIT_S = float(os.getenv("HUMAN_CHECK_WAIT_S", "300"))


def wait_for_human_check(still_shown: Callable[[], bool], should_stop: Callable[[], bool] = lambda: False,
                         label: str = "the site", bring_front: Optional[Callable[[], None]] = None) -> bool:
    """
    A CAPTCHA is on screen: leave it there, bring Edge forward, tell Boss, and
    wait while Boss completes it in that window. True once it is gone (the
    caller carries on with the same job), False on timeout or stop.
    Veronica never solves it herself.
    """
    if bring_front:
        try:
            bring_front()
        except Exception:
            pass
    msg = f"{label} is showing a 'verify you are human' check - complete it in the Edge window and I'll continue."
    logger.info("Waiting up to %ds for Boss: %s", HUMAN_CHECK_WAIT_S, msg)
    try:
        from autojob.core.veronica_ui_bridge import emit

        emit({"type": "log", "label": msg, "tone": "warning"})
        emit({"type": "agent", "id": "naukri", "status": "waiting", "action": "waiting for you to solve the CAPTCHA"})
    except Exception:
        pass
    try:
        import winsound

        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
    except Exception:
        pass
    deadline = time.time() + HUMAN_CHECK_WAIT_S
    while time.time() < deadline:
        if should_stop():
            return False
        time.sleep(2.5)
        try:
            if not still_shown():
                time.sleep(2.0)  # the page usually reloads once the check passes
                if not still_shown():
                    logger.info("Human check on %s is done; continuing", label)
                    try:
                        from autojob.core.veronica_ui_bridge import emit

                        emit({"type": "agent", "id": "naukri", "status": "active", "action": "CAPTCHA done, continuing"})
                    except Exception:
                        pass
                    return True
        except Exception as exc:
            logger.debug("Human-check probe failed: %s", exc)
    logger.info("Human check on %s still there after %ds", label, HUMAN_CHECK_WAIT_S)
    return False


# ---------------------------------------------------------------------------
# Page scripts
# ---------------------------------------------------------------------------

# Stamps the apply window with data-vro-root and describes it. `roots` is a
# list of [css selector, text regex or ""]; the last visible match wins (the
# top-most dialog).
_JS_FIND_ROOT = r"""
(roots) => {
  document.querySelectorAll('[data-vro-root]').forEach(e => e.removeAttribute('data-vro-root'));
  const shown = el => {
    const r = el.getBoundingClientRect(), st = getComputedStyle(el);
    return r.width > 40 && r.height > 40 && st.visibility !== 'hidden' && st.display !== 'none' && st.opacity !== '0';
  };
  for (const [sel, text] of roots) {
    let list;
    try { list = Array.from(document.querySelectorAll(sel)).filter(shown); } catch (e) { continue; }
    const re = text ? new RegExp(text, 'i') : null;
    for (let i = list.length - 1; i >= 0; i--) {
      const el = list[i];
      const t = (el.innerText || '');
      if (re && !re.test(t.slice(0, 5000))) continue;
      el.setAttribute('data-vro-root', '1');
      const h = el.querySelector('h1, h2, h3, [role=heading]');
      const p = el.querySelector('[role=progressbar], progress');
      return {
        heading: h ? (h.innerText || '').trim().replace(/\s+/g, ' ').slice(0, 150) : '',
        progress: p ? String(p.getAttribute('aria-valuenow') || p.getAttribute('value') || p.value || '') : '',
        text: t.replace(/\s+/g, ' ').slice(0, 3000),
      };
    }
  }
  return null;
}
"""

# Every question inside the stamped root. Radios (and checkbox groups) become
# one field each with their options; inputs are stamped data-vro-fid="<id>"
# and options data-vro-fid="<id>:<n>".
_JS_FIELDS = r"""
() => {
  const root = document.querySelector('[data-vro-root]') || document.body;
  document.querySelectorAll('[data-vro-fid]').forEach(e => e.removeAttribute('data-vro-fid'));
  const clean = s => (s || '').replace(/\s+/g, ' ').trim();
  const dedupe = s => { const m = s.match(/^(.{4,}?)\s*\1$/); return m ? m[1] : s; };
  const shown = el => {
    if (!el) return false;
    const r = el.getBoundingClientRect(), st = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && st.visibility !== 'hidden' && st.display !== 'none';
  };
  const byId = id => { try { return document.getElementById(id); } catch (e) { return null; } };
  const ownLabel = el => {
    const lb = el.getAttribute('aria-labelledby');
    if (lb) {
      const t = lb.split(/\s+/).map(i => clean((byId(i) || {}).innerText)).filter(Boolean).join(' ');
      if (t) return t;
    }
    if (el.labels && el.labels.length) {
      const t = clean(Array.from(el.labels).map(l => l.innerText).join(' '));
      if (t) return t;
    }
    if (el.id) {
      const l = root.querySelector('label[for="' + CSS.escape(el.id) + '"]');
      if (l && clean(l.innerText)) return clean(l.innerText);
    }
    const a = el.getAttribute('aria-label');
    if (a) return clean(a);
    const wrap = el.closest('label');
    if (wrap && clean(wrap.innerText)) return clean(wrap.innerText);
    return '';
  };
  const CONT = '[data-test-form-element], .fb-dash-form-element, .jobs-easy-apply-form-element, '
    + '.ia-Questions-item, [class*="Questions-item"], fieldset, [role=radiogroup], [role=group], '
    + '[class*="question" i], [class*="form-group" i], [class*="FormField"], [class*="formField"], '
    + '[class*="form-field" i], [class*="form-element" i]';
  const boxOf = el => (el.parentElement && el.parentElement.closest(CONT)) || el.parentElement;
  const questionIn = (box, exclude) => {
    if (!box) return '';
    const lg = box.querySelector('legend, [data-test-form-builder-radio-button-form-component__title], '
      + '[class*="__title"], [class*="label" i]:not(label):not(input), h3, h4');
    let t = lg ? clean(lg.innerText) : '';
    if (!t || exclude.includes(t)) {
      const lines = (box.innerText || '').split('\n').map(clean).filter(Boolean);
      t = lines.find(l => !exclude.includes(l) && l.length > 1) || '';
    }
    return dedupe(t).slice(0, 300);
  };
  const errorNear = (el, box) => {
    for (const id of (el.getAttribute('aria-describedby') || '').split(/\s+/).filter(Boolean)) {
      const n = byId(id);
      if (n && shown(n) && /error|invalid|required|please|enter|must|select|valid|answer/i.test(n.innerText || '')) {
        return clean(n.innerText).slice(0, 200);
      }
    }
    if (box) {
      for (const n of box.querySelectorAll('[role=alert], .artdeco-inline-feedback--error, [class*="error" i], '
                                         + '[id*="error" i], [class*="invalid" i]')) {
        if (n.matches('input, select, textarea, label')) continue;
        if (shown(n) && clean(n.innerText)) return clean(n.innerText).slice(0, 200);
      }
    }
    return '';
  };
  const requiredOf = (el, q, box) => {
    if (/\(optional\)|\boptional\s*$/i.test(q)) return false;
    return !!(el.required || el.getAttribute('aria-required') === 'true' || /\*/.test(q)
      || (box && box.querySelector('[class*="required" i], abbr[title*="required" i]')));
  };
  const kindOf = el => {
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    if (tag === 'select') return 'select';
    if (type === 'radio' || el.getAttribute('role') === 'radio') return 'radio';
    if (type === 'checkbox' || el.getAttribute('role') === 'checkbox') return 'checkbox';
    if (type === 'file') return 'file';
    if (tag === 'textarea' || el.isContentEditable) return 'textarea';
    if (el.getAttribute('role') === 'combobox' || el.getAttribute('aria-autocomplete') || el.hasAttribute('list')) return 'combobox';
    if (type === 'number' || /numeric/i.test(el.id || '') || /numeric|decimal/.test(el.getAttribute('inputmode') || '')) return 'number';
    if (type === 'date' || /dd\s*\/\s*mm|mm\s*\/\s*dd|yyyy/i.test(el.getAttribute('placeholder') || '')) return 'date';
    if (type === 'email') return 'email';
    if (type === 'tel') return 'tel';
    return 'text';
  };
  const PLACEHOLDER = /^\W*(select( an)?( option)?|choose( one| an option)?|please select|--+)?\W*$/i;

  const out = [], groups = new Map(), boxIds = new Map();
  const els = root.querySelectorAll('input, select, textarea, [contenteditable="true"], [role=combobox], [role=radio], [role=checkbox]');
  for (const el of els) {
    if (el.closest('[data-vro-fid]') && !el.matches('input[type=radio], input[type=checkbox]')) continue;
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    if (tag === 'input' && ['hidden', 'submit', 'button', 'image', 'reset', 'password', 'search'].includes(type)) continue;
    if (el.disabled || el.getAttribute('aria-disabled') === 'true' || el.readOnly) continue;
    // The site's own search box / header is never part of the application.
    if (el.closest('[role=search], form[action*="search" i], header, nav, [class*="global-nav" i]')) continue;
    const kind = kindOf(el);
    const box = boxOf(el);
    const lab = el.labels && el.labels[0];
    const visible = shown(el) || (['radio', 'checkbox', 'file'].includes(kind) && (shown(lab) || shown(box)));
    if (!visible) continue;

    if (kind === 'radio' || kind === 'checkbox') {
      const gbox = el.closest('fieldset, [role=radiogroup], [role=group], [data-test-form-element], '
        + '.fb-dash-form-element, .ia-Questions-item, [class*="Questions-item"]') || box;
      let key = kind === 'radio' && el.name ? 'r:' + el.name : null;
      if (!key) {
        if (!boxIds.has(gbox)) boxIds.set(gbox, boxIds.size);
        key = kind + ':' + boxIds.get(gbox);
      }
      const optText = ownLabel(el) || clean(el.value) || clean(el.innerText);
      let g = groups.get(key);
      if (!g) {
        const many = gbox ? gbox.querySelectorAll('input[type=' + kind + '], [role=' + kind + ']').length > 1 : false;
        g = {id: String(out.length), kind: kind === 'radio' ? 'radio' : (many ? 'checkboxes' : 'checkbox'),
             options: [], checked: [], _box: gbox, _el: el};
        groups.set(key, g);
        out.push(g);
      }
      el.setAttribute('data-vro-fid', g.id + ':' + g.options.length);
      g.options.push(optText);
      if (el.checked || el.getAttribute('aria-checked') === 'true') g.checked.push(optText);
      continue;
    }

    const id = String(out.length);
    el.setAttribute('data-vro-fid', id);
    const q = dedupe(ownLabel(el) || questionIn(box, []) || clean(el.getAttribute('placeholder')));
    let value = '', options = [];
    if (kind === 'select') {
      options = Array.from(el.options).map(o => clean(o.text));
      const cur = el.options[el.selectedIndex];
      value = cur && el.value !== '' && !PLACEHOLDER.test(clean(cur.text)) ? clean(cur.text) : '';
    } else if (el.isContentEditable) {
      value = clean(el.innerText);
    } else if (kind === 'file') {
      value = el.files && el.files.length ? el.files[0].name : '';
    } else {
      value = clean(el.value || (el.getAttribute('role') === 'combobox' && tag !== 'input' ? el.innerText : ''));
    }
    out.push({id, kind, question: q, options, value, required: requiredOf(el, q, box), error: errorNear(el, box)});
  }

  for (const g of groups.values()) {
    const q = g.kind === 'checkbox' ? (g.options[0] || questionIn(g._box, [])) : questionIn(g._box, g.options);
    g.question = q;
    g.value = g.kind === 'checkbox' ? (g.checked.length ? 'true' : '') : g.checked.join(' | ');
    g.required = requiredOf(g._el, questionIn(g._box, g.options) + ' ' + q, g._box);
    g.error = errorNear(g._el, g._box);
    delete g._box; delete g._el;
  }
  return out.slice(0, 60);
}
"""

# Visible buttons inside the root, stamped data-vro-btn.
_JS_BUTTONS = r"""
() => {
  const root = document.querySelector('[data-vro-root]') || document.body;
  document.querySelectorAll('[data-vro-btn]').forEach(e => e.removeAttribute('data-vro-btn'));
  const clean = s => (s || '').replace(/\s+/g, ' ').trim();
  const out = [];
  for (const b of root.querySelectorAll('button, [role=button], input[type=submit], input[type=button]')) {
    const r = b.getBoundingClientRect(), st = getComputedStyle(b);
    if (r.width < 1 || r.height < 1 || st.visibility === 'hidden' || st.display === 'none') continue;
    const text = clean(b.innerText || b.value || '');
    const aria = clean(b.getAttribute('aria-label') || b.getAttribute('title') || '');
    if (!text && !aria) continue;
    const disabled = !!(b.disabled || b.getAttribute('aria-disabled') === 'true');
    b.setAttribute('data-vro-btn', String(out.length));
    out.push({i: out.length, text: text.slice(0, 80), aria: aria.slice(0, 120), disabled});
    if (out.length >= 80) break;
  }
  return out;
}
"""

# Dialogs, overlays and banners on top of the page that are NOT the apply
# window. Each is stamped data-vro-pop=<n>, its buttons data-vro-pbtn="<n>:<i>".
# `roots` are the apply-window candidates: anything matching them is never
# reported, so the sweep can never close the application itself. hasForm
# marks popups with inputs; those are only closed when a rule knows them.
_JS_POPUPS = r"""
(roots) => {
  document.querySelectorAll('[data-vro-pop]').forEach(e => e.removeAttribute('data-vro-pop'));
  document.querySelectorAll('[data-vro-pbtn]').forEach(e => e.removeAttribute('data-vro-pbtn'));
  const root = document.querySelector('[data-vro-root]');
  const clean = s => (s || '').replace(/\s+/g, ' ').trim();
  const isApplyWindow = el => roots.some(([sel, text]) => {
    try {
      const m = el.matches(sel) || !!el.querySelector(sel);
      return m && (!text || new RegExp(text, 'i').test((el.innerText || '').slice(0, 5000)));
    } catch (e) { return false; }
  });
  const SEL = '[role=dialog], [role=alertdialog], [aria-modal="true"], .artdeco-modal, [class*="modal" i], '
    + '[class*="popup" i], [class*="popover" i], [class*="drawer" i], [class*="lightbox" i], [class*="toast" i], '
    + '[class*="interstitial" i], [id*="modal" i], [id*="popup" i]';
  const out = [];
  for (const el of document.querySelectorAll(SEL)) {
    if (out.some(p => p.el.contains(el))) continue;
    if (root && el.contains(root)) continue;
    // Inside the apply window only a real dialog counts (its header/body parts match "modal" too).
    if (root && root.contains(el) && !el.matches('[role=dialog], [role=alertdialog], [aria-modal="true"]')) continue;
    if (el.matches('[role=listbox], [role=menu], [role=tooltip]') || el.closest('[role=listbox], [role=menu]')) continue;
    const r = el.getBoundingClientRect(), st = getComputedStyle(el);
    if (r.width < 120 || r.height < 50 || st.visibility === 'hidden' || st.display === 'none' || st.opacity === '0') continue;
    if (r.bottom < 0 || r.top > innerHeight || r.right < 0 || r.left > innerWidth) continue;
    const text = clean(el.innerText).slice(0, 700);
    if (!text || isApplyWindow(el)) continue;
    const n = out.length;
    el.setAttribute('data-vro-pop', String(n));
    const buttons = [];
    for (const b of el.querySelectorAll('button, [role=button], a, input[type=button], input[type=submit]')) {
      const br = b.getBoundingClientRect();
      if (br.width < 1 || br.height < 1) continue;
      b.setAttribute('data-vro-pbtn', n + ':' + buttons.length);
      buttons.push({text: clean(b.innerText || b.value || '').slice(0, 60),
                    aria: clean(b.getAttribute('aria-label') || b.getAttribute('title') || '').slice(0, 80),
                    href: b.tagName === 'A' ? (b.getAttribute('href') || '') : ''});
      if (buttons.length >= 25) break;
    }
    const hasForm = Array.from(el.querySelectorAll('input:not([type=hidden]), select, textarea'))
      .some(i => { const b = i.getBoundingClientRect(); return b.width > 0 && b.height > 0; });
    const blocking = r.width > innerWidth * 0.35 && r.height > innerHeight * 0.25
      || st.position === 'fixed' && r.width > innerWidth * 0.6;
    out.push({el, n, text, buttons, blocking, hasForm});
  }
  return out.map(({el, ...rest}) => rest);
}
"""

# A human check anywhere on the page: those are Boss's to solve, never ours.
_JS_CAPTCHA = r"""
() => {
  for (const f of document.querySelectorAll('iframe')) {
    const src = (f.getAttribute('src') || '') + ' ' + (f.getAttribute('title') || '');
    const r = f.getBoundingClientRect();
    if (r.width > 30 && r.height > 30 && /recaptcha|hcaptcha|challenges\.cloudflare|turnstile|arkoselabs|funcaptcha|captcha/i.test(src)) return true;
  }
  const t = (document.body ? document.body.innerText : '').slice(0, 4000);
  return /verify (that )?you('| a)re (a )?human|are you a robot|press (and|&) hold|confirm you('| a)re not a robot|additional verification required|let'?s confirm you are human/i.test(t);
}
"""


# LinkedIn (2025+) renders the Easy Apply window inside a shadow root
# (#interop-outlet). document.querySelector() cannot see into it, so every run
# reported "the apply window closed" while the window was open. These helpers
# search the page and every open shadow root; _deep() rewrites the scripts
# above to use them (a page without shadow DOM behaves exactly as before).
_JS_DEEP = r"""
  const __roots = () => {
    const out = [document];
    const walk = r => { for (const el of r.querySelectorAll('*')) if (el.shadowRoot) { out.push(el.shadowRoot); walk(el.shadowRoot); } };
    walk(document);
    return out;
  };
  const __qsa = sel => { const out = []; for (const r of __roots()) out.push(...r.querySelectorAll(sel)); return out; };
  const __qs = sel => __qsa(sel)[0] || null;
  const __byId = (id, near) => {
    const rn = near && near.getRootNode ? near.getRootNode() : null;
    return (rn && rn.getElementById && rn.getElementById(id)) || document.getElementById(id);
  };
  const __contains = (a, b) => {  // a contains b, across shadow boundaries
    for (let n = b; n; n = n.parentNode || n.host) if (n === a) return true;
    return false;
  };
  const __pageText = () => __roots().map(r => r === document ? (document.body ? document.body.innerText : '')
    : Array.from(r.children).map(c => c.innerText || '').join('\n')).join('\n');
"""


def _deep(js: str) -> str:
    """Make a page script see inside open shadow roots (see _JS_DEEP)."""
    head, sep, body = js.partition("=> {")
    body = (body.replace("document.querySelectorAll(", "__qsa(")
                .replace("document.querySelector(", "__qs(")
                .replace("document.getElementById(id)", "__byId(id, root)")
                .replace("(document.body ? document.body.innerText : '')", "__pageText()")
                .replace("p.el.contains(el)", "__contains(p.el, el)")
                .replace("el.contains(root)", "__contains(el, root)")
                .replace("root.contains(el)", "__contains(root, el)"))
    return head + sep + _JS_DEEP + body


_JS_FIND_ROOT, _JS_FIELDS, _JS_BUTTONS, _JS_POPUPS, _JS_CAPTCHA = (
    _deep(_JS_FIND_ROOT), _deep(_JS_FIELDS), _deep(_JS_BUTTONS), _deep(_JS_POPUPS), _deep(_JS_CAPTCHA))
_JS_PAGE_TEXT = _deep("() => { return (document.body ? document.body.innerText : '').slice(0, 6000); }")

# What is on screen when no apply window is found - written to the log so a
# site change can be diagnosed from one failed run.
_JS_DIAG = _deep(r"""() => {
  const vis = el => { const r = el.getBoundingClientRect(); return r.width > 40 && r.height > 40; };
  const dialogs = __qsa('[role=dialog], [aria-modal="true"], .artdeco-modal').filter(vis)
    .map(d => (d.className || d.getAttribute('role') || '').toString().slice(0, 60) + ': '
              + (d.innerText || '').replace(/\s+/g, ' ').slice(0, 80));
  const hosts = __roots().slice(1).map(r => r.host.tagName.toLowerCase() + (r.host.id ? '#' + r.host.id : ''));
  return {dialogs: dialogs.slice(0, 5), shadow_hosts: hosts.slice(0, 10)};
}""")


# ---------------------------------------------------------------------------
# Classifying buttons and popups (pure, unit tested)
# ---------------------------------------------------------------------------

_BTN_RULES = [
    ("submit", re.compile(r"^submit\b(?!.*\b(later|draft)\b)|^send (my |your |the )?application\b", re.I)),
    ("review", re.compile(r"^review\b", re.I)),
    ("next", re.compile(r"^(next|continue|save (and|&) continue|proceed|next step)\b", re.I)),
]


def classify_button(button: dict) -> Optional[str]:
    """'submit' | 'review' | 'next' | None for one button of the apply window."""
    labels = [str(button.get("aria") or ""), str(button.get("text") or "")]
    for kind, rule in _BTN_RULES:
        if any(rule.search(l.strip()) for l in labels if l.strip()):
            return kind
    return None


def pick_button(buttons: list[dict]) -> tuple[Optional[str], Optional[dict]]:
    """The button that moves the flow forward: Submit, else Review, else Next."""
    found: dict[str, dict] = {}
    for b in buttons:
        kind = classify_button(b)
        if kind and kind not in found and not b.get("disabled"):
            found[kind] = b
    for kind in ("submit", "review", "next"):
        if kind in found:
            return kind, found[kind]
    return None, None


@dataclass
class PopupRule:
    name: str
    text: re.Pattern
    buttons: tuple[str, ...] = ()  # preferred button labels, in order


POPUP_RULES = [
    PopupRule("captcha", re.compile(r"verify (that )?you('| a)re (a )?human|are you a robot|security (check|verification)|"
                                    r"captcha|press (and|&) hold|unusual activity", re.I)),
    PopupRule("login", re.compile(r"\b(sign in|log ?in) to (apply|continue)|join (linkedin )?to apply|session (has )?"
                                  r"expired|please (sign|log) ?in again", re.I)),
    PopupRule("limit", re.compile(r"(reached|exceeded|hit).{0,60}\blimit\b|daily (application |submission )?limit|"
                                  r"limit daily submissions|too many applications", re.I)),
    PopupRule("safety", re.compile(r"job search safety|safety reminder|be careful|stay safe|scam", re.I),
              ("continue applying", "continue", "i understand", "got it", "ok")),
    PopupRule("sent", re.compile(r"application (was |has been )?(sent|submitted)|your application was sent|"
                                 r"applied successfully|successfully applied|you applied", re.I),
              ("done", "close", "dismiss", "not now", "ok", "got it", "no thanks")),
    PopupRule("save_application", re.compile(r"save (this|your) application\??|discard (this )?application", re.I),
              ("discard",)),
    PopupRule("leave", re.compile(r"(are you sure|do you want) (you want )?to (leave|exit|quit)|unsaved changes|"
                                  r"leave (this )?page", re.I),
              ("stay", "cancel", "no", "keep applying", "continue applying")),
    PopupRule("promo", re.compile(r"premium|try (it )?(for )?free|upgrade|get the app|download (the |our )?app|open (in|the) "
                                  r"app|turn on (notifications|job alerts)|notifications?|rate (your|us)|how was your|"
                                  r"feedback|survey|become a pro|\bneo\b|what'?s new|new feature|share your profile|"
                                  r"open to work|add (a |more )?skills?|update your (profile|resume)|complete your "
                                  r"profile|resume score|boost|recommended for you|similar jobs|job alert|"
                                  r"cookie|we use cookies", re.I),
              ("not now", "no thanks", "no, thanks", "maybe later", "skip", "dismiss", "close", "cancel", "later",
               "got it", "reject", "decline", "ok")),
]

_CLOSE_LABEL = re.compile(r"^(×|✕|x|close|dismiss|close (this )?(dialog|modal|popup|window)|cancel)$", re.I)


def classify_popup(popup: dict) -> Optional[PopupRule]:
    text = str(popup.get("text") or "")
    for rule in POPUP_RULES:
        if rule.text.search(text):
            return rule
    return None


def popup_button(popup: dict, rule: Optional[PopupRule]) -> Optional[int]:
    """Index of the button to press for this popup (None = nothing safe to press)."""
    buttons = popup.get("buttons") or []
    labels = [(" ".join(str(b.get("text") or "").split()).lower(), " ".join(str(b.get("aria") or "").split()).lower())
              for b in buttons]
    if rule:
        for wanted in rule.buttons:
            for i, (t, a) in enumerate(labels):
                if any(l == wanted or l.startswith(wanted + " ") for l in (t, a)) and \
                        not buttons[i].get("href", "").startswith("http"):
                    return i
    for i, (t, a) in enumerate(labels):
        if _CLOSE_LABEL.match(t) or _CLOSE_LABEL.match(a) or re.match(r"^(dismiss|close)\b", a):
            return i
    return None


# ---------------------------------------------------------------------------
# One apply flow
# ---------------------------------------------------------------------------

@dataclass
class FlowSpec:
    """What the engine needs to know about one board's apply flow."""
    portal: str
    roots: list[list[str]]                 # [css, text regex] candidates for the apply window
    success: re.Pattern                    # page / dialog text once the application went through
    hosts: tuple[str, ...]                 # Submit is only ever clicked on these hosts
    success_url: Optional[re.Pattern] = None
    root_url: Optional[re.Pattern] = None  # the apply window only counts on pages / frames whose URL matches
    max_steps: int = 14
    abandon: Optional[Callable[[Any], None]] = None  # page -> None: leave a half-done application
    extra: dict = field(default_factory=dict)


class FlowResult(dict):
    """{"status": applied | needs_user | failed | paused | limit | stopped | dry_run, ...}"""


def _host_ok(url: str, hosts: tuple[str, ...]) -> bool:
    host = (urlparse(url or "").hostname or "").lower()
    return any(host == h or host.endswith("." + h) for h in hosts)


def _pause(ms: tuple[int, int] = (350, 900)) -> None:
    time.sleep(random.uniform(ms[0], ms[1]) / 1000)


class ApplyFlow:
    """Fills one open apply flow on `sess` (browser_service.BrowserSession)."""

    def __init__(self, sess, spec: FlowSpec, answerer: BoardAnswerer,
                 should_stop: Callable[[], bool] = lambda: False, dry_run: bool = False,
                 on_popup: Optional[Callable[[str], None]] = None):
        self.sess = sess
        self.spec = spec
        self.answerer = answerer
        self.should_stop = should_stop
        self.dry_run = dry_run
        self.on_popup = on_popup
        self.filled: list[str] = []
        self._frame = None

    # -- page access (owner thread) ---------------------------------------

    def _frames(self) -> list:
        page = self.sess._page
        frames = [page.main_frame]
        for fr in page.frames[1:]:
            try:
                if not fr.is_detached() and re.search(r"apply|indeed|linkedin|naukri", fr.url or "", re.I):
                    frames.append(fr)
            except Exception:
                continue
        return frames

    def find_root(self) -> Optional[dict]:
        def _find():
            for fr in self._frames():
                if self.spec.root_url and not self.spec.root_url.search(fr.url or ""):
                    continue
                try:
                    info = fr.evaluate(_JS_FIND_ROOT, self.spec.roots)
                except Exception:
                    continue
                if info:
                    self._frame = fr
                    return {**info, "url": self.sess._page.url}
            self._frame = None
            return None
        return self.sess.submit(_find)

    def _in_root(self, script: str, arg: Any = None):
        def _run():
            fr = self._frame or self.sess._page.main_frame
            return fr.evaluate(script, arg) if arg is not None else fr.evaluate(script)
        return self.sess.submit(_run)

    def fields(self) -> list[dict]:
        return self._in_root(_JS_FIELDS) or []

    def buttons(self) -> list[dict]:
        return self._in_root(_JS_BUTTONS) or []

    def page_text(self) -> str:
        def _t():
            page = self.sess._page
            try:
                body = page.evaluate(_JS_PAGE_TEXT) or ""
            except Exception:
                body = ""
            return body + "\n" + (page.url or "")
        return self.sess.submit(_t)

    def captcha(self) -> bool:
        def _c():
            for fr in [self.sess._page.main_frame, *self._frames()[1:]]:
                try:
                    if fr.evaluate(_JS_CAPTCHA):
                        return True
                except Exception:
                    continue
            return False
        return bool(self.sess.submit(_c))

    def _human_check_shown(self) -> bool:
        if self.captcha():
            return True
        popups = self.sess.submit(lambda: self.sess._page.evaluate(_JS_POPUPS, self.spec.roots)) or []
        return any(getattr(classify_popup(p), "name", None) == "captcha" for p in popups)

    def wait_human(self) -> bool:
        """CAPTCHA in the flow: wait on screen for Boss to finish it. True = carry on with this job."""
        return wait_for_human_check(self._human_check_shown, self.should_stop, self.spec.portal.title(),
                                    bring_front=lambda: self.sess.submit(lambda: self.sess._page.bring_to_front()))

    def handle_popups(self) -> Optional[str]:
        """
        Close what covers the flow. Returns the name of a popup the caller must
        act on (captcha, login, limit, sent), else None.
        """
        if self.captcha():
            return "captcha"
        for _ in range(4):
            popups = self.sess.submit(lambda: self.sess._page.evaluate(_JS_POPUPS, self.spec.roots)) or []
            acted = False
            for p in popups:
                rule = classify_popup(p)
                name = rule.name if rule else None
                if name in ("captcha", "login", "limit"):
                    return name
                if name is None and (not p.get("blocking") or p.get("hasForm")):
                    # A page section that only looks like a popup, or an unknown
                    # form: never closed blindly (it could be the application).
                    continue
                idx = popup_button(p, rule)
                if idx is None:
                    if p.get("blocking"):
                        logger.info("[%s] popup with nothing safe to press: %s", self.spec.portal, p["text"][:120])
                    continue
                label = (p["buttons"][idx].get("text") or p["buttons"][idx].get("aria") or "")[:40]
                logger.info("[%s] popup '%s' (%s): pressing '%s'", self.spec.portal, p["text"][:80], name, label)
                try:
                    self.sess.submit(lambda n=p["n"], i=idx: self.sess._page.locator(
                        f'[data-vro-pbtn="{n}:{i}"]').first.click(timeout=5000))
                except Exception as exc:
                    logger.debug("Popup button click failed: %s", exc)
                    continue
                if self.on_popup:
                    self.on_popup(name or "popup")
                _pause((500, 1100))
                if name == "sent":
                    return "sent"
                acted = True
                break
            if not acted:
                return None
        return None

    # -- filling ----------------------------------------------------------

    def _loc(self, key: str):
        fr = self._frame or self.sess._page.main_frame
        return fr.locator(f'[data-vro-fid="{key}"]').first

    def _fill(self, f: dict, value: Any) -> Optional[str]:
        """Put one answer into the page. Returns an error message, or None."""
        kind = f.get("kind")
        fid = str(f["id"])

        def _do():
            if kind in ("radio", "checkboxes"):
                wanted = value if isinstance(value, list) else [value]
                for w in wanted:
                    idx = next((i for i, o in enumerate(f["options"]) if o == w), None)
                    if idx is None:
                        raise ValueError(f"no option '{w}'")
                    el = self._loc(f"{fid}:{idx}")
                    try:
                        el.check(force=True, timeout=5000)
                    except Exception:
                        el.evaluate("e => (e.labels && e.labels[0] || e).click()")
                    if not el.is_checked():
                        el.evaluate("e => (e.labels && e.labels[0] || e).click()")
                return None
            if kind == "checkbox":
                el = self._loc(f"{fid}:0")
                want = bool(value)
                if el.is_checked() != want:
                    try:
                        el.set_checked(want, force=True, timeout=5000)
                    except Exception:
                        el.evaluate("e => (e.labels && e.labels[0] || e).click()")
                return None
            el = self._loc(fid)
            el.scroll_into_view_if_needed(timeout=5000)
            if kind == "file":
                if value == UPLOAD_CV and self.answerer.cv_file:
                    el.set_input_files(str(self.answerer.cv_file))
                return None
            if kind == "select":
                bs._choose_option(self.sess, el, str(value))
                return None
            if kind == "combobox":
                try:
                    bs._choose_option(self.sess, el, str(value))
                    return None
                except ValueError:
                    # A typeahead with no suggestion for it: keep the typed text.
                    if el.evaluate("e => e.tagName") != "INPUT":
                        raise
                    self.sess._page.keyboard.press("Escape")
            bs._fast_fill(self.sess, el, str(value))
            try:
                el.press("Tab", timeout=2000)  # sites validate on blur
            except Exception:
                pass
            return None

        try:
            self.sess.submit(_do)
            self.filled.append(f"{str(f.get('question'))[:60]} = {str(value)[:40]}")
            return None
        except Exception as exc:
            return bs._short_error({"error": str(exc)})

    @staticmethod
    def needs_answer(f: dict) -> bool:
        kind = f.get("kind")
        has = bool(str(f.get("value") or "").strip())
        if f.get("error"):
            return True
        if kind == "checkbox":
            return True  # rules decide (consent on, follow/newsletter off); others are left alone
        if kind == "file":
            return bool(f.get("required")) and not has
        return not has

    def _answer_and_fill(self, fields: list[dict]) -> list[dict]:
        """Answer what needs it and fill it. Returns the required fields left unanswered."""
        todo = []
        for f in fields:
            if not self.needs_answer(f):
                continue
            if f.get("kind") == "checkbox" and not (f.get("required") or f.get("error")):
                # Optional box: only the rules touch it (consent on; follow / newsletter off).
                rule = self.answerer.rule_answer(f, fields)
                if rule is not None and rule != bool(f.get("value")):
                    self._fill(f, rule)
                continue
            todo.append(f)
        if not todo:
            return []
        answers = self.answerer.answer_fields(todo, step_fields=fields)
        missing = []
        for f in todo:
            value = answers.get(str(f["id"]))
            if f.get("kind") == "checkbox":
                if value is None:
                    if not f.get("value"):
                        missing.append(f)
                    continue
                if (value is True) == bool(f.get("value")):
                    continue
            if value is None:
                if f.get("required") or f.get("error"):
                    missing.append(f)
                continue
            if f.get("kind") in ("select", "radio") and value not in real_options(f.get("options") or []):
                missing.append(f)
                continue
            err = self._fill(f, value)
            if err:
                logger.info("[%s] could not fill '%s': %s", self.spec.portal, str(f.get("question"))[:80], err)
                if f.get("required"):
                    missing.append({**f, "error": err})
            _pause((150, 450))
        return missing

    def _fix_errors(self, fields: list[dict]) -> list[dict]:
        """After a step did not move: fix the fields the site complains about. Returns the ones left broken."""
        broken = [f for f in fields if f.get("error")]
        left = []
        for f in broken:
            key = self.answerer._cache_key(f)
            fixed = self.answerer.fix_for_error(f, f.get("value"))
            if fixed is None:
                # Ask again with the site's complaint attached (cache bypassed).
                self.answerer._cache.pop(key, None)
                fixed = self.answerer.answer_fields([f], step_fields=fields).get(str(f["id"]))
            if fixed is None or str(fixed) == str(f.get("value")):
                left.append(f)
                continue
            self.answerer._cache[key] = fixed  # later jobs asking the same get the accepted form
            if self._fill(f, fixed):
                left.append(f)
        return left

    # -- the loop ---------------------------------------------------------

    def _signature(self, info: Optional[dict], fields: list[dict]) -> str:
        if not info:
            return "none"
        return "|".join([info.get("heading", ""), info.get("progress", ""), info.get("url", ""),
                         ";".join(str(f.get("question"))[:40] for f in fields[:12])])

    def _succeeded(self) -> bool:
        text = self.page_text()
        if self.spec.success_url and self.spec.success_url.search(self.sess.submit(lambda: self.sess._page.url) or ""):
            return True
        return bool(self.spec.success.search(text))

    def _wait_for_change(self, before: str, timeout_s: float = 9.0) -> tuple[str, Optional[dict], list[dict]]:
        deadline = time.time() + timeout_s
        info, fields = None, []
        while time.time() < deadline:
            time.sleep(0.7)
            info = self.find_root()
            fields = self.fields() if info else []
            sig = self._signature(info, fields)
            if sig != before:
                return sig, info, fields
            if any(f.get("error") for f in fields):
                break
        return self._signature(info, fields), info, fields

    def abandon(self) -> None:
        if self.spec.abandon:
            try:
                self.sess.submit(lambda: self.spec.abandon(self.sess._page))
            except Exception as exc:
                logger.debug("Could not leave the application cleanly: %s", exc)

    def run(self) -> FlowResult:
        same_steps = 0
        for step in range(1, self.spec.max_steps + 1):
            if self.should_stop():
                self.abandon()
                return FlowResult(status="stopped")

            # Root first: the popup sweep skips the stamped apply window.
            self.find_root()
            popup = self.handle_popups()
            if popup == "captcha" and self.wait_human():
                self.find_root()
                popup = self.handle_popups()
            if popup in ("captcha", "login", "limit"):
                return FlowResult(status="limit" if popup == "limit" else "paused", reason=popup)
            if popup == "sent":
                return FlowResult(status="applied", confirmation="the site showed 'application sent'")

            info = self.find_root()
            # The window can take a few seconds to open (slow network, a popup first).
            deadline = time.time() + (8 if step == 1 else 3)
            while info is None and time.time() < deadline:
                time.sleep(1.0)
                popup = self.handle_popups() if step == 1 else None
                if popup == "captcha" and self.wait_human():
                    deadline = time.time() + 8
                    info = self.find_root()
                    continue
                if popup in ("captcha", "login", "limit"):
                    return FlowResult(status="limit" if popup == "limit" else "paused", reason=popup)
                if popup == "sent":
                    return FlowResult(status="applied", confirmation="the site showed 'application sent'")
                info = self.find_root()
            if info is None:
                if self._succeeded():
                    return FlowResult(status="applied", confirmation="confirmation on the page")
                try:
                    diag = self.sess.submit(lambda: self.sess._page.evaluate(_JS_DIAG))
                except Exception as exc:
                    diag = str(exc)
                logger.warning("[%s] no apply window found at step %d; on screen: %s", self.spec.portal, step, diag)
                return FlowResult(status="failed", reason="The apply window closed before the application was sent.")

            fields = self.fields()
            missing = self._answer_and_fill(fields)
            if missing:
                questions = [str(f.get("question") or "?")[:200] for f in missing]
                self.abandon()
                return FlowResult(status="needs_user", questions=questions,
                                  reason="Questions your CV/profile do not answer: " + "; ".join(questions[:4]))

            kind, button = pick_button(self.buttons())
            if kind is None:
                # Maybe a popup we could not classify sits on top: one more sweep, then give up.
                if self.handle_popups() is None and pick_button(self.buttons())[0] is None:
                    self.abandon()
                    return FlowResult(status="failed",
                                      reason=f"No Next / Review / Submit button on step '{info.get('heading')}'.")
                continue

            if kind == "submit":
                url = self.sess.submit(lambda: self.sess._page.url)
                if not _host_ok(url, self.spec.hosts):
                    self.abandon()
                    return FlowResult(status="failed", reason=f"Submit is on an unexpected site ({url[:80]}).")
                if self.dry_run:
                    self.abandon()
                    return FlowResult(status="dry_run", filled=self.filled[-30:],
                                      reason="Dry run: everything filled, Submit NOT clicked.")

            before = self._signature(info, fields)
            logger.info("[%s] step %d '%s': %s", self.spec.portal, step, info.get("heading", "")[:60],
                        button.get("aria") or button.get("text"))
            try:
                self.sess.submit(lambda i=button["i"]: (self._frame or self.sess._page.main_frame).locator(
                    f'[data-vro-btn="{i}"]').first.click(timeout=8000))
            except Exception as exc:
                logger.info("[%s] button click failed: %s", self.spec.portal, bs._short_error({"error": str(exc)}))

            if kind == "submit":
                return self._after_submit()

            sig, info2, fields2 = self._wait_for_change(before)
            if sig != before:
                same_steps = 0
                continue
            # Did not move: read what the site complains about and fix it once.
            same_steps += 1
            left = self._fix_errors(fields2)
            if left or same_steps >= 2:
                errors = [f"{str(f.get('question'))[:80]}: {f.get('error')}" for f in (left or fields2) if f.get("error")]
                if left and all(f.get("error") for f in left):
                    questions = [str(f.get("question") or "?")[:200] for f in left]
                    for f in left:
                        self.answerer._remember_unanswered(str(f.get("question") or ""),
                                                           real_options(f.get("options") or []))
                    self.abandon()
                    return FlowResult(status="needs_user", questions=questions,
                                      reason="The site rejected the answers to: " + "; ".join(errors[:3]))
                if same_steps >= 2:
                    self.abandon()
                    return FlowResult(status="failed",
                                      reason="Stuck on step '" + str(info.get("heading")) + "'"
                                             + (": " + "; ".join(errors[:3]) if errors else ""))
        self.abandon()
        return FlowResult(status="failed", reason=f"Still not submitted after {self.spec.max_steps} steps.")

    def _after_submit(self) -> FlowResult:
        deadline = time.time() + 15
        while time.time() < deadline:
            time.sleep(1.0)
            if self._succeeded():
                self.handle_popups()  # close the "application sent" dialog
                return FlowResult(status="applied", confirmation="the site confirmed the application")
            self.find_root()
            popup = self.handle_popups()
            if popup == "sent":
                return FlowResult(status="applied", confirmation="the site showed 'application sent'")
            if popup == "captcha" and self.wait_human():
                deadline = time.time() + 15  # the submit goes through once the check passes
                continue
            if popup in ("captcha", "login"):
                return FlowResult(status="paused", reason=popup)
            info = self.find_root()
            if info:
                fields = self.fields()
                errors = [f"{str(f.get('question'))[:80]}: {f['error']}" for f in fields if f.get("error")]
                if errors:
                    self.abandon()
                    return FlowResult(status="failed", reason="The site refused the submit: " + "; ".join(errors[:3]))
        return FlowResult(status="unconfirmed",
                          reason="Submit was clicked but the site showed no confirmation - check this job in the "
                                 "browser; it is NOT counted as applied.")
