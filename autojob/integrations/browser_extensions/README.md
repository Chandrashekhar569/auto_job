# Browser extensions

Drop **unpacked** Edge/Chrome extensions in here as folders. Any folder
containing a `manifest.json` is loaded automatically the next time Veronica
opens Edge.

```
browser_extensions/
  ublock/
    manifest.json
    ...
```

## Is an extension actually needed?

Mostly no. Playwright already drives the page through CDP, so it can read the
DOM, wait for elements, intercept network traffic and inject scripts — an
extension adds no capability on top of that.

The two things that genuinely slowed automation down are handled without one:

- **Cookie walls and overlay modals** intercepting clicks — handled by an init
  script that runs before page scripts on every navigation
  (`_JS_AUTODISMISS` in `browser_service.py`).
- **Round-trip latency**, which dominated everything else — handled by
  `browser_batch`, which performs many actions per model call.

## Where one does help

A content blocker. Ad iframes and interstitials on job boards cause
misdirected clicks, and blocking them at network level is more reliable than
hiding them afterwards. uBlock Origin works well:

1. Download the Chromium release from
   https://github.com/gorhill/uBlock/releases (`uBlock0_x.y.z.chromium.zip`).
2. Unzip it into `browser_extensions/ublock/` so `manifest.json` sits directly
   inside that folder.
3. Restart Veronica. The startup log will report how many extensions loaded.

Extensions only load in headed mode, which is how this session always runs.
