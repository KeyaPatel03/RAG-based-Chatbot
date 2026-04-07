#!/usr/bin/env python3
"""
webScrap_from_urls_txt.py  (UPDATED WITH METADATA GENERATION)

- Reads URLs from urls.txt and clgUrls.txt (one URL per line; '#' ignored).
- Uses aiohttp + certifi for SSL.
- Uses Playwright as a robust renderer with extra fallbacks.
- Extracts readable text.
- Generates 'scraped_metadata.json' mapping doc_id -> url -> filepath.
"""

import io
import asyncio
import ssl
import certifi
import aiohttp
from aiohttp import ClientTimeout, TCPConnector
from bs4 import BeautifulSoup, NavigableString
from pathlib import Path
from urllib.parse import urlparse
import sys
import json
import hashlib
import subprocess
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from pypdf import PdfReader

try:
    from playwright_stealth import stealth
except Exception:
    stealth = None

try:
    import requests
except Exception:
    requests = None


# ---------- CONFIG ----------
OUTPUT_DIR = Path("outputs_aiohttp")
OUTPUT_DIR.mkdir(exist_ok=True)

SAVE_RENDERED_FOR_DEBUG = True
PLAYWRIGHT_TIMEOUT_MS = 30000
PW_WAIT_FOR_FUNCTION_TIMEOUT_MS = 8000
TCP_LIMIT_PER_HOST = 5

BLOCKED_HTTP_STATUSES = {401, 403, 406, 409, 410, 412, 429}
FIREFOX_FIRST_DOMAINS = {"studentaid.gov"}
PLAYWRIGHT_INSTALL_ATTEMPTED = False
# ----------------------------


# ---------- helpers ----------
def load_urls_from_file():
    url_files = [Path("urls.txt"), Path("clgUrls.txt")]
    all_urls = []
    loaded_from = []

    for file_path in url_files:
        if not file_path.exists():
            continue
        lines = file_path.read_text(encoding="utf-8").splitlines()
        urls = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
        all_urls.extend(urls)
        loaded_from.append((file_path.name, len(urls)))

    if not loaded_from:
        print("[error] No URL file found. Create urls.txt or clgUrls.txt with one URL per line.")
        return []

    unique_urls = list(dict.fromkeys(all_urls))
    sources = ", ".join(f"{name}:{count}" for name, count in loaded_from)
    print(f"[info] Loaded {len(unique_urls)} unique URLs ({sources})")
    return unique_urls


def clean_filename(url: str, suffix: str = ".txt") -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_")
    name = (parsed.netloc + ("_" + path if path else ""))[:200]
    name = "".join(c if c.isalnum() or c in "._- " else "_" for c in name)
    return f"{name}{suffix}"


def looks_js_driven(html: str) -> bool:
    lower = (html or "").lower()
    if "<noscript" in lower or "enable javascript" in lower:
        return True
    if 'id="root"' in lower or 'id="__next"' in lower or 'data-reactroot' in lower:
        return True
    soup = BeautifulSoup(html or "", "html.parser")
    body_text = soup.get_text(" ", strip=True)
    if len(body_text) < 800:
        scripts = soup.find_all("script")
        if len(scripts) >= 1:
            return True
    return False


def is_probable_block_page(text: str) -> bool:
    lower = (text or "").lower()
    patterns = (
        "access denied",
        "forbidden",
        "request blocked",
        "bot detection",
        "captcha",
        "verify you are human",
        "cloudflare",
        "akamai",
        "incident id",
    )
    return any(p in lower for p in patterns)


def is_probable_spa_shell_text(text: str) -> bool:
    lower = (text or "").lower()
    shell_markers = (
        "you need to enable javascript to run this app",
        "<fsa-root",
        "loading...",
    )
    return any(marker in lower for marker in shell_markers)


def maybe_install_playwright_browsers(reason: str) -> bool:
    global PLAYWRIGHT_INSTALL_ATTEMPTED
    if PLAYWRIGHT_INSTALL_ATTEMPTED:
        return False

    lower_reason = (reason or "").lower()
    if "executable doesn't exist" not in lower_reason:
        return False

    PLAYWRIGHT_INSTALL_ATTEMPTED = True
    print("[playwright] Missing browser executables. Attempting one-time install...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "firefox", "chromium"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            print("[playwright] Browser install completed.")
            return True
        print(f"[playwright] Browser install failed (exit {result.returncode}).")
        if result.stderr:
            print(result.stderr[-1200:])
    except Exception as e:
        print(f"[playwright] Browser install error: {e}")
    return False


def extract_text_from_pdf_bytes(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        text_parts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                text_parts.append(t.strip())
        return "\n\n".join(text_parts)
    except Exception as e:
        print(f"[error] PDF extraction failed: {e}")
        return ""


# ---------------- extractor ----------------
def extract_content_from_html(html: str) -> str:
    if not html:
        return ""

    if isinstance(html, tuple) and html[0] == "IS_PDF":
        pdf_data = html[1]
        pdf_text = extract_text_from_pdf_bytes(pdf_data)
        if len(pdf_text) > 100:
            return pdf_text
        html = ""

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "aside", "svg"]):
        tag.decompose()
    for hidden in soup.select('[style*="display:none"], [style*="visibility:hidden"], [hidden]'):
        hidden.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id="content")
        or soup.find(id="fsa-main-content")
        or soup.find(attrs={"role": "main"})
        or soup.select_one(".layout-content")
    )
    root = main if main else (soup.body or soup)

    lines = []

    def append_text_block(txt):
        if not txt:
            return
        parts = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        if not parts:
            return
        lines.extend(parts)

    for elem in root.descendants:
        if isinstance(elem, NavigableString):
            continue
        name = getattr(elem, "name", None)
        if not name:
            continue
        if name in ("h1", "h2", "h3", "h4"):
            txt = elem.get_text(separator="\n", strip=True)
            if txt:
                lines.append("")
                lines.append(txt.upper())
                lines.append("")
        elif name == "p":
            txt = elem.get_text(separator="\n", strip=True)
            if txt:
                append_text_block(txt)
                lines.append("")
        elif name == "li":
            txt = elem.get_text(" ", strip=True)
            if txt:
                lines.append(f"• {txt}")
        elif name == "pre":
            txt = elem.get_text("\n", strip=True)
            if txt:
                lines.append(txt)
                lines.append("")
        elif name == "br":
            lines.append("")

    cleaned = []
    prev = None
    for ln in lines:
        ln_stripped = (ln or "").rstrip()
        if ln_stripped == "":
            if prev == "":
                continue
            cleaned.append("")
            prev = ""
        else:
            if ln_stripped == prev:
                continue
            cleaned.append(ln_stripped)
            prev = ln_stripped
    return "\n".join(cleaned).strip()


# ---------- HTTP fetch ----------
async def fetch(session: aiohttp.ClientSession, url: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://google.com",
    }
    timeout = ClientTimeout(total=30)
    try:
        async with session.get(url, headers=headers, timeout=timeout, allow_redirects=True) as resp:
            status = resp.status
            content_type = resp.headers.get("Content-Type", "").lower()
            print(f"[fetch] {url} -> HTTP {status} ({content_type})")
            if "application/pdf" in content_type:
                data = await resp.read()
                return status, ("IS_PDF", data)
            text = await resp.text(errors="ignore")
            return status, text
    except Exception as e:
        print(f"[error] fetching {url}: {e}")
        return None, ""


async def fetch_bytes(session: aiohttp.ClientSession, url: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117 Safari/537.36",
    }
    timeout = ClientTimeout(total=60)
    try:
        async with session.get(url, headers=headers, timeout=timeout, allow_redirects=True) as resp:
            print(f"[fetch_bytes] {url} -> HTTP {resp.status}")
            if resp.status == 200:
                return resp.status, await resp.read()
            return resp.status, None
    except Exception as e:
        print(f"[error] fetching bytes {url}: {e}")
        return None, None


# ---------- Alternative fallback rendering ----------
def render_with_alternative_fallback(url: str) -> str:
    if requests is None:
        return ""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
        resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        if resp.status_code == 200 and resp.text:
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            body_text = soup.get_text(" ", strip=True)
            if body_text and len(body_text) > 100 and "loading" not in body_text.lower()[:50]:
                print(f"[alt-fallback] {url} has {len(body_text)} chars of usable content")
                return resp.text
            print(f"[alt-fallback] {url} text too short ({len(body_text)} chars) or looks like loading page")
    except Exception as e:
        print(f"[alt-fallback error] {e}")
    return ""


# ---------- Playwright rendering ----------
async def render_with_playwright(playwright, url: str, timeout_ms: int = PLAYWRIGHT_TIMEOUT_MS) -> str:
    async def try_with_browser(browser_type, launch_args=None):
        try:
            browser = await browser_type.launch(headless=True, args=launch_args or [])
        except Exception as e:
            if maybe_install_playwright_browsers(str(e)):
                browser = await browser_type.launch(headless=True, args=launch_args or [])
            else:
                raise
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 720},
            ignore_https_errors=True,
        )
        page = await context.new_page()

        if stealth is not None:
            try:
                await stealth(page)
            except Exception:
                pass

        try:
            for wait_mode in ("networkidle", "load", "domcontentloaded"):
                try:
                    await page.goto(url, wait_until=wait_mode, timeout=timeout_ms)
                except PlaywrightTimeoutError:
                    print(f"[playwright timeout - {wait_mode}] continuing to next mode.")
                except Exception as nav_err:
                    print(f"[playwright navigation warning - {wait_mode}] {nav_err}")

                await asyncio.sleep(2.0)

                hydration_ok = False
                try:
                    print(f"[playwright] Waiting for content hydration on {url}...")
                    await page.wait_for_function(
                        "document.body.innerText.length > 500 || document.querySelector('h1') || document.querySelector('main')",
                        timeout=PW_WAIT_FOR_FUNCTION_TIMEOUT_MS,
                    )
                    hydration_ok = True
                    print(f"[playwright] Hydration condition met for {url}")
                except Exception:
                    print(f"[playwright wait_for_function] Timed out ({PW_WAIT_FOR_FUNCTION_TIMEOUT_MS}ms) - page may be unresponsive or slow.")

                if not hydration_ok:
                    continue

                try:
                    await page.evaluate(
                        """
                        () => {
                            const candidates = document.querySelectorAll(
                                'button[aria-controls], [role="button"][aria-controls], .accordion-button, .usa-accordion__button, .accordion-toggle, .toggle'
                            );
                            candidates.forEach((b) => {
                                try {
                                    const type = (b.getAttribute('type') || '').toLowerCase();
                                    const cls = (b.className || '').toString().toLowerCase();
                                    const aria = (b.getAttribute('aria-expanded') || '').toLowerCase();
                                    const label = (b.getAttribute('aria-label') || '').toLowerCase();

                                    const looksAccordion = cls.includes('accordion') || cls.includes('toggle') || b.hasAttribute('aria-controls');
                                    const isSafeType = type === '' || type === 'button';
                                    const isNavLike = cls.includes('search') || cls.includes('menu') || label.includes('search') || label.includes('menu');

                                    if (looksAccordion && isSafeType && !isNavLike && aria !== 'true') {
                                        b.click();
                                    }
                                } catch (e) {}
                            });

                            document.querySelectorAll('.accordion-panel, .usa-accordion__content, [data-testid="accordion-content"], [id^="accordion"], .panel, .collapse, .accordion__content').forEach((el) => {
                                try {
                                    el.style.display = 'block';
                                    el.style.visibility = 'visible';
                                    el.removeAttribute('hidden');
                                    el.setAttribute('aria-hidden', 'false');
                                } catch (e) {}
                            });
                        }
                        """
                    )
                    await asyncio.sleep(0.35)
                except Exception as e:
                    print(f"[playwright evaluate expand error] {e}")

                selectors = ("main", "article", "#content", ".layout-content", "body")
                for sel in selectors:
                    try:
                        html = await page.inner_html(sel, timeout=1500)
                        if html and len(html) > 50:
                            if SAVE_RENDERED_FOR_DEBUG:
                                (OUTPUT_DIR / clean_filename(url, suffix=f"_rendered_{sel}.html")).write_text(html, encoding="utf-8")
                            return html
                    except Exception:
                        continue

                try:
                    full = await page.content()
                    if full and len(full) > 50:
                        if SAVE_RENDERED_FOR_DEBUG:
                            (OUTPUT_DIR / clean_filename(url, suffix="_rendered_full.html")).write_text(full, encoding="utf-8")
                        return full
                except Exception as e:
                    print(f"[playwright full content error] {e}")

            try:
                body_text = await page.evaluate("() => document.body && document.body.innerText ? document.body.innerText : ''")
                if body_text and len(body_text) > 50:
                    if SAVE_RENDERED_FOR_DEBUG:
                        (OUTPUT_DIR / clean_filename(url, suffix="_rendered_body_text.html")).write_text(f"<pre>{body_text}</pre>", encoding="utf-8")
                    return f"<body><pre>{body_text}</pre></body>"
            except Exception:
                pass

            return ""
        finally:
            try:
                await context.close()
            except Exception:
                pass
            try:
                await browser.close()
            except Exception:
                pass

    domain = urlparse(url).netloc.lower()
    firefox_first = any(domain == d or domain.endswith(f".{d}") for d in FIREFOX_FIRST_DOMAINS)

    if firefox_first:
        try:
            print(f"[playwright] Using Firefox-first strategy for {domain}")
            html = await try_with_browser(playwright.firefox)
            if html:
                return html
        except Exception as e:
            print(f"[playwright firefox top-level error] {e}")

    try:
        chromium_args = ["--disable-blink-features=AutomationControlled", "--disable-http2"]
        html = await try_with_browser(playwright.chromium, launch_args=chromium_args)
        if html:
            return html
    except Exception as e:
        print(f"[playwright chromium error] {e}")

    if not firefox_first:
        try:
            html = await try_with_browser(playwright.firefox)
            if html:
                return html
        except Exception as e:
            print(f"[playwright firefox top-level error] {e}")

    try:
        req_ctx = await playwright.request.new_context()
        try:
            resp = await req_ctx.get(url, timeout=timeout_ms)
            print(f"[playwright.request] {url} -> HTTP {getattr(resp, 'status', None)}")
            text = await resp.text()
            if text and len(text) > 50:
                if SAVE_RENDERED_FOR_DEBUG:
                    (OUTPUT_DIR / clean_filename(url, suffix="_playwright_request.html")).write_text(text, encoding="utf-8")
                return text
        finally:
            await req_ctx.dispose()
    except Exception as e:
        print(f"[playwright.request fallback error] {e}")

    print(f"[playwright] final fallback failed for {url}")
    return ""


async def process_url(session, playwright, url):
    if not url:
        return None

    print(f"Processing: {url}")
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
    doc_id = f"doc_{url_hash}"

    if url.lower().endswith(".pdf"):
        status, content = await fetch_bytes(session, url)
        if status == 200 and content:
            text = extract_text_from_pdf_bytes(content)
            if len(text) > 100:
                out_name = clean_filename(url, suffix=".txt")
                out_path = OUTPUT_DIR / out_name
                out_path.write_text(text, encoding="utf-8")
                print(f"[saved PDF] {out_path}")
                return {
                    "doc_id": doc_id,
                    "url": url,
                    "source": "pdf",
                    "file_path": str(out_path),
                    "status": "success",
                }

    status, html_text = await fetch(session, url)

    if isinstance(html_text, tuple) and html_text[0] == "IS_PDF":
        pdf_data = html_text[1]
        text = extract_text_from_pdf_bytes(pdf_data)
        if len(text) > 100:
            out_name = clean_filename(url, suffix=".txt")
            out_path = OUTPUT_DIR / out_name
            out_path.write_text(text, encoding="utf-8")
            print(f"[saved] PDF extracted from Content-Type ({len(text)} chars)")
            return {
                "doc_id": doc_id,
                "url": url,
                "source": "pdf",
                "file_path": str(out_path),
                "status": "success",
            }
        html_text = ""

    blocked_status = status in BLOCKED_HTTP_STATUSES if status is not None else False
    need_render = blocked_status or looks_js_driven(html_text)
    content = ""

    if need_render:
        if blocked_status:
            print(f"[notice] {url} returned HTTP {status}; forcing Playwright render.")
        else:
            print(f"[notice] {url} appears JS-driven; using Playwright renderer.")

        try:
            rendered_html = await render_with_playwright(playwright, url)
            if not rendered_html:
                print(f"[playwright] rendered_html empty for {url} - attempting alternative fallback.")
                rendered_html = render_with_alternative_fallback(url)
                if not rendered_html:
                    print(f"[alt-fallback] also failed - falling back to server HTML from aiohttp fetch.")
                    rendered_html = html_text or ""
            content = extract_content_from_html(rendered_html)
        except Exception as e:
            print(f"[rendering error] {e} - falling back to server HTML extraction.")
            content = extract_content_from_html(html_text or "")
    else:
        content = extract_content_from_html(html_text or "")

    if not content or len(content.strip()) < 50:
        print(f"[retry] extraction insufficient for {url}; trying relaxed body text fallback and playwright.request fallback.")
        if html_text:
            soup = BeautifulSoup(html_text, "html.parser")
            body_text = soup.get_text("\n", strip=True)
            if body_text and len(body_text) > 50 and not is_probable_spa_shell_text(body_text):
                content = body_text

        if not content or len(content.strip()) < 50:
            try:
                req_ctx = await playwright.request.new_context()
                try:
                    resp = await req_ctx.get(url, timeout=PLAYWRIGHT_TIMEOUT_MS)
                    print(f"[playwright.request-second-pass] {url} -> HTTP {getattr(resp, 'status', None)}")
                    text = await resp.text()
                    if text and len(text) > 50:
                        extracted = extract_content_from_html(text)
                        if extracted and not is_probable_spa_shell_text(extracted):
                            content = extracted
                finally:
                    await req_ctx.dispose()
            except Exception as e:
                print(f"[playwright.request-second-pass error] {e}")

    if not content:
        print(f"[no-content] No extracted text for {url}.")
        return None

    if is_probable_block_page(content):
        print(f"[blocked] Extracted text for {url} still appears blocked/interstitial; skipping save.")
        return None

    out_name = clean_filename(url, suffix=".txt")
    out_path = OUTPUT_DIR / out_name
    out_path.write_text(content, encoding="utf-8")
    print(f"[saved] {out_path} (chars: {len(content)})")

    return {
        "doc_id": doc_id,
        "url": url,
        "source": "web",
        "file_path": str(out_path),
        "status": "success",
    }


async def main(urls):
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = TCPConnector(limit_per_host=TCP_LIMIT_PER_HOST, ssl=ssl_context)

    async with aiohttp.ClientSession(connector=connector) as session:
        async with async_playwright() as pw:
            tasks = [process_url(session, pw, u) for u in urls]
            results = await asyncio.gather(*tasks)

            valid_stats = [r for r in results if r]

            output_meta = Path("scraped_metadata.json")
            with open(output_meta, "w", encoding="utf-8") as f:
                json.dump(valid_stats, f, indent=2)

            print(f"\n[DONE] Saved metadata for {len(valid_stats)} documents to {output_meta}")


if __name__ == "__main__":
    urls = load_urls_from_file()
    if not urls:
        sys.exit(1)
    asyncio.run(main(urls))
