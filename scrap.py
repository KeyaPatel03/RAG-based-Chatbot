#!/usr/bin/env python3
"""
webScrap_from_urls_txt.py  (UPDATED WITH METADATA GENERATION)

- Reads URLs from urls.txt and clgUrls.txt (one URL per line; '#' ignored).
- Uses aiohttp + certifi for SSL.
- Uses Playwright as a robust renderer.
- Extracts readable text.
- Generates 'scraped_metadata.json' mapping doc_id -> url -> filepath.
"""
from pypdf import PdfReader
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
import uuid
import hashlib
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# ---------- CONFIG ----------
OUTPUT_DIR = Path("outputs_aiohttp")
OUTPUT_DIR.mkdir(exist_ok=True)

SAVE_RENDERED_FOR_DEBUG = True
PLAYWRIGHT_TIMEOUT_MS = 30000
TCP_LIMIT_PER_HOST = 5
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
    soup = BeautifulSoup(html, "html.parser")
    body_text = soup.get_text(" ", strip=True)
    if len(body_text) < 800:
        scripts = soup.find_all("script")
        if len(scripts) >= 1:
            return True
    return False

def is_pdf_url(url: str) -> bool:
    return url.lower().endswith(".pdf")

def extract_text_from_pdf_bytes(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    text_parts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            text_parts.append(t.strip())
    return "\n\n".join(text_parts)

# ---------------- extractor ----------------
def extract_content_from_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "aside", "svg"]):
        tag.decompose()
    for hidden in soup.select('[style*="display:none"], [style*="visibility:hidden"], [hidden]'):
        hidden.decompose()
    main = soup.find("main") or soup.find("article") or soup.find(id="content") or soup.select_one(".layout-content")
    root = main if main else (soup.body or soup)
    lines = []
    def append_text_block(txt):
        if not txt: return
        parts = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        if not parts: return
        lines.extend(parts)
    for elem in root.descendants:
        if isinstance(elem, NavigableString): continue
        name = getattr(elem, "name", None)
        if not name: continue
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
            if txt: lines.append(f"• {txt}")
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
            if prev == "": continue
            cleaned.append("")
            prev = ""
        else:
            if ln_stripped == prev: continue
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
            if "application/pdf" in content_type or url.lower().endswith(".pdf"):
                data = await resp.read()
                return status, data
            text = await resp.text(errors="ignore")
            return status, text
    except Exception as e:
        print(f"[error] fetching {url}: {e}")
        return None, ""

# ---------- Playwright rendering ----------
async def render_with_playwright(playwright, url: str, timeout_ms: int = PLAYWRIGHT_TIMEOUT_MS) -> str:
    async def try_with_browser(browser_type, launch_args=None):
        browser = await browser_type.launch(headless=True, args=launch_args or [])
        context = await browser.new_context(user_agent="Mozilla/5.0 ...", ignore_https_errors=True)
        page = await context.new_page()
        try:
            for wait_mode in ("networkidle", "load", "domcontentloaded"):
                try:
                    await page.goto(url, wait_until=wait_mode, timeout=timeout_ms)
                except Exception: pass
                await asyncio.sleep(0.6)
                try:
                    await page.evaluate("""() => {
                        document.querySelectorAll('button, [role="button"], .accordion-button').forEach(b => { try { b.click(); } catch(e) {} });
                        document.querySelectorAll('.accordion-panel, [hidden]').forEach(el => { el.hidden = false; el.style.display = 'block'; });
                    }""")
                    await asyncio.sleep(0.35)
                except Exception: pass
                
                selectors = ("main", "article", "#content", ".layout-content", "body")
                for sel in selectors:
                    try:
                        html = await page.inner_html(sel, timeout=1500)
                        if html and len(html) > 50:
                            if SAVE_RENDERED_FOR_DEBUG:
                                (OUTPUT_DIR / clean_filename(url, suffix=f"_rendered_{sel}.html")).write_text(html, encoding="utf-8")
                            await context.close()
                            await browser.close()
                            return html
                    except Exception: continue
                
                try:
                    full = await page.content()
                    if full and len(full) > 50:
                        await context.close()
                        await browser.close()
                        return full
                except Exception: pass
            return ""
        finally:
            await context.close()
            await browser.close()

    try:
        html = await try_with_browser(playwright.chromium)
        if html: return html
    except Exception: pass
    
    try:
        html = await try_with_browser(playwright.chromium, launch_args=["--disable-features=NetworkService", "--no-sandbox"])
        if html: return html
    except Exception: pass

    try:
        req_ctx = await playwright.request.new_context()
        resp = await req_ctx.get(url, timeout=timeout_ms)
        text = await resp.text()
        await req_ctx.dispose()
        if text and len(text) > 50: return text
    except Exception: pass

    try:
        html = await try_with_browser(playwright.firefox)
        if html: return html
    except Exception: pass
    
    return ""

async def process_url(session, playwright, url):
    # ---------- main processing ----------
    # Create deterministic ID based on URL
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
    doc_id = f"doc_{url_hash}"
    status_code, html = await fetch(session, url)

    # PDF
    if isinstance(html, (bytes, bytearray)) or url.lower().endswith(".pdf"):
        print(f"[pdf] Detected PDF: {url}")
        try:
            if not isinstance(html, (bytes, bytearray)):
                async with session.get(url) as resp:
                    html = await resp.read()
            text = extract_text_from_pdf_bytes(html)
            if text.strip():
                out_name = clean_filename(url, suffix=".txt")
                out_path = OUTPUT_DIR / out_name
                out_path.write_text(text, encoding="utf-8")
                print(f"[saved PDF] {out_path}")
                return {
                    "doc_id": doc_id,
                    "url": url,
                    "source": "pdf",
                    "file_path": str(out_path),
                    "status": "success"
                }
        except Exception as e:
            print(f"[pdf error] {url}: {e}")
        return None

    if status_code is None: return None

    need_render = looks_js_driven(html)
    content = ""

    if need_render:
        print(f"[notice] {url} appears JS-driven; using Playwright.")
        try:
            rendered_html = await render_with_playwright(playwright, url)
            content = extract_content_from_html(rendered_html or html or "")
        except Exception:
            content = extract_content_from_html(html or "")
    else:
        content = extract_content_from_html(html or "")

    # Retry fallback
    if not content or len(content.strip()) < 50:
        if html:
            soup = BeautifulSoup(html, "html.parser")
            body_text = soup.get_text("\n", strip=True)
            if body_text and len(body_text) > 50: content = body_text

    if not content:
        print(f"[no-content] No extracted text for {url}.")
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
        "status": "success"
    }

async def main(urls):
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = TCPConnector(limit_per_host=TCP_LIMIT_PER_HOST, ssl=ssl_context)

    async with aiohttp.ClientSession(connector=connector) as session:
        async with async_playwright() as pw:
            tasks = [process_url(session, pw, u) for u in urls]
            results = await asyncio.gather(*tasks)
            
            # Filter successful results
            valid_stats = [r for r in results if r]
            
            output_meta = Path("scraped_metadata.json")
            with open(output_meta, "w") as f:
                json.dump(valid_stats, f, indent=2)
            
            print(f"\n[DONE] Saved metadata for {len(valid_stats)} documents to {output_meta}")

if __name__ == "__main__":
    urls = load_urls_from_file()
    if not urls:
        sys.exit(1)
    asyncio.run(main(urls))
