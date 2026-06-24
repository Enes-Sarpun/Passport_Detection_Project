"""
PRADO passport-image scraper for the Passport Computer Vision project.

Goal
----
Download ONLY passport documents (and their images) from the EU PRADO public
register (https://www.consilium.europa.eu/prado/) so the images can later be
labelled and used to train a YOLO + OCR pipeline.

Verified site structure (June 2026)
------------------------------------
* PRADO is behind a Cloudflare "managed challenge" ("Bir dakika lütfen...").
  Plain requests / vanilla Selenium are blocked, so we drive the page with
  `undetected-chromedriver`, which clears the challenge automatically.
* Country list  : search-by-document-country.html
                  -> <a class="prado-link" href=".../prado-documents/<CC>/index.html">
* Country page  : prado-documents/<CC>/index.html
                  -> category links, the passport one reads "A - Passport (N)"
                     href ".../prado-documents/<CC>/A/docs-per-category.html"
* Category page : .../A/docs-per-category.html
                  -> <img src=".../prado/images/<DOC-CODE>/<id>_thumb.jpg?v=..."
                         alt="<DOC-CODE>, <subtype>">
* Full-res image: drop the "_thumb" / "_archived" suffix from the file name.

Images are downloaded with `requests`, reusing the browser's Cloudflare
cookies + User-Agent. Files go on disk; SQLite stores the metadata + path.
"""

import re
import time
import logging
import sqlite3
from pathlib import Path
from urllib.parse import urlparse

import requests
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
START_URL = "https://www.consilium.europa.eu/prado/en/search-by-document-country.html"

DB_PATH = "europa_data.db"
IMAGE_ROOT = Path("data/images")

CHALLENGE_TIMEOUT = 60      # max seconds to wait for the Cloudflare challenge
PAGE_SETTLE = 2.0           # small pause after a page is ready
POLITE_DELAY = 1.5          # pause between documents (be nice to the server)
HEADLESS = False            # keep visible; Cloudflare is friendlier to real windows

# Only the passport category. On a country page the passport entry's text is
# like "A - Passport (16)" and its link points to a docs-per-category page.
PASSPORT_TEXT = re.compile(r"passport", re.IGNORECASE)

# Optional: limit how many countries to process (handy for a first test run).
# Set to None to scrape every country.
COUNTRY_LIMIT = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prado")


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def setup_database(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS europa_data (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            country      TEXT,
            doc_code     TEXT,
            doc_type     TEXT,
            Name         TEXT,
            Surname      TEXT,
            Descriptions TEXT,
            date         TEXT,
            image_path   TEXT UNIQUE,
            source_url   TEXT
        )
        """
    )
    conn.commit()
    return conn


def save_record(conn, *, country, doc_code, doc_type, description,
                image_path, source_url):
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO europa_data
                (country, doc_code, doc_type, Name, Surname,
                 Descriptions, date, image_path, source_url)
            VALUES (?, ?, ?, '', '', ?, '', ?, ?)
            """,
            (country, doc_code, doc_type, description,
             str(image_path), source_url),
        )
        conn.commit()
    except sqlite3.Error as exc:
        log.error("DB insert failed for %s: %s", image_path, exc)


# ---------------------------------------------------------------------------
# Browser
# ---------------------------------------------------------------------------
def setup_driver(headless: bool = HEADLESS) -> uc.Chrome:
    options = uc.ChromeOptions()
    options.add_argument("--window-size=1400,1000")
    if headless:
        options.add_argument("--headless=new")
    # NOTE: do NOT override the User-Agent here; a custom UA makes the
    # Cloudflare challenge fail. Let undetected-chromedriver manage it.
    driver = uc.Chrome(options=options)
    driver.set_page_load_timeout(90)
    return driver


def _challenge_active(driver) -> bool:
    title = (driver.title or "").lower()
    return any(k in title for k in ("dakika", "just a moment", "moment", "attente"))


def open_page(driver, url, *, expect_css=None, expect_img_prefix=None) -> bool:
    """Navigate to `url` and wait until the Cloudflare challenge clears and the
    expected content is present. Returns True on success."""
    driver.get(url)
    deadline = time.time() + CHALLENGE_TIMEOUT
    while time.time() < deadline:
        time.sleep(2)
        if _challenge_active(driver):
            continue
        if expect_css and not driver.find_elements(By.CSS_SELECTOR, expect_css):
            continue
        if expect_img_prefix:
            ready = any(expect_img_prefix in (im.get_attribute("src") or "")
                        for im in driver.find_elements(By.TAG_NAME, "img"))
            if not ready:
                continue
        time.sleep(PAGE_SETTLE)
        return True
    log.warning("Timed out waiting for %s", url)
    return False


def build_requests_session(driver) -> requests.Session:
    """Reuse the browser's Cloudflare cookies + UA so image GETs aren't blocked."""
    session = requests.Session()
    ua = driver.execute_script("return navigator.userAgent;")
    session.headers.update({"User-Agent": ua, "Referer": START_URL})
    for cookie in driver.get_cookies():
        session.cookies.set(cookie["name"], cookie["value"],
                            domain=cookie.get("domain"))
    return session


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------
def get_country_links(driver) -> list[dict]:
    """Return [{'code': 'AUT', 'name': '...', 'url': '.../AUT/index.html'}, ...]."""
    if not open_page(driver, START_URL, expect_css="a.prado-link"):
        return []
    countries, seen = [], set()
    for a in driver.find_elements(By.CSS_SELECTOR, "a.prado-link"):
        href = a.get_attribute("href") or ""
        if "prado-documents/" not in href or not href.endswith("/index.html"):
            continue
        if href in seen:
            continue
        seen.add(href)
        m = re.search(r"prado-documents/([A-Z]{3})/index\.html", href)
        code = m.group(1) if m else "UNK"
        # a.text includes a trailing document-counter; keep only the label line.
        lines = (a.text or "").splitlines()
        name = lines[0].strip() if lines else code
        name = re.sub(r"\s*\d+\s*$", "", name).strip() or code
        countries.append({"code": code, "name": name, "url": href})
    log.info("Found %d countries.", len(countries))
    return countries


def get_passport_category_url(driver, country: dict) -> str | None:
    """Open a country page and return the 'A - Passport' category URL."""
    if not open_page(driver, country["url"], expect_css="a.prado-link"):
        return None
    for a in driver.find_elements(By.CSS_SELECTOR, "a.prado-link"):
        text = (a.text or "").strip()
        href = a.get_attribute("href") or ""
        # The single passport category link aggregates all passport sub-types.
        if PASSPORT_TEXT.search(text) and "docs-per-category" in href:
            log.info("  %s passport category: %s", country["code"], text)
            return href
    log.info("  %s: no passport category found.", country["code"])
    return None


def get_passport_images(driver, category_url: str) -> list[dict]:
    """Return [{'thumb': url, 'doc_code': ..., 'subtype': ...}, ...]."""
    if not open_page(driver, category_url, expect_img_prefix="/prado/images/"):
        return []
    images, seen = [], set()
    for im in driver.find_elements(By.TAG_NAME, "img"):
        src = im.get_attribute("src") or ""
        if "/prado/images/" not in src or src in seen:
            continue
        seen.add(src)
        # URL: .../prado/images/<DOC-CODE>/<id>_thumb.jpg?v=...
        path_parts = urlparse(src).path.split("/")
        doc_code = path_parts[-2] if len(path_parts) >= 2 else "UNKNOWN"
        alt = im.get_attribute("alt") or ""
        subtype = alt.split(",")[-1].strip() if "," in alt else ""
        images.append({"thumb": src, "doc_code": doc_code, "subtype": subtype})
    log.info("    %d passport image(s) on category page.", len(images))
    return images


def download_image(session, url: str, save_path: Path) -> bool:
    """Try full-resolution first, then the thumbnail."""
    clean = url.split("?")[0]
    full = clean.replace("_thumb", "").replace("_archived", "")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    for candidate in (full, clean):
        try:
            r = session.get(candidate, timeout=30)
            if r.status_code == 200 and r.headers.get("Content-Type", "").startswith("image"):
                save_path.write_bytes(r.content)
                return True
        except requests.RequestException as exc:
            log.debug("download error %s: %s", candidate, exc)
    return False


def safe_name(*parts: str) -> str:
    raw = "_".join(p for p in parts if p)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_") or "image"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    conn = setup_database()
    driver = setup_driver()
    total = 0
    try:
        countries = get_country_links(driver)
        if not countries:
            log.error("No countries found - the challenge may not have cleared.")
            return
        if COUNTRY_LIMIT:
            countries = countries[:COUNTRY_LIMIT]
            log.info("COUNTRY_LIMIT set: processing %d country/ies.", len(countries))

        session = build_requests_session(driver)

        for country in countries:
            cat_url = get_passport_category_url(driver, country)
            if not cat_url:
                continue
            images = get_passport_images(driver, cat_url)
            session = build_requests_session(driver)  # refresh cookies

            for idx, img in enumerate(images, start=1):
                img_id = Path(urlparse(img["thumb"]).path).stem.replace("_thumb", "")
                fname = safe_name(img["doc_code"], img_id) + ".jpg"
                save_path = IMAGE_ROOT / country["code"] / "passport" / fname
                if save_path.exists():
                    continue
                if download_image(session, img["thumb"], save_path):
                    total += 1
                    save_record(
                        conn,
                        country=country["code"],
                        doc_code=img["doc_code"],
                        doc_type=img["subtype"],
                        description=country["name"],
                        image_path=save_path,
                        source_url=cat_url,
                    )
                    log.info("    saved %s", save_path)
                else:
                    log.warning("    FAILED %s", img["thumb"])
            time.sleep(POLITE_DELAY)

        log.info("Done. %d passport image(s) downloaded.", total)
    finally:
        try:
            driver.quit()
        except Exception:
            pass
        conn.close()


if __name__ == "__main__":
    main()




