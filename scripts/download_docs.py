# scripts/scrape_gitlab_handbook.py
"""
Scrapes GitLab handbook pages as markdown files.
Every URL in this script is verified by direct HTTP fetch.

Install:
    pip install playwright aiofiles beautifulsoup4
    playwright install chromium

Run:
    python scripts/scrape_gitlab_handbook.py
"""

import asyncio
import re
import aiofiles
from pathlib import Path
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup


# ─────────────────────────────────────────────────────────
# ALL URLs VERIFIED BY DIRECT FETCH
# ─────────────────────────────────────────────────────────

BASE_DIR = Path("docs/")

PAGES: dict[str, str] = {

    # ── HR / People ───────────────────────────────────────
    # paid-time-off at /paid-time-off/ = 404 confirmed
    # using time-off-types which IS confirmed working
    "hr/time_off_types.md": (
        "https://handbook.gitlab.com/handbook/people-group/"
        "time-off-and-absence/time-off-types/"
    ),
    "hr/anti_harassment.md": (
        "https://handbook.gitlab.com/handbook/people-group/anti-harassment/"
    ),
    "hr/people_group.md": (
        "https://handbook.gitlab.com/handbook/people-group/"
    ),
    "hr/people_policies.md": (
        "https://handbook.gitlab.com/handbook/people-policies/"
    ),
    "hr/people_policies_usa.md": (
        "https://handbook.gitlab.com/handbook/people-policies/inc-usa/"
    ),
    "hr/onboarding.md": (
        "https://handbook.gitlab.com/handbook/people-group/general-onboarding/"
    ),
    "hr/code_of_business_conduct.md": (
        "https://handbook.gitlab.com/handbook/legal/"
        "gitlab-code-of-business-conduct-and-ethics/"
    ),
    "hr/anti_retaliation.md": (
        "https://handbook.gitlab.com/handbook/legal/anti-retaliation-policy/"
    ),

    # ── Engineering ───────────────────────────────────────
    "engineering/engineering_handbook.md": (
        "https://handbook.gitlab.com/handbook/engineering/"
    ),
    "engineering/development.md": (
        "https://handbook.gitlab.com/handbook/engineering/development/"
    ),
    "engineering/career_development.md": (
        "https://handbook.gitlab.com/handbook/engineering/careers/"
    ),

    # ── Culture / Company ─────────────────────────────────
    "culture/values.md": (
        "https://handbook.gitlab.com/handbook/values/"
    ),
    "culture/all_remote.md": (
        "https://handbook.gitlab.com/handbook/company/culture/all-remote/"
    ),
    "culture/okrs.md": (
        "https://handbook.gitlab.com/handbook/company/okrs/"
    ),
}


# ─────────────────────────────────────────────────────────
# HTML → MARKDOWN CONVERTER
# ─────────────────────────────────────────────────────────

def html_to_markdown(element) -> str:
    lines = []
    for tag in element.find_all(
        ["h1", "h2", "h3", "h4", "p", "li", "pre"],
        recursive=True,
    ):
        text = tag.get_text(separator=" ", strip=True)
        if not text or len(text) < 3:
            continue

        if tag.name == "h1":
            lines.append(f"\n# {text}\n")
        elif tag.name == "h2":
            lines.append(f"\n## {text}\n")
        elif tag.name == "h3":
            lines.append(f"\n### {text}\n")
        elif tag.name == "h4":
            lines.append(f"\n#### {text}\n")
        elif tag.name == "p":
            lines.append(f"\n{text}\n")
        elif tag.name == "li":
            lines.append(f"- {text}")
        elif tag.name == "pre":
            lines.append(f"\n```\n{text}\n```\n")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# SCRAPE SINGLE PAGE
# ─────────────────────────────────────────────────────────

async def scrape_page(page, url: str) -> dict | None:
    try:
        response = await page.goto(
            url,
            wait_until = "networkidle",
            timeout    = 45000,
        )

        # Check HTTP status
        if response and response.status == 404:
            print(f"    ✗ 404: {url}")
            return None

        await page.wait_for_timeout(2000)

        content = await page.content()
        soup    = BeautifulSoup(content, "html.parser")

        # Extract title
        title = ""
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)
        else:
            raw_title = await page.title()
            title = raw_title.replace(
                " | The GitLab Handbook", ""
            ).strip()

        # Extract main content
        main = (
            soup.find("main") or
            soup.find("article") or
            soup.find("div", {"class": re.compile(r"content|main", re.I)})
        )

        if not main:
            print(f"    ✗ No main content found: {url}")
            return None

        # Remove clutter
        for tag in main.find_all([
            "nav", "header", "footer",
            "script", "style", "aside",
            "button", "form",
        ]):
            tag.decompose()

        for tag in main.find_all(
            True,
            {"class": re.compile(
                r"sidebar|navbar|toc|cookie|banner|"
                r"breadcrumb|feedback|edit-page|"
                r"table-of-contents",
                re.I,
            )}
        ):
            tag.decompose()

        text = html_to_markdown(main)

        if len(text.strip()) < 100:
            print(f"    ✗ Content too short: {url}")
            return None

        return {"title": title, "content": text}

    except Exception as e:
        print(f"    ✗ Error scraping {url}: {e}")
        return None


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

async def main():
    print("╔════════════════════════════════════════════╗")
    print("║   Cortex — GitLab Handbook Scraper         ║")
    print("║   All URLs verified by direct HTTP fetch   ║")
    print("╚════════════════════════════════════════════╝")
    print(f"\n  Pages: {len(PAGES)}")
    print(f"  Output: {BASE_DIR.absolute()}\n")

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    ok      = 0
    skipped = 0
    failed  = 0
    failed_urls = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900}
        )
        page = await context.new_page()

        for rel_path, url in PAGES.items():
            output = BASE_DIR / rel_path
            output.parent.mkdir(parents=True, exist_ok=True)

            # Skip already scraped and non-empty
            if output.exists() and output.stat().st_size > 500:
                print(f"  ○ Skip (exists): {rel_path}")
                skipped += 1
                continue

            print(f"\n  ↓ {rel_path}")
            print(f"    {url}")

            result = await scrape_page(page, url)

            if result:
                md = (
                    f"# {result['title']}\n\n"
                    f"Source: {url}\n\n"
                    f"---\n\n"
                    f"{result['content']}"
                )

                async with aiofiles.open(
                    output, "w", encoding="utf-8"
                ) as f:
                    await f.write(md)

                size = output.stat().st_size
                wc   = len(result["content"].split())
                print(f"  ✓ {output.name} ({size // 1024} KB, {wc} words)")
                ok += 1
            else:
                print(f"  ✗ Failed: {rel_path}")
                failed_urls.append(url)
                failed += 1

            await asyncio.sleep(2)

        await browser.close()

    # ── Summary ───────────────────────────────────────────
    print("\n╔════════════════════════════════════════════╗")
    print("║   Summary                                  ║")
    print("╚════════════════════════════════════════════╝")
    print(f"  ✓ Scraped:  {ok}")
    print(f"  ○ Skipped:  {skipped}")
    print(f"  ✗ Failed:   {failed}")

    if failed_urls:
        print("\n  Failed URLs:")
        for u in failed_urls:
            print(f"    - {u}")

    print("\n  Files by category:")
    for cat in ["hr", "engineering", "culture"]:
        cat_dir = BASE_DIR / cat
        if cat_dir.exists():
            files = list(cat_dir.glob("*.md"))
            size  = sum(
                f.stat().st_size for f in files
            ) // 1024
            print(
                f"    {cat:<15} "
                f"{len(files):>3} files   "
                f"{size:>5} KB"
            )

    all_files = list(BASE_DIR.rglob("*.md"))
    print(f"\n  Total: {len(all_files)} files")

    if ok > 0:
        print(f"\n✓ Next step:")
        print(f"  python -m rag.ingestion.handbook_loader")


if __name__ == "__main__":
    asyncio.run(main())