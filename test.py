import asyncio
from playwright.async_api import async_playwright
import json
from urllib.parse import urljoin, urlparse

async def scrape_page_components(url):
    """
    Scrapes all components from a given URL including text, links, images, and documents (async).
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until='networkidle')

        scraped_data = {
            'url': url,
            'title': '',
            'headings': {},
            'paragraphs': [],
            'links': [],
            'images': [],
            'documents': [],
            'external_resources': []
        }

        # Extract page title (now with await)
        scraped_data['title'] = await page.title()

        # Extract all headings (h1-h6)
        for i in range(1, 7):
            headings = await page.query_selector_all(f'h{i}')
            scraped_data['headings'][f'h{i}'] = [                await h.inner_text() for h in headings
            ]

        # Extract all paragraphs
        paragraphs = await page.query_selector_all('p')
        scraped_data['paragraphs'] = [            await p.inner_text() for p in paragraphs if (await p.inner_text()).strip()
        ]

        # Extract all links
        links = await page.query_selector_all('a')
        for link in links:
            href = await link.get_attribute('href')
            text = (await link.inner_text()).strip()
            if href:
                absolute_url = urljoin(url, href)
                scraped_data['links'].append({
                    'text': text,
                    'url': absolute_url,
                    'is_external': urlparse(url).netloc != urlparse(absolute_url).netloc
                })

        # Extract all images
        images = await page.query_selector_all('img')
        for img in images:
            src = await img.get_attribute('src')
            alt = await img.get_attribute('alt') or ''
            if src:
                absolute_url = urljoin(url, src)
                scraped_data['images'].append({
                    'src': absolute_url,
                    'alt': alt
                })

        # Extract document links (PDF, DOC, DOCX, XLS, XLSX, etc.)
        document_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.zip', '.csv']
        all_links = await page.query_selector_all('a')
        for link in all_links:
            href = await link.get_attribute('href')
            if href:
                absolute_url = urljoin(url, href)
                if any(absolute_url.lower().endswith(ext) for ext in document_extensions):
                    scraped_data['documents'].append({
                        'text': (await link.inner_text()).strip(),
                        'url': absolute_url,
                        'type': next((ext for ext in document_extensions if absolute_url.lower().endswith(ext)), 'unknown')
                    })

        # Extract external resources (CSS, JS, fonts, etc.)
        stylesheets = await page.query_selector_all('link[rel="stylesheet"]')
        for style in stylesheets:
            href = await style.get_attribute('href')
            if href:
                scraped_data['external_resources'].append({
                    'type': 'css',
                    'url': urljoin(url, href)
                })

        scripts = await page.query_selector_all('script[src]')
        for script in scripts:
            src = await script.get_attribute('src')
            if src:
                scraped_data['external_resources'].append({
                    'type': 'javascript',
                    'url': urljoin(url, src)
                })

        await browser.close()
        return scraped_data

# Async main entry point
async def main():
    target_url = "https://example.com"
    print(f"Scraping: {target_url}")
    data = await scrape_page_components(target_url)

    with open('scraped_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n--- Scraping Summary ---")
    print(f"Title: {data['title']}")
    print(f"Total Links: {len(data['links'])}")
    print(f"Total Images: {len(data['images'])}")
    print(f"Total Documents: {len(data['documents'])}")
    print(f"Total External Resources: {len(data['external_resources'])}")
    print(f"\nData saved to scraped_data.json")

if __name__ == "__main__":
    asyncio.run(main())
