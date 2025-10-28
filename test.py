from playwright.sync_api import sync_playwright
import json
from urllib.parse import urljoin, urlparse

def scrape_page_components(url):
    """
    Scrapes all components from a given URL including text, links, images, and documents
    """
    
    with sync_playwright() as p:
        # Launch browser in headless mode
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # Navigate to the URL
        page.goto(url, wait_until='networkidle')
        
        # Initialize data structure to store scraped data
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
        
        # Extract page title
        scraped_data['title'] = page.title()
        
        # Extract all headings (h1-h6)
        for i in range(1, 7):
            headings = page.query_selector_all(f'h{i}')
            scraped_data['headings'][f'h{i}'] = [h.inner_text() for h in headings]
        
        # Extract all paragraphs
        paragraphs = page.query_selector_all('p')
        scraped_data['paragraphs'] = [p.inner_text() for p in paragraphs if p.inner_text().strip()]
        
        # Extract all links
        links = page.query_selector_all('a')
        for link in links:
            href = link.get_attribute('href')
            text = link.inner_text().strip()
            if href:
                absolute_url = urljoin(url, href)
                scraped_data['links'].append({
                    'text': text,
                    'url': absolute_url,
                    'is_external': urlparse(url).netloc != urlparse(absolute_url).netloc
                })
        
        # Extract all images
        images = page.query_selector_all('img')
        for img in images:
            src = img.get_attribute('src')
            alt = img.get_attribute('alt') or ''
            if src:
                absolute_url = urljoin(url, src)
                scraped_data['images'].append({
                    'src': absolute_url,
                    'alt': alt
                })
        
        # Extract document links (PDF, DOC, DOCX, XLS, XLSX, etc.)
        document_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.zip', '.csv']
        all_links = page.query_selector_all('a')
        for link in all_links:
            href = link.get_attribute('href')
            if href:
                absolute_url = urljoin(url, href)
                if any(absolute_url.lower().endswith(ext) for ext in document_extensions):
                    scraped_data['documents'].append({
                        'text': link.inner_text().strip(),
                        'url': absolute_url,
                        'type': next((ext for ext in document_extensions if absolute_url.lower().endswith(ext)), 'unknown')
                    })
        
        # Extract external resources (CSS, JS, fonts, etc.)
        stylesheets = page.query_selector_all('link[rel="stylesheet"]')
        for style in stylesheets:
            href = style.get_attribute('href')
            if href:
                scraped_data['external_resources'].append({
                    'type': 'css',
                    'url': urljoin(url, href)
                })
        
        scripts = page.query_selector_all('script[src]')
        for script in scripts:
            src = script.get_attribute('src')
            if src:
                scraped_data['external_resources'].append({
                    'type': 'javascript',
                    'url': urljoin(url, src)
                })
        
        # Close browser
        browser.close()
        
        return scraped_data


# Example usage
if __name__ == "__main__":
    target_url = "https://example.com"
    
    print(f"Scraping: {target_url}")
    data = scrape_page_components(target_url)
    
    # Save to JSON file
    with open('scraped_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n--- Scraping Summary ---")
    print(f"Title: {data['title']}")
    print(f"Total Links: {len(data['links'])}")
    print(f"Total Images: {len(data['images'])}")
    print(f"Total Documents: {len(data['documents'])}")
    print(f"Total External Resources: {len(data['external_resources'])}")
    print(f"\nData saved to scraped_data.json")
