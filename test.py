browser = await p.chromium.launch(headless=False)
context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")
page = await context.new_page()
await page.set_extra_http_headers({
    "accept-language": "en-US,en;q=0.9"
})
await page.goto("https://your-url.com", wait_until="networkidle")
