from playwright.sync_api import sync_playwright

web_url = "https://en.wikipedia.org/wiki/Lego_Batman:_Legacy_of_the_Dark_Knight"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(web_url)
    print(page.title())
    page.screenshot(path=".\\web_screenshot.png", full_page=True)
    with open("page_content.html", "w", encoding="utf-8") as w:
        w.write(page.content())
    browser.close()