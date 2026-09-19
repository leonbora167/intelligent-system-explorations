from playwright.sync_api import sync_playwright


def url_to_html(web_url):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(web_url)
        with open("page_content.html", "w", encoding="utf-8") as w:
            w.write(page.content())
        browser.close()