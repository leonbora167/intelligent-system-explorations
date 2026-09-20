from playwright.sync_api import sync_playwright


def url_to_html(web_url):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        try: #Some pages are not text based like tiktok which are causing errors
            page.goto(web_url)
        except:
            return 0 #Page wont load
        with open("page_content.html", "w", encoding="utf-8") as w:
            w.write(page.content())
        browser.close()
    return 1