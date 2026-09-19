from web_helpers import query_to_url
from playwright_helper import url_to_html
from bs4 import BeautifulSoup


def html_extract():
    content = ''
    with open("page_content.html", "r", encoding="utf-8") as f:
        html_content = f.read() 
    soup = BeautifulSoup(html_content, "html.parser")

    page_main_title = soup.title.string
    content = content + ' Title is : ' + page_main_title + '\n'
    all_paragraphs = soup.find_all("p")
    for index, paragraph in enumerate(all_paragraphs):
        content = str(index) + '\t' + paragraph.text + "\n"
    return content
        

user_query = "Metal Gear Solid"

url_lists = query_to_url(user_query)

for web_url in url_lists:
    url_to_html(web_url)
    page_paragraph_content = html_extract()
    print(page_paragraph_content)
    