from langchain_ollama import ChatOllama 
from ddgs import DDGS


model_name = "qwen3.5:4b"

llm = ChatOllama(model = model_name,
                 temperature = 0.1)

def query_to_url(query):
    '''
    Takes one query from the user
    Returns a list of the web url it has found
    '''
    results = DDGS().text(query, 
                      safesearch = "off",
                      backend = "auto",
                      max_results=5)
    url_list = []

    for index in range(len(results)):
        web_url = results[index]["href"]
        url_list.append(web_url)

    return url_list