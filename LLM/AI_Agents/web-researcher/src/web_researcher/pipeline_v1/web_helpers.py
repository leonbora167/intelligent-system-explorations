from langchain_ollama import ChatOllama 
from langchain_core.messages import SystemMessage, HumanMessage
from ddgs import DDGS


model_name = "qwen3.5:4b"

summarizer_llm = ChatOllama(model = model_name,
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

def paragraph_summary(page_content, system_instructions, temperature, user_query):
    payload = f"USER QUERY : {user_query} \n\nPAGE CONTENT : \n\n{page_content}"
    prompt = [
        SystemMessage(content=system_instructions),
        HumanMessage(content=payload)
    ]
    response = summarizer_llm.invoke(prompt)
    return response