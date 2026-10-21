from langchain_ollama import ChatOllama 
from langchain_core.messages import SystemMessage, HumanMessage
from ddgs import DDGS
import os
from phoenix.otel import register
from opentelemetry import trace

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"

tracer_provider = register(protocol="http/protobuf", 
                           set_global_tracer_provider=True,
                            project_name="web-researcher")
tracer = tracer_provider.get_tracer(__name__)

summarizer_model_name = "llama3.2:1b"
scorer_model_name = "qwen3.5:4b"

summarizer_llm = ChatOllama(model = summarizer_model_name,
                 temperature = 0.1)

scorer_model = ChatOllama(model = scorer_model_name,
                          temperature = 0.5)

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
    with tracer.start_as_current_span("summarizer model",
                                    openinference_span_kind="chain") as span:
            span.set_input(prompt)
            span.set_output(response)
            span.set_status("Response from summariser model received")
    return response

def summary_scorer(summaries, system_instructions, user_query):
    payload = f"USER QUERY : {user_query} \n\nSUMMARIES : \n\n{summaries}"
    prompt = [
        SystemMessage(content=system_instructions),
        HumanMessage(content = payload)
    ]
    response = scorer_model.invoke(prompt)
    with tracer.start_as_current_span("Scorer model",
                                    openinference_span_kind="chain") as span:
            span.set_input(prompt)
            span.set_output(response)
            span.set_status("Response from summariser model received")
    return response