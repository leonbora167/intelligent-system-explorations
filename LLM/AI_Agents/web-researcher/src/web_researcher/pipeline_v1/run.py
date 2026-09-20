from web_helpers import query_to_url
from playwright_helper import url_to_html
from bs4 import BeautifulSoup
from web_helpers import paragraph_summary
import yaml
from tqdm import tqdm
from web_helpers import summary_scorer

config_path = "llm_config.yaml"
summarization_model = "llama3.2:1b"
scorer_model = "gemma4:34b"

def html_extract():
    content = ''
    with open("page_content.html", "r", encoding="utf-8") as f:
        html_content = f.read() 
    soup = BeautifulSoup(html_content, "html.parser")

    page_main_title = soup.title.string
    content = content + ' Title is : ' + page_main_title + '\n'
    all_paragraphs = soup.find_all("p")
    for index, paragraph in enumerate(all_paragraphs):
        content = content + str(index) + '\t' + paragraph.text + "\n"
    return content

def get_prompt_config(config_path, task, model_name):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        task_config = config.get("tasks", {}).get(task)
        if not task_config:
            raise ValueError(f"Task {task} not found in the config {config_path}")
        #print(task_config)
        model_config = task_config.get("models", {}).get(model_name)
        if not model_config:
            raise ValueError(f"Model {model_name} not found in config {config_path}")
        return model_config

def get_metadata(llm_response):
    response_metadata = llm_response.response_metadata
    total_duration = response_metadata["total_duration"]
    load_duration = response_metadata["load_duration"]
    usage_metadata = llm_response.usage_metadata
    input_tokens = usage_metadata["input_tokens"]
    output_tokens = usage_metadata["output_tokens"]
    total_tokens = usage_metadata["total_tokens"]

user_query = "Metal Gear Solid"

url_lists = query_to_url(user_query)
print("URL Lists are ", url_lists)
progress_bar_description = f"Summarising Web Pages for {user_query}"

for web_url in tqdm(url_lists, 
                    desc=progress_bar_description,
                    colour="green",
                    dynamic_ncols=True,
                    unit="site"):

    summaries = ''

    webpage_load = url_to_html(web_url)
    if webpage_load == 0:
        continue
    page_paragraph_content = html_extract()
    model_config = get_prompt_config(config_path, "summarization", summarization_model)
    system_instructions = model_config.get("system_instructions")
    #print(system_instructions)
    temperature = float(model_config.get("temperature"))
    
    llm_response = paragraph_summary(page_paragraph_content, system_instructions, temperature, user_query)
    page_summary = llm_response.content

    #Placeholder to write and save llm metadata in future

    summaries = summaries + page_summary + "\n"

print("All summaries gathered")


model_config = get_prompt_config(config_path, "content_evaluator", scorer_model)
system_instructions = model_config.get("system_instructions")
temperature = float(model_config.get("temperature"))

llm_response = summary_scorer(summaries, system_instructions, user_query)
print("Evaluation Score is :- ",llm_response.content)

model_config = get_prompt_config(config_path, "summarization", summarization_model)
system_instructions = model_config.get("system_instructions")
final_summary = paragraph_summary(page_paragraph_content, system_instructions, temperature, user_query)
print("Final Summary by the model is ", final_summary)