import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir = "models",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Reasoning model for summarisation loaded")

df = pd.read_csv("outputs.csv")
global_context = ""

for index in range(0, len(df)):
    response = f"Video Part - {index} \n" + df.iloc[index]["Response"]
    global_context = global_context + response + "\n\n"

prompt = '''
There is a video which has been separated into small chunks and then the description of each chunk has been generated.

You will be provided each chunk in the format "Video Part - <video part number> \n<Description of this part of the video>...."

Your task is of a summarisation agent. Complete the task according to the follwing rules :- 

##Rules
1. Do not hallucinate and make up any answers that contains information not present in the video part descriptions. 
2. Keep the summary consistent with the overall tone of the descriptions given to you.
3. I want you to look at all the video parts descriptions, analyze and understand them to create a coherent structure. 
4. Return the overall summary of the videos together as one. 

Context and Description of each video parts :-


'''

messages = [
    {"role": "system", "content": "You are a text summarisation agent. "},
    {"role": "user", "content": prompt + global_context}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
content = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

with open("summary.txt", "w") as out:
    out.write(content)                                                                                                              