from fastapi import FastAPI 
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel

app = FastAPI()


model_name = "Qwen/Qwen3-0.6B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                        cache_dir = "models")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir = "models",
    torch_dtype="auto",
    device_map="auto"
)

print("Model Loaded")

def run_inference(prompt, model, tokenizer):
    # prepare the model input
    #prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        temperature = 0.7, 
        top_p = 0.9, 
        do_sample=True
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    #print("thinking content:", thinking_content)
    #print("content:", content)

    return {"thinking_content" : thinking_content, 
            "content" : content}


class Item(BaseModel):
    user_prompt : str

@app.post("/inference")
async def inference(json_struct : Item):
    response = run_inference(json_struct.user_prompt, model, tokenizer)
    return response