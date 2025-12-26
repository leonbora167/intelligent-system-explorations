from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_path = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(model_path,
                                                    cache_dir = "models",
                                                    torch_dtype=torch.bfloat16).to("cuda")
print("SMolVLM2 Processor Loaded")
print("SmolVLM2 - 256M Model Loaded")

def model1(prompt):
    messages = prompt
    inputs = processor.apply_chat_template(messages,add_generation_prompt=True,
                                           tokenize=True,return_dict=True,
                                           return_tensors="pt",).to(model.device, dtype=torch.bfloat16)
    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=512)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,)

    response = generated_texts[0]
    return response

