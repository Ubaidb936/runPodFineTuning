from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoTokenizer,
)
from peft import PeftModel, PeftConfig
import torch
import gradio as gr

d_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
local_model_path = "outputs/checkpoint-100"     # Path to the combined weights

# Loading the base Model
config = PeftConfig.from_pretrained(local_model_path)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, 
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=d_map,
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)


def inferance(query: str, model, tokenizer, temp = 1.0, limit = 200) -> str:
  device = "cuda:0"

  prompt_template = """
  Below is an instruction that describes a task. Write a response that appropriately completes the request.
  ### Question:
  {query}

  ### Answer:
  """
  prompt = prompt_template.format(query=query)

  encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

  model_inputs = encodeds.to(device)

  generated_ids = model.generate(**model_inputs, max_new_tokens=int(limit), temperature=temp, do_sample=True, pad_token_id=tokenizer.eos_token_id)
  decoded = tokenizer.batch_decode(generated_ids)
  return (decoded[0])



def predict(temp, limit, text):
    prompt = text
    out = inferance(prompt, model, tokenizer, temp = 1.0, limit = 200)
    return out

pred = gr.Interface(
    predict,
    inputs=[
        gr.Slider(0.001, 10, value=0.1, label="Temperature"),
        gr.Slider(1, 1024, value=128, label="Token Limit"),
        gr.Textbox(
            label="Input",
            lines=1,
            value="#### Human: What's the capital of Australia?#### Assistant: ",
        ),
    ],
    outputs='text',
)

pred.launch(share=True)