from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoTokenizer,
)
from peft import PeftModel, PeftConfig
import torch

d_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
local_model_path = "outputs/checkpoint-80"     # Path to the combined weights

# Loading the base Model
config = PeftConfig.from_pretrained(local_model_path)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, 
    return_dict=True, 
    load_in_4bit=True, 
    device_map=d_map,
    ignore_mismatched_sizes=True
)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# load the base model with the Lora model
model = PeftModel.from_pretrained(model, local_model_path)

merged = model.merge_and_unload()

merged.save_pretrained("outputs/merged")

tokenizer.save_pretrained("outputs/merged")