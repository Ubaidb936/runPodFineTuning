from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel, PeftConfig

import torch

d_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
lora_path = "outputs/checkpoint-20"     # Path to the combined weights
repo_name = "Financial_Analyst"            # HuggingFace repo name
hf_token = "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"


# LOADING MODEL IN N(4, 8.....) BIT
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

config = PeftConfig.from_pretrained(lora_path)

print(config.base_model_name_or_path)

model = AutoModelForCausalLM.from_pretrained(
  config.base_model_name_or_path,
  return_dict=True,  
  torch_dtype="auto",
  use_cache=False, # set to False as we're going to use gradient checkpointing
  quantization_config=bnb_config,
  device_map=d_map,
  trust_remote_code=True,
  ignore_mismatched_sizes=True  
)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)

model = PeftModel.from_pretrained(model, lora_path)

merged = model.merge_and_unload()

merged.push_to_hub(repo_name, token=hf_token)
tokenizer.push_to_hub(repo_name, token=hf_token)















