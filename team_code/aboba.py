# Load model directly
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("DAMO-NLP-SG/VideoLLaMA2-7B")