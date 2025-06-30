import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv
import time

load_dotenv()

# Configuration - Set your preferences here
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
USE_4BIT = True  # Enable 4-bit quantization
MAX_CONTEXT_SIZE = 4096

# 4-bit quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=USE_4BIT,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
) if USE_4BIT else None

# Model loading with safety checks
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=torch.float16 if not USE_4BIT else None,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
    )
    print("🔥 Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    raise

def format_prompt(task: str, context: str) -> str:
    return f"""<s>[INST] You are an expert document assistant. {task}
Context: {context} [/INST]"""

def generate_response(prompt: str, max_tokens: int = 512) -> str:
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_SIZE).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"⚠️ Generation error: {e}")
        return "Error processing request"

def summarize(chunks: list[str]) -> str:
    context = "\n\n".join(chunks[:8])[:6000]
    task = "Create a comprehensive summary highlighting key points, conclusions, and recommendations."
    prompt = format_prompt(task, context)
    return generate_response(prompt, max_tokens=512)

def answer_qa(question: str, context: str) -> str:
    task = f"Answer the question concisely and accurately based ONLY on the provided context. Question: {question}"
    prompt = format_prompt(task, context)
    return generate_response(prompt, max_tokens=256)