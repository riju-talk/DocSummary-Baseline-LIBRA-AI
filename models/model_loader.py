from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()
class QAModel:
    def __init__(self):
        self.model_name = os.getenv('MODEL_NAME')
        self.tokenizer = None
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load quantized model for CPU"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            # Enable CPU optimizations
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")
            
    def generate_response(self, prompt: str, max_length: int = 512) -> str:
        """Generate response with CPU-friendly settings"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")
            
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            raise RuntimeError(f"Generation failed: {str(e)}")
            
    def summarize(self, text: str) -> str:
        prompt = f"""Summarize the following document concisely:
        
        {text}
        
        Summary:"""
        return self.generate_response(prompt, max_length=256)
        
    def answer_question(self, context: str, question: str) -> str:
        prompt = f"""Answer the question based on the context below. Be concise.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        return self.generate_response(prompt, max_length=200)