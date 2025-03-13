"""
Llama-Instruct Model Wrapper
Provides an interface to a locally loaded Llama-based instruction-tuned model
from Hugging Face as a replacement for OpenAI's API.
"""

import os
import logging
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logger = logging.getLogger("ragbot")

class LlamaInstructModel:
    """Wrapper for Llama-based instruction-tuned models from Hugging Face"""
    
    def __init__(self, model_name="NousResearch/Llama-2-7b-chat-hf", 
                 device="auto", 
                 use_4bit=True,
                 max_new_tokens=500):
        """
        Initialize the Llama model.
        
        Args:
            model_name: HuggingFace model name/path
            device: 'cpu', 'cuda', 'auto' for automatic detection
            use_4bit: Whether to use 4-bit quantization (reduced memory usage)
            max_new_tokens: Maximum number of tokens to generate
        """
        logger.info(f"Initializing Llama model: {model_name}")
        
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        
        # Determine device placement
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Configure quantization for memory efficiency
        quantization_config = None
        if use_4bit and self.device == "cuda":
            logger.info("Using 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        
        # Load the tokenizer and model
        try:
            logger.info("Loading tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                padding_side="left",
                trust_remote_code=True
            )
            
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Loading model")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=self.device if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                logger.info("Moving model to CPU")
                self.model = self.model.to("cpu")
                
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise
            
    def generate_response(self, system_prompt: str, user_message: str) -> str:
        """
        Generate a response using the Llama model.
        
        Args:
            system_prompt: System instructions/context
            user_message: User query
            
        Returns:
            Generated text response
        """
        try:
            logger.info("Generating response with Llama model")
            
            # Format the prompt according to the model's expected format
            if "llama-2" in self.model_name.lower():
                # Llama-2 chat format
                prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST]"
            elif "mistral" in self.model_name.lower():
                # Mistral Instruct format
                prompt = f"<s>[INST] {system_prompt}\n\n{user_message} [/INST]"
            elif "llama-3" in self.model_name.lower():
                # Llama-3 format
                prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_message}\n<|assistant|>"
            else:
                # Generic instruction format
                prompt = f"System: {system_prompt}\n\nUser: {user_message}\n\nAssistant:"
            
            logger.debug(f"Formatted prompt: {prompt[:100]}...")
            
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            
            # Move inputs to the correct device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's reply
            if "llama-2" in self.model_name.lower():
                response = response.split("[/INST]")[-1].strip()
            elif "mistral" in self.model_name.lower():
                response = response.split("[/INST]")[-1].strip()
            elif "llama-3" in self.model_name.lower():
                response = response.split("<|assistant|>")[-1].strip()
            else:
                response = response.split("Assistant:")[-1].strip()
            
            logger.info(f"Generated response: {response[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return "I apologize, but I encountered an error while generating a response. Please try again."
            
    def chat_completion_create(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Generate a chat completion response similar to OpenAI API format.
        
        Args:
            messages: List of messages in the format [{"role": "system", "content": "..."}, ...]
            
        Returns:
            Dict with response in a format similar to OpenAI's API
        """
        try:
            # Extract system prompt and user message
            system_content = ""
            user_content = ""
            
            # Process messages to extract system and user content
            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                elif msg["role"] == "user":
                    # For simplicity, we'll use the last user message
                    user_content = msg["content"]
            
            # If no system prompt was provided, use a default one
            if not system_content:
                system_content = "You are a helpful, harmless, and precise assistant."
                
            # Generate the response
            assistant_reply = self.generate_response(system_content, user_content)
            
            # Format response to match OpenAI's structure
            response = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": assistant_reply
                        }
                    }
                ]
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}", exc_info=True)
            # Return a minimal response structure with error message
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I apologize, but I encountered an error while processing your request."
                        }
                    }
                ]
            } 