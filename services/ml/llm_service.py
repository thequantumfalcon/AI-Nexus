"""
LLM Service with GPU-Accelerated Inference
Supports LLaMA 3.3 and other transformer models with quantization
"""

import torch
from typing import List, Dict, Optional, Iterator, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM service"""
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct"  # Default model
    device_map: str = "auto"  # Auto GPU allocation
    load_in_4bit: bool = True  # 4-bit quantization
    load_in_8bit: bool = False  # 8-bit quantization
    max_length: int = 4096  # Maximum sequence length
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    use_cache: bool = True
    trust_remote_code: bool = False
    torch_dtype: str = "auto"  # or "float16", "bfloat16"


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_return_sequences: int = 1
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


class LLMService:
    """
    GPU-accelerated LLM service with quantization support
    
    Features:
    - 4-bit/8-bit quantization for memory efficiency
    - Multi-GPU support via device_map
    - Streaming generation
    - Batch inference
    - Integration with RAG pipeline
    """
    
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        use_local_model: bool = False,
        local_model_path: Optional[str] = None
    ):
        """
        Initialize LLM service
        
        Args:
            config: LLM configuration
            use_local_model: Use local model instead of downloading
            local_model_path: Path to local model files
        """
        self.config = config or LLMConfig()
        self.use_local_model = use_local_model
        self.local_model_path = local_model_path
        
        self.model = None
        self.tokenizer = None
        self.device = None
        
        logger.info(f"Initializing LLM service: {self.config.model_name}")
        
        # Check GPU availability
        self._check_gpu()
        
        # Load model and tokenizer
        if not use_local_model:
            self._load_model()
        else:
            logger.info("Using local model mode - call load_local_model() manually")
    
    def _check_gpu(self):
        """Check GPU availability and memory"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available - LLM will use CPU (slow)")
            self.device = torch.device("cpu")
            return
        
        self.device = torch.device("cuda")
        
        # Log GPU info
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / 1e9  # GB
            logger.info(
                f"GPU {i}: {props.name} - "
                f"{total_mem:.1f}GB VRAM, "
                f"Compute {props.major}.{props.minor}"
            )
    
    def _load_model(self):
        """Load model and tokenizer from Hugging Face"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            # Configure quantization
            quantization_config = None
            if self.config.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info("Using 4-bit quantization (NF4)")
            elif self.config.load_in_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
                logger.info("Using 8-bit quantization")
            
            # Determine torch dtype
            if self.config.torch_dtype == "auto":
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            else:
                torch_dtype = getattr(torch, self.config.torch_dtype)
            
            logger.info(f"Loading model: {self.config.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map=self.config.device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=self.config.trust_remote_code,
                low_cpu_mem_usage=True
            )
            
            self.model.eval()  # Set to evaluation mode
            
            logger.info("Model loaded successfully")
            self._log_model_info()
            
        except ImportError as e:
            logger.error(f"Required packages not installed: {e}")
            logger.error("Install with: pip install transformers bitsandbytes accelerate")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Model loading failed - using mock mode")
            self._use_mock_model()
    
    def _use_mock_model(self):
        """Use mock model for testing without downloading"""
        logger.info("Using mock model (no actual LLM)")
        self.model = None
        self.tokenizer = None
    
    def _log_model_info(self):
        """Log model information"""
        if self.model is None:
            return
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(
            f"Model parameters: {total_params / 1e9:.2f}B total, "
            f"{trainable_params / 1e9:.2f}B trainable"
        )
        
        # Log memory usage
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                logger.info(
                    f"GPU {i} memory: {allocated:.2f}GB allocated, "
                    f"{reserved:.2f}GB reserved"
                )
    
    def load_local_model(self, model_path: str):
        """
        Load model from local path
        
        Args:
            model_path: Path to local model directory
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading local model from: {model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=self.config.device_map,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            
            self.model.eval()
            logger.info("Local model loaded successfully")
            self._log_model_info()
            
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
        return_full_text: bool = False
    ) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            generation_config: Generation configuration
            return_full_text: Return full text (prompt + generation)
            
        Returns:
            Generated text
        """
        if self.model is None:
            # Mock generation for testing
            return f"[Mock response to: {prompt[:50]}...]"
        
        gen_config = generation_config or GenerationConfig()
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=gen_config.max_new_tokens,
                temperature=gen_config.temperature,
                top_p=gen_config.top_p,
                top_k=gen_config.top_k,
                repetition_penalty=gen_config.repetition_penalty,
                do_sample=gen_config.do_sample,
                num_return_sequences=gen_config.num_return_sequences,
                pad_token_id=gen_config.pad_token_id or self.tokenizer.pad_token_id,
                eos_token_id=gen_config.eos_token_id or self.tokenizer.eos_token_id,
                use_cache=self.config.use_cache
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Remove prompt if requested
        if not return_full_text and prompt in generated_text:
            generated_text = generated_text.replace(prompt, "", 1).strip()
        
        return generated_text
    
    def generate_stream(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None
    ) -> Iterator[str]:
        """
        Generate text with streaming output
        
        Args:
            prompt: Input prompt
            generation_config: Generation configuration
            
        Yields:
            Generated text chunks
        """
        if self.model is None:
            # Mock streaming
            mock_response = f"Mock streaming response to: {prompt[:50]}..."
            for word in mock_response.split():
                yield word + " "
            return
        
        gen_config = generation_config or GenerationConfig()
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Streaming generation
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True
        )
        
        generation_kwargs = {
            **inputs,
            "max_new_tokens": gen_config.max_new_tokens,
            "temperature": gen_config.temperature,
            "top_p": gen_config.top_p,
            "top_k": gen_config.top_k,
            "repetition_penalty": gen_config.repetition_penalty,
            "do_sample": gen_config.do_sample,
            "streamer": streamer,
            "pad_token_id": gen_config.pad_token_id or self.tokenizer.pad_token_id,
            "eos_token_id": gen_config.eos_token_id or self.tokenizer.eos_token_id
        }
        
        # Generate in separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Yield tokens as they're generated
        for text in streamer:
            yield text
        
        thread.join()
    
    def batch_generate(
        self,
        prompts: List[str],
        generation_config: Optional[GenerationConfig] = None
    ) -> List[str]:
        """
        Generate text for multiple prompts (batch inference)
        
        Args:
            prompts: List of input prompts
            generation_config: Generation configuration
            
        Returns:
            List of generated texts
        """
        if self.model is None:
            return [f"[Mock batch response {i}]" for i in range(len(prompts))]
        
        gen_config = generation_config or GenerationConfig()
        
        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=gen_config.max_new_tokens,
                temperature=gen_config.temperature,
                top_p=gen_config.top_p,
                top_k=gen_config.top_k,
                repetition_penalty=gen_config.repetition_penalty,
                do_sample=gen_config.do_sample,
                pad_token_id=gen_config.pad_token_id or self.tokenizer.pad_token_id,
                eos_token_id=gen_config.eos_token_id or self.tokenizer.eos_token_id
            )
        
        # Decode outputs
        generated_texts = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        
        # Remove prompts
        results = []
        for prompt, generated in zip(prompts, generated_texts):
            if prompt in generated:
                generated = generated.replace(prompt, "", 1).strip()
            results.append(generated)
        
        return results
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        generation_config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Chat completion (conversation format)
        
        Args:
            messages: List of message dicts with 'role' and 'content'
                     e.g., [{"role": "user", "content": "Hello!"}]
            generation_config: Generation configuration
            
        Returns:
            Assistant's response
        """
        # Format messages into prompt
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback formatting
            prompt = self._format_chat_messages(messages)
        
        return self.generate(prompt, generation_config)
    
    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into prompt (fallback)"""
        formatted = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                formatted.append(f"System: {content}")
            elif role == 'user':
                formatted.append(f"User: {content}")
            elif role == 'assistant':
                formatted.append(f"Assistant: {content}")
        
        formatted.append("Assistant:")
        return "\n\n".join(formatted)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.model is None:
            return {
                "status": "mock",
                "model_name": self.config.model_name,
                "loaded": False
            }
        
        info = {
            "model_name": self.config.model_name,
            "loaded": True,
            "device": str(self.device),
            "quantization": "4-bit" if self.config.load_in_4bit else (
                "8-bit" if self.config.load_in_8bit else "none"
            )
        }
        
        # Add GPU info
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_memory"] = {}
            for i in range(torch.cuda.device_count()):
                info["gpu_memory"][f"gpu_{i}"] = {
                    "allocated_gb": torch.cuda.memory_allocated(i) / 1e9,
                    "reserved_gb": torch.cuda.memory_reserved(i) / 1e9
                }
        
        return info


# Global LLM service instance
_llm_service: Optional[LLMService] = None


def get_llm_service(
    config: Optional[LLMConfig] = None,
    force_new: bool = False
) -> LLMService:
    """
    Get global LLM service instance (singleton)
    
    Args:
        config: Optional LLM configuration
        force_new: Force creation of new instance
        
    Returns:
        Global LLM service instance
    """
    global _llm_service
    
    if _llm_service is None or force_new:
        _llm_service = LLMService(config)
    
    return _llm_service
