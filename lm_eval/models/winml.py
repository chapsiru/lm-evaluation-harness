"""
WinML backend for lm-eval-harness with NPU/GPU/CPU support.
    
This backend leverages Windows Machine Learning (WinML) to run models on various 
hardware backends including NPUs, GPUs, and CPUs. It's particularly useful for
running inference on Windows devices with dedicated Neural Processing Units.

Example usage:
    lm_eval --model winml --model_args pretrained=path/to/onnx/model.onnx,device=npu --tasks hellaswag

Supported devices:
    - npu: Neural Processing Unit (recommended for efficiency)
    - gpu: Graphics Processing Unit
    - cpu: Central Processing Unit
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
import onnxruntime_genai as og

if TYPE_CHECKING:
    from lm_eval.api.instance import Instance

eval_logger = logging.getLogger(__name__)


@register_model("winml")
class WindowsML(TemplateLM):
    """
    WindowsML backend for lm-eval-harness with NPU/GPU/CPU support.

    This model class provides integration with Windows Machine Learning (WindowsML)
    to enable evaluation on NPUs and other Windows-optimized hardware.
    """
    
    _DEFAULT_MAX_LENGTH = 2048

    @classmethod
    def create_from_arg_obj(
        cls, arg_dict: Dict[str, Any], additional_config: Optional[Dict[str, Any]] = None
    ) -> "WindowsML":
        """
        Override to properly merge dictionaries and avoid duplicate keyword arguments.
        
        Args:
            arg_dict: Dictionary containing model arguments
            additional_config: Optional dictionary containing additional configuration
            
        Returns:
            Instance of WindowsML class
        """
        # Merge the dictionaries, with additional_config taking precedence
        merged_config = {**(arg_dict or {})}
        if additional_config:
            # Filter out None values and merge
            filtered_additional = {k: v for k, v in additional_config.items() if v is not None}
            merged_config.update(filtered_additional)
        
        return cls(**merged_config)

    def __init__(
        self,
        pretrained: str,
        device: str = "cpu",
        max_length: Optional[int] = 4096,
        batch_size: int = 1,
        max_batch_size: int = 64,
    ) -> None:
        """
        Initialize WindowsML model.
        
        Args:
            pretrained: Path to ONNX model file or directory containing model files
            device: Target device ('npu', 'gpu', 'cpu')
            max_length: Maximum sequence length
            batch_size: Batch size for inference
            max_batch_size: Maximum batch size for auto-batching
        """
        super().__init__()
        
        # Validate and import dependencies
        self._validate_dependencies()
        
        # Store configuration
        self.pretrained = pretrained
        self.device = device.lower()
        self.max_length = max_length or self._DEFAULT_MAX_LENGTH
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        
        self._fix_winrt_runtime()

        # Initialize Windows ML execution providers
        self._register_winml_providers_to_genai()
        
        # Setup device and execution providers using new Windows ML API
        self._setup_winml_devices_and_providers()
        
        # Load and compile ONNX model
        self._load_and_compile_model(pretrained)
        
        eval_logger.info(f"Available EP devices: {len(self.ep_device_map)} execution providers")

    def _validate_dependencies(self) -> None:
        """
        Validate that required dependencies are available.
        
        Raises:
            ImportError: If required dependencies are not installed
        """
        try:
            import onnxruntime_genai as og
            self.og = og
            eval_logger.info(f"ONNX Runtime GenAI version: {og.__version__}")
        except ImportError as e:
            raise ImportError(
                "ONNX Runtime GenAI is required for WinML backend. "
                "Install with: pip install onnxruntime-genai"
            ) from e
        
        # Also import regular ONNX Runtime for EP registration
        try:
            import onnxruntime as ort
            self.ort = ort
            eval_logger.info(f"ONNX Runtime version: {ort.__version__}")
        except ImportError as e:
            raise ImportError(
                "ONNX Runtime is also required for execution provider registration. "
                "Install with: pip install onnxruntime"
            ) from e

    def _fix_winrt_runtime(self):
        """
        This function removes the msvcp140.dll from the winrt-runtime package.
        So it does not cause issues with other libraries.
        """
        from importlib import metadata
        site_packages_path = Path(str(metadata.distribution('winrt-runtime').locate_file('')))
        dll_path = site_packages_path / 'winrt' / 'msvcp140.dll'
        if dll_path.exists():
            dll_path.unlink()

    def _register_winml_providers_to_genai(self) -> bool:
        """
        Register Windows ML execution providers to ONNX Runtime GenAI.
        
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import (
                InitializeOptions,
                initialize
            )
            import winui3.microsoft.windows.ai.machinelearning as winml
            
            with initialize(options=InitializeOptions.ON_NO_MATCH_SHOW_UI):
                catalog = winml.ExecutionProviderCatalog.get_default()
                providers = catalog.find_all_providers()
                for provider in providers:
                    provider.ensure_ready_async().get()
                    # Register to GenAI instead of regular ONNX Runtime
                    self.og.register_execution_provider_library(provider.name, provider.library_path)
                    eval_logger.info(f"Registered {provider.name} to ONNX Runtime GenAI")
            
            return True
        except ImportError as e:
            eval_logger.warning(f"Windows ML import error: {e}")
            return False
        except Exception as e:
            eval_logger.warning(f"Error registering providers to GenAI: {e}")
            return False

    def _setup_winml_devices_and_providers(self) -> None:
        """
        Setup execution providers using Windows ML device enumeration API.
        
        This method queries available devices and builds a mapping of execution providers."""
        try:
            # Get available EP devices using Windows ML API
            ep_devices = self.ort.get_ep_devices()
            self.ep_device_map = {}
            
            # Build device map
            for device in ep_devices:
                ep_name = device.ep_name
                if ep_name not in self.ep_device_map:
                    self.ep_device_map[ep_name] = []
                self.ep_device_map[ep_name].append(device)
            
            # Log available devices
            eval_logger.info("Available execution provider devices:")
            for name, devices in self.ep_device_map.items():
                eval_logger.info(f"Execution Provider: {name}")
                for device in devices:
                    try:
                        device_type = self.ort.OrtHardwareDeviceType(device.device.type).name
                        eval_logger.info(f" | Vendor: {device.ep_vendor:<16} | Device Type: {device_type:<8}")
                    except Exception:
                        eval_logger.info(f" | Vendor: {device.ep_vendor:<16} | Device Type: Unknown")
            
        except Exception as e:
            eval_logger.warning(f"Windows ML device enumeration failed: {e}")
            eval_logger.info("Falling back to legacy provider selection")
            self.ep_device_map = {}

    def _load_and_compile_model(self, model_path: str) -> None:
        """
        Load and optionally compile ONNX model with ONNX Runtime GenAI.
        
        Args:
            model_path: Path to ONNX model file or directory
            
        Raises:
            FileNotFoundError: If model path is not found or invalid
            Exception: If model loading fails
        """
        model_path = Path(model_path)
        
        # Handle different input formats
        if model_path.is_file() and model_path.suffix == '.onnx':
            input_model_path = model_path.parent  # GenAI expects directory
        elif model_path.is_dir():
            input_model_path = model_path
        else:
            raise FileNotFoundError(f"Model path {model_path} not found or invalid")
        
        # Load model using ONNX Runtime GenAI with proper batch configuration
        try:
            eval_logger.info(f"Loading model with ONNX Runtime GenAI from: {input_model_path}")
            
            # Create config with batch size
            config = self.og.Config(str(input_model_path))
            
            # Configure search options with batch size
            search_config = {
                "batch_size": self.max_batch_size,  # Use max_batch_size for model initialization
                "num_beams": 1  # Default to greedy search
            }
            
            # Apply search configuration overlay
            import json
            config.overlay(json.dumps({"search": search_config}))
            
            # Load model and tokenizer using GenAI with config
            self.genai_model = self.og.Model(config)
            self.genai_tokenizer = self.og.Tokenizer(self.genai_model)
            
            eval_logger.info(f"Model loaded with max batch size: {self.max_batch_size}")
            eval_logger.info("Model and tokenizer loaded successfully with ONNX Runtime GenAI")
            
            # Store model info
            self.model_path = input_model_path
            
        except Exception as e:
            eval_logger.error(f"Failed to load model with ONNX Runtime GenAI from {input_model_path}: {e}")
            raise
    
    @property
    def eot_token_id(self) -> int:
        """
        Get the end-of-text token ID.
        
        Returns:
            End-of-text token ID from the GenAI tokenizer
        """
        # GenAI tokenizer uses eos_token_id
        return self.genai_tokenizer.eos_token_id
    
    @property
    def max_gen_toks(self) -> int:
        """
        Get the maximum number of tokens to generate.
        
        Returns:
            Maximum generation tokens (default: 4096)
        """
        return 4096

    def tok_encode(self, string: str, left_truncate_len: Optional[int] = None, add_special_tokens: bool = True) -> List[int]:
        """
        Tokenize string and return token IDs.
        
        Args:
            string: Input string to tokenize
            left_truncate_len: If provided, truncate from the left to this length
            add_special_tokens: Whether to add special tokens (note: GenAI tokenizer handles this automatically)
            
        Returns:
            List of token IDs
        """
        # Use GenAI tokenizer for consistency with model inference
        encoding = self.genai_tokenizer.encode(string)

        # Handle left truncation if requested
        if left_truncate_len is not None and len(encoding) > left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def tok_decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            tokens: List of token IDs to decode
            
        Returns:
            Decoded text string
        """
        return self.genai_tokenizer.decode(tokens)

    def _run_batch_logits_inference(self, prompts: List[str]) -> List[np.ndarray]:
        """
        Run batch inference using ONNX Runtime GenAI to get logits for multiple prompts.
        
        Args:
            prompts: List of input text strings
            
        Returns:
            List of logits arrays, one for each prompt
        """
        if not prompts:
            return []
        
        try:
            batch_size = len(prompts)
            
            # Create generator parameters with dynamic batch size
            params = self.og.GeneratorParams(self.genai_model)
            params.set_search_options(max_length=4096, do_sample=False)
            
            # Create generator
            generator = self.og.Generator(self.genai_model, params)
            
            # Use encode_batch for batch tokenization
            input_tokens_batch = self.genai_tokenizer.encode_batch(prompts)
            
            # Append batch tokens to generator
            generator.append_tokens(input_tokens_batch)
            
            # Get logits for the entire batch
            full_logits_tensor = generator.get_output("logits")
            logits_array = np.array(full_logits_tensor, dtype=np.float32)
            
            # Extract logits for each item in the batch
            results = []
            for i in range(batch_size):
                if len(logits_array.shape) == 4:  # (batch_size, num_beams, seq_len, vocab_size)
                    item_logits = logits_array[i, 0]  # Take first beam for greedy
                elif len(logits_array.shape) == 3:  # (batch_size, seq_len, vocab_size)
                    item_logits = logits_array[i]
                else:
                    raise ValueError(f"Unexpected logits shape: {logits_array.shape}")
                
                results.append(item_logits)
            
            eval_logger.debug(f"Processed batch of {batch_size} prompts")
            return results
            
        except Exception as e:
            eval_logger.error(f"Batch logits inference failed: {e}")
            # Fallback to single item processing
            results = []
            for prompt in prompts:
                try:
                    single_result = self._run_batch_logits_inference([prompt])
                    results.append(single_result[0] if single_result else np.empty((0, 0), dtype=np.float32))
                except Exception:
                    results.append(np.empty((0, 0), dtype=np.float32))
            return results

    def _loglikelihood_tokens(
        self, 
        requests: List["Instance"], 
        disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood for tokens using batch operations.
        
        Args:
            requests: List of instances containing context and continuation tokens
            disable_tqdm: Whether to disable progress bar
            
        Returns:
            List of tuples containing (log_likelihood, is_greedy) for each request
        """
        if not requests:
            return []
        
        # Prepare batch data
        prompts = []
        request_info = []
        
        for request in requests:
            _, context_enc, continuation_enc = request
            
            if len(continuation_enc) == 0:
                request_info.append((0, 0, True))  # empty continuation
                prompts.append("")  # placeholder
                continue
            
            # Combine context and continuation
            context_text = self.genai_tokenizer.decode(context_enc)
            continuation_text = self.genai_tokenizer.decode(continuation_enc)
            full_text = context_text + continuation_text
            
            prompts.append(full_text)
            request_info.append((len(context_enc), len(continuation_enc), False))
        
        # Run batch inference
        results = []
        batch_logits = self._run_batch_logits_inference(prompts)
        
        for i, logits in enumerate(tqdm(batch_logits, disable=disable_tqdm, desc="Computing log-likelihoods")):
            context_len, continuation_len, is_empty = request_info[i]
            
            if is_empty:
                results.append((0.0, True))
                continue
            
            try:
                if context_len >= logits.shape[0]:
                    results.append((0.0, False))
                    continue
                
                # Get logits for continuation positions
                start_idx = max(0, context_len - 1)
                end_idx = min(logits.shape[0], context_len + continuation_len - 1)
                
                if start_idx >= end_idx:
                    results.append((0.0, False))
                    continue
                
                # Get the original request for token access
                _, context_enc, continuation_enc = requests[i]
                
                cont_logits = logits[start_idx:end_idx, :]
                cont_tokens = np.array(continuation_enc[:end_idx - start_idx])
                
                # Calculate log probabilities
                log_probs = torch.log_softmax(torch.from_numpy(cont_logits), dim=-1)
                log_likelihood = sum(log_probs[j, token] for j, token in enumerate(cont_tokens))
                
                # Check if greedy (highest probability tokens)
                greedy_tokens = torch.argmax(log_probs, dim=-1).numpy()
                is_greedy = np.array_equal(greedy_tokens, cont_tokens)
                
                results.append((float(log_likelihood), bool(is_greedy)))
                
            except Exception as e:
                eval_logger.warning(f"Failed to compute loglikelihood for item {i}: {e}")
                results.append((0.0, False))
        
        return results

    def loglikelihood_rolling(self, requests: List["Instance"], disable_tqdm: bool = False) -> List[float]:
        """
        Compute rolling log-likelihood for perplexity using batch operations.
        
        Args:
            requests: List of instances containing text sequences
            disable_tqdm: Whether to disable progress bar
            
        Returns:
            List of average log-likelihood values for each request
        """
        if not requests:
            return []
        
        # Prepare prompts for batch processing
        prompts = []
        token_info = []
        
        for request in requests:
            string = request.args[0]
            tokens = self.tok_encode(string)
            
            if len(tokens) <= 1:
                prompts.append("")  # placeholder
                token_info.append(([], True))  # empty tokens flag
            else:
                prompts.append(string)
                token_info.append((tokens, False))
        
        # Run batch inference
        batch_logits = self._run_batch_logits_inference(prompts)
        results = []
        
        for i, logits in enumerate(tqdm(batch_logits, disable=disable_tqdm, desc="Computing rolling log-likelihoods")):
            tokens, is_empty = token_info[i]
            
            if is_empty:
                results.append(0.0)
                continue
            
            try:
                if logits.shape[0] == 0:
                    results.append(0.0)
                    continue
                
                # Calculate log-likelihood for all tokens except the first
                total_log_likelihood = 0.0
                valid_tokens = 0

                # logits[i] predicts token[i+1]
                for j in range(min(len(tokens) - 1, logits.shape[0])):
                    logit_vector = logits[j, :]
                    log_probs = torch.log_softmax(torch.from_numpy(logit_vector), dim=-1)
                    target_token = tokens[j + 1]  # Next token to predict
                    
                    if 0 <= target_token < len(log_probs):
                        total_log_likelihood += float(log_probs[target_token])
                        valid_tokens += 1
                
                # Average log-likelihood per token
                avg_log_likelihood = total_log_likelihood / max(valid_tokens, 1)
                results.append(avg_log_likelihood)
                
            except Exception as e:
                eval_logger.warning(f"Failed to compute rolling loglikelihood for item {i}: {e}")
                results.append(0.0)
        
        return results

    def generate_until(self, requests: List["Instance"], disable_tqdm: bool = False) -> List[str]:
        """
        Generate text until stopping criteria using batch operations.
        
        Args:
            requests: List of generation requests with context and generation kwargs
            disable_tqdm: Whether to disable progress bar
            
        Returns:
            List of generated text strings
        """
        if not requests:
            return []
        
        # Prepare batch data
        prompts = []
        gen_configs = []
        
        for request in requests:
            context, gen_kwargs = request.args
            
            max_gen_toks = gen_kwargs.get('max_gen_toks', self.max_gen_toks)
            until = gen_kwargs.get('until', [])
            
            prompts.append(context)
            gen_configs.append((max_gen_toks, until))
        
        # Run batch generation
        results = self._run_batch_generation(prompts, gen_configs, disable_tqdm)
        return results
    
    def _run_batch_generation(
        self, 
        prompts: List[str], 
        gen_configs: List[Tuple[int, List[str]]], 
        disable_tqdm: bool = False
    ) -> List[str]:
        """
        Run batch text generation using ONNX Runtime GenAI.
        
        Args:
            prompts: List of input prompts
            gen_configs: List of (max_tokens, stop_sequences) tuples
            disable_tqdm: Whether to disable progress bar
            
        Returns:
            List of generated text strings
        """
        if not prompts:
            return []
        
        try:
            batch_size = len(prompts)
            
            # Use the maximum token limit across all requests
            max_tokens = max(config[0] for config in gen_configs)
            
            # Create generator parameters
            params = self.og.GeneratorParams(self.genai_model)
            params.set_search_options(max_length=int(max_tokens), do_sample=False)
            
            # Create generator
            generator = self.og.Generator(self.genai_model, params)
            
            # Use encode_batch for batch tokenization
            input_tokens_batch = self.genai_tokenizer.encode_batch(prompts)
            
            # Append batch tokens to generator
            generator.append_tokens(input_tokens_batch)
            
            # Generate tokens for the entire batch
            start_time = time.time()
            while not generator.is_done():
                generator.generate_next_token()
            
            # Extract results for each item in the batch
            results = []
            for i in tqdm(range(batch_size), disable=disable_tqdm, desc="Processing batch generation"):
                try:
                    # Get the sequence for this batch item
                    full_sequence = generator.get_sequence(i)
                    input_length = len(input_tokens_batch[i])
                    
                    # Extract only the generated tokens
                    if len(full_sequence) > input_length:
                        generated_tokens = full_sequence[input_length:]
                        generated_text = self.genai_tokenizer.decode(generated_tokens)
                        
                        # Apply stopping criteria specific to this request
                        _, stop_sequences = gen_configs[i]
                        if stop_sequences:
                            for stop_seq in stop_sequences:
                                if stop_seq in generated_text:
                                    generated_text = generated_text.split(stop_seq)[0]
                                    break
                        
                        results.append(generated_text)
                    else:
                        results.append("")
                        
                except Exception as e:
                    eval_logger.warning(f"Failed to extract generation for batch item {i}: {e}")
                    results.append("")
            
            eval_logger.debug(f"Batch generation completed for {batch_size} prompts in {time.time() - start_time:.2f}s")
            return results
            
        except Exception as e:
            eval_logger.error(f"Batch generation failed: {e}")
            # Fallback to individual processing
            results = []
            for i, (prompt, (max_tokens, stop_sequences)) in enumerate(zip(prompts, gen_configs)):
                try:
                    single_result = self._run_batch_generation([prompt], [(max_tokens, stop_sequences)])
                    results.append(single_result[0] if single_result else "")
                except Exception:
                    results.append("")
            return results