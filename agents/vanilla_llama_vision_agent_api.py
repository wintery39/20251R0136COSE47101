from typing import Dict, List, Any
import os

import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from agents.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline

import vllm

# Configuration constants
AICROWD_SUBMISSION_BATCH_SIZE = 8

# GPU utilization settings 
# Change VLLM_TENSOR_PARALLEL_SIZE during local runs based on your available GPUs
# For example, if you have 2 GPUs on the server, set VLLM_TENSOR_PARALLEL_SIZE=2. 
# You may need to uncomment the following line to perform local evaluation with VLLM_TENSOR_PARALLEL_SIZE>1. 
# os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

#### Please ensure that when you submit, VLLM_TENSOR_PARALLEL_SIZE=1. 
VLLM_TENSOR_PARALLEL_SIZE = 1 
VLLM_GPU_MEMORY_UTILIZATION = 0.85 

# These are model specific parameters to get the model to run on a single NVIDIA L40s GPU
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 2

MAX_GENERATION_TOKENS = 75

class APILlamaVisionModel(BaseAgent):
    """
    LlamaVisionModel is an implementation of BaseAgent using Meta's Llama 3.2 Vision models.
    
    This agent processes image-based queries using the Llama-3.2-11B-Vision-Instruct model
    and generates responses based on the visual content. It leverages vLLM for efficient,
    batched inference and supports multi-turn conversations.
    
    The model handles formatting of prompts and processes both single-turn and multi-turn
    conversations in a standardized way.
    
    Attributes:
        search_pipeline (UnifiedSearchPipeline): Pipeline for searching relevant information.
            Note: The web-search will be disabled in case of Task 1 (Single-source Augmentation) - so only image-search can be used in that case.
        model_name (str): Name of the Hugging Face model to use.
        max_gen_len (int): Maximum generation length for responses.
        llm (vllm.LLM): The vLLM model instance for inference.
        tokenizer: The tokenizer associated with the model.
    """
    
    def __init__(
        self, search_pipeline: UnifiedSearchPipeline, model_name="meta-llama/Llama-3.2-11B-Vision-Instruct", max_gen_len=64
    ):
        """
        Initialize the agent
        
        Args:
            search_pipeline (UnifiedSearchPipeline): A pipeline for searching web and image content.
                Note: The web-search will be disabled in case of Task 1 (Single-source Augmentation) - so only image-search can be used in that case.
            model_name (str): Hugging Face model name to use for vision-language processing.
            max_gen_len (int): Maximum generation length for model outputs.
        """
        super().__init__(search_pipeline)
        self.model_name = model_name
        self.max_gen_len = max_gen_len
        
        self.initialize_models()
        
    def initialize_models(self):
        """
        Initialize the vLLM model and tokenizer with appropriate settings.
        
        This configures the model for vision-language tasks with optimized
        GPU memory usage and restricts to one image per prompt.
        """
        # Initialize the model with vLLM
        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE, 
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION, 
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=True,
            limit_mm_per_prompt={
                "image": 1 
            } # In the CRAG-MM dataset, every conversation has at most 1 image
        )
        self.tokenizer = self.llm.get_tokenizer()
        
        print("Loaded models")

    def get_batch_size(self) -> int:
        """
        Determines the batch size used by the evaluator when calling batch_generate_response.
        
        The evaluator uses this value to determine how many queries to send in each batch.
        Valid values are integers between 1 and 16.
        
        Returns:
            int: The batch size, indicating how many queries should be processed together 
                 in a single batch.
        """
        return AICROWD_SUBMISSION_BATCH_SIZE

    def get_api_results(self, origin_image: Image.Image, k=5) -> str:
        image_li = []
        
        image_li.append(origin_image)

        width, height = origin_image.size

        # 중간 지점
        mid_x = width // 2
        mid_y = height // 2

        # 각 영역 자르기 (crop의 인자는 (left, upper, right, lower))
        divide_coordinates = [
            (0, 0, mid_x, mid_y),  # 왼쪽 위
            (mid_x, 0, width, mid_y),  # 오른쪽 위
            (0, mid_y, mid_x, height),  # 왼쪽 아래
            (mid_x, mid_y, width, height)  # 오른쪽 아래
        ]

        for x, y, w, h in divide_coordinates:
            cropped_image = origin_image.crop((x, y, w, h))
            image_li.append(cropped_image)
        
        results = []
        
        for image in image_li:
            response = self.search_pipeline(image, k = 2)
            if response is None:
                continue
            
            results.extend(response)

        final_response_str = ""
        for result in response:
            if float(result["score"]) < 0.75:
                break

            for entity in result['entities']:
                for idx, (key, value) in enumerate(entity.items()):
                    if idx < 5:
                        break
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            final_response_str += f"{subkey}: {subvalue}\n"
                    else:
                        final_response_str += f"{key}: {value}\n"
        
        return final_response_str

    def prepare_formatted_prompts(self, queries: List[str], images: List[Image.Image], message_histories: List[List[Dict[str, Any]]]) -> List[str]:
        """
        Prepare formatted prompts for the model by applying the chat template.
        
        This method formats the prompts according to Llama's chat template,
        including system prompts, images, conversation history, and the current query.
        
        Args:
            queries (List[str]): List of user questions or prompts.
            images (List[Image.Image]): List of PIL Image objects to analyze.
            message_histories (List[List[Dict[str, Any]]]): List of conversation histories.
                Each conversation history is a list of message dictionaries with the
                following structure:
                - For user messages: {"role": "user", "content": "user message text"}
                - For assistant messages: {"role": "assistant", "content": "assistant response"}
                
                For multi-turn conversations, the history contains all previous turns.
                For single-turn queries, the history will be an empty list.
                
        Returns:
            List[str]: List of formatted prompts ready for the model.
        """
        formatted_prompts = []

        for query_idx, (query, image) in enumerate(zip(queries, images)):
            message_history = message_histories[query_idx]
            
            # Structure messages with image placeholder
            SYSTEM_PROMPT = ("You are a helpful assistant that truthfully answers user questions about the provided image."
                           "Keep your response concise and to the point. If you don't know the answer, respond with 'I don't know'.")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image"}]}
            ]

            results = self.get_api_results(image, k=2)

            if results:
                API_PROMPT = (
                    "The following information may help you better understand the context. "
                    "It provides factual reference details related to the task. "
                    "You can use this information to improve the quality or accuracy of your response, "
                    "but feel free to ignore it if it is not relevant.\n\n"
                )
                API_PROMPT += results
                messages.append({"role": "user", "content": API_PROMPT})
                
            # Add history if exists - only relevant for multi-turn conversations
            if message_history:
                messages = messages + message_history

            # Add query to the messages
            messages += [{"role": "user", "content": query}]
            
            # Apply chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

            formatted_prompts.append(formatted_prompt)

        return formatted_prompts
  
 

    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        """
        Generate responses for a batch of queries with associated images.
        
        This method is the main entry point called by the evaluator. It handles
        preparing the prompts, combining them with images, and generating responses
        through the vLLM model.
        
        Args:
            queries (List[str]): List of user questions or prompts.
            images (List[Image.Image]): List of PIL Image objects, one per query. 
                The evaluator will ensure that the dataset rows which have just
                image_url are populated with the associated image.
            message_histories (List[List[Dict[str, Any]]]): List of conversation histories,
                one per query. Each history is a list of message dictionaries with
                'role' and 'content' keys in the following format:
                
                - For single-turn conversations: Empty list []
                - For multi-turn conversations: List of previous message turns in the format:
                  [
                    {"role": "user", "content": "first user message"},
                    {"role": "assistant", "content": "first assistant response"},
                    {"role": "user", "content": "follow-up question"},
                    {"role": "assistant", "content": "follow-up response"},
                    ...
                  ]
                
        Returns:
            List[str]: List of generated responses, one per input query.
        """        
        # Prepare prompts and image data
        formatted_prompts = self.prepare_formatted_prompts(queries, images, message_histories)
        
        # Create input list with multimodal data
        inputs = [{
            "prompt": prompt,
            "multi_modal_data": {
                "image": img
            }
        } for prompt, img in zip(formatted_prompts, images)]

        # Generate responses
        outputs = self.llm.generate(
            inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=MAX_GENERATION_TOKENS,
                skip_special_tokens=True
            )
        )

        responses = [output.outputs[0].text for output in outputs]
        return responses
