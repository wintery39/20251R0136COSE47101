from typing import Dict, List, Any
import os

import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from agents.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline

# Configuration constants
AICROWD_SUBMISSION_BATCH_SIZE = 8

# GPU utilization settings - transformers 버전
MAX_GENERATION_TOKENS = 75

class LlamaVisionFinetuned(BaseAgent):
    """
    LlamaVisionModel is an implementation of BaseAgent using Meta's Llama 3.2 Vision models.
    
    This agent processes image-based queries using the Llama-3.2-11B-Vision-Instruct model
    and generates responses based on the visual content. It leverages transformers for 
    inference and supports multi-turn conversations.
    
    The model handles formatting of prompts and processes both single-turn and multi-turn
    conversations in a standardized way.
    
    Attributes:
        search_pipeline (UnifiedSearchPipeline): Pipeline for searching relevant information.
            Note: The web-search will be disabled in case of Task 1 (Single-source Augmentation) - so only image-search can be used in that case.
        model_name (str): Name of the Hugging Face model to use.
        max_gen_len (int): Maximum generation length for responses.
        model (MllamaForConditionalGeneration): The transformers model instance for inference.
        processor (AutoProcessor): The processor for handling images and text.
    """
    
    def __init__(
        self, search_pipeline: UnifiedSearchPipeline, model_name="KUInformatics/aicrowd_susik", max_gen_len=64
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.initialize_models()
        
    def initialize_models(self):
        """
        Initialize the transformers model and processor with appropriate settings.
        
        This configures the model for vision-language tasks with optimized
        GPU memory usage.
        """
        # Initialize the model and processor with transformers
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        
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

    def prepare_formatted_prompts(self, queries: List[str], images: List[Image.Image], message_histories: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
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
            List[List[Dict[str, Any]]]: List of formatted message lists ready for the model.
        """
        formatted_prompts = []

        for query_idx, (query, image) in enumerate(zip(queries, images)):
            message_history = message_histories[query_idx]
            
            # Structure messages with image placeholder
            SYSTEM_PROMPT = (
                """You are an extremely cautious and precise vision-language assistant. Your absolute primary objective is to avoid any incorrect statements. Risk avoidance is paramount.

                ### RULE 1: The 'Zero Doubt' Decision Gate
                1. **Internal Self-Correction & Declaration**:
                   After meticulously analyzing all available information (the image, the user's query), you MUST internally and explicitly decide: "Zero Doubt and Absolutely Confident" or "Any Doubt (however small) leads to Uncertain".
                2. **Output Policy**:
                   • If your internal decision is **"Any Doubt (however small) leads to Uncertain"** → you MUST respond *exactly* with: `I don't know.`
                   • If your internal decision is **"Zero Doubt and Absolutely Confident"** → you MAY proceed to answer clearly, concisely, and use no more than 3 sentences.

                ### RULE 2: Criteria for "Zero Doubt and Absolutely Confident" - EXTREMELY STRICT
                You can ONLY declare "Zero Doubt and Absolutely Confident" if **ALL of the following conditions are *flawlessly and unequivocally* met, without requiring *any* inference, assumption, or interpretation beyond what is explicitly stated or shown**:
                (a) **Direct, Explicit, and Unmistakable Visual Evidence**: The image *itself* MUST contain direct, explicit, and unmistakable visual evidence that fully and unambiguously answers the question. If there's any room for interpretation of visual cues, it's "Any Doubt".
                (b) **Literal Match for Question Scope**: The user's question must be answerable *solely* and *literally* either from this direct visual evidence OR from provided  information that *perfectly and literally matches* the visual context and *directly and explicitly answers* the question. No inferential steps are allowed to bridge gaps.
                (c) **Absolute Consistency and No Conflicts**: There must be absolutely no conflicting information, no discrepancies, and no ambiguities whatsoever between the image, and the chat history. The slightest hint of inconsistency defaults to "Any Doubt".
                (d) **Strict Adherence to Provided Information**: Do NOT use any external knowledge, general knowledge, or learned associations beyond what is explicitly present in the image. If the answer isn't *explicitly and literally* in the provided materials, it's "Any Doubt".
                (e) **Rejection of Inferential Reasoning**: If answering the question requires *any* form of inference, logical deduction (beyond simple matching), interpretation of nuanced details, or filling in missing information, you MUST classify it as "Any Doubt". Only direct factual recall from the provided context is permitted for an "Absolutely Confident" answer.

                ### RULE 3: Reasoning Style & Output Content
                1. **Internal Thought Process**: Your internal reasoning should be a silent, meticulous check against each criterion in RULE 2.
                2. **Output Protocol**:
                   • If "Any Doubt (however small) leads to Uncertain", your entire output MUST be `I don't know.` Nothing else.
                   • If "Zero Doubt and Absolutely Confident", provide *only* the direct, factual answer. Do NOT output your reasoning, a rephrasing of the question, or any an
                   cillary conversational text.

                ULTIMATE DIRECTIVE: Incorrect answers are heavily penalized (-1). `I don't know.` is neutral (0). Therefore, if there is *any possibility whatsoever* that your answer *might* be even slightly off, you MUST default to `I don't know.` Prioritize absolute accuracy and error avoidance over informativeness or appearing helpful. Do not guess. Do not speculate. Do not infer.
                """
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image"}]}
            ]
            
            # Add history if exists - only relevant for multi-turn conversations
            if message_history:
                messages = messages + message_history

            # Add query to the messages
            messages += [{"role": "user", "content": query}]
            
            formatted_prompts.append(messages)

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
        through the transformers model.
        
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
        # Prepare formatted prompts
        formatted_prompts = self.prepare_formatted_prompts(queries, images, message_histories)
        
        responses = []
        
        # Process each query individually (can be batched if needed)
        for prompt_messages, image in zip(formatted_prompts, images):
            # Apply chat template to get the formatted text
            formatted_text = self.processor.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            
            # Process inputs
            inputs = self.processor(
                text=formatted_text,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                generate_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_GENERATION_TOKENS,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # Decode response (exclude input tokens)
            generated_text = self.processor.batch_decode(
                generate_ids[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            responses.append(generated_text.strip())
        
        return responses