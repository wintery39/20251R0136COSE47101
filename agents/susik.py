from typing import Dict, List, Any
import os

import torch
from PIL import Image
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

# Number of search results to retrieve
NUM_SEARCH_RESULTS = 3

class RAGAgentSusik(BaseAgent):
    """
    SimpleRAGAgent demonstrates all the basic components you will need to create your 
    RAG submission for the CRAG-MM benchmark.
    Note: This implementation is not tuned for performance, and is intended for demonstration purposes only.
    
    This agent enhances responses by retrieving relevant information through a search pipeline
    and incorporating that context when generating answers. It follows a two-step approach:
    1. First, batch-summarize all images to generate effective search terms
    2. Then, retrieve relevant information and incorporate it into the final prompts
    
    The agent leverages batched processing at every stage to maximize efficiency.
    
    Note:
        This agent requires a search_pipeline for RAG functionality. Without it,
        the agent will raise a ValueError during initialization.
    
    Attributes:
        search_pipeline (UnifiedSearchPipeline): Pipeline for searching relevant information.
        model_name (str): Name of the Hugging Face model to use.
        max_gen_len (int): Maximum generation length for responses.
        llm (vllm.LLM): The vLLM model instance for inference.
        tokenizer: The tokenizer associated with the model.
    """

    def __init__(
        self, 
        search_pipeline: UnifiedSearchPipeline, 
        model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct", 
        max_gen_len: int = 64
    ):
        """
        Initialize the RAG agent with the necessary components.
        
        Args:
            search_pipeline (UnifiedSearchPipeline): A pipeline for searching web and image content.
                Note: The web-search will be disabled in case of Task 1 (Single-source Augmentation) - so only image-search can be used in that case.
                      Hence, this implementation of the RAG agent is not suitable for Task 1 (Single-source Augmentation).
            model_name (str): Hugging Face model name to use for vision-language processing.
            max_gen_len (int): Maximum generation length for model outputs.
            
        Raises:
            ValueError: If search_pipeline is None, as it's required for RAG functionality.
        """
        super().__init__(search_pipeline)
        
        if search_pipeline is None:
            raise ValueError("Search pipeline is required for RAG agent")
            
        self.model_name = model_name
        self.max_gen_len = max_gen_len
        
        self.initialize_models()
        
    def initialize_models(self):
        """
        Initialize the vLLM model and tokenizer with appropriate settings.
        
        This configures the model for vision-language tasks with optimized
        GPU memory usage and restricts to one image per prompt, as 
        Llama-3.2-Vision models do not handle multiple images well in a single prompt.
        
        Note:
            The limit_mm_per_prompt setting is critical as the current Llama vision models
            struggle with multiple images in a single conversation.
            Ref: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct/discussions/43#66f98f742094ed9e5f5107d4
        """
        print(f"Initializing {self.model_name} with vLLM...")
        
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
        
        print("Models loaded successfully")

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
    
    def batch_summarize_images(self, images: List[Image.Image]) -> List[str]:
        """
        Generate brief summaries for a batch of images to use as search keywords.
        
        This method efficiently processes all images in a single batch call to the model,
        resulting in better performance compared to sequential processing.
        
        Args:
            images (List[Image.Image]): List of images to summarize.
            
        Returns:
            List[str]: List of brief text summaries, one per image.
        """
        # Prepare image summarization prompts in batch
        summarize_prompt = (
            "This image will be used for a web search to find more information. "
            "Describe the key objects, scene, and any recognizable landmarks or context in the image "
            "as a comma-separated list of keywords. Focus on nouns and proper nouns."
            "Example: Eiffel Tower, Paris, daytime, tourists, clear sky"
        )
        
        inputs = []
        for image in images:
            messages = [
                {"role": "system", "content": (
                    "You are an expert image analyst. Your task is to generate a concise, keyword-rich summary of the provided image. "
                    "These keywords will be used to perform a web search to gather more information related to the image and the user's query. "
                    "Accuracy and relevance of the keywords are crucial for a successful search."
                )},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": summarize_prompt}]},
            ]
            
            # Format prompt using the tokenizer
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {
                    "image": image
                }
            })
        # Generate summaries in a single batch 
        outputs = self.llm.generate(
            inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=30,  # Short summary only
                skip_special_tokens=True
            )
        )
        
        # Extract and clean summaries
        summaries = [output.outputs[0].text.strip() for output in outputs]
        print(f"Generated {len(summaries)} image summaries")
        return summaries
    
    def prepare_rag_enhanced_inputs(
        self, 
        queries: List[str], 
        images: List[Image.Image], 
        image_summaries: List[str],
        message_histories: List[List[Dict[str, Any]]]
    ) -> List[dict]:
        """
        Prepare RAG-enhanced inputs for the model by retrieving relevant information in batch.
        
        This method:
        1. Uses image summaries combined with queries to perform effective searches
        2. Retrieves contextual information from the search_pipeline
        3. Formats prompts incorporating this retrieved information
        
        Args:
            queries (List[str]): List of user questions.
            images (List[Image.Image]): List of images to analyze.
            image_summaries (List[str]): List of image summaries for search.
            message_histories (List[List[Dict[str, Any]]]): List of conversation histories.
            
        Returns:
            List[dict]: List of input dictionaries ready for the model.
        """
        # Batch process search queries
        search_results_batch = []
        
        # Create combined search queries for each image+query pair
        search_queries = [f"{query} {summary}" for query, summary in zip(queries, image_summaries)]
        
        # Retrieve relevant information for each query
        for i, search_query in enumerate(search_queries):
            results = self.search_pipeline(search_query, k=NUM_SEARCH_RESULTS)
            search_results_batch.append(results)
        
        # Prepare formatted inputs with RAG context for each query
        inputs = []
        for idx, (query, image, message_history, search_results) in enumerate(
            zip(queries, images, message_histories, search_results_batch)
        ):
            # Create system prompt with RAG guidelines
            # 수정된 시스템 프롬프트: 페르소나 부여, 정보 활용 방식 구체화
            SYSTEM_PROMPT = (
                """You are an extremely cautious and precise vision-language assistant. Your absolute primary objective is to avoid any incorrect statements. Risk avoidance is paramount.

                ### RULE 1: The 'Zero Doubt' Decision Gate
                1. **Internal Self-Correction & Declaration**:
                   After meticulously analyzing all available information (the image, the user's query, chat history, and any provided search snippets), you MUST internally and explicitly decide: "Zero Doubt and Absolutely Confident" or "Any Doubt (however small) leads to Uncertain".
                2. **Output Policy**:
                   • If your internal decision is **"Any Doubt (however small) leads to Uncertain"** → you MUST respond *exactly* with: `I don't know.`
                   • If your internal decision is **"Zero Doubt and Absolutely Confident"** → you MAY proceed to answer clearly, concisely, and use no more than 3 sentences.

                ### RULE 2: Criteria for "Zero Doubt and Absolutely Confident" - EXTREMELY STRICT
                You can ONLY declare "Zero Doubt and Absolutely Confident" if **ALL of the following conditions are *flawlessly and unequivocally* met, without requiring *any* inference, assumption, or interpretation beyond what is explicitly stated or shown**:
                (a) **Direct, Explicit, and Unmistakable Visual Evidence**: The image *itself* MUST contain direct, explicit, and unmistakable visual evidence that fully and unambiguously answers the question. If there's any room for interpretation of visual cues, it's "Any Doubt".
                (b) **Literal Match for Question Scope**: The user's question must be answerable *solely* and *literally* either from this direct visual evidence OR from provided search information that *perfectly and literally matches* the visual context and *directly and explicitly answers* the question. No inferential steps are allowed to bridge gaps.
                (c) **Absolute Consistency and No Conflicts**: There must be absolutely no conflicting information, no discrepancies, and no ambiguities whatsoever between the image, the search snippets, and the chat history. The slightest hint of inconsistency defaults to "Any Doubt".
                (d) **Strict Adherence to Provided Information**: Do NOT use any external knowledge, general knowledge, or learned associations beyond what is explicitly present in the image or the provided search snippets. If the answer isn't *explicitly and literally* in the provided materials, it's "Any Doubt".
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

            # RAG 컨텍스트 도입부 수정 제안 (선택 사항, 더 강한 경고)
            rag_context = ""
            if search_results:
                # rag_context = "Here is some additional information that may help you answer:\n\n" # 기존
                rag_context = "WARNING: The following search snippets have been retrieved. They may be irrelevant, misleading, or incomplete. Evaluate them with extreme caution against the strict criteria in RULE 2:\n\n" # 수정 제안
                for i, result in enumerate(search_results):
                    snippet = result.get('page_snippet', '')
                    if snippet:
                        rag_context += f"[Info {i+1}] {snippet}\n\n"

                
            # Structure messages with image and RAG context
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image"}]}
            ]
            
            # Add conversation history for multi-turn conversations
            if message_history:
                messages = messages + message_history
                
            # Add RAG context as a separate user message if available
            if rag_context:
                messages.append({"role": "user", "content": rag_context})
                
            # Add the current query
            messages.append({"role": "user", "content": query})
            
            # Apply chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {
                    "image": image
                }
            })
        
        return inputs

    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        """
        Generate RAG-enhanced responses for a batch of queries with associated images.
        
        This method implements a complete RAG pipeline with efficient batch processing:
        1. First batch-summarize all images to generate search terms
        2. Then retrieve relevant information using these terms
        3. Finally, generate responses incorporating the retrieved context
        
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
        print(f"Processing batch of {len(queries)} queries with RAG")
        
        # Step 1: Batch summarize all images for search terms
        image_summaries = self.batch_summarize_images(images)
        
        # Step 2: Prepare RAG-enhanced inputs in batch
        rag_inputs = self.prepare_rag_enhanced_inputs(
            queries, images, image_summaries, message_histories
        )
        
        # Step 3: Generate responses using the batch of RAG-enhanced prompts
        print(f"Generating responses for {len(rag_inputs)} queries")
        outputs = self.llm.generate(
            rag_inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=MAX_GENERATION_TOKENS,
                skip_special_tokens=True
            )
        )
        
        # Extract and return the generated responses
        responses = [output.outputs[0].text for output in outputs]
        print(f"Successfully generated {len(responses)} responses")
        return responses
