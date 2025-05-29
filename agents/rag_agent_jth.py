from typing import Dict, List, Any
import os
from PIL import Image

import torch
import vllm

from agents.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline

AICROWD_SUBMISSION_BATCH_SIZE = 8
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_GPU_MEMORY_UTILIZATION = 0.85
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 2
MAX_GENERATION_TOKENS = 75
NUM_SEARCH_RESULTS = 3

class SimpleRAGAgentJTH(BaseAgent):
    def __init__(
        self,
        search_pipeline: UnifiedSearchPipeline,
        model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
        max_gen_len: int = 64,
    ):
        super().__init__(search_pipeline)
        if search_pipeline is None:
            raise ValueError("Search pipeline is required for RAG agent")

        self.model_name = model_name
        self.max_gen_len = max_gen_len
        self.initialize_models()

    def initialize_models(self):
        print(f"[vLLM] Loading {self.model_name}")
        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=True,
            limit_mm_per_prompt={"image": 1},
        )
        self.tokenizer = self.llm.get_tokenizer()
        print("[vLLM] Ready!")

    def get_batch_size(self) -> int:
        return AICROWD_SUBMISSION_BATCH_SIZE

    def batch_summarize_images(self, images: List[Image.Image]) -> List[str]:
        summarize_prompt = (
            "Please summarize the image with one concise sentence describing its key elements."
        )
        inputs = []
        for im in images:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that accurately describes images. "
                        "Your responses will be used only as search keywords."
                    ),
                },
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": summarize_prompt}]},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs.append({"prompt": prompt, "multi_modal_data": {"image": im}})

        outs = self.llm.generate(
            inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0.1, top_p=0.9, max_tokens=30, skip_special_tokens=True
            ),
        )

        return [o.outputs[0].text.strip() for o in outs]

    def prepare_rag_enhanced_inputs(
        self,
        queries: List[str],
        images: List[Image.Image],
        image_summaries: List[str],
        histories: List[List[Dict[str, Any]]],
    ) -> List[dict]:
        search_queries = [f"{q} {s}" for q, s in zip(queries, image_summaries)]
        search_batches = [self.search_pipeline(qry, k=NUM_SEARCH_RESULTS) for qry in search_queries]

        inputs = []
        for q, im, hist, sr in zip(queries, images, histories, search_batches):
            rag_ctx = ""
            if sr:
                rag_ctx = "Below is some additional information that may help answer:\n\n"
                for idx, r in enumerate(sr, 1):
                    snip = r.get("page_snippet", "")
                    if snip:
                        rag_ctx += f"[Info {idx}] {snip}\n\n"

            SYSTEM_PROMPT = (
                "You are a concise, truthful multimodal assistant. "
                "Answer ONLY if the information is explicitly present "
                "in the image or the provided passages. "
                "If not 100% sure, respond exactly with: I don't know."
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image"}]},
            ]
            if hist:
                messages.extend(hist)
            if rag_ctx:
                messages.append({"role": "user", "content": rag_ctx})
            messages.append({"role": "user", "content": q})

            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs.append({"prompt": prompt, "multi_modal_data": {"image": im}})

        return inputs

    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        print(f"Processing batch of {len(queries)} queries with RAG")
        image_summaries = self.batch_summarize_images(images)
        rag_inputs = self.prepare_rag_enhanced_inputs(
            queries, images, image_summaries, message_histories
        )
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
        responses = [output.outputs[0].text.strip() for output in outputs]
        print(f"Successfully generated {len(responses)} responses")
        return responses


