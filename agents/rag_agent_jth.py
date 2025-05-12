# rag_agent_jth.py
from typing import List, Dict, Any
import re, collections
from PIL import Image
import vllm

from agents.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline

# ---------------- 전역 설정 ----------------
AICROWD_SUBMISSION_BATCH_SIZE = 8

VLLM_TENSOR_PARALLEL_SIZE   = 1     # 제출 시 1
VLLM_GPU_MEMORY_UTILIZATION = 0.85

MAX_MODEL_LEN         = 8192
MAX_NUM_SEQS          = 2
MAX_GENERATION_TOKENS = 75

NUM_SEARCH_RESULTS    = 5           # ← recall ↑
# -------------------------------------------------

class SimpleRAGAgentJTH(BaseAgent):
    """
    CRAG-MM 제출용 - JTH 버전 (보수적 답변 + soft-support self-검증)
    """

    # ---------- 초기화 ----------
    def __init__(
        self,
        search_pipeline: UnifiedSearchPipeline,
        model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
    ):
        super().__init__(search_pipeline)
        if search_pipeline is None:
            raise ValueError("search_pipeline is required")

        self.model_name = model_name
        self._init_models()

    def _init_models(self):
        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size   = VLLM_TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization = VLLM_GPU_MEMORY_UTILIZATION,
            max_model_len          = MAX_MODEL_LEN,
            max_num_seqs           = MAX_NUM_SEQS,
            trust_remote_code      = True,
            dtype                  = "bfloat16",
            enforce_eager          = True,
            limit_mm_per_prompt    = {"image": 1},
        )
        self.tokenizer = self.llm.get_tokenizer()

    # ---------- 배치 크기 ----------
    def get_batch_size(self) -> int:
        return AICROWD_SUBMISSION_BATCH_SIZE

    # ---------- 이미지 요약 ----------
    def batch_summarize_images(self, images: List[Image.Image]) -> List[str]:
        prompt_txt = ("Describe the main objects or landmarks in this image "
                      "as comma-separated keywords.")
        inputs = []
        for im in images:
            msgs = [
                {"role": "system", "content": "Return only keywords."},
                {"role": "user",
                 "content": [{"type": "image"},
                             {"type": "text", "text": prompt_txt}]},
            ]
            p = self.tokenizer.apply_chat_template(msgs, True, tokenize=False)
            inputs.append({"prompt": p, "multi_modal_data": {"image": im}})

        outs = self.llm.generate(
            inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0.1, top_p=0.9,
                max_tokens=30, skip_special_tokens=True,
            ),
        )
        return [o.outputs[0].text.strip() for o in outs]

    # ---------- soft-support (간단 키워드 검사) ----------
    def _soft_support(self, ctx: str, ans: str, k: int = 2) -> bool:
        toks = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", ans) if len(t) >= 4]
        uniq = list(collections.OrderedDict.fromkeys(toks))
        return sum(1 for t in uniq if t in ctx.lower()) >= k

    # LLM + soft support 이중 검증
    def verify_answer(self, ctx: str, ques: str, ans: str) -> bool:
        msgs = [
            {"role": "system",
             "content": "Return 'yes' if ANSWER is fully supported by CONTEXT, else 'no'."},
            {"role": "user",
             "content": f"CONTEXT:\n{ctx}\n\nQUESTION:\n{ques}\n\nANSWER:\n{ans}\n\nSupported?"},
        ]
        p = self.tokenizer.apply_chat_template(msgs, True, tokenize=False)
        out = self.llm.generate(
            [{"prompt": p}],
            sampling_params=vllm.SamplingParams(
                temperature=0.2, max_tokens=5, skip_special_tokens=True),
        )[0].outputs[0].text.strip().lower()
        return out.startswith("y") or self._soft_support(ctx, ans)

    # ---------- RAG 입력 ----------
    def _build_inputs(
        self,
        queries: List[str],
        images: List[Image.Image],
        img_sums: List[str],
        histories: List[List[Dict[str, Any]]],
    ):
        search_qs   = [f"{q} {s}" for q, s in zip(queries, img_sums)]
        search_bat  = [self.search_pipeline(q, k=NUM_SEARCH_RESULTS) for q in search_qs]

        inputs = []
        for q, im, hist, sr in zip(queries, images, histories, search_bat):
            rag_ctx = ""
            if sr:
                rag_ctx = ("Below are retrieved snippets (may include noise):\n\n" +
                           "\n\n".join(f"[Info {i+1}] {r.get('page_snippet','')}"
                                       for i, r in enumerate(sr) if r.get("page_snippet")))

            SYS = ("You are a cautious V-L assistant. "
                   "Use ONLY explicit evidence from image/snippets. "
                   "If not 100% sure, respond exactly: I don't know")

            msgs = [{"role": "system", "content": SYS},
                    {"role": "user",   "content": [{"type": "image"}]}]
            if hist:
                msgs += hist
            if rag_ctx:
                msgs.append({"role": "user", "content": rag_ctx})
            msgs.append({"role": "user", "content": q})

            prompt = self.tokenizer.apply_chat_template(msgs, True, tokenize=False)
            inputs.append({"prompt": prompt, "multi_modal_data": {"image": im}})
        return inputs, search_bat

    # ---------- 메인 ----------
    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        histories: List[List[Dict[str, Any]]],
    ) -> List[str]:

        img_sums = self.batch_summarize_images(images)
        rag_inputs, search_batches = self._build_inputs(
            queries, images, img_sums, histories
        )

        # 1) 두 후보 생성 → logprob 큰 것 선택
        sampl = vllm.SamplingParams(
            temperature=0.1, top_p=0.9,
            max_tokens=MAX_GENERATION_TOKENS,
            skip_special_tokens=True, n=2,
        )
        outs = self.llm.generate(rag_inputs, sampl)
        drafts = [max(o.outputs, key=lambda o_: o_.cumulative_logprob).text.strip()
                  for o in outs]

        # 2) 검증 → 실패 시 I don't know
        finals = []
        for ans, q, sr in zip(drafts, queries, search_batches):
            ctx = "\n".join(r.get("page_snippet","") for r in sr if r.get("page_snippet"))
            finals.append(ans if self.verify_answer(ctx, q, ans) else "I don't know")

        return finals

