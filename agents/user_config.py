from agents.rag_agent import SimpleRAGAgent
from agents.random_agent import RandomAgent
from agents.vanilla_llama_vision_agent_api import APILlamaVisionModel
from agents.susik2_finetuned import LlamaVisionFinetuned

# UserAgent = RandomAgent
# UserAgent = SimpleRAGAgent
#UserAgent = APILlamaVisionModel
UserAgent = LlamaVisionFinetuned
