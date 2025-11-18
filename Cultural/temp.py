import os
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import socket
import torch
from typing import Optional, Union

def load_hf_causal_lm_pipeline(
        repo_id : str,
        model_dir : Optional[Union[str, os.PathLike]] = None,
        task : str = "text-generation",
        max_new_tokens : int = 1024,
        temperature : float = 0.6,
        top_p : float = 0.9,
        top_k : int = 20,
        repetition_penalty : float = 1.2,
        torch_dtype = torch.bfloat16,
        expose_prompt : bool = False
):
    hostname = socket.gethostname()
    # 목포대 서버에서는 volume에 모델파일들을 관리
    if hostname == 'ubuntu' and model_dir is None:
        model_dir = "/volume/hf_cache/hub"

    model = AutoModelForCausalLM.from_pretrained(repo_id, cache_dir=model_dir, torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, cache_dir = model_dir)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    pipe = pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        return_full_text = expose_prompt
    )
    return HuggingFacePipeline(pipeline=pipe)