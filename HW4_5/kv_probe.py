import os, sys, time
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"   # allow testing beyond the model's 32k limit (memory test only)
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

ctx, do_run = int(sys.argv[1]), sys.argv[2] == "run"
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="float16", max_model_len=ctx,
          gpu_memory_utilization=0.80, enforce_eager=True, max_num_seqs=8)   # same settings as HW4.a

if do_run:   # real batch-8 request at this context length
    gen = 32
    prompts = [TokensPrompt(prompt_token_ids=[9707] * (ctx - gen)) for _ in range(8)]
    sp = SamplingParams(temperature=0.0, max_tokens=gen, ignore_eos=True)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    done = sum(len(o.outputs[0].token_ids) == gen for o in outs)
    print(f"RESULT_RUN ctx={ctx} completed={done}/8 seconds={dt:.2f}", flush=True)
print("RESULT_OK", flush=True)
