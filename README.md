# What Sub-Billion Models Don’t Know About Brazil

**Probing and adapting Qwen3-0.6B for offline K-12 use in Brazilian Portuguese**

Daniel Fazzioni, Daniel M. Pedrozo, Bryan L. M. de Oliveira, Ivo P. de Medeiros, Sávio Teles, Adailton F. de Araújo
Universidade Federal de Goiás (UFG) · Centro de Excelência em Inteligência Artificial (CEIA)
*BRACIS 2026*

> [!WARNING]
> **Single-turn only.** The SFT and RL models were trained exclusively on single-turn exchanges (one question, one answer). Multi-turn dialogue was not part of training, so quality may degrade after the first turn. Send each question as a fresh conversation.

> [!CAUTION]
> **No safety alignment.** The pipeline has no dedicated safety stage. Do not expose these models to students without additional safeguards against harmful content and hallucination.

---

## Overview

Sub-billion-parameter models can run offline on the laptops already in Brazilian public schools, but do they know enough about Brazil to help a student? Off-the-shelf **Qwen3-0.6B** scores **40.9%** on an aggregate Brazilian benchmark suite, yet answers at most **11%** of direct factual questions about Brazil correctly. Aggregate benchmarks measure *skill*: following instructions, parsing multiple choice, writing fluent Portuguese. They miss *knowledge*.

This repository introduces the **Brazilian Knowledge Probe (BKP)** to measure knowledge separately from skill, and a three-stage adaptation pipeline: **CPT → SFT → RL (GRPO)**. Tracking both axes shows a knowledge–skill trade-off:

- **CPT** installs Brazilian facts: capitals go from 45% to 100% (base format).
- **SFT** restores the chat format but loses part of that knowledge and drops aggregate accuracy by 9.3 pp.
- **RL** recovers skill to 42.8% (+1.9 pp over baseline) and brings more knowledge into chat: +63 pp on capitals over the instruct baseline.

---

## Released resources

| Resource | Hugging Face | Description |
|---|---|---|
| **BKP** (benchmark) | [`Fazzioni/qa-brazil-bench`](https://huggingface.co/datasets/Fazzioni/qa-brazil-bench) | 116 single-answer factual questions about Brazil, in Portuguese |
| **CPT** | [`CEIA-POSITIVO/BKP-Qwen-0.6B-CPT`](https://huggingface.co/CEIA-POSITIVO/BKP-Qwen-0.6B-CPT) | Stage 1: Qwen3-0.6B-Base after continued pre-training. **Not chat-aligned**, so prompt it with raw text |
| **CPT+SFT** | [`CEIA-POSITIVO/BKP-Qwen-0.6B-CPT-SFT`](https://huggingface.co/CEIA-POSITIVO/BKP-Qwen-0.6B-CPT-SFT) | Stage 2: chat model trained on student Q&A with `<think>` reasoning |
| **CPT+SFT+RL** | [`CEIA-POSITIVO/BKP-Qwen-0.6B-CPT-SFT-RL`](https://huggingface.co/CEIA-POSITIVO/BKP-Qwen-0.6B-CPT-SFT-RL) | Stage 3: SFT model after GRPO. Recommended for use |

---

## Quick start

### Chat models (CPT+SFT, CPT+SFT+RL)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "CEIA-POSITIVO/BKP-Qwen-0.6B-CPT-SFT-RL"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")

# Single turn only: one question per conversation
messages = [{"role": "user", "content": "Qual é a capital do Paraná?"}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
)
outputs = model.generate(inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True))
```

The chat models were trained to reason inside a `<think>…</think>` block before the final answer.

### CPT model (raw text)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "CEIA-POSITIVO/BKP-Qwen-0.6B-CPT"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")

prompt = "Pergunta: Qual é a capital do Paraná?\nResposta:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=16)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

### Benchmark

```python
from datasets import load_dataset

bkp = load_dataset("Fazzioni/qa-brazil-bench")
print(bkp)
```

---

## Brazilian Knowledge Probe (BKP)

116 factual questions with a single short answer (a city, an acronym, a person, a work), covering only facts that are specific to Brazil. A multilingual model therefore can’t answer them by transferring general world knowledge. Many institution and literature items describe the entity or the work, so the model has to map the description to the name instead of expanding a memorized acronym.

| Category | Items | Example (translated) |
|---|---:|---|
| State capitals | 27 | “What is the capital of the state of Amazonas?” |
| Federal institutions | 30 | “Which institution oversees the legal profession and runs the bar exam?” |
| Presidents | 29 | “Who was the president of Brazil in the year 1976?” |
| Literature | 30 | “Which Suassuna play blends Catholic tradition and cordel through João Grilo?” |

**Evaluation protocols.** Each checkpoint is probed in its native regime, with 8 generations per item, reporting accuracy and pass@k (k ∈ {1, 2, 4, 8}):

- **Protocol A (base models):** raw text with 5 same-category question–answer examples.
- **Protocol B (chat models):** the model’s native chat template, with the same 5 examples formatted as prior user/assistant turns.

Absolute values across the two protocols are not directly comparable. Compare trajectories within each protocol.

---

## Adaptation pipeline

| Stage | Data | Setup |
|---|---|---|
| **1 · CPT** | ~300B tokens: GigaVerbo, PT-BR web (75%) · `cnmoro/reasoning-v1-20m-portuguese`, PT-BR reasoning traces (15%) · Infiwebmath-4plus / Finemath-4plus, math (10%) | NVIDIA NeMo AutoModel; sequence length 4096; global batch 768; peak LR 3×10⁻⁵, cosine decay; AdamW; bf16; ~10⁵ updates |
| **2 · SFT** | ~400k real student questions answered by Qwen3-8B (with `<think>` traces), kept only when the answer matches a human-validated reference | Cross-entropy on the answer span (thinking included); LR 4×10⁻⁶; 2 epochs |
| **3 · RL (GRPO)** | Held-out prompts from the same educational distribution | G = 8 rollouts per prompt; reward from a frozen **gpt-oss-20B** judge with a reference answer; checkpoints at 60 and 100 steps |

**GRPO reward.** `r = s_judge + s_fmt`

| Component | Signal | Weight |
|---|---|---:|
| `s_judge` | task_coverage | +0.30 |
| | relative_quality | +0.30 |
| | hallucination | −0.20 |
| | internal_coherence | +0.10 |
| | usefulness | +0.10 |
| `s_fmt` | `<think>` present | +0.15 |
| | `</think>` present | +0.15 |

A fully hallucinated rollout, or one not in Brazilian Portuguese, gets `s_judge = 0`.

**Why this order.** Early runs that applied GRPO without prior adaptation degraded the model’s Portuguese, producing invented words that blend languages. Applying GRPO directly to the CPT checkpoint (skipping SFT) led to reward hacking within tens of steps: rollouts collapsed onto judge-pleasing patterns while held-out accuracy stayed flat. At this scale, SFT before RL is load-bearing.

---

## Results

### Knowledge: BKP accuracy (%)

| Category | Qwen3-0.6B-Base (A) | CPT (A) | Qwen3-0.6B instruct (B) | CPT+SFT (B) | CPT+SFT+RL (B) |
|---|---:|---:|---:|---:|---:|
| Capitals | 45 | **100** | 11 | 70 | **74** |
| Institutions | 24 | **72** | 0 | 17 | ≈17 |
| Literature | 0.4 | 17 | 3 | 7 | **17** |
| Presidents | 6.5 | **27** | 3 | 7 | **14** |

*(A) = Protocol A, raw text; (B) = Protocol B, chat, pass@1. CPT+SFT+RL shows the best of the 60- and 100-step checkpoints: 100 steps for capitals and literature, 60 for presidents. Institutions stay near 17% at every post-training checkpoint.*

### Skill: aggregate Brazilian suite (%, chat template, no CoT)

| Task | Qwen3-0.6B instruct | CPT+SFT | CPT+SFT+RL (60 steps) | CPT+SFT+RL (100 steps) |
|---|---:|---:|---:|---:|
| MMLU-PT | 42.4 | 29.6 | 35.0 | 36.6 |
| Vestibular | 42.6 | 28.7 | 35.6 | 40.2 |
| OAB | 32.2 | 24.2 | 30.7 | 32.1 |
| BLUEX | 41.2 | 22.4 | 31.1 | 32.2 |
| ENEM | 48.0 | 27.1 | 37.1 | 40.7 |
| ASSIN2 | 50.6 | 56.2 | 56.7 | 62.1 |
| Math | 62.9 | 20.1 | 32.5 | 35.3 |
| Capitals-QA | 7.4 | 44.4 | 66.7 | 62.9 |
| **Mean** | **40.9** | **31.6** | **40.7** | **42.8** |

### On-device

Measured on a **Positivo Vision C15M** school laptop (Intel Celeron N4500, 2 cores; 4 GB RAM; 64 GB eMMC; stock Windows 11): Qwen3-0.6B at FP16 runs at **8–9 tokens/s**, with a time-to-first-token of **~0.6 s**.

---

## Key findings

1. **Measure knowledge and skill separately.** Aggregate scores hide the factual gap: 40.9% on the suite, at most 11% on the BKP.
2. **The stages negotiate a trade-off.** CPT installs knowledge, SFT loses part of it in chat (as does Qwen’s own instruct release), and RL wins it back: +63 pp on capitals and +1.9 pp overall.
3. **Installed is not accessible.** Institutions reach 72% after CPT but stay near 17% in chat at every checkpoint. Part of the knowledge is in the model but doesn’t come out in conversation.

## Limitations

- Small probe (116 items, ~10 pp bootstrap confidence intervals) and one training seed per checkpoint.
- BKP items come from public sources (e.g., Wikipedia-PT, Brasil Escola) that may overlap the CPT corpus, so CPT gains are an upper bound for unseen facts.
- Pipeline tested only on the Qwen3 family.
- Single-turn training only; no safety-alignment stage.

## Future work

- A training recipe that improves skill and knowledge together.
- Making knowledge installed during CPT reachable in the chat format.

---

## Citation

```bibtex
@inproceedings{fazzioni2026subbillion,
  title     = {What Sub-Billion Models Don't Know About Brazil},
  author    = {Fazzioni, Daniel and Pedrozo, Daniel M. and de Oliveira, Bryan L. M. and de Medeiros, Ivo P. and Teles, S{\'a}vio and de Ara{\'u}jo, Adailton F.},
  booktitle = {Proceedings of the Brazilian Conference on Intelligent Systems (BRACIS)},
  year      = {2026}
}
```

## Acknowledgments

This work was supported by the Center of Excellence in Artificial Intelligence (CEIA) at the Federal University of Goiás (UFG), the Brazilian Company of Research and Industrial Innovation (EMBRAPII), and Positivo Tecnologia.

Contact: fazzioni@egresso.ufg.br
