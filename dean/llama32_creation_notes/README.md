# Llama 3.2 3B Creation Notes

This folder separates three different goals:

1. Understand how Meta created `llama3.2:3b`.
2. Understand what Ollama is running locally.
3. Build a realistic local model project of your own.

Important: recreating `llama3.2:3b` from scratch is not realistic on a normal
computer. Meta reports that Llama 3.2 3B training used large GPU clusters and
hundreds of thousands of H100 GPU-hours. The realistic path is to start from an
existing open model, then customize it with prompts, retrieval, or LoRA/QLoRA
fine-tuning.

Recommended reading order:

1. `01_how_llama32_3b_was_created.md`
2. `02_what_ollama_is_running.md`
3. `03_realistic_paths_to_your_own_local_llm.md`
4. `04_first_experiments.md`

Primary sources:

- Meta Llama 3.2 model card: https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md
- Hugging Face model card: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- Ollama model page: https://ollama.com/library/llama3.2
