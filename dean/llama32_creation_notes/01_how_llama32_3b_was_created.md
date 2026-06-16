# How Llama 3.2 3B Was Created

Meta describes Llama 3.2 as a family of pretrained and instruction-tuned
autoregressive transformer language models. The text-only family includes 1B
and 3B sizes. The 3B model has about 3.21 billion parameters.

## The High-Level Recipe

Llama 3.2 3B was created in stages:

1. Architecture design
   - Autoregressive transformer model.
   - Grouped-Query Attention for more efficient inference.
   - Shared embeddings.
   - Long context support in the full text-only model family.

2. Large-scale pretraining
   - Trained on a large mixture of publicly available online data.
   - Meta reports up to 9 trillion pretraining tokens for the family.
   - Knowledge cutoff: December 2023.

3. Distillation from larger models
   - For the 1B and 3B models, Meta used logits from Llama 3.1 8B and 70B
     during model development.
   - In plain English: larger teacher models helped train the smaller model.

4. Post-training / alignment
   - Supervised Fine-Tuning, or SFT.
   - Rejection Sampling, or RS.
   - Direct Preference Optimization, or DPO.
   - The instruction-tuned version is the one meant for chat and assistant use.

5. Quantization
   - Smaller local versions are compressed so they use less RAM/VRAM.
   - Meta describes 4-bit groupwise weight quantization and 8-bit activation
     quantization for their mobile-oriented quantized models.
   - Ollama commonly serves quantized variants so a multi-billion-parameter
     model can run on a laptop.

## Why Recreating It From Scratch Is Not Practical

Meta reports roughly 460k H100 GPU-hours for Llama 3.2 3B training, plus more
work for quantization and fine-tuning. That is data-center scale.

So your practical goal should not be:

> Train a new Llama 3.2 3B equivalent from random weights.

Your practical goal should be one of:

> Customize an existing local model.

> Fine-tune a small open model on your domain.

> Build a domain assistant around a local model using tools, retrieval, and
> validation.

For your dean spreadsheet assistant, the best path is probably a local model
plus strong structured prompting, examples, validation, and maybe LoRA
fine-tuning later.
