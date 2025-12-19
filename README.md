# vLLM 4-bit Quantization Notebooks

Two Jupyter notebooks are included to help you quantize Llama Instruct models to 4-bit with vLLM’s compression tools, serve them, make a sample call, and run a quick 100-sample accuracy probe.

## Files
- `gptq_quantization.ipynb` – W4A16 weight-only quantization with GPTQ (llm-compressor one-shot flow).
- `awq_quantization.ipynb` – W4A16 Activation-Aware Quantization (AWQ) flow.

## Prerequisites
- NVIDIA GPU with compute capability ≥ 8.0 (Ampere or newer).
- Python 3.10+ and a writable environment (e.g., `python -m venv .venv && source .venv/bin/activate`).
- Optional: `HUGGINGFACE_HUB_TOKEN` exported if the model is gated.

## Quick start
1) Launch Jupyter (for example):
   ```bash
   pip install jupyterlab
   jupyter lab
   ```
2) Open either notebook and run the first cell to install dependencies:
   - `vllm`, `llm-compressor`, `transformers`, `datasets`, `accelerate`.
3) Adjust the model ID if desired (defaults to `meta-llama/Llama-3.2-1B-Instruct`) and the calibration slice size.
4) Run the quantization cell (GPTQ or AWQ) to produce a local 4-bit checkpoint (saved to `llama-gptq-w4a16/` or `llama-awq-w4a16/`).
5) Serve with vLLM (from a terminal):
   ```bash
   vllm serve ./llama-gptq-w4a16 --max-model-len 4096 --tensor-parallel-size 1 --port 8000 --api-key dummy
   ```
6) Call the served model using the OpenAI-compatible endpoint via the notebook’s example cell.

## Benchmarking
Each notebook includes a quick accuracy probe on 100 samples of `tweet_eval/sentiment`, prompting for a one-word label and reporting exact-match accuracy plus prediction distribution. Increase calibration samples (e.g., 512+) if you need better fidelity.

## Tips
- Keep the chat template when building calibration data; avoid adding extra BOS tokens.
- Increase `NUM_CALIBRATION_SAMPLES` and `MAX_SEQUENCE_LENGTH` if you have headroom for higher quality.
- Use `--tensor-parallel-size` to span multiple GPUs when serving larger models.
