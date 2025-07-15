docker run -d \
  --runtime=nvidia --gpus=all --name coder \
  -v /home/llm/model/qwen/Qwen3-32B-AWQ:/model/Qwen3-32B-AWQ \
  -p 8888:8000 --cpuset-cpus 0-55 \
  --ulimit memlock=-1 --ulimit stack=67108864 --restart always --ipc=host \
  vllm/vllm-openai:v0.8.5 \
  --model /model/Qwen3-32B-AWQ --served-model-name coder \
  --tensor-parallel-size 4 --quantization awq --dtype auto \
  --max-model-len 32768 --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.96 \
  --block-size 16 \
  --enable-prefix-caching \
  --swap-space 64 \
  --max-num-seqs 64