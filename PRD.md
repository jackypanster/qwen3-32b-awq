
RTX 2080 Ti（Turing架构，Compute Capability 7.5） 不支持 FlashAttention-2
目标：在 4×RTX 2080 Ti（CC 7.5，单卡22 GB显存，共88 GB显存）+ Ubuntu 24.04 上，物理内存512G，SSD 2T，使用 vLLM 0.8.5 部署 Qwen3-32B-AWQ，尽量压榨硬件性能，优先提升上下文处理能力，允许牺牲延迟和稳定性。
主要要求：
- 显存利用率尽量高，优先吞吐和上下文长度
- 支持最大32K tokens上下文
- 稳定性不做要求，可接受偶发崩溃或重启
已知限制：
- 2080 Ti 不支持 FlashAttention-2，仅支持 FlashAttention-1（需自行编译）或普通 attention
- Turing 架构部分新特性不支持
- 需关注 KV Cache 占用和 swap 空间设置
2080 Ti 不支持 Flash-Attn-2，NVIDIA-SMI 570.153.02             Driver Version: 570.153.02     CUDA Version: 12.8  