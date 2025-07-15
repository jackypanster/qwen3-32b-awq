# Qwen3-32B-AWQ vLLM 高效推理部署（多卡2080Ti极限压榨版）

## 项目简介

本项目旨在利用 4×RTX 2080 Ti（单卡22GB显存，合计88GB）服务器，基于 vLLM 0.8.5 框架，极限压榨硬件性能，部署 Qwen3-32B-AWQ 大模型，实现最大化上下文长度（32K tokens）、高吞吐推理。允许牺牲延迟和稳定性，适合科研、极限测试、低成本大模型推理场景。

---

## 硬件与系统环境

- GPU：4 × RTX 2080 Ti（Turing架构，Compute Capability 7.5，单卡22GB显存）
- CPU：建议 56 核以上
- 内存：512GB
- 存储：SSD 2TB（建议 NVMe）
- 操作系统：Ubuntu 24.04
- 驱动：NVIDIA 570.153.02
- CUDA：12.8

---

## 主要需求与目标

- **显存利用率**：尽量吃满全部显存，优先吞吐和上下文长度
- **上下文长度**：支持最大 32K tokens
- **稳定性**：不做强要求，可接受偶发崩溃或重启
- **吞吐/延迟**：允许牺牲延迟以换取更大吞吐和上下文

---

## 已知限制

- 2080 Ti 不支持 FlashAttention-2，仅支持 FlashAttention-1（需自行编译）或 XFormers/普通 attention
- Turing 架构部分新特性不支持，部分优化无效
- 需关注 KV Cache 占用和 swap 空间设置，极限 context 下易 OOM
- AWQ 量化在 vLLM 0.8.5 上未完全优化，部分场景下 FP16 反而更快

---

## 快速部署

### 1. 模型准备

请将 Qwen3-32B-AWQ 模型权重（已做 AWQ 量化）下载并放置于本地目录 `/home/llm/model/qwen/Qwen3-32B-AWQ`。

### 2. 启动命令（run.sh）

```bash
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
```

### 3. 参数说明

| 参数 | 说明 |
|------|------|
| --tensor-parallel-size 4 | 4卡并行，分摊模型参数 |
| --quantization awq | 启用 AWQ 权重量化，节省显存 |
| --dtype auto | 自动选择最优精度（建议FP16） |
| --max-model-len 32768 | 最大支持32K tokens上下文 |
| --max-num-batched-tokens 32768 | 单批最大tokens数，影响吞吐与显存 |
| --gpu-memory-utilization 0.96 | 吃满显存，极限压榨 |
| --block-size 16 | KV Cache分块，适合大context |
| --enable-prefix-caching | 启用前缀缓存，提升长对话效率 |
| --swap-space 64 | 允许64GB swap空间，防止KV Cache爆显存 |
| --max-num-seqs 64 | 最大并发序列数，建议根据显存调整 |

---

## API 使用示例

### OpenAI Chat API 格式

```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "coder",
    "messages": [
      {"role": "system", "content": "你是一个聪明的AI助手。"},
      {"role": "user", "content": "请写一个Python冒泡排序。"}
    ],
    "max_tokens": 512,
    "temperature": 0.2
  }'
```

> **注意**：max_tokens 建议设置为 512~2048，确保返回完整答案。极限 context 下，max_tokens 过大易 OOM。

---

## 性能调优建议

1. **AWQ/FP16 性能对比**  
   - 实测 AWQ 和 FP16 的吞吐与延迟，部分场景下 FP16 更快。
2. **XFormers/FlashAttention-1**  
   - 2080 Ti 不支持 FlashAttention-2，建议尝试编译 flash-attn-1 或直接用 XFormers。
3. **KV Cache 管理**  
   - swap-space 可适当调大，但过大时延迟会显著上升。
   - 关注日志中的 KV Cache 占用，适时调整 max-num-batched-tokens、max-num-seqs。
4. **多实例分卡**  
   - 若业务允许，可将4卡分为2实例各2卡，提升整体利用率。
5. **自动降级与监控**  
   - 建议在API层加自动降级机制，OOM时自动缩短context，提升服务可用性。
   - 用 supervisor/pm2/watchdog 脚本监控容器，崩溃自动重启。

---

## 常见问题与排查

- **Q: 返回内容不完整？**  
  A: 增大 API 请求的 max_tokens 参数，或缩短输入 context。
- **Q: OOM/崩溃？**  
  A: 降低 max-model-len、max-num-batched-tokens、max-num-seqs，或增大 swap-space。
- **Q: 推理慢？**  
  A: 2080 Ti 性能有限，XFormers 性能低于新卡，建议减少并发或缩短 context。
- **Q: NCCL 报错/死锁？**  
  A: 检查 --disable-custom-all-reduce 参数，或升级 NCCL 版本。

---

## 参考资料

- [vLLM 官方文档](https://vllm.readthedocs.io/)
- [Qwen3-32B-AWQ 模型主页](https://huggingface.co/Qwen/Qwen3-32B-AWQ)
- [FlashAttention 项目](https://github.com/Dao-AILab/flash-attention)
- [AWQ 量化论文与实现](https://github.com/mit-han-lab/llm-awq)

## 附录：测试脚本与日志说明

### test.sh：并发/极限上下文压力测试脚本

test.sh 脚本用于自动化测试模型在极限上下文（如32K tokens）和高并发（如10并发）下的响应能力、稳定性和吞吐。

- 先用 python 生成 32000 tokens 的超长 prompt（prompt32k.txt）。
- 构造标准 OpenAI Chat API 请求（payload.json），max_tokens 可自定义。
- 用 ThreadPoolExecutor 并发发起多路 curl 请求，统计每个请求的耗时。
- 可用于评估模型在极限输入下的响应速度、是否 OOM、返回内容完整性等。

**用法示例：**
```bash
# 生成超长 prompt
python3 -c "print(' '.join(['token'] * 32000))" > prompt32k.txt
# 运行并发测试
bash test.sh
```

**分析建议：**
- 关注每个🏁输出的耗时，评估高并发下的延迟。
- 检查返回内容是否完整，是否有报错或 OOM。
- 可调整 max_tokens、并发数，探索系统极限。

---

### log.output：vLLM 启动与运行日志

log.output 记录了 vLLM 容器的完整启动、加载模型、分配显存、KV Cache、并发能力、API 路由等详细信息。

**主要作用：**
- 验证参数是否正确生效（如 max-model-len、tensor-parallel-size、swap-space 等）。
- 检查硬件资源利用率、显存分配、KV Cache 占用。
- 发现潜在警告（如 FlashAttention-2 不支持、AWQ 性能警告、swap 空间过大等）。
- 记录最大并发能力、API 路由、采样参数等。

**分析建议：**
- 启动后关注 WARNING/ERROR，及时调整参数。
- 关注 KV Cache、swap 空间、最大并发等指标，结合 test.sh 结果优化部署。
- 日志可辅助定位 OOM、死锁、性能瓶颈等问题。

---