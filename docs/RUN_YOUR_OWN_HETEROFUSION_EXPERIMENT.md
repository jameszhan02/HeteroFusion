# 用自己的模型、LoRA 和数据运行 HeteroFusion

本文给出从实验设计、数据准备、配置、训练到评估的完整流程。命令默认从
HeteroFusion 仓库根目录执行。

## 1. 先明确 HeteroFusion 在合并什么

一次实验至少包含以下四部分：

1. **Target base model**：最终承载融合结果的基础模型。
2. **Target LoRA**：`initial_target_lora`，必须属于 target base model；它是融合的锚点。
3. **Source LoRA**：一个或多个希望迁移到 target 上的 LoRA。source 可以来自不同模型家族，
   训练时只读取其 LoRA 权重，不加载 source base model。
4. **带答案的融合数据**：用于优化融合网络。当前实现计算
   `LM loss + lambda_reg * RDM loss`，因此每条样本必须有参考答案。

最终产物仍然是 **target base model 可加载的 PEFT LoRA adapter**，不是 source
模型的 checkpoint，也不是默认就能独立运行的完整模型。

建议先写清楚自己的实验定义：

```text
Target base model:      ______________________________
Target LoRA 及其任务:   ______________________________
Source LoRA 及其任务:   ______________________________
融合训练数据:           ______________________________
独立验证/测试数据:      ______________________________
最终评价指标:           ______________________________
```

### `main` 和 `replay` 的含义

- `type: main` 是融合阶段的主体数据。配置中必须至少有一个 main 数据集。
- `type: replay` 用于保留 target 原能力或其他 source 能力，减少遗忘。
- 数值型 `ratio` 表示相对于 **全部 main 样本数** 的采样比例。例如 main 有 500 条，
  `ratio: 0.2` 会从该 replay 数据集中抽取 100 条。
- `ratio: all` 表示使用该 replay 数据集的全部样本。
- 多个 main 数据集会先拼接，再据此计算每个 replay 的抽样数量。

融合数据已经参与梯度计算，不能再作为无偏测试集。至少保留一份完全不参与融合的
test 集；调参时最好再单独保留 validation 集。

## 2. 环境准备

推荐在带 CUDA 的 Linux 环境中运行：

```bash
conda create -n infer_train python=3.10 -y
conda activate infer_train
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers peft safetensors pyyaml fire tqdm numpy datasets accelerate sentencepiece scipy einops
```

确认环境和 GPU：

```bash
python -c "import torch, transformers, peft; print('torch=', torch.__version__); print('transformers=', transformers.__version__); print('peft=', peft.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

当前 `main.py` 固定用 `bfloat16` 和 `device_map="auto"` 加载 target base model。
GPU 需要支持 BF16；如果显存不足，应优先减小 `block_size`、`cutoff_len`、
`batch_size` 或 source 数量。直接在 CPU 上运行通常不实用。

## 3. 准备 target 和 source LoRA

每个 adapter 目录至少应包含：

```text
adapter_name/
├── adapter_config.json
└── adapter_model.safetensors    # 或 adapter_model.bin
```

### 3.1 必须满足的兼容条件

- `base_model_path` 必须是 `initial_target_lora` 实际训练时使用的基础模型。
- target LoRA 必须能由 PEFT 正常加载到 target base model。
- 同一次融合中的 target/source LoRA 应使用相同 rank；当前融合编码器不能可靠处理
  不同 rank 的张量。
- target/source 必须有可对应的模块后缀，例如都包含 `q_proj`、`v_proj` 等。
- 当前层匹配依赖权重键中的 `layers.<数字>.` 命名。没有这种层编号的架构需要先修改
  `src/trainer.py` 的匹配逻辑。
- source 可以具有不同的层数、hidden size 和模型家族，但只有成功匹配到的层与模块
  会参与融合。
- 建议所有 adapter 使用普通 LoRA，且不要依赖未被保存/迁移的额外
  `modules_to_save` 权重。

先查看每个 adapter 的配置：

```bash
python -m json.tool /path/to/target_lora/adapter_config.json
python -m json.tool /path/to/source_lora/adapter_config.json
```

重点比较 `r`、`target_modules`、`task_type` 和 target adapter 的
`base_model_name_or_path`。

### 3.2 如果手里只有完整微调模型

仓库提供 `tools/full_model_delta_to_lora.py`，可用截断 SVD 将完整模型与其原始 base
之间的差分近似成 LoRA：

```bash
python tools/full_model_delta_to_lora.py \
  --base-model /data/shared_ckpt/Llama-3.2-1B \
  --trained-model /data/shared_ckpt/Llama-3.2-1B-Instruct \
  --output-dir /data/shared_ckpt/Llama-3.2-1B-Instruct_r32 \
  --rank 32 \
  --lora-alpha 64 \
  --dtype float32 \
  --device cuda:0 \
  --save-tokenizer
```

```bash
python3 tools/create_zero_lora.py \
  --base-model /path/to/target_base_model \
  --output-dir adapters/target_zero_lora \
  --rank 64 --mode both
```

```bash
python3 tools/create_random_lora.py \
  --base-model /data/shared_ckpt/Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir /data/shared_ckpt/Qwen/Qwen2.5-1.5B-random-lora \
  --rank 64 --match-norm-to /data/shared_ckpt/Qwen/grpo-baseline-qwen2.5-1.5b-lora
```

```bash
python tools/merge_lora_to_full_model.py \
  --base-model /data/shared_ckpt/Llama-3.2-1B-Instruct \
  --adapter /data/shared_ckpt/r64_baseline_lora/r64_qwen-baseline-gms8k/merged_lora \
  --output-dir /data/shared_ckpt/r64_llama-baseline-gms8k-full \
  --dtype bfloat16 \
  --device cuda:0
```

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  configs/heterofusion/custom/two_ckpt_merge_template.yaml
```

对每个 target/source 完整模型分别执行一次，并确保 `--base-model` 是该模型微调前的
准确 checkpoint。显存允许时使用 `float32` 做差分。该转换是有损的，且默认不能表达
norm、embedding、bias、LM head 等变化；能重新训练时，优先直接训练 LoRA。

## 4. 准备融合数据

### 4.1 推荐的数据格式

创建例如 `data/custom/my_target_replay.json`：

```json
[
  {
    "instruction": "请解答下面的问题，并只输出最终答案。",
    "input": "12 + 30 等于多少？",
    "output": "42"
  },
  {
    "instruction": "请解答下面的问题，并只输出最终答案。",
    "input": "9 × 7 等于多少？",
    "output": "63"
  }
]
```

也可以使用 JSONL，每行一个同结构的 JSON 对象。字段含义为：

- `instruction`：任务指令；
- `input`：当前样本输入，可为空字符串；
- `output`：参考回答，不能为空，它会变成计算 LM loss 的 label；
- `history`：可选的多轮历史。

这里不是让模型先自由生成回答再比较，而是 SFT teacher forcing：prompt token 默认被
mask，主要对 `output` 的 token 计算交叉熵。

数据格式、标签名称和输出格式应与原 target/source LoRA 的训练任务一致。例如分类任务
不要一部分用 `positive`、另一部分用 `1`；抽取任务的字段顺序和分隔符也应统一。

### 4.2 在 `dataset_info.json` 注册数据

在 `data/dataset_info.json` 顶层 JSON 对象中增加：

```json
"my_target_replay": {
  "file_name": "custom/my_target_replay.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output"
  }
}
```

注意保持整个文件仍是合法 JSON，前一个条目后需要逗号。检查方法：

```bash
python -m json.tool data/dataset_info.json > /dev/null
```

PowerShell 可使用：

```powershell
python -m json.tool data/dataset_info.json | Out-Null
```

### 4.3 数据划分建议

建议每个任务至少准备：

```text
fusion/replay split   用于 HeteroFusion 优化，包含答案
validation split      用于选择超参数，不进入训练
test split            最终只评一次，不进入训练或调参
```

小规模起跑可先为每个任务抽取 100～300 条有代表性的 fusion/replay 样本，但应保持类别、
长度和难度覆盖。正式结论需要对数据量和随机种子做消融。

## 5. 创建自己的 YAML 配置

复制模板：

```bash
cp configs/heterofusion/custom/two_ckpt_merge_template.yaml \
   configs/heterofusion/custom/my_experiment.yaml
```

PowerShell：

```powershell
Copy-Item configs/heterofusion/custom/two_ckpt_merge_template.yaml `
  configs/heterofusion/custom/my_experiment.yaml
```

下面是一份可修改的完整示例：

```yaml
experiment_name: my_heterofusion_experiment
output_dir: outputs/custom/my_heterofusion_experiment

# 最终 fused LoRA 属于这个 base model。
base_model_path: ${MODEL_ROOT}/my_target_base

# 必须是由上面的 target base model 训练得到的 LoRA。
initial_target_lora: ${ADAPTER_ROOT}/target_task_r64

data_global:
  dataset_dir: data
  # 必须与 target base model 的聊天模板匹配，例如 llama3、qwen。
  template: llama3
  cutoff_len: 1024
  batch_size: 1
  num_workers: 4

tasks:
  - task_name: fuse_source_tasks_tail_b_only
    source_lora_paths:
      - ${ADAPTER_ROOT}/source_task_a_r64
      - ${ADAPTER_ROOT}/source_task_b_r64

    # 使用成功对齐层中的 100%；也可写 0.5 或 "1/2"。
    transfer_ratio: "1"
    # 可选：tail、head、uniform、naive。
    layer_alignment: tail

    datasets:
      # 至少一个 main；这里通常放当前最希望迁移/优化的 source 任务数据。
      - name: my_source_a_replay
        type: main
      # 保留 target 原任务能力。
      - name: my_target_replay
        type: replay
        ratio: 0.5
      # 其他 source 任务也可作为 replay。
      - name: my_source_b_replay
        type: replay
        ratio: all

    training:
      # 同一 pipeline 的多个 task 若希望复用 transfer net，需要保持该名称和结构参数一致。
      fusion_group_name: my_transfer_r64
      block_size: 512
      embed_dim: 1024
      num_heads: 8
      max_position_embeddings: 4096
      num_epochs: 3
      lr: 5.0e-05
      alpha_init: 0.3
      gradient_accumulation_steps: 8
      mu_gate: 0.1
      lambda_reg: 0.005
      mu_target: 0.0
      sigma_target: 1.0
      num_projections: 2048
      # 可选：b_only、a_only、ab_joint。建议先从 b_only 开始。
      update_mode: b_only
      seed: 42
    seed: 42
```

### 5.1 关键参数说明

| 参数              | 含义                         | 起始建议                        |
| ----------------- | ---------------------------- | ------------------------------- |
| `template`        | target tokenizer 的对话模板  | 必须按 target 模型选择          |
| `cutoff_len`      | token 截断长度               | 先用 512/1024，保证答案未被截掉 |
| `batch_size`      | 单步 batch                   | 显存紧张时用 1                  |
| `transfer_ratio`  | 使用对齐层的比例             | `"1"`                           |
| `layer_alignment` | 异构模型层对齐方式           | `tail`                          |
| `block_size`      | LoRA 权重分块行数            | rank 64 可先用 512              |
| `embed_dim`       | 融合网络隐藏维度             | 1024                            |
| `num_heads`       | attention 头数               | 8，且需整除 `embed_dim`         |
| `num_epochs`      | 融合训练轮数                 | 3                               |
| `lr`              | 融合网络学习率               | `5e-5`                          |
| `alpha_init`      | 预测 LoRA 增量初始缩放       | 0.3                             |
| `lambda_reg`      | RDM 正则权重                 | 0.005                           |
| `update_mode`     | 更新 target LoRA 的 A/B 矩阵 | `b_only`                        |

`max_position_embeddings` 约束融合网络内部 block 序列长度，不等同于文本的
`cutoff_len`。如果 block 数或多 source 拼接后超过它，需要调大。

当前训练器仅在 `(step + 1) % gradient_accumulation_steps == 0` 时调用 optimizer。
每个 epoch 尾部不足一个完整累积周期的梯度不会单独更新。因此应保证每个 epoch 的
dataloader batch 数至少达到累积步数，最好能整除它；小数据 smoke test 可临时设为 1。

### 5.2 多 task pipeline 与一次多 source 融合

- 在同一个 task 的 `source_lora_paths` 中列出多个 source：这些 source 共同参与一次融合。
- 在 `tasks:` 中配置多个 task：按顺序执行；前一个 task 输出的 `merged_lora` 会成为
  后一个 task 的 target LoRA。

第一次实验建议只写一个 task，便于定位问题。

### 5.3 示例：两个 LoRA checkpoint 跑 GSM8K 融合

如果已经有一个 target LoRA 和一个 source LoRA，例如：

```text
target base:   /data/shared_ckpt/Llama-3.2-1B
target LoRA:   /data/adapters/Llama-1B-Instruct-r64
source LoRA:   /data/adapters/OLMo2-1B-teacher-r64
fusion data:   gsm8k_fusion_train_200
```

可以直接从模板开始：

```bash
cp configs/heterofusion/custom/two_ckpt_merge_template.yaml \
   configs/heterofusion/custom/llama32_olmo2_gsm8k.yaml
```

配置中最关键的是三类路径：

```yaml
experiment_name: llamda3.2_1b_gsm8k_heterfusion
output_dir: /data/adapters/llamda3.2_1b_offical_Instuct_heterfusion_r64_gsm8k

base_model_path: /data/shared_ckpt/Llama-3.2-1B
initial_target_lora: /data/adapters/Llama-1B-Instruct-r64

tasks:
  - task_name: two_ckpt_tail_b_only
    source_lora_paths:
      - /data/adapters/OLMo2-1B-teacher-r64
    layer_alignment: tail
    datasets:
      - name: gsm8k_fusion_train_200
        type: main
    training:
      fusion_group_name: custom_two_ckpt_transfer
      update_mode: b_only
```

这里 `base_model_path` 必须是 `initial_target_lora` 实际训练时的 base。如果
`/data/adapters/Llama-1B-Instruct-r64` 是基于 `Llama-3.2-1B-Instruct` 训练的，
则 `base_model_path` 也应改成对应的 Instruct base，而不是普通 base。

`gsm8k_fusion_train_200` 需要先写入 `data/dataset_info.json`。仓库提供了转换脚本：

```bash
python tools/prepare_gsm8k_fusion_dataset.py \
  --input data/genome_tasks/gsm8k/train.jsonl \
  --output data/gsm8k_fusion/gsm8k_fusion_train_200.json \
  --dataset-info data/dataset_info.json \
  --dataset-name gsm8k_fusion_train_200 \
  --limit 200 \
  --shuffle
```

如果该数据集名称已经存在并确认要覆盖注册项，追加
`--overwrite-dataset-entry`。转换后的数据仍是带答案的 SFT/fusion 数据，不能再作为
无偏测试集。

正式跑之前建议先把 smoke-test 配置改小：

```yaml
data_global:
  batch_size: 1
  num_workers: 0

training:
  num_epochs: 1
  gradient_accumulation_steps: 1
  num_projections: 128
```

启动命令：

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  configs/heterofusion/custom/llama32_olmo2_gsm8k.yaml
```

如果使用 `uv` 管理环境：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python main.py \
  configs/heterofusion/custom/llama32_olmo2_gsm8k.yaml
```

输出 adapter 位于：

```text
/data/adapters/llamda3.2_1b_offical_Instuct_heterfusion_r64_gsm8k/two_ckpt_tail_b_only/merged_lora/
```

该目录仍然是 PEFT LoRA adapter，推理或继续合并时需要搭配同一个 target
`base_model_path` 使用。若日志中没有任何 `Init HeteroFusion transfer net`，通常说明
target/source 的层名或 LoRA 模块后缀没有匹配成功，应先检查两个 adapter 的
`adapter_config.json`、rank 和 `target_modules`。

## 6. 运行前检查

设置路径变量：

```bash
export MODEL_ROOT=/path/to/base_models
export ADAPTER_ROOT=/path/to/adapters
```

PowerShell：

```powershell
$env:MODEL_ROOT = "D:\models"
$env:ADAPTER_ROOT = "D:\adapters"
```

如果没有设置 `ADAPTER_ROOT`，代码会将 `${ADAPTER_ROOT}` 回退到仓库内的
`adapters/`。`MODEL_ROOT` 没有对应回退值，必须设置或直接在 YAML 中写路径。

运行前逐项确认：

- target base、target LoRA、所有 source LoRA 路径存在；
- 每个 adapter 有 `adapter_config.json` 和权重文件；
- target/source rank 相同，模块后缀有交集；
- 数据文件存在，注册名称与 YAML 完全一致；
- `output` 非空，且 `cutoff_len` 不会把答案全部截断；
- `template` 与 target 模型匹配；
- 输出目录不会与原始 adapter 目录混用；
- fusion、validation、test 数据没有泄漏。

## 7. 先做 smoke test

正式训练前，复制一份 smoke-test YAML，并暂时设置：

```yaml
data_global:
  batch_size: 1
  num_workers: 0

training:
  num_epochs: 1
  gradient_accumulation_steps: 1
  num_projections: 128
```

同时给 main/replay 注册小样本文件，先验证数据解析、LoRA 匹配、前向、反向和保存全部
能够完成。启动：

```bash
CUDA_VISIBLE_DEVICES=0 python main.py configs/heterofusion/custom/my_experiment_smoke.yaml
```

PowerShell：

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
python main.py configs/heterofusion/custom/my_experiment_smoke.yaml
```

日志中至少应看到：

```text
Loading Base Model
Loading Adapter Weights
Init HeteroFusion transfer net
Starting HeteroFusion training
LM=... RDM=... alpha=...
Starting fused adapter export
Final Combined Adapter is located at: .../merged_lora
```

如果没有任何 `Init HeteroFusion transfer net`，通常表示 target/source 的层名或模块名
没有匹配成功，不应继续正式实验。

## 8. 正式运行

```bash
CUDA_VISIBLE_DEVICES=0 python main.py configs/heterofusion/custom/my_experiment.yaml \
  2>&1 | tee logs_my_experiment.txt
```

PowerShell：

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
python main.py configs/heterofusion/custom/my_experiment.yaml 2>&1 |
  Tee-Object -FilePath logs_my_experiment.txt
```

单 task 的输出位置为：

```text
<output_dir>/<task_name>/merged_lora/
├── adapter_config.json
└── adapter_model.safetensors
```

训练过程中关注：

- `LM` 是否为有限值并总体下降；
- `RDM` 是否为有限值；
- `alpha` 是否发生变化且没有爆炸；
- 是否出现 OOM、NaN、零个匹配模块或数据样本被全部丢弃；
- 最终目录是否同时包含 adapter config 和权重。

当前实现只保存最终 fused adapter，不保存 optimizer、scheduler 或中间训练 checkpoint；
中断后不能从中间 step 原样续训。

## 9. 加载并测试 fused LoRA

最重要的是比较同一个 target base 上的至少三个条件：

1. target base model；
2. target base + 原始 target LoRA；
3. target base + fused LoRA。

如果条件允许，再加入 source 模型 + source LoRA 作为参考上界。所有条件必须使用相同
prompt、解码参数、测试样本和指标。

下面是最小加载检查；把 prompt 和模板改成自己的任务格式：

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_path = "/path/to/target_base"
adapter_path = "outputs/custom/my_heterofusion_experiment/fuse_source_tasks_tail_b_only/merged_lora"

tokenizer = AutoTokenizer.from_pretrained(base_path)
base = AutoModelForCausalLM.from_pretrained(
    base_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, adapter_path)
model.eval()

prompt = "请解答：12 + 30 等于多少？"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

正式评价应调用自己任务原有的 evaluator，而不是只看训练 loss 或几条人工输出。

### 使用仓库的 GENOME evaluator

只有当任务属于外部 GENOME evaluator 支持的任务时，才使用：

```bash
export GENOME_ROOT=/path/to/GENOME
python tools/run_genome_merged_eval.py \
  --model-path /path/to/target_base \
  --lora-path outputs/custom/my_heterofusion_experiment/fuse_source_tasks_tail_b_only/merged_lora \
  --task gsm8k \
  --split test \
  --output-dir infer_results/my_experiment/gsm8k \
  --work-dir /tmp/heterofusion_genome_eval
```

自定义任务并不会因为写入 HeteroFusion YAML 就自动获得 evaluator；需要复用原任务的
评价脚本，或自行实现 exact match、F1、accuracy、ROUGE 等合适指标。

## 10. 可选：导出独立完整模型

如果部署环境不方便同时加载 base + PEFT adapter，可将 fused LoRA 合入 target base：

```bash
python tools/merge_lora_to_full_model.py \
  --base-model /data/shared_ckpt/Qwen/Qwen2.5-1.5B \
  --adapter /data/shared_ckpt/Qwen/Qwen2.5-Math-1.5B-Instruct_r64 \
  --output-dir /data/shared_ckpt/Qwen/Qwenmath_r64_recover \
  --dtype bfloat16 \
  --device cuda:0
```

`--output-dir` 必须为空目录或尚不存在，不能覆盖 base model 或 adapter。该工具默认执行
PEFT safe merge 并保存 tokenizer。

## 11. 推荐的正式实验矩阵

为了判断提升确实来自 HeteroFusion，至少运行：

| 实验                       | 用途                         |
| -------------------------- | ---------------------------- |
| Target base                | 基础能力下界                 |
| Target LoRA                | 融合前锚点                   |
| Fused LoRA                 | 主结果                       |
| Target LoRA 继续做普通 SFT | 控制“只是用了更多数据”的影响 |
| 不同随机种子               | 估计方差                     |

建议的首轮消融：

- `layer_alignment`: `tail` 与 `uniform`；
- `update_mode`: `b_only` 与 `ab_joint`；
- `alpha_init`: 0.1、0.3、0.5；
- `lambda_reg`: 0、0.005、0.01；
- replay 比例：0、0.1、0.5、`all`。

不要用 test 集挑选这些参数。用 validation 选定配置后，再在 test 上报告一次最终结果。

## 12. 常见问题

### `No adapter found`

source/target 目录中缺少 `adapter_model.safetensors` 或 `adapter_model.bin`，或者 YAML
路径没有正确展开。先打印环境变量并检查绝对路径。

### PEFT 加载 target LoRA 时报 shape mismatch

`base_model_path` 不是 target LoRA 所属的准确基础模型，或者 target modules/rank 与
checkpoint 不一致。不要把 source base 当作 `base_model_path`。

### 融合网络没有初始化，训练立即结束

target/source 权重键没有匹配到相同的 `layers.N.<module>` 后缀，或 target adapter 的
模块无法映射到运行时 PEFT 模型。检查 safetensors 的键名和 `target_modules`。

### CUDA OOM

依次尝试：降低 `block_size`、`cutoff_len`、`batch_size`、`embed_dim`、
`num_projections`、source 数量或 `transfer_ratio`。`block_size` 越小通常越省单层线性投影
参数，但 block 序列会更长，应同时留意 `max_position_embeddings`。

### loss 是 NaN

检查数据中是否存在空答案或异常超长样本；降低学习率和 `alpha_init`；确认 adapter
权重本身无 NaN/Inf；先用一个 source、`b_only`、`gradient_accumulation_steps: 1`
完成 smoke test。

### 训练结束但效果变差

常见原因包括任务 prompt/label 格式不一致、replay 比例失衡、数据泄漏导致错误调参、
source 本身质量不足、层对齐方式不合适，或融合收益其实来自额外 SFT 数据。应结合独立
test、普通 SFT 控制组和任务级指标判断，而不能只依据训练 loss。

## 13. 最终复现实验应保存的内容

每次正式运行建议保存：

- 完整 YAML 配置；
- target/source adapter 的路径、版本与 `adapter_config.json`；
- base model 的准确 revision；
- fusion/validation/test 数据版本和样本 ID；
- Python、PyTorch、Transformers、PEFT 和 CUDA 版本；
- 随机种子、GPU 型号、完整日志；
- 原始 target LoRA、fused LoRA 及全部基线的逐任务指标；
- 最终 `merged_lora/` 和评价输出。

完成上述记录后，实验才具备可复现性，也能区分“真正的跨模型 LoRA 融合收益”和
“额外监督数据带来的继续微调收益”。
