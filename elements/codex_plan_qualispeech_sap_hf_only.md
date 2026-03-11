
# Codex 实现计划：QualiSpeech → SAP 跨数据集增广实验（Interspeech 级别，一站式自动化，Hugging Face-only）

## 0. 这份计划的定位

这份计划的目标很明确：把你当前论文里的思路，扩成一套能直接交给 Codex 落地的研究代码规范。最终代码库需要做到：

- 从原始数据开始自动解析；
- 自动构建 SAP 目标任务与 QualiSpeech 辅助任务；
- 自动完成基线、JT、FT、比例消融、补充控制实验；
- 自动在 dev 上选最优 checkpoint；
- 自动跑 test；
- 自动汇总多 seed 结果、导出表格、画图、做显著性分析；
- 自动产出论文主表、消融表、附录图和预测文件；
- 所有 SSL encoder **完全基于 Hugging Face Transformers / Hugging Face Hub**；
- **禁止依赖 fairseq**，也不要在代码里留下任何 fairseq 兼容分支。

这份计划默认目标会议就是 Interspeech，因此我会把“paper-faithful 复现”和“reviewer 会追问的补充实验”一起写进去，但两条线路要在配置层清晰分开，保证主实验结果可复现、补充实验可扩展、工程实现不混乱。

---

## 1. 硬约束（Codex 必须逐条执行）

### 1.1 研究问题与实验范围

需要支持以下 QualiSpeech → SAP 数据增广对：

1. `Naturalness -> Naturalness`
2. `Continuity -> Inappropriate silences`
3. `Distortion -> Distorted vowels`
4. `Distortion -> Imprecise consonants`
5. `Listening effort -> Intelligibility`
6. `Overall quality -> Intelligibility`

### 1.2 主方法只保留两种

和 PDF 保持一致，主实验只实现两种跨数据集方法：

- `JT`：Joint Training
- `FT`：Fine-Tuning（两阶段）

不要在主表里再引入第三种新方法；所有额外设计都放到 reviewer control / supplementary suite。

### 1.3 基线系统

需要实现以下 encoder 家族的 SAP-only 基线：

- `WavLM`
  - Base
  - Base+
  - Large
- `wav2vec 2.0`
  - Base
  - Large*
  - Large+
- `HuBERT`
  - Base
  - Large

### 1.4 增广系统

需要在同样的 encoder 集合上，加入 QualiSpeech 做 JT 和 FT。

### 1.5 消融实验

其他设置固定，只改变加入 QualiSpeech 的比例：

- `0.25`
- `0.5`
- `0.75`
- `1.0`
- `1.5`
- `2.0`

这里的 `1.0` 必须定义成“论文默认设置”的基准比例，具体在后文写死。

### 1.6 工程约束

- **SSL encoder 部分只用 Hugging Face**
- **不安装 fairseq**
- **不调用 fairseq checkpoint loader**
- **不依赖 torchaudio 内置 bundle 代替 HF**
- **不依赖 speechbrain / s3prl 去间接包装 fairseq**
- 统一使用：
  - `transformers`
  - `huggingface_hub`
  - `datasets`（可选）
  - `torch`
  - `torchaudio`（只做 I/O / resample，不做模型加载）

### 1.7 自动化目标

最终仓库要支持一条命令完成完整流程，例如：

```bash
make all
```

或：

```bash
python scripts/run_pipeline.py --suite configs/suite/all.yaml
```

执行完成后自动得到：

- 所有中间 manifest
- 所有 run 目录
- best ckpt
- test 预测
- 汇总 csv / json
- png / pdf 图
- latex 表格
- 一份总报告 markdown

---

## 2. 从论文与数据里固定下来的默认事实

这一部分是给 Codex 的“默认配置来源”。

### 2.1 论文主方法默认设置

主实验的默认设置按 PDF 来：

- 输入：原始波形
- SSL encoder 输出 frame-level contextual representations
- pooling：time mean pooling
- 回归头：两层前馈网络
- 激活：ReLU
- dropout：有
- 输出：单连续标量
- encoder 端到端可训练
- 优化目标：MSE
- 指标：MSE / LCC / SRCC
- 优化器：Adam 或 AdamW 都可，但默认建议 `AdamW`
- 默认学习率：`1e-5`
- 默认 weight decay：`0.01`

### 2.2 论文里的尺度关系

- SAP 标签：`1~7`，分数越高，病理越严重
- QualiSpeech 标签：`1~5`，分数越高，语音质量越好

因此：

- JT 需要把 QualiSpeech 标签映射到 SAP 标度
- FT stage 1 用 QualiSpeech 原始标度
- FT stage 2 切换到 SAP 原始标度

### 2.3 论文里的 JT 标度映射

JT 里按论文公式写死默认映射：

```text
sap_aligned = 1 + (5 - qs_score) * 6 / 4
```

也就是：

- `5 -> 1`
- `1 -> 7`

这个映射对所有你指定的 6 个 pair 都先采用同样的“反向线性映射”作为 paper-faithful 默认方案。

### 2.4 SAP 数据划分事实

根据 PDF：

- 原始 SAP test 不可用
- `dev` 被当作 test
- Naturalness：
  - train subset = 5040
  - val = 500（从 train 抽）
  - test = 714（来自 dev）
- Intelligibility：
  - train subset = 5046
  - val = 500（从 train 抽）
  - test = 716（来自 dev）

对你新增的 SAP 目标维度（Inappropriate silences / Distorted vowels / Imprecise consonants），同样采用这个 protocol：

- 从 `train` 中取该维度有标签的 utterance
- 再固定随机种子抽一个 val 子集
- `dev` 中该维度有标签的 utterance 全部作为 test

### 2.5 QualiSpeech 事实

从 `train.csv` 可以看到至少有这些监督列：

- `Speed`
- `Naturalness`
- `Background noise`
- `Distortion`
- `Listening effort`
- `Continuity`
- `Overall quality`
- `Feeling of voice`

以及描述列：

- `Noise Description`
- `Distortion description`
- `Unnatural pause`
- `Natural language description`

本项目主实验只需要监督列中的：

- `Naturalness`
- `Distortion`
- `Listening effort`
- `Continuity`
- `Overall quality`

其余列先保留到 manifest 里，后面可做分析。

### 2.6 SAP JSON 里已经观察到的坑

从目录结构和示例 JSON 可以确定需要处理这些问题：

- `dev/` 根目录存在与说话人目录内同名的 json 副本；
- SAP `Ratings` 可能为空列表；
- 维度名有尾部空格；
- 维度拼写存在错误，例如 `Intelligbility `;
- 某些维度只出现在特定病种或特定说话人上；
- prompt 类型不只一种，至少会出现：
  - `Novel Sentences`
  - `Spontaneous Speech Prompts`
  - `Digital Assistant Commands`

这些坑都要在数据解析层一次性兜住。

---

## 3. HUGGING FACE-ONLY 方案（这部分要写得非常硬）

这是你刚刚新增的关键要求，我建议 Codex 直接按下面做。

## 3.1 模型注册表必须完全基于 HF model id

创建统一注册表 `src/models/model_registry.py`，只允许出现以下 HF 模型：

### wav2vec 2.0

- `facebook/wav2vec2-base`
- `facebook/wav2vec2-large-lv60`
- `facebook/wav2vec2-large-robust`

### HuBERT

- `facebook/hubert-base-ls960`
- `facebook/hubert-large-ll60k`

### WavLM

- `microsoft/wavlm-base`
- `microsoft/wavlm-base-plus`
- `microsoft/wavlm-large`

## 3.2 论文模型名到 HF 模型的映射

给 Codex 一个显式映射表：

| 论文别名 | 工程别名 | HF model id |
|---|---|---|
| wav2vec 2.0 Base | `w2v2_base` | `facebook/wav2vec2-base` |
| wav2vec 2.0 Large* | `w2v2_large_lv60` | `facebook/wav2vec2-large-lv60` |
| wav2vec 2.0 Large+ | `w2v2_large_robust` | `facebook/wav2vec2-large-robust` |
| HuBERT Base | `hubert_base` | `facebook/hubert-base-ls960` |
| HuBERT Large | `hubert_large` | `facebook/hubert-large-ll60k` |
| WavLM Base | `wavlm_base` | `microsoft/wavlm-base` |
| WavLM Base+ | `wavlm_base_plus` | `microsoft/wavlm-base-plus` |
| WavLM Large | `wavlm_large` | `microsoft/wavlm-large` |

### 3.2.1 说明

- `wav2vec2-base` 是 HF 上的 pretraining 版 base；
- `wav2vec2-large-lv60` 对应 Libri-Light 路线；
- `wav2vec2-large-robust` 对应 Libri-Light + CommonVoice + Switchboard + Fisher 的多域预训练路线；
- `hubert-base-ls960` / `hubert-large-ll60k` 是 HuBERT 的标准 HF 特征抽取 checkpoint；
- `wavlm-base` / `wavlm-base-plus` / `wavlm-large` 都有官方 HF model card。

## 3.3 统一加载接口

创建 `src/models/hf_ssl_backbone.py`，暴露统一接口：

```python
class HFSSLBackbone(nn.Module):
    def __init__(self, model_name: str, cache_dir: str | None = None, revision: str | None = None):
        ...
    def forward(self, input_values, attention_mask=None):
        ...
```

要求：

1. 使用 `AutoConfig.from_pretrained`
2. 使用 `AutoModel.from_pretrained`
3. 使用 `AutoProcessor.from_pretrained` 或 `AutoFeatureExtractor.from_pretrained`
4. 对 WavLM 允许 fallback 到 `Wav2Vec2Processor` / `Wav2Vec2FeatureExtractor`
5. 输出统一为：
   - `last_hidden_state`
   - `hidden_states`（可选）
   - `frame_mask`（下采样后的 mask）
   - `pooled_embedding`

### 3.3.1 不允许出现的代码

仓库里不应该出现这些 import：

```python
import fairseq
from fairseq import ...
from fairseq.checkpoint_utils import ...
```

也不要出现：

```python
torch.hub.load("pytorch/fairseq", ...)
```

### 3.3.2 reproducibility 方案

为避免 HF 模型后续更新影响复现，增加两层机制：

- 配置里允许写 `revision`
- 第一次成功下载后，把真实 commit hash 解析出来，保存到：
  - `results/metadata/resolved_model_revisions.yaml`

主实验建议：

- 默认使用公开 `main`
- 生成一次 resolved revisions
- 之后所有正式跑表都锁定到具体 revision

## 3.4 Processor / Feature Extractor 统一策略

强烈建议 Codex 统一走下面逻辑：

```python
try:
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
except Exception:
    try:
        processor = AutoFeatureExtractor.from_pretrained(model_id, revision=revision)
    except Exception:
        processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, revision=revision)
```

原因：

- 这三类模型都吃 16kHz raw waveform；
- WavLM 文档明确建议使用 `Wav2Vec2Processor` 做特征处理；
- HuBERT / Wav2Vec2 的 HF 文档都支持 `AutoModel` / `from_pretrained` 的标准加载方式；
- 这样整个模型层完全是 Hugging Face 风格，没有 fairseq 遗留。

## 3.5 下采样 mask 与 mean pooling

这一点要明确写给 Codex，因为 HF audio encoder 的 pooling 很容易写错。

输入端的 `attention_mask` 是 sample-level 的，encoder 输出是 frame-level 的，下采样之后长度会变短。因此 pooling 必须：

1. 从 processor 得到 sample-level `attention_mask`
2. 调用 encoder 的 helper，把它变成 feature-level mask
3. 做 masked mean pooling

推荐做法：

```python
feat_mask = model._get_feature_vector_attention_mask(
    hidden_states.shape[1], attention_mask
)
pooled = (hidden_states * feat_mask.unsqueeze(-1)).sum(1) / feat_mask.sum(1, keepdim=True)
```

如果某个 HF 版本的 helper 不可用，再写一个统一 fallback，根据 conv stride 规则手算。

## 3.6 内部增强与随机性控制

为了减少不同 encoder 的隐藏差异，建议主实验默认：

- `apply_spec_augment = False`
- `layerdrop = 0.0`

具体做法：

- 加载 config 后覆盖这些字段
- 保存 resolved config 到 run 目录

这样可以避免 HF 模型默认训练行为悄悄变化，提升可控性。

### 3.6.1 reviewer supplement

另开一个补充实验：
- 恢复 encoder 原始 `apply_spec_augment`
- 对最佳 1~2 个 encoder 做对照

放 supplementary 即可，不进主表。

---

## 4. 仓库结构（建议直接按这个生成）

```text
project_root/
├── README.md
├── Makefile
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── paths.yaml
│   ├── defaults.yaml
│   ├── models/
│   │   ├── w2v2_base.yaml
│   │   ├── w2v2_large_lv60.yaml
│   │   ├── w2v2_large_robust.yaml
│   │   ├── hubert_base.yaml
│   │   ├── hubert_large.yaml
│   │   ├── wavlm_base.yaml
│   │   ├── wavlm_base_plus.yaml
│   │   └── wavlm_large.yaml
│   ├── tasks/
│   │   ├── sap_naturalness.yaml
│   │   ├── sap_inappropriate_silences.yaml
│   │   ├── sap_distorted_vowels.yaml
│   │   ├── sap_imprecise_consonants.yaml
│   │   └── sap_intelligibility.yaml
│   ├── pairs/
│   │   ├── qs_nat_to_sap_nat.yaml
│   │   ├── qs_cont_to_sap_sil.yaml
│   │   ├── qs_dist_to_sap_vowel.yaml
│   │   ├── qs_dist_to_sap_cons.yaml
│   │   ├── qs_effort_to_sap_intel.yaml
│   │   └── qs_overall_to_sap_intel.yaml
│   ├── experiments/
│   │   ├── baseline.yaml
│   │   ├── jt.yaml
│   │   ├── ft.yaml
│   │   ├── ratio_ablation.yaml
│   │   └── reviewer_controls.yaml
│   └── suite/
│       ├── smoke.yaml
│       ├── baselines.yaml
│       ├── main.yaml
│       ├── ablation_all_models.yaml
│       ├── reviewer.yaml
│       └── all.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── sap/
│   │   ├── qualispeech/
│   │   ├── splits/
│   │   └── stats/
│   └── cache/
├── src/
│   ├── cli/
│   ├── data/
│   ├── models/
│   ├── tasks/
│   ├── samplers/
│   ├── trainers/
│   ├── eval/
│   ├── analysis/
│   ├── plots/
│   ├── tables/
│   └── utils/
├── scripts/
│   ├── prepare_all.py
│   ├── run_experiment.py
│   ├── run_suite.py
│   ├── summarize_results.py
│   ├── export_tables.py
│   ├── export_figures.py
│   └── package_report.py
├── results/
│   ├── runs/
│   ├── summaries/
│   ├── tables/
│   ├── figures/
│   └── reports/
└── tests/
    ├── test_sap_parser.py
    ├── test_qs_parser.py
    ├── test_pair_builder.py
    ├── test_hf_backbone.py
    ├── test_train_smoke.py
    └── test_summary.py
```

---

## 5. 数据处理层设计

## 5.1 SAP 解析器

实现 `src/data/parse_sap.py`，输出这些标准化文件：

1. `sap_utterances.parquet`
2. `sap_labels_long.parquet`
3. `sap_labels_wide.parquet`
4. `sap_integrity_report.json`
5. `sap_dimension_stats.csv`

### 5.1.1 `sap_utterances.parquet` 字段

每条 utterance 一行，至少包含：

- `dataset = "sap"`
- `split_original`
- `speaker_id`
- `speaker_dir`
- `speaker_json_path`
- `utt_id`
- `audio_filename`
- `audio_path`
- `etiology`
- `block_number`
- `created`
- `created_or_modified`
- `comment`
- `prompt_text`
- `transcript`
- `prompt_category`
- `prompt_subcategory`
- `num_ratings`
- `has_any_rating`
- `duration_sec`
- `sample_rate`
- `num_samples`

### 5.1.2 `sap_labels_long.parquet` 字段

每个 `(utt_id, dimension)` 一行：

- `utt_id`
- `speaker_id`
- `split_original`
- `dimension_raw`
- `dimension_canonical`
- `label`
- `label_min = 1`
- `label_max = 7`

### 5.1.3 `sap_labels_wide.parquet`

每个 utterance 一行，按 canonical dimension 展开列。

## 5.2 SAP 维度名 canonicalization

新增 `src/data/sap_dimension_map.py`，做统一映射：

- 去首尾空格
- 压缩中间多余空格
- 全小写
- 空格改下划线
- 纠正常见拼写错误

至少覆盖：

- `intelligbility` -> `intelligibility`
- `inappropriate silences` -> `inappropriate_silences`
- `imprecise consonants` -> `imprecise_consonants`
- `distorted vowels` -> `distorted_vowels`
- `breathy voice (continuous)` -> `breathy_voice_continuous`
- `short rushes of speech` -> `short_rushes_of_speech`
- `pitch level` -> `pitch_level`
- `variable rate` -> `variable_rate`
- `reduced stress` -> `reduced_stress`
- `harsh voice` -> `harsh_voice`

未知维度不要静默丢弃，要：

- 原样保留
- 记入 `unknown_dimensions.log`

## 5.3 SAP dev 重复 json 处理

对 `dev/` 根目录中的 json 和说话人目录内 json：

1. 优先读说话人目录内的 json
2. 如果根目录存在同名 json：
   - 计算 hash
   - 内容相同：记成 duplicate，忽略
   - 内容不同：直接报错终止

## 5.4 SAP 空 Ratings

`Ratings=[]` 的 utterance 保留到 `sap_utterances.parquet`，方便统计覆盖率，但构造任务时再按目标维度过滤。

## 5.5 SAP prompt 元信息必须保留

原因：

- reviewer 可能会问 read vs spontaneous 的差异；
- Inappropriate silences / Continuity 对 spontaneous prompt 可能更敏感；
- 你后面很可能想做 prompt-category breakdown。

因此在最终 analysis 层要支持：

- `Novel Sentences`
- `Spontaneous Speech Prompts`
- `Digital Assistant Commands`
- 其他未见类别自动纳入

## 5.6 QualiSpeech 解析器

实现 `src/data/parse_qualispeech.py`，输入：

- `data/train`
- `data/val`
- `data/test`
- `train.csv`
- `val.csv`
- `test.csv`

输出：

1. `qs_train.parquet`
2. `qs_val.parquet`
3. `qs_test.parquet`
4. `qs_integrity_report.json`
5. `qs_stats.csv`

### 5.6.1 标准字段

每条音频一行：

- `dataset = "qualispeech"`
- `split_original`
- `utt_id`
- `audio_filename`
- `audio_path`
- `duration_sec`
- `sample_rate`
- `num_samples`
- `speed`
- `naturalness`
- `background_noise`
- `distortion`
- `listening_effort`
- `continuity`
- `overall_quality`
- `feeling_of_voice`
- `noise_description`
- `distortion_description`
- `unnatural_pause`
- `natural_language_description`

### 5.6.2 列名 canonicalization

CSV 原列名带空格，统一改成 snake_case：

- `Background noise` -> `background_noise`
- `Listening effort` -> `listening_effort`
- `Overall quality` -> `overall_quality`
- 其他类似处理

## 5.7 音频读入与重采样

统一数据层函数：

```python
load_audio(path) -> waveform_float32_mono_16k
```

规则：

- 全部转单声道
- 全部重采样到 `16_000`
- float32
- 保留原始完整时长

### 5.7.1 主实验禁止静音裁剪

尤其对这些任务：

- `Inappropriate silences`
- `Continuity`

静音本身就是目标现象之一，所以主实验不要做：

- VAD trimming
- leading / trailing silence trimming
- energy-based cut

后面如果要做补充实验，可单独开一个 `trim_silence_control`，放 supplementary。

---

## 6. 任务层设计

## 6.1 SAP 目标任务定义

需要支持以下 5 个 SAP 目标任务：

1. `sap_naturalness`
2. `sap_inappropriate_silences`
3. `sap_distorted_vowels`
4. `sap_imprecise_consonants`
5. `sap_intelligibility`

其中 Intelligibility 会被两个不同 QualiSpeech 维度配对。

## 6.2 QualiSpeech 辅助任务定义

需要支持以下 5 个辅助维度：

1. `qs_naturalness`
2. `qs_continuity`
3. `qs_distortion`
4. `qs_listening_effort`
5. `qs_overall_quality`

## 6.3 pair 定义

显式写成 6 个 pair config：

```yaml
pair_id: qs_nat_to_sap_nat
sap_target: naturalness
qs_aux: naturalness
```

其余 5 个同理。

## 6.4 pair builder 行为

实现 `src/tasks/pair_builder.py`，输出 pair-specific manifests：

- `sap_train_task.parquet`
- `sap_val_task.parquet`
- `sap_test_task.parquet`
- `qs_train_aux.parquet`
- `qs_val_aux.parquet`
- `pair_metadata.json`

### 6.4.1 `pair_metadata.json` 里至少包含

- `pair_id`
- `sap_target_dim`
- `qs_aux_dim`
- `sap_train_n`
- `sap_val_n`
- `sap_test_n`
- `qs_train_n`
- `qs_val_n`
- `label_range_sap`
- `label_range_qs`
- `jt_mapping_formula`
- `split_protocol`
- `random_seed`

---

## 7. 数据切分协议

## 7.1 paper-faithful split（主实验默认）

主实验默认保留和 PDF 尽可能一致的协议：

- `test = SAP dev`
- `val = 从 SAP train 中抽样`
- `train = SAP train 剩余部分`

对 Naturalness / Intelligibility：

- 强制固定 `val_size = 500`
- 和 PDF 对齐

对新增 SAP target：

- 如果该维度 train 样本数足够，默认也取 `500`
- 若不足，则取 `min(500, round(0.1 * train_n))`，并写入 metadata

### 7.1.1 抽样策略

默认建议：

- `utterance-level stratified sampling by label`
- 保持 label histogram 尽量接近整体分布
- 固定随机种子
- 输出 split 索引文件

## 7.2 speaker-disjoint val（reviewer control）

补充实现一个更严格协议：

- 从 SAP train 中按 speaker 划出 validation speakers
- 目标仍尽量接近 500 utterances
- 保持标签分布尽量接近

这个协议不要覆盖主实验，只放 supplementary，用来回答 reviewer 关于 speaker leakage 的问题。

---

## 8. 比例消融的精确定义

这个部分必须定义得非常清楚，避免 Codex 自己猜。

## 8.1 ratio 的语义

`aux_ratio` 定义为“辅助数据量相对于该方法默认辅助数据量的倍数”。

### 8.1.1 对 JT

论文默认 JT 是 1:1，也就是辅助数据量与当前 SAP target train 数量相同。

因此：

```text
jt_aux_n(r) = round(r * sap_train_n)
```

例子：

- `r=1.0`：与论文默认一致
- `r=0.25`：四分之一的 QS 辅助样本
- `r=2.0`：两倍于 SAP train 数量的 QS 辅助样本

## 8.1.2 对 FT

论文默认 FT 的 stage 1 用完整 QualiSpeech 训练集做预训练。

因此：

```text
ft_aux_n(r) = round(r * qs_full_train_n_for_aux_dim)
```

例子：

- `r=1.0`：完整 QS train，和论文默认一致
- `r=0.25`：四分之一 QS train
- `r=2.0`：两倍有效暴露量，需要重复抽样 / oversampling

## 8.2 小于 1 的情况

- 无放回抽样
- 固定 seed
- 尽量保持辅助标签分布

## 8.3 大于 1 的情况

- 先完整遍历全部辅助样本
- 余下部分用有放回抽样补足
- 记录：
  - `effective_n`
  - `unique_n`
  - `oversample_factor`

## 8.4 分布保持策略

抽样时至少保持：

- 目标辅助维度的 label histogram

可选再保持：

- `id` 前缀分布
- 音频时长分布分位数

如果以后能恢复 QualiSpeech 的 speech category 元数据（synthetic / simulated / real），再把 source composition 也加入分层抽样。

---

## 9. 训练任务定义

## 9.1 Baseline：SAP-only

每个 SAP 目标任务、每个 encoder 都要跑：

```text
encoder x sap_target
```

输出：

- best val ckpt
- test predictions
- metrics
- calibration plot
- confusion-like severity heatmap（可选）

### 9.1.1 Baseline 总数

- 8 encoders
- 5 SAP targets

共 `40` 个 baseline run / seed。

## 9.2 JT：paper-faithful 单头联合回归

实现 `src/trainers/jt_trainer.py`

### 9.2.1 训练数据

- 一个 SAP target dataset
- 一个 QS aux dataset
- 按 ratio 构造辅助集
- 将 QS 标签映射到 SAP 1~7 标度
- 合并成单一回归任务

### 9.2.2 标签来源字段

每条样本必须带：

- `domain = sap / qualispeech`
- `task_dim`
- `label_raw`
- `label_aligned`
- `label_for_loss`

### 9.2.3 loss

默认单一 `MSELoss`

### 9.2.4 验证与 checkpoint 选择

- 只在 `SAP val` 上做 model selection
- primary metric = `val_mse`
- tie-break：
  1. `higher val_lcc`
  2. `higher val_srcc`

## 9.3 FT：两阶段预训练 + 微调

实现 `src/trainers/ft_trainer.py`

### 9.3.1 Stage 1

- 数据：QS aux train / val
- 标签：QS 原始 1~5 标度
- 输出：stage1 best ckpt

### 9.3.2 Stage 2

- 加载 stage1 best ckpt
- 数据：SAP target train / val
- 标签：SAP 原始 1~7 标度
- 输出：stage2 best ckpt
- 最后在 SAP test 上评估

### 9.3.3 默认 head 行为

paper-faithful 默认：

- **不重置 regression head**
- 直接从 stage1 连续微调到 stage2

补充实验再做：

- `reset_head_stage2 = true`
- `reset_last_linear_only = true`

---

## 10. 模型头与 pooling 设计

## 10.1 主模型结构

```text
waveform
 -> HF SSL encoder
 -> masked mean pooling
 -> MLP hidden layer
 -> ReLU
 -> Dropout
 -> Linear scalar output
```

## 10.2 回归头默认参数

由于 PDF 没给出隐藏层宽度，建议写成配置可控，默认：

- `head_hidden_dim = encoder_hidden_size`
- `dropout = 0.1`

也支持：

- `head_hidden_dim = 512`
- `dropout = 0.2`

但主实验用统一默认值，所有 encoder 共享规则。

## 10.3 输出范围

训练阶段：

- 不对输出加 hard clip

评估阶段：

- 计算两版指标：
  - `raw_pred`
  - `clipped_pred`（clip 到目标标签范围）

主表默认使用：

- `raw_pred`

附录里额外给出：

- `clipped_pred`

这样 reviewer 如果质疑 out-of-range prediction，也有备用结果。

---

## 11. 训练细节与优化

## 11.1 优化器

主实验默认：

```yaml
optimizer: adamw
lr: 1e-5
weight_decay: 0.01
betas: [0.9, 0.999]
eps: 1e-8
```

## 11.2 scheduler

PDF 没写 scheduler，建议工程上实现两种：

- `none`（paper-faithful 默认）
- `cosine_with_warmup`（supplementary）

主实验默认 `none`，避免超出论文设置太多。

## 11.3 训练时长与早停

建议默认：

### baseline / JT / FT stage2
- `max_epochs = 30`
- `patience = 5`

### FT stage1
- `max_epochs = 20`
- `patience = 4`

如果数据规模较大、收敛较慢，再从配置调整。

## 11.4 mixed precision

默认启用：

- `bf16` 优先
- 不支持就 `fp16`
- 再不支持就 `fp32`

## 11.5 gradient accumulation

必须支持，因为 large model + 长 utterance 很容易爆显存。

## 11.6 dynamic batching

不要用固定样本数 batch。建议按总音频秒数做 batch sampler：

- base 模型：每卡 `max_total_sec ~ 180`
- large 模型：每卡 `max_total_sec ~ 90`
- 实际写成配置项

### 11.6.1 为什么要这样

SAP 有 read，也有 spontaneous，时长波动很大。固定 batch size 会让显存抖动明显，动态秒数 batch 更稳定。

## 11.7 gradient checkpointing

对 large 模型默认支持开关：

- `gradient_checkpointing = true`

## 11.8 DDP

需要支持：

- 单卡
- 多卡 DDP

每个 run 目录都要保存：

- world size
- effective batch size
- accumulation steps

---

## 12. 评估层设计

## 12.1 主指标

主表必须输出：

- `MSE`
- `LCC`
- `SRCC`

## 12.2 补充指标

建议顺手算上：

- `MAE`
- `CCC`
- `RMSE`

主表可以不放，summary 里保留。

## 12.3 per-utterance prediction 文件

每个 test run 都要输出：

```csv
utt_id,speaker_id,audio_path,y_true,y_pred,y_pred_clipped,domain,target_dim,encoder,method,pair_id,ratio,seed
```

## 12.4 bootstrap 置信区间

对 test metrics 做 paired bootstrap：

- `n_bootstrap = 10000`
- 输出 95% CI

## 12.5 显著性检验

推荐实现：

- 对 MSE 差异做 paired permutation test
- 对 LCC / SRCC 差异做 bootstrap difference CI

主表可加：

- `*`
- `**`

并在脚注写检验方式。

---

## 13. 结果汇总与论文制表

## 13.1 汇总主键

统一 run 唯一键：

```text
protocol / encoder / method / sap_target / qs_aux / pair_id / ratio / seed / split_protocol / stage
```

## 13.2 每个 run 目录结构

```text
results/runs/{run_id}/
├── config_resolved.yaml
├── train_log.csv
├── val_metrics.csv
├── best.ckpt
├── last.ckpt
├── test_metrics.json
├── test_predictions.csv
├── model_info.json
└── plots/
```

## 13.3 主表

建议自动导出以下表：

### Table A：SAP-only baseline
- 行：SAP target
- 列：encoder
- 指标：MSE / LCC / SRCC

### Table B：ratio=1 主结果
- 行：pair × method
- 列：encoder
- 指标：MSE / LCC / SRCC
- 同时给出相对 baseline 提升

### Table C：ratio 消融
- 行：ratio
- 列：metric
- 每个 pair + encoder 生成一张子表
- 同时再导出汇总版：按 pair 聚合、按 encoder 聚合

### Table D：reviewer controls
- dual-head JT
- SAP multi-task
- shuffled-label control
- speaker-disjoint val

## 13.4 图

至少自动生成这些图：

1. `ratio -> MSE` 折线图
2. `ratio -> LCC` 折线图
3. `ratio -> SRCC` 折线图
4. 每个 pair 的 improvement over baseline
5. encoder 大小 vs gain
6. prediction vs ground truth 散点图
7. residual histogram
8. severity-bin breakdown
9. prompt-category breakdown
10. per-etiology breakdown（如果样本量允许）

---

## 14. 主实验矩阵（要非常清楚）

## 14.1 SAP-only baseline

对以下 5 个目标都跑：

- Naturalness
- Inappropriate silences
- Distorted vowels
- Imprecise consonants
- Intelligibility

对以下 8 个 encoder 都跑：

- wavlm_base
- wavlm_base_plus
- wavlm_large
- w2v2_base
- w2v2_large_lv60
- w2v2_large_robust
- hubert_base
- hubert_large

## 14.2 ratio=1 主实验

对以下 6 个 pair：

- qs_nat_to_sap_nat
- qs_cont_to_sap_sil
- qs_dist_to_sap_vowel
- qs_dist_to_sap_cons
- qs_effort_to_sap_intel
- qs_overall_to_sap_intel

对以下 2 个 method：

- JT
- FT

对以下 8 个 encoder：

- 同上 8 个

## 14.3 全比例消融

对同样的 6 pair × 2 method × 8 encoder，跑：

- 0.25
- 0.5
- 0.75
- 1.0
- 1.5
- 2.0

## 14.4 seed 方案

建议至少：

- 主表：`3 seeds`
- reviewer 最终补充实验：`3 seeds`
- 如果算力允许，最佳模型再跑 `5 seeds`

默认种子建议：

- `13`
- `17`
- `23`

再开一个 stronger suite：

- `13, 17, 23, 29, 41`

---

## 15. reviewer 视角下，最值得补的实验

下面这些是我以 Interspeech reviewer 身份最建议补的内容。建议它们全部实现到代码库里，但主文只放最关键的几项，其余进附录。

## 15.1 Dual-head JT control（非常重要）

### 动机
论文已经解释 FT > JT 的原因可能来自 label misalignment 和 shared head 冲突。reviewer 很可能会问：

> 你们的 JT 表现差，究竟是联合训练本身不合适，还是“单头 + 线性映射”这套实现限制了 JT？

### 做法
实现一个 control：

- 共享 encoder
- SAP head 一个
- QS head 一个
- batch 中混合 SAP/QS
- loss：
  - SAP 样本走 SAP head
  - QS 样本走 QS head
- encoder 参数共享
- 不做 QS->SAP 线性映射

### 价值
这个实验能直接回答：
- FT 的优势来自“阶段式训练”
- 还是来自“避免共享回归头冲突”

这是最像 reviewer 追问的补充实验之一。

## 15.2 SAP-only multi-task control（非常重要）

### 动机
reviewer 会问：

> 增益增益是否来自“多任务学习”本身？如果只在 SAP 内部联合训练多个病理维度，会不会一样有效？

### 做法
实现一个 SAP-only multi-task baseline：

- 一个共享 encoder
- 多个 SAP target head
- 只用 SAP 数据
- 对 missing label 维度做 masked loss

### 对照
拿这个结果和：
- SAP-only single-task baseline
- QualiSpeech FT / JT

进行比较。

### 价值
如果跨域增广仍优于 SAP-only multi-task，论点会更强。

## 15.3 label-shuffled control（很重要）

### 动机
要验证辅助 supervision 的语义性，而非仅仅是多看了额外英语音频。

### 做法
对最强 1~2 个 encoder 做：

- 保持音频不变
- 随机打乱 QS 辅助标签
- 重新跑 FT / JT

### 期望
如果 shuffled 明显退化，说明增益来自有意义的 perceptual transfer。

## 15.4 random-pair / bad-pair control

### 动机
你现在的 6 个 pair 有明显语义假设。reviewer 可能会问：

> 如果随便配一个无关维度也能涨，那说明 pair 设计没有说服力。

### 做法
加 1~2 个明显偏弱的 pair 作为 negative control，例如：

- `Continuity -> Naturalness`
- `Distortion -> Inappropriate silences`

只需要选最强 2 个 encoder 跑即可。

## 15.5 stage2 head reset ablation

### 动机
FT 里 stage1 和 stage2 的标签范围不同，reviewer 可能会问 head 初始化是否影响结果。

### 做法
比较：

- `reuse_full_head`
- `reset_last_linear_only`
- `reset_full_head`

## 15.6 freeze schedule ablation

### 动机
小数据上 end-to-end tuning 未必总是最好。

### 做法
对最佳 1~2 个 encoder 比较：

- `freeze_encoder`
- `unfreeze_last_4_layers`
- `full_finetune`

## 15.7 speaker-disjoint validation control

### 动机
主实验的 val 从 train 抽 utterance，可能和 train 共享 speaker。reviewer 很容易问这一点。

### 做法
用 speaker-disjoint val 重跑最关键结果：

- baseline
- best FT pair
- best JT pair

## 15.8 severity-bin analysis（很值得放图）

### 动机
PDF 已经指出 Intelligibility 严重不平衡。reviewer 会关心增益到底出现在哪些 severity 段。

### 做法
对 test 按 label 分 bin：

- 1
- 2
- 3
- 4+
  或保持原始 1~7

输出：

- per-bin MAE / MSE
- per-bin count
- paired improvement

### 价值
这能解释模型究竟帮助了重度样本、轻度样本，还是只在主流类别上变好。

## 15.9 prompt-category breakdown（很值得）

### 动机
SAP 含 read / spontaneous / digital assistant prompts。增益可能只集中在某一类。

### 做法
按 `prompt_category` 分析：

- baseline
- best FT
- best JT

### 价值
这能帮助你回答：
- transfer 对 spontaneous speech 是否更有效？
- 对 read speech 是否更稳定？

## 15.10 etiology breakdown

### 动机
不同病种的语音表现不同。reviewer 会关心跨域 transfer 是否只对 Parkinson 有利。

### 做法
按 `etiology` 分层统计：

- count
- metric
- improvement

样本太少的病种可以只放附录。

## 15.11 ordinal-loss supplementary

### 动机
SAP / QS 标签本质是有序等级，用纯 MSE 会被 reviewer 追问。

### 做法
加一个 supplementary 对照：

- `MSE`
- `Huber`
- `CORAL` 或 ordinal regression loss（只对 best encoder 跑）

### 价值
不一定放主文，但实现后很有底气。

## 15.12 现代 encoder 对照（补充，不进主表也可以）

### 动机
Interspeech 2026 的 reviewer 可能会问：
- 只和 wav2vec2 / HuBERT / WavLM 对比是否足够？
- 有没有更现代的 HF audio backbone？

### 建议
补一个可选 supplementary benchmark：

- `Wav2Vec2-BERT 2.0` 或另一个公开 HF audio SSL encoder

只跑：
- SAP-only baseline
- 最强 FT pair

这项不属于你现在必须跑的主矩阵，但代码结构最好预留 registry 扩展位。

---

## 16. 训练编排与自动运行

## 16.1 suite 层

把实验拆成多个 suite：

### `smoke`
- 1 个 encoder
- 1 个 SAP target baseline
- 1 个 pair 的 JT / FT
- ratio=1
- 单 seed

### `baselines`
- 全部 encoder × 全部 SAP targets
- 单 seed或多 seed

### `main`
- 全部 encoder × 6 pair × 2 methods × ratio=1

### `ablation_all_models`
- 全部 encoder × 6 pair × 2 methods × 6 ratios

### `reviewer`
- dual-head JT
- SAP multi-task
- shuffled control
- speaker-disjoint val
- stage2 head reset
- freeze schedule

### `all`
- prepare
- baselines
- main
- ablation
- reviewer
- summarize
- export

## 16.2 skip-if-complete

每个 run 前检查：

- `test_metrics.json` 是否存在
- `status = complete` 是否写入 metadata

已完成则跳过。

## 16.3 resume

训练中断后支持：

- 自动从 `last.ckpt` 恢复
- 保留 optimizer state
- 保留 scheduler state
- 保留 current epoch

## 16.4 失败重试

对容易失败的环节加入：

- HF 下载失败自动重试
- 单个 run OOM 自动降低 batch 秒数再重启一次
- 若仍失败，标成 `failed`，后续汇总跳过但保留日志

---

## 17. 汇总脚本必须做的事情

实现 `scripts/summarize_results.py`：

1. 扫描所有 run 目录
2. 读取 resolved config
3. 读取 test metrics
4. 合并成 `all_results_long.csv`
5. 聚合成：
   - `all_results_mean_std.csv`
   - `best_per_encoder.csv`
   - `best_per_pair.csv`
   - `ratio_curves.csv`
6. 生成显著性结果：
   - `significance_tests.csv`

---

## 18. 画图脚本必须做的事情

实现 `scripts/export_figures.py`，至少输出：

- `fig_ratio_ablation_mse_{pair}.png`
- `fig_ratio_ablation_lcc_{pair}.png`
- `fig_ratio_ablation_srcc_{pair}.png`
- `fig_gain_over_baseline_{target}.png`
- `fig_prompt_breakdown_{pair}.png`
- `fig_severity_bin_{target}.png`
- `fig_encoder_scale_vs_gain.png`

同一张图尽量做到：

- 风格统一
- 颜色映射固定
- 方法线型固定（baseline / JT / FT）
- 输出 png + pdf 双格式

---

## 19. Makefile / CLI 设计

建议最少提供这些命令：

```bash
make prepare
make smoke
make baselines
make main
make ablation
make reviewer
make summarize
make figures
make tables
make report
make all
```

以及脚本入口：

```bash
python scripts/prepare_all.py --config configs/paths.yaml
python scripts/run_suite.py --suite configs/suite/main.yaml
python scripts/summarize_results.py
python scripts/export_tables.py
python scripts/export_figures.py
python scripts/package_report.py
```

---

## 20. 你这篇 Interspeech 稿件里我最建议强调的叙事

如果我是 reviewer，我最希望在最终稿里看到的逻辑链条是：

1. 先证明 SAP-only baseline 足够扎实；
2. 再证明跨域增广确实稳定提升；
3. 再证明提升和语义匹配相关；
4. 再证明 FT 优于 JT 的原因来自 label / head 冲突，同时排除“JT 实现受限”这一种解释；
5. 再证明收益在 imbalance 更严重、感知更接近的维度上更明显；
6. 最后用 prompt / severity / etiology breakdown 说明这类 transfer 何时最有效。

只要这条链条闭合，整篇 paper 的说服力会强很多。

---

## 21. 给 Codex 的实现顺序（必须按这个顺序开发）

### Phase 1：数据层
1. SAP parser
2. QS parser
3. manifest 标准化
4. split builder
5. pair builder
6. 数据完整性测试

### Phase 2：HF 模型层
1. model registry
2. processor loader
3. HFSSLBackbone
4. pooling / mask
5. regression head
6. 单 batch forward smoke test

### Phase 3：训练层
1. baseline trainer
2. JT trainer
3. FT trainer
4. checkpoint / early stopping / resume

### Phase 4：评估层
1. metrics
2. test prediction export
3. bootstrap CI
4. significance test

### Phase 5：编排层
1. run_experiment.py
2. run_suite.py
3. skip-if-complete
4. failed run logging

### Phase 6：分析层
1. summarize_results.py
2. export_tables.py
3. export_figures.py
4. package_report.py

### Phase 7：reviewer controls
1. dual-head JT
2. SAP multi-task
3. shuffled-label control
4. speaker-disjoint val
5. head reset / freeze ablation
6. prompt / severity / etiology breakdown

---

## 22. 验收标准（Codex 完成后逐条检查）

只要以下项目全部满足，我会认为代码库达到可用标准。

### 22.1 数据层
- SAP train/dev 能完整解析
- dev 重复 json 被正确处理
- `Intelligbility` 被规范到 `intelligibility`
- 空 Ratings 不会导致崩溃
- QualiSpeech CSV 列名全部规范化
- 所有音频路径都能对上

### 22.2 模型层
- 所有 8 个 encoder 都能从 HF 成功加载
- 仓库里没有 fairseq 依赖
- pooling mask 正确
- 单卡 / 多卡都能 forward

### 22.3 训练层
- baseline / JT / FT 都能正常训练
- FT stage1 -> stage2 串联正常
- dev 自动选 best ckpt
- 中断后可恢复

### 22.4 实验层
- 40 个 baseline run 能自动编排
- 6 pair × 2 method × 8 encoder 的 main run 能自动编排
- 6 个 ratio 的消融能自动编排
- reviewer suite 可单独运行

### 22.5 分析层
- 自动汇总 mean/std
- 自动导出 csv / latex 表
- 自动画图
- 自动给出最佳配置索引
- 自动生成一份总报告

### 22.6 复现层
- 所有 seed 固定
- model revision 固定
- config_resolved.yaml 落盘
- 每个 run 都能回溯到原始参数

---

## 23. 我对这套计划的最终建议

如果你的目标就是 Interspeech，我会建议你把这套代码库的优先级排成这样：

### 第一优先级
- SAP-only baseline 全量跑稳
- 6 个 pair 的 JT / FT 全量跑稳
- ratio=1 主结果和 6 档比例消融全自动产出

### 第二优先级
- dual-head JT
- SAP multi-task
- shuffled-label control
- speaker-disjoint val

### 第三优先级
- prompt / severity / etiology breakdown
- head reset / freeze schedule
- ordinal-loss supplementary
- 现代 encoder supplementary

这样安排最符合论文投稿节奏，也最符合 reviewer 的关注点。

---

## 24. 最终一句话给 Codex

请按这份计划实现一个 **完全基于 Hugging Face、禁止 fairseq、从数据处理到训练再到汇总出图都可一键执行** 的实验仓库；主实验严格覆盖 SAP-only baseline、QualiSpeech 增广的 JT/FT、6 个指定 pair、8 个 encoder、6 档比例消融，并额外实现 reviewer 关心的 dual-head JT、SAP multi-task、label-shuffle control、speaker-disjoint validation、prompt/severity/etiology breakdown 等补充实验。
