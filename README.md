# arxiv-digest

基于 Hugging Face Daily Papers 的个性化中文论文日报系统。系统先做本地关键词预筛选，再用 OpenAI-compatible LLM 判断真实相关性并翻译 abstract；只对 Top-K 高相关论文获取全文和生成深度解读，最后保存 Markdown，并可选择推送飞书。

旧的 `scripts/generate_digest.py` 三源工作流仍然保留。

## Installation

项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python 环境：

```bash
uv sync --dev
uv run pytest -q
```

支持 Python 3.11 及以上版本。

## Configure topics

编辑 `config/topics.json`：

```json
{
  "topics": [
    {
      "name": "Video Reasoning",
      "keywords": [
        "video reasoning",
        "video understanding",
        "temporal reasoning"
      ]
    },
    {
      "name": "Multimodal Agents",
      "keywords": [
        "multimodal agent",
        "vision-language agent"
      ]
    }
  ],
  "exclude_keywords": []
}
```

旧格式 `{"topics": ["LLM agent", "RLHF"]}` 仍然支持。过滤器检查 title、abstract、HF summary 和 HF keywords，并归一化大小写、空格和短横线。可在 `config/preferences.json` 中设置 `keyword_filter.enabled=false` 暂停预筛选。

## Configure LLM

复制 `.env.example` 中的变量并在 shell 或 secret manager 中设置：

```bash
export SCREENING_LLM_BASE_URL="https://openrouter.ai/api/v1"
export SCREENING_LLM_API_KEY="..."
export SCREENING_LLM_MODEL="openai/gpt-5-nano"

export DEEP_LLM_BASE_URL="https://openrouter.ai/api/v1"
export DEEP_LLM_API_KEY="..."
export DEEP_LLM_MODEL="google/gemini-3.8-flash"
```

screening 模型负责 relevance、novelty、impact、abstract 翻译和 TL;DR；deep 模型只处理 Top-K 全文总结。两端接口都需兼容 OpenAI `/chat/completions`。如果未配置带前缀的新变量，系统会回退到旧的 `LLM_BASE_URL/LLM_API_KEY/LLM_MODEL` 单模型配置。API key 只能来自环境变量；不要把 `.env` 或真实 secret commit 到仓库。

## Configure Feishu

用户侧配置步骤：

```text
飞书群 → 群设置 → 机器人 → 添加机器人 → 自定义机器人 → 获取 Webhook
```

如果开启了签名校验，同时保存机器人 secret：

```bash
export FEISHU_WEBHOOK_URL="..."
export FEISHU_WEBHOOK_SECRET="..."  # optional
```

在 `config/preferences.json` 中将 `delivery.feishu.enabled` 改为 `true`，即可让默认 daily run 推送。系统按论文边界生成 interactive cards，并用 digest hash receipt 防止 cron 重跑造成重复消息。

## Run locally

完整默认流程：

```bash
uv run python scripts/run_daily.py
```

小规模验证，明确禁止投递：

```bash
uv run python scripts/run_daily.py \
  --date today \
  --max-papers 5 \
  --deep-top-k 1 \
  --no-delivery
```

指定历史日期：

```bash
uv run python scripts/run_daily.py --date 2026-09-03
```

## Dry run

`--dry-run` 仍会抓取、筛选、调用 LLM 和生成 Markdown，但只构建飞书 payload，不发送：

```bash
uv run python scripts/run_daily.py --deliver feishu --dry-run
```

## Run without delivery

```bash
uv run python scripts/run_daily.py --no-delivery
```

## Force cache refresh

```bash
uv run python scripts/run_daily.py \
  --force-screening \
  --force-deep-summary
```

只有明确需要重复投递时才增加 `--force-delivery`。

## Resend existing digest

重发不会重新调用 HF、arXiv、LLM 或全文总结：

```bash
uv run python scripts/resend_daily_digest.py \
  memory/digests/2026-09-03.md \
  --feishu
```

如 receipt 已记录相同 digest，默认跳过；使用 `--force` 显式重发。

## Tests

普通测试完全离线，并默认排除 live marker：

```bash
uv run pytest -q
uv run pytest -m "not live" -q
uv run pytest -q tests/e2e/test_daily_pipeline_offline.py -vv
```

## Live tests

Live tests 必须显式 opt in：

```bash
uv run pytest -m live tests/live/test_huggingface_live.py -vv
uv run pytest -m live tests/live/test_fulltext_live.py -vv
```

真实 LLM 测试仅在三个 `LLM_*` 变量都存在时执行，且只 screening 一篇短样例：

```bash
uv run pytest -m live tests/live/test_llm_live.py -vv
```

飞书 live test 只读取 `FEISHU_TEST_WEBHOOK_URL`，绝不使用 production webhook；它发送一条带 `[TEST]` 标题的消息：

```bash
uv run pytest -m live tests/live/test_feishu_live.py -vv
```

完整真实 smoke test 应先禁用投递：

```bash
uv run python scripts/run_daily.py \
  --date today \
  --max-papers 5 \
  --deep-top-k 1 \
  --no-delivery
```

确认 `memory/runs/<date>/run.json` 和 `memory/digests/<date>.md` 后，再使用 `--deliver feishu`。

## Scheduled execution

仓库包含 `.github/workflows/daily-digest.yml`，默认每天 **UTC+8 上午 10:00** 自动执行。GitHub Actions cron 使用 UTC，因此配置为：

```yaml
schedule:
  - cron: "0 2 * * *"
```

Scheduled workflow 只从默认分支运行，GitHub 在繁忙时可能延迟几分钟。也可以在 Actions 页面通过 `workflow_dispatch` 手动触发。运行状态缓存会保存 screening、全文、deep summary 和飞书 receipt，手动重跑同一日报不会重复调用 LLM 或重复推送。

如果改用主机 cron：

```bash
scripts/run_daily.sh --no-delivery
```

Cron 示例：

```cron
0 9 * * * /path/to/arxiv-digest/scripts/run_daily.sh
```

Cron 使用服务器本地时区。请在主机或 crontab 中显式设置 `TZ`；系统不会假定服务器位于 Asia/Shanghai 或 Asia/Singapore。还需确保 cron 环境能找到 `uv` 和所需的环境变量。

## Directory structure

```text
arxiv_digest/                 核心模块
  sources/                    HF Daily 和 arXiv metadata
  llm/                        client、screening、deep summary
  fulltext/                   HF/HTML/PDF reader
  rendering/                  Markdown digest
  delivery/                   Feishu webhook
scripts/run_daily.py          新 daily CLI
scripts/generate_digest.py    保留的 legacy CLI
tests/e2e/                    完全离线端到端测试
tests/live/                   opt-in 真实服务测试
memory/analysis/              screening cache
memory/fulltext/              PDF 与正文 cache
memory/deep_summaries/        deep summary cache
memory/digests/               Markdown 日报
memory/runs/                  每次运行报告
memory/delivery/              防重复投递 receipt
```

## Failure recovery

- HF source 完全失败：本次 run 失败，不生成伪造日报。
- 单篇 arXiv enrichment 失败：保留 HF metadata。
- 单篇 screening 失败：记录 error，继续其他论文。
- 全文失败：保留中文 abstract/TL;DR，不进行 deep summary。
- deep summary 失败：使用 compact summary，不中止日报。
- 飞书失败：本地 digest 已先保存，可使用 resend 命令补发。
- 重复运行：有效 screening、fulltext 和 deep-summary cache 会被复用；相同 digest 不会重复推送。

## Legacy commands

以下接口继续可用：

```bash
uv run python scripts/generate_digest.py --raw
uv run python scripts/generate_digest.py \
  --rerank-json memory/llm_scores.json \
  --output memory/daily_digest.md
```

## License

MIT
