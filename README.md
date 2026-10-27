# PraisonAI-Nemo

Run a [PraisonAI](https://github.com/MervinPraison/PraisonAI) agent team as an
[NVIDIA NeMo Agent Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit) (NAT) workflow:
declared in NAT YAML, served by a Nemotron NIM, with per-call token usage in NAT's profiler.

> **Status: early spike (0.0.1.dev0), not on PyPI yet.** It registers with NAT and was tested
> against a local OpenAI-compatible stub. The build plan, verified facts and acceptance criteria
> are in the tracking issue.

## Install (from source)

```bash
pip install -e .            # pulls nvidia-nat-core>=1.8,<2 and praisonaiagents[llm]
nat info components | grep praisonai
```

Distribution name: `nemo-agent-toolkit-praisonai` · import: `nat.plugins.praisonai` ·
entry point: `nat.plugins` → `nat_praisonai` (NAT's third-party convention).

## Run

```bash
NVIDIA_API_KEY=... nat run --config_file examples/nemotron_team.yml --input "What day is it?"
```

The example uses `nvidia/nemotron-3.5-lightning-30b-a3b` (some accounts get 404 on Super 120B chat).

## What it registers

| `_type` / wrapper | Kind | Notes |
|---|---|---|
| `praisonai_team` | function / workflow | Builds fresh agents + an `AgentTeam` per request; a failed run raises |
| `praisonai` clients for `nim`, `openai`, `litellm` | LLM client | Yield an `Agent(llm={...})` dict |
| `praisonai` | tool wrapper | NAT functions become PraisonAI tools; calls run on NAT's event loop |
| profiler | — | `LLM_START`/`LLM_END` with token usage and `TOOL_START`/`TOOL_END` events |

## Known limits

- PraisonAI has no per-call LLM *start* signal yet, so profiler LLM latency reads 0.
- The profiler imports NAT models not exported from `nat.plugin_api`; pinned `<2`.
- `framework` is sent as `None` with `"praisonai"` in metadata, because NAT's field only accepts its own enum.

## License

MIT
