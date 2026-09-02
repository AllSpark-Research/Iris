# miroflow-tools

MCP servers for the two tools the Iris agent uses, plus the `ToolManager` that
spawns them over stdio and normalizes their results.

| Server | Exposes | Requires |
| --- | --- | --- |
| `searching_serper_mcp_server` | `web_search`, `scrape_website` | `SERPER_API_KEY` + `SUMMARY_LLM_*` |
| `searching_jina_mcp_server` | `scrape_website` | `JINA_API_KEY` + `SUMMARY_LLM_*` |
| `searching_zhipu_mcp_server` | `web_search`, `scrape_website` | `ZHIPU_AI_KEY` + `SUMMARY_LLM_*` |

An agent config picks the backends and blacklists the overlapping tools so that
exactly one server owns each tool name — see `conf/agent/` in the repo root.

All three force LLM extraction on `scrape_website` — it returns a condensed
view of the page against the caller's `info_to_extract`, never raw markdown —
so `SUMMARY_LLM_*` is required, not optional, whichever backend you pick.

Derived from [MiroFlow](https://github.com/MiroMindAI/MiroFlow) (Apache 2.0).
