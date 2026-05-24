# Bitext Customer Service Data Analyst Agent

Assignment 3 — LangGraph ReAct agent over the [Bitext customer support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset). Tasks 1–3 plus bonuses (Streamlit UI, query recommender) are implemented.

## Setup (under 5 minutes)

```bash
cd bitext-agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set NEBIUS_API_KEY
python main.py
```

First run downloads ~27k rows from Hugging Face and caches them under `data/bitext.parquet`.

## CLI

```bash
python main.py
python main.py --session alice
python main.py --session alice --user alice
```

- `--session` — LangGraph `thread_id`; same value restores conversation after restart (`data/checkpoints.sqlite`).
- `--user` — profile file under `data/profiles/` (defaults to `--session`).

Interactive loop. Type `quit` to exit. The CLI prints:

- Router decision (`structured` / `unstructured` / `out_of_scope`)
- Each tool call and observation
- Final answer

### Session memory (Task 2a)

Follows the course notebook pattern ([agent_memory.ipynb](../agent_memory.ipynb)): compile with a checkpointer, pass only the **new** user message each turn, reuse `thread_id` via `--session`. We use **SqliteSaver** instead of `MemorySaver` so state survives restarts.

Example:

```text
You: How many complaints are in the dataset?
…
You: What about refunds?
```

### User profile (Task 2b)

Distilled facts (name, interests) live in `data/profiles/<user>.json`, separate from chat checkpoints. Updated after each in-scope turn. Ask: *What do you remember about me?*

## Model

**`meta-llama/Llama-3.3-70B-Instruct`** via [Nebius Token Factory](https://api.tokenfactory.nebius.com/v1/).

We use the same model for routing and the ReAct agent. We tried smaller models on the platform (e.g. Gemma 3 27B), but tool calling and answer quality were worse. Llama 3.3 70B is described as optimized for chat quality, and in our tests it picked tools reliably—important for multi-step queries like filter → count. It felt like the right balance between cost and performance for this assignment.

## Architecture

```mermaid
flowchart TD
    start([User query]) --> router[router_node]
    router -->|out_of_scope| decline[decline_node]
    router -->|structured or unstructured| agent[react_agent]
    agent --> tools[tool_node]
    tools --> agent
    agent -->|final answer| endNode([END])
    decline --> endNode
```

1. **Router** — classifies the query before any tools run. Out-of-scope questions get a fixed decline (no general knowledge).
2. **ReAct agent** — Nebius LLM with tools; max **12** agent steps, then a fallback message.
3. **Tools** — composable filters and queries over the dataset.

**Filter state:** `filter_by_*` tools narrow a working view (row IDs in `working_row_ids`, checkpointed per session) so multi-step chains and follow-ups like “show me 3 more” keep the same filter.

## Tools

| Tool | Purpose |
|------|---------|
| `list_categories` | All category names |
| `list_intents` | Intent names, optional category filter |
| `filter_by_category` | Restrict working view to a category |
| `filter_by_intent` | Restrict working view to an intent |
| `reset_filters` | Clear filters |
| `count_rows` | Row count in current view |
| `sample_examples` | Random instruction/response pairs |
| `intent_distribution` | Intent counts in current view |

Multi-step example: `filter_by_intent("get_refund")` → `count_rows()`.

## MCP server (Task 3)

Expose dataset tools via [FastMCP](https://gofastmcp.com/) for MCP clients (Cursor, Claude Desktop, etc.).

### Start the server

```bash
python mcp_server.py
```

Or with the FastMCP CLI:

```bash
fastmcp run mcp_server.py
```

The server uses **stdio** transport (default): the client spawns this process and talks over stdin/stdout. The dataset is loaded once at startup.

**Exposed tools (7):** `list_categories`, `list_intents`, `filter_by_category`, `filter_by_intent`, `count_rows`, `sample_examples`, `intent_distribution`.

Stateful filters work within one MCP session: call `filter_by_intent` then `count_rows` (same as the CLI agent).

### Connect from Cursor

Add to your Cursor MCP settings (`.cursor/mcp.json` or Settings → MCP):

```json
{
  "mcpServers": {
    "bitext-analyst": {
      "command": "/absolute/path/to/bitext-agent/.venv/bin/python",
      "args": ["/absolute/path/to/bitext-agent/mcp_server.py"]
    }
  }
}
```

Replace paths with your machine’s paths.

### Example: call one tool

In Cursor chat (with the MCP server enabled), ask the model to use the `list_categories` tool.

Or test from the terminal:

```bash
python scripts/test_mcp.py
```

Manual chain example:

1. `filter_by_intent` with `intent="get_refund"`
2. `count_rows` → should return `997`

## Bonus A — Streamlit UI

```bash
streamlit run streamlit_app.py
```

- Chat interface with assistant responses
- **Reasoning steps** expander (router, tool calls, observations)
- **Sidebar:** Session ID and User ID (same semantics as CLI `--session` / `--user`)

## Bonus B — Query recommender

Ask: *What should I query next?* (also in the CLI)

1. Agent **suggests** a follow-up query — does **not** run it yet
2. You can refine (*"I'd rather see examples instead"*)
3. Reply **yes** / **go ahead** / **do it** to execute the suggested query

Works in both `streamlit run streamlit_app.py` and `python main.py`.

## Example queries

- What categories exist in the dataset?
- How many refund requests did we get?
- Show me 5 examples of the SHIPPING category.
- Summarize how agents respond to complaint intents.
- What is the distribution of intents in the ACCOUNT category?
- Who is the president of France? (out-of-scope)
