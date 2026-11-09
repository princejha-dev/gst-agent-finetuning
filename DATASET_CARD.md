---
license: apache-2.0
language:
- en
- hi
pretty_name: GST Tool-Calling (India)
size_categories:
- 1K<n<10K
task_categories:
- text-generation
tags:
- function-calling
- tool-use
- india
- gst
- hinglish
- synthetic
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: test
    path: data/test.jsonl
---

# GST Tool-Calling (India)

Synthetic multi-turn dataset for fine-tuning small LLMs to act as an Indian GST assistant that **calls tools correctly**, chains them, asks for missing details, and declines out-of-scope requests. Prompts are in English and Hinglish (Romanised Hindi).

## Why synthetic and programmatic
Every tool result is produced by deterministic Python code (`generate_dataset.py`), so calculations, GSTIN checksums and due dates are exact. No LLM-generated numbers.

## Tools (8)
`calculate_gst`, `split_gst`, `determine_supply_type`, `get_gst_rate`, `validate_gstin`, `calculate_interest`, `get_filing_due_date`, `calculate_net_gst_payable`. Schemas are in `tools.json` (OpenAI function-calling format).

## Example types
| Category | What it teaches |
|---|---|
| Single tool call (7 tools) | Pick the right tool and fill arguments from natural language |
| `multi_rate_then_calc` | Sequential chain: look up rate, then calculate |
| `multi_invoice_breakup` | Parallel calls, then a dependent call using earlier outputs |
| `get_gst_rate_unknown` | Handle "not found" without inventing a rate |
| `clarify` | Ask one question when required arguments are missing |
| `concept` | Answer GST concept questions with **no** tool call |
| `out_of_scope` | Decline unrelated or tax-evasion requests |

## Format
One JSON object per line:
- `messages`: chat list with roles `system`, `user`, `assistant` (with `tool_calls`), `tool`
- `gold_calls`: JSON string of the expected tool calls, for evaluation
- `category`, `lang` (`en` / `hi` = Hinglish), `unseen_template`, `id`

`tool_calls[].function.arguments` is stored as a **JSON string** (keeps the Arrow schema stable across tools). Parse with `json.loads` before applying a chat template that expects a dict.

## Splits
- `train`: ~2.4k examples
- `test`: ~0.6k examples. It contains (a) ~10% random holdout and (b) rows written from **prompt templates never used in training** (`unseen_template=true`) to measure generalisation beyond memorised phrasing. No prompt appears in both splits.

## Limitations
- Rate table is a small **illustrative subset** following GST 2.0 (effective 22 Sep 2025). Rates change; verify against official notifications.
- Prompts come from templates, so real user language is more varied. Plan to add a small hand-written eval set.
- `validate_gstin` checks format and checksum only, not whether a GSTIN is registered.
- Due dates are the standard dates for regular monthly filers (no QRMP, holidays or extensions).
- Not tax advice.
