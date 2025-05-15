import os
import pandas as pd
import asyncio
from dataclasses import dataclass
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_vertex import GoogleVertexProvider
from pathlib import Path
import json

# Load environment variables
load_dotenv()
GCP_REGION = "us-central1"

# Set up Vertex AI provider
provider = None
try:
    provider = GoogleVertexProvider(region=GCP_REGION)
except Exception:
    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if sa_path:
        provider = GoogleVertexProvider(service_account_file=sa_path, region=GCP_REGION)
if provider is None:
    raise RuntimeError("❌ No valid Vertex AI credentials found.")

# Set up Gemini model
model = GeminiModel("gemini-2.5-pro-preview-03-25", provider=provider)

@dataclass
class PatchDeps:
    api_key: str  # Not used, required for compatibility

# Main pipeline
async def full_pipeline(input_csv: str):
    csv_base = Path(input_csv).stem
    output_csv = f"{csv_base}_conflict_analysis.csv"
    output_md = f"{csv_base}_discovered_categories.md"
    output_json = f"{csv_base}_discovered_categories.json"

    df = pd.read_csv(input_csv)
    cve_list = df["CVE"] if "CVE" in df.columns else ["UNKNOWN"] * len(df)

    rows = []
    for i, row in df.iterrows():
        upstream = str(row.get("upstream_patch_diff", "") or "").strip()
        downstream = str(row.get("downstream_patch_diff", "") or "").strip()
        error = str(row.get("error_message", "") or "").strip()
        if upstream and downstream and error:
            rows.append({
                "index": i + 1,
                "cve": cve_list[i] if i < len(cve_list) else "UNKNOWN",
                "upstream": upstream,
                "downstream": downstream,
                "error": error
            })

    all_input = "".join(
        f"### Patch Conflict {r['index']}\n"
        f"**Upstream Patch Diff:**\n```diff\n{r['upstream']}\n```\n"
        f"**Patch Error Message:**\n```\n{r['error']}\n```\n"
        f"**Downstream Patch (ground truth):**\n```diff\n{r['downstream']}\n```\n\n"
        for r in rows
    )

    summarization_prompt = """
You are analyzing kernel patch conflicts during backporting.

Each block includes:
- An upstream patch (which failed),
- A downstream patch (that succeeded),
- An error message showing why the upstream patch failed.

Compare the upstream and downstream patches, and identify the type of conflict. Use one of the following categories:

| Label   | Category                        | Description |
|---------|----------------------------------|-------------|
| Type I  | Location Only                    | Statements are reordered or appear in different positions but the logic and structure remain unchanged. |
| Type II | Namespace Only                   | Only variable, function, or class names differ. Logic and structure are semantically equivalent. |
| Type III| Namespace + Location Changes     | Mix of renaming and reordering. Though structurally different, logic is still equivalent. |
| Type IV | Significant Logical/Structural   | Substantive semantic differences — such as added logic, missing conditions, or altered control flow. |
| Type V  | File Missing                     | Patch fails to apply because the expected file is completely missing or never existed in the downstream version. |

---

For each case:
1. Write a 1–2 sentence description of the issue.
2. Assign the most appropriate **Type I–V** label and category.
3. Provide a short explanation for why you selected that category.

Return results in the following markdown sections:

1. Conflict Table:
| CVE | Description | Category | Explanation |

2. Category Descriptions Table:
| Category | Description |

3. Summary Table:
| Category | Count |
""" + all_input

    summarizer = Agent(model)
    response = await summarizer.run(summarization_prompt)

    with open(output_md, "w") as f:
        f.write(response.data)
    print(f"✅ Saved full markdown to: {output_md}")

    # Parse markdown
    lines = response.data.splitlines()
    conflict_rows, category_rows, summary_rows = [], [], []
    section = None

    for line in lines:
        if "| CVE | Description | Category | Explanation |" in line:
            section = "conflicts"
            continue
        elif "| Category | Description |" in line and "Descriptions" in section if section else "":
            section = "categories"
            continue
        elif "| Category | Count |" in line:
            section = "summary"
            continue
        if line.startswith("|---") or not line.startswith("|"):
            continue

        parts = [p.strip() for p in line.split("|")[1:-1]]
        if section == "conflicts" and len(parts) == 4:
            conflict_rows.append(dict(cve=parts[0], description=parts[1], category=parts[2], explanation=parts[3]))
        elif section == "categories" and len(parts) == 2:
            category_rows.append(dict(category=parts[0], description=parts[1]))
        elif section == "summary" and len(parts) == 2:
            try:
                summary_rows.append(dict(Category=parts[0], Count=int(parts[1].replace("**", "").strip())))
            except ValueError:
                continue

    # Save to CSV
    conflict_df = pd.DataFrame(conflict_rows)
    df_output = df.copy()
    if not conflict_df.empty:
        df_output["llm_conflict_description"] = conflict_df["description"]
        df_output["llm_conflict_category"] = conflict_df["category"]
        df_output["llm_conflict_explanation"] = conflict_df["explanation"]
        df_output.to_csv(output_csv, index=False)
        print(f"✅ Saved enriched CSV to: {output_csv}")
    else:
        print("❌ Conflict results couldn't be parsed. Please check LLM output.")

    # Save to JSON
    json_output = {
        "conflicts": conflict_rows,
        "categories": category_rows,
        "summary": summary_rows
    }
    with open(output_json, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"✅ Saved structured JSON to: {output_json}")

if __name__ == "__main__":
    asyncio.run(full_pipeline("linux_kernel_2025_cves.csv"))
