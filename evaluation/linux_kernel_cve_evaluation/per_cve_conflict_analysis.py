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
from collections import Counter

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

async def full_pipeline(input_csv: str):
    csv_base = Path(input_csv).stem
    output_csv = f"{csv_base}_conflict_analysis.csv"
    output_md = f"{csv_base}_discovered_categories.md"
    output_json = f"{csv_base}_discovered_categories.json"

    df = pd.read_csv(input_csv)
    if "CVE" not in df.columns:
        raise ValueError("❌ Input CSV must contain a 'CVE' column.")

    summarizer = Agent(model)
    all_conflict_rows = []
    all_category_rows = []

    # Initialize output DataFrame with both grouped and individual columns
    df_output = df.copy()
    for prefix in ["grouped", "individual"]:
        df_output[f"llm_conflict_description_{prefix}"] = ""
        df_output[f"llm_conflict_category_{prefix}"] = ""
        df_output[f"llm_conflict_explanation_{prefix}"] = ""

    category_counter = Counter()

    with open(output_md, "w") as md_out:
        for cve, group in df.groupby("CVE"):
            rows = []
            for i, row in group.iterrows():
                upstream = str(row.get("upstream_patch_diff", "") or "").strip()
                downstream = str(row.get("downstream_patch_diff", "") or "").strip()
                error = str(row.get("error_message", "") or "").strip()
                if upstream and downstream and error:
                    rows.append({
                        "index": i + 1,
                        "upstream": upstream,
                        "downstream": downstream,
                        "error": error
                    })

            if not rows:
                continue

            all_input = "".join(
                f"### Conflict Example {r['index']}\n"
                f"**Upstream Patch Diff:**\n```diff\n{r['upstream']}\n```\n"
                f"**Patch Error Message:**\n```\n{r['error']}\n```\n"
                f"**Downstream Patch (ground truth):**\n```diff\n{r['downstream']}\n```\n\n"
                for r in rows
            )

            summarization_prompt = f"""
You are analyzing patch conflicts for CVE: {cve}.

Compare each failed upstream patch and its successful downstream counterpart. Then classify the conflict using one of:

| Label   | Category                    | Description                                                                 |
|---------|-----------------------------|-----------------------------------------------------------------------------|
| Type I  | Location Only               | Statements are reordered but logic is unchanged.                           |
| Type II | Namespace Only              | Variable/function/class names differ, logic is equivalent.                 |
| Type III| Namespace + Location Change| Mix of renaming and reordering.                                            |
| Type IV | Significant Logic/Structure| Control flow or logic meaningfully changes.                                |
| Type V  | File Missing                | Patch fails because the file doesn't exist in downstream.                  |

Output:

1. Conflict Table:
| CVE | Description | Category | Explanation |

2. Category Descriptions:
| Category | Description |

3. Summary Table:
| Category | Count |
""" + all_input

            response = await summarizer.run(summarization_prompt)
            md_out.write(f"## CVE: {cve}\n\n")
            md_out.write(response.data + "\n\n")

            lines = response.data.splitlines()
            section = None

            for line in lines:
                if "| CVE | Description | Category | Explanation |" in line:
                    section = "conflicts"
                    continue
                elif "| Category | Description |" in line:
                    section = "categories"
                    continue
                elif "| Category | Count |" in line:
                    section = "summary"
                    continue
                if line.startswith("|---") or not line.startswith("|"):
                    continue

                parts = [p.strip() for p in line.split("|")[1:-1]]
                if section == "conflicts" and len(parts) == 4:
                    all_conflict_rows.append(dict(cve=parts[0], description=parts[1], category=parts[2], explanation=parts[3]))
                elif section == "categories" and len(parts) == 2:
                    all_category_rows.append(dict(category=parts[0], description=parts[1]))

    # Recompute category counts from grouped output
    conflict_df = pd.DataFrame(all_conflict_rows)
    category_counter = Counter(conflict_df["category"])

    # Update grouped columns
    for i, row in conflict_df.iterrows():
        match_idx = df_output[df_output["CVE"] == row["cve"]].index[i]
        df_output.at[match_idx, "llm_conflict_description_grouped"] = row["description"]
        df_output.at[match_idx, "llm_conflict_category_grouped"] = row["category"]
        df_output.at[match_idx, "llm_conflict_explanation_grouped"] = row["explanation"]

    # Row-level (individual) analysis
    for i, row in df.iterrows():
        upstream = str(row.get("upstream_patch_diff", "") or "").strip()
        downstream = str(row.get("downstream_patch_diff", "") or "").strip()
        error = str(row.get("error_message", "") or "").strip()
        if not (upstream and downstream and error):
            continue

        prompt = f"""
You are analyzing a single failed patch attempt for CVE: {row['CVE']}.

Upstream Patch Diff:
```diff
{upstream}
Patch Error Message:

go
Copy
Edit
{error}
Downstream Patch Diff (ground truth):

diff
Copy
Edit
{downstream}
Classify the conflict using one of the following categories:

Type I: Location Only

Type II: Namespace Only

Type III: Namespace + Location Change

Type IV: Significant Logic/Structure

Type V: File Missing

Output in this format: Category: ... Description: ... Explanation: ... """ response = await summarizer.run(prompt) parsed = response.data.strip().splitlines() cat = next((line.split(":", 1)[1].strip() for line in parsed if line.startswith("Category:")), "") desc = next((line.split(":", 1)[1].strip() for line in parsed if line.startswith("Description:")), "") expl = next((line.split(":", 1)[1].strip() for line in parsed if line.startswith("Explanation:")), "")

python
Copy
Edit
    df_output.at[i, "llm_conflict_category_individual"] = cat
    df_output.at[i, "llm_conflict_description_individual"] = desc
    df_output.at[i, "llm_conflict_explanation_individual"] = expl

# Save final results
df_output.to_csv(output_csv, index=False)
print(f"✅ Saved enriched CSV to: {output_csv}")

summary_rows = [{"Category": k, "Count": v} for k, v in category_counter.items()]
json_output = {
    "conflicts": all_conflict_rows,
    "categories": all_category_rows,
    "summary": summary_rows
}
with open(output_json, "w") as f:
    json.dump(json_output, f, indent=2)
print(f"✅ Saved structured JSON to: {output_json}")
print(f"✅ Saved full markdown to: {output_md}")
if name == "main": asyncio.run(full_pipeline("linux_kernel_2025_cves.csv"))