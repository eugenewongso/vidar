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
                    all_conflict_rows.append(dict(cve=parts[0], description=parts[1], category=parts[2], explanation=parts[3]))
                    category_counter[parts[2]] += 1
                elif section == "categories" and len(parts) == 2:
                    all_category_rows.append(dict(category=parts[0], description=parts[1]))

    # Save CSV
    conflict_df = pd.DataFrame(all_conflict_rows)
    df_output = df.copy()
    df_output["llm_conflict_description"] = ""
    df_output["llm_conflict_category"] = ""
    df_output["llm_conflict_explanation"] = ""

    for i, row in conflict_df.iterrows():
        match_idx = df_output[df_output["CVE"] == row["cve"]].index[i]
        df_output.at[match_idx, "llm_conflict_description"] = row["description"]
        df_output.at[match_idx, "llm_conflict_category"] = row["category"]
        df_output.at[match_idx, "llm_conflict_explanation"] = row["explanation"]

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

if __name__ == "__main__":
    asyncio.run(full_pipeline("linux_kernel_2025_cves.csv"))
