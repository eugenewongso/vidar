# active_learning_loop.py
import json
import re
from datetime import datetime
import pandas as pd
import numpy as np
import sys
from tree_sitter import Language, Parser

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ------------------------
# Helper functions
# ------------------------
def count_diff_lines(diff_text):
    return sum(
        1 for line in diff_text.splitlines()
        if (line.strip().startswith('+') and not line.strip().startswith('+++')) or
           (line.strip().startswith('-') and not line.strip().startswith('---'))
    )

def extract_inline_conflict_lines(text):
    blocks = re.findall(r'<<<<<<<.*?\n(.*?)>>>>>>>', text, re.DOTALL)
    return sum(len(block.splitlines()) for block in blocks)

def parse_patch_error(error_msg):
    if "can't find file to patch" in error_msg:
        return "file_missing"
    elif "hunk FAILED" in error_msg:
        return "hunk_failed"
    elif "Skipping patch" in error_msg:
        return "skipped"
    else:
        return "other"

def get_days_apart(date1_str, date2_str):
    fmt = "%Y-%m-%d %H:%M:%S %z"
    try:
        date1 = datetime.strptime(date1_str, fmt)
        date2 = datetime.strptime(date2_str, fmt)
        return abs((date2 - date1).days)
    except:
        return None

def extract_file_name(diff_text):
    match = re.search(r'^---\s+(.+)', diff_text, re.MULTILINE)
    if match:
        return match.group(1).replace('.rej', '').strip()
    return None

# -------- Tree-sitter AST Enhancer --------
try:
    C_LANGUAGE = Language('build/my-languages.so', 'c')
    PARSER = Parser()
    PARSER.set_language(C_LANGUAGE)

    def estimate_ast_features(text):
        try:
            tree = PARSER.parse(bytes(text, "utf8"))
            root_node = tree.root_node

            functions = set()
            scope_change = 0

            def walk_and_count(node):
                nonlocal scope_change
                total = 1  # count this node
                if node.type == "function_definition":
                    functions.add(node.start_byte)
                    scope_change = 1
                for child in node.children:
                    total += walk_and_count(child)
                return total

            node_count = walk_and_count(root_node)
            return node_count, scope_change, len(functions)

        except Exception as e:
            print("Tree-sitter parsing failed, falling back:", e)
            return 0, 0, 0

except Exception as fallback_error:
    print("\u26a0\ufe0f Tree-sitter not available, using regex instead:", fallback_error)

    def estimate_ast_features(text):
        function_defs = re.findall(r'\b(?:[\w\*\s]+)?\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\([^)]*\)\s*\{', text)
        class_defs = re.findall(r'\bclass\s+\w+', text)
        changed_function_count = len(set(function_defs))
        total_structure_changes = len(function_defs) + len(class_defs)
        return total_structure_changes, int(changed_function_count > 0), changed_function_count

# ------------------------
# Feature extraction
# ------------------------
def extract_features_from_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    features_list = []

    for category in ["cves_with_all_failures", "cves_with_partial_failures"]:
        for cve in data.get(category, []):
            cve_id = cve.get("cve_url", "").split("/")[-1].replace(".json", "")
            for attempt in cve.get("patch_attempts", []):
                upstream_date = attempt.get("upstream_commit_date", "")
                for result in attempt.get("patch_results", []):
                    features = {
                        "cve_id": cve_id,
                        "result": result.get("result", "failure"),
                        "num_failed_hunks": result.get("total_failed_hunks", 0),
                        "patching_error_type": parse_patch_error(result.get("error", "")),
                        "has_rej_file": int("No rejected diff content found." not in result.get("rej_file_content", "")),
                        "has_inline_conflict": int("No conflict markers found." not in result.get("inline_merge_conflict", "")),
                        "upstream_commit_date": upstream_date,
                        "commit_date": result.get("commit_date", ""),
                        "days_between_commits": get_days_apart(upstream_date, result.get("commit_date", "")),
                    }

                    rej = result.get("rej_file_content", "")
                    inline = result.get("inline_merge_conflict", "")
                    features["loc_diff_size"] = count_diff_lines(rej) + count_diff_lines(inline)
                    features["inline_conflict_lines"] = extract_inline_conflict_lines(inline)
                    features["conflict_file_name"] = extract_file_name(rej)

                    ast_node_diff_count, function_scope_change, number_of_changed_functions = estimate_ast_features(rej + "\n" + inline)
                    features["ast_node_diff_count"] = ast_node_diff_count
                    features["function_scope_change"] = function_scope_change
                    features["number_of_changed_functions"] = number_of_changed_functions

                    features_list.append(features)

    return pd.DataFrame(features_list)

# ------------------------
# Main execution
# ------------------------
if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else "merge_conflict_data.json"
    df = extract_features_from_json(json_path)
    df.to_csv("features.csv", index=False)
    print("\u2705 Features extracted and saved to features.csv")

    try:
        labeled_df = pd.read_csv("evaluation_dataset/labeled_features.csv")
        labeled_df = labeled_df.dropna(subset=["label"])

        categorical_cols = ["patching_error_type", "conflict_file_name"]
        label_encoders = {col: LabelEncoder().fit(df[col].fillna("").tolist() + labeled_df[col].fillna("").tolist()) for col in categorical_cols}
        for col in categorical_cols:
            df[col] = label_encoders[col].transform(df[col].fillna(""))
            labeled_df[col] = label_encoders[col].transform(labeled_df[col].fillna(""))

        feature_cols = [
            "loc_diff_size", "inline_conflict_lines", "num_failed_hunks",
            "patching_error_type", "has_inline_conflict", "conflict_file_name",
            "ast_node_diff_count", "function_scope_change", "number_of_changed_functions"
        ]

        X_train = labeled_df[feature_cols]
        y_train = labeled_df["label"]

        base = DecisionTreeClassifier(max_depth=3)
        clf = AdaBoostClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        X_unlabeled = df[feature_cols]
        probs = clf.predict_proba(X_unlabeled)
        max_confidence = np.max(probs, axis=1)
        uncertainty = 1 - max_confidence

        df["uncertainty"] = uncertainty
        df["predicted_label"] = clf.predict(X_unlabeled)
        df["confidence"] = max_confidence

        df = df[df["predicted_label"] != "COMMENTS"]
        df.to_csv("features_with_predictions.csv", index=False)
        print("\u2705 Saved features_with_predictions.csv with all predicted labels and confidence scores")

    except FileNotFoundError:
        print("\u26a0\ufe0f 'labeled_features.csv' not found. Skipping model training.")
