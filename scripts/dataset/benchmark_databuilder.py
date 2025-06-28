import pandas as pd
from datasets import load_dataset
import random
from utils.config_loader import load_yaml_config, load_path_registry
from scripts.dataset.extractor import extract_fields
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
default_output_path = os.path.join(
    project_root, "data/outputs/benchmark_data.csv"
)

# Sampling plan: dataset name → desired count 
TARGET_COUNTS = {
    "ajibawa-2023/Python-Code-23k-ShareGPT": 1500 * 25 // 100,  # 375
    "xlangai/DS-1000": 1500 * 15 // 100,                        # 225
    "bigcode/bigcodebench": 1500 * 25 // 100,                   # 375
    "Multilingual-Multimodal-NLP/McEval-Instruct": 1500 * 35 // 100,  # 525
}

def build_dataset(save_path=default_output_path):
    paths = load_path_registry()
    config = load_yaml_config(paths["datasets"])
    dataset_info = config.get("datasets", [])
    all_data = []

    for ds in dataset_info:
        name = ds["name"]
        split = ds["split"]

        if name not in TARGET_COUNTS:
            print(f"Skipping {name}, not part of sampling plan.")
            continue

        target = TARGET_COUNTS[name]
        print(f"\nLoading {name} [{split}], target samples: {target}")

        try:
            dataset = load_dataset(name, split=split)
            if name == "Multilingual-Multimodal-NLP/McEval-Instruct":
                # For McEval-Instruct, we sample only Python examples
                examples_all = dataset if isinstance(dataset, list) else list(dataset)
                examples = [ex for ex in examples_all if ex.get("language", "").lower() == "python"]
                n_available = len(examples)
                print(f"Filtered Python examples: {n_available}")
                n_sample = min(target, n_available)
            else:
                examples = dataset if isinstance(dataset, list) else list(dataset)
                n_available = len(examples)
                print(f"Available examples: {n_available}")
                n_sample = min(target, n_available)
            
            sampled = random.sample(examples, n_sample)
            print(f"Sampled {n_sample} from {n_available} available examples.")
            count = 0
            for ex in sampled:
                nl, code, test = extract_fields(ex, name)
                if nl and code:
                    all_data.append({
                        "dataset": name,
                        "instruction": nl,
                        "code": code,
                        "test": test,
                    })
                    count += 1

            print(f"Collected {count}/{n_sample} valid samples from {name}")

        except Exception as e:
            print(f"Error loading {name}: {e}")

    df = pd.DataFrame(all_data)
    df.to_csv(save_path, index=False)
    print(f"\nFinal CSV saved at {save_path} with {len(df)} rows.")
    return df

if __name__ == "__main__":
    build_dataset()
