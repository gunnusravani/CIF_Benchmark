from tqdm import tqdm
import pandas as pd
from utils.config_loader import load_yaml_config, load_path_registry
from utils.openai_utils import map_items_to_categories

paths = load_path_registry()
constraint_tasks = load_yaml_config(paths["constraint_tasks"])
categories = load_yaml_config(paths["constraint_categories"])
prompt_config = load_yaml_config(paths["constraint_prompts"])

defaults = constraint_tasks["constraint_mapping"]
input_path = defaults["input_path"]
output_path = defaults["output_path"]
model = defaults["model"]

df = pd.read_csv(input_path)

prompt_template = prompt_config["map_items_to_categories"]
category_data = categories["constraint_categories_v2"]

# Apply mapping
df["Mapped_Characteristics"] = tqdm(
    df.apply(
        lambda row: map_items_to_categories(
            row, "Characteristics_List", category_data, prompt_template
        ),
        axis=1,
    )
)
df["Mapped_Constraints"] = tqdm(
    df.apply(
        lambda row: map_items_to_categories(
            row, "constraints", category_data, prompt_template
        ),
        axis=1,
    )
)

df.to_csv(output_path, index=False)
print("✅ Done mapping and saved file.")
