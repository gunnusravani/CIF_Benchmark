import os
import openai
from dotenv import load_dotenv
from utils.io_utils import load_csv, save_yaml
from utils.config_loader import load_yaml_config, load_path_registry
from utils.openai_utils import generate_constraint_categories, format_list_for_prompt

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def main():
    paths = load_path_registry()
    constraint_tasks = load_yaml_config(paths["constraint_tasks"])
    defaults = constraint_tasks["constraint_category_generation"]
    prompt_config = load_yaml_config(defaults["prompt_path"])

    df = load_csv(defaults["input_path"])
    model = defaults.get("model", "gpt-3.5-turbo")
    prompt_template = prompt_config["constraint_category_generation"]

    all_characteristics = sorted(
        set([item for sublist in df["Characteristics_List"] for item in eval(sublist)])
    )
    all_constraints = sorted(
        set([item for sublist in df["constraints"] for item in eval(sublist)])
    )

    # Save prompt to log
    prompt_log_path = defaults.get(
        "prompt_log_path", "data/outputs/constraint_category_prompt_log.txt"
    )
    os.makedirs(os.path.dirname(prompt_log_path), exist_ok=True)

    categories = generate_constraint_categories(
        characteristics_list=all_characteristics,
        constraints_list=all_constraints,
        prompt_template=prompt_template,
        model=model,
    )
    print(categories)
    with open(prompt_log_path, "w") as log_file:
        log_file.write("===== Prompt Sent to Model =====\n")
        log_file.write(prompt_template + "\n\n")
        log_file.write(format_list_for_prompt("Characteristics", all_characteristics))
        log_file.write("\n")
        log_file.write(format_list_for_prompt("Constraints", all_constraints))
        log_file.write("====Generated Constraints====\n")
        log_file.write(format_list_for_prompt(categories))

    if categories:
        save_yaml(categories, defaults["output_path"])
        print(f"Saved constraint categories to {defaults['output_path']}")
    else:
        print("No categories generated.")


if __name__ == "__main__":
    main()
