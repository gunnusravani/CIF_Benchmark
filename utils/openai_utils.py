import openai
import os
import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
# rits_key = os.getenv("RITS_API_KEY")
# base_url = os.getenv("BASE_URL")
# client = OpenAI(api_key=rits_key, base_url=base_url)
client = OpenAI(api_key=openai_api_key)


def convert_text_to_list(text, prompt_template, model="gpt-4o-mini"):
    if not isinstance(text, str) or not text.strip():
        return []

    prompt = prompt_template + "\n\n" + text.strip()

    # try:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    output = response.choices[0].message.content[9:-3]
    print(output)
    parsed = eval(output.strip(), {"__builtins__": None}, {})
    if isinstance(parsed, list):
        return parsed
    return [output.strip()]


import openai
import yaml


def format_list_for_prompt(name, items):
    formatted = f"{name} (total: {len(items)}):\n"
    for item in items:
        formatted += f"- {item}\n"
    return formatted


def generate_constraint_categories(
    characteristics_list, constraints_list, prompt_template, model="gpt-4o-mini"
):
    characteristics_text = format_list_for_prompt(
        "Characteristics", characteristics_list
    )
    constraints_text = format_list_for_prompt("Constraints", constraints_list)

    prompt = prompt_template + "\n\n" + characteristics_text + "\n" + constraints_text

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = response.choices[0].message.content
        # Remove Markdown code fencing if present
        if content.startswith("```yaml"):
            content = content.replace("```yaml", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
        print(content)
        parsed = yaml.safe_load(content)
        return parsed
    except Exception as e:
        print(f" Error during OpenAI call: {e}")
        return {}
