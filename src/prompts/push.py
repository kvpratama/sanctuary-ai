"""CLI script to push core Langchain prompts to LangSmith.

Run this script directly via ``python -m src.prompts.push``
to upload or update the fallback prompts into the LangSmith hub.
"""

from dotenv import load_dotenv

from src.prompts.manager import push_eval_prompts

if __name__ == "__main__":
    load_dotenv()
    push_eval_prompts()
