import os
from dotenv import load_dotenv

import google.generativeai as genai

from src.utils.settings import ConstantSettings

load_dotenv()

api_key = os.getenv("GOOGLE_GEMINI_API_TOKEN")


def llm_response(query: str, context: str) -> str:
    prompt = ConstantSettings.RESPONSE_PROMPT.format(query, context)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash-002")
    response = model.generate_content(prompt)
    return response.text


if __name__ == '__main__':
    print(llm_response("greet me in nepali language", "my name is anishka"))
