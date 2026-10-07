import torch
from openai import OpenAI

class LocalGenerator:

    def __init__(self, model_name: str = "liquid/lfm2.5-1.2b", base_url: str = "http://localhost:1234/v1"):
        self.model_name = model_name
        self.base_url = base_url

        self.client = OpenAI(base_url=self.base_url, api_key="not-needed")

    def generate(self, context: str, query: str) -> str:

        prompt = f"""Answer the following question based on the provided context.
If the answer is not present in the context, respond with "I don't know."

--- CONTEXT ---
{context}
-----------------

Question: {query}
Answer:"""
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": query}
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip() or "No response generated."