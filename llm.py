from ollama import chat

class OllamaLLM:
    """
    Wrapper for calling a local Ollama model with streaming disabled by default.
    """
    def __init__(self, model_name="qwen2.5:3b-instruct", stream=False):
        self.model_name = model_name
        self.stream = stream

    def invoke(self, prompt: str) -> str:
        response = chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=self.stream,
        )

        if self.stream:
            text_out = ""
            for chunk in response:
                text_out += chunk.get("message", {}).get("content", "")
            return text_out
        else:
            return response["message"]["content"]
