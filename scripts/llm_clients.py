import os
import time
import logging
from typing import Tuple
from urllib import response

LOGGER = logging.getLogger("LLM-Client")

MAX_EDGE_TOKENS = 1024
TEMPERATURE = 0
TOP_P = 1


class LLMClient:
    def generate(self, prompt: str, instruction: str, multi_edge: bool) -> Tuple[int, int, str]:
        raise NotImplementedError


class OpenAIClient(LLMClient):
    def __init__(self, model_id: str):
        from openai import AzureOpenAI
        import os

        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2025-01-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

        self.model_id = model_id

    def generate(self, prompt: str, instruction: str, multi_edge: bool):
        stop_token = "\n\n" if multi_edge else "\n"

        is_gpt5 = "gpt-5" in self.model_id

        kwargs = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt},
            ],
        }

        if is_gpt5:
            kwargs["max_completion_tokens"] = MAX_EDGE_TOKENS
        else:
            kwargs["max_tokens"] = MAX_EDGE_TOKENS
            kwargs["temperature"] = TEMPERATURE
            kwargs["top_p"] = TOP_P
            kwargs["stop"] = stop_token

        result = self.client.chat.completions.create(**kwargs)

        total_tokens = result.usage.total_tokens
        completion_tokens = result.usage.completion_tokens
        completion_string = result.choices[0].message.content

        LOGGER.info(f"Using model {self.model_id}")

        return total_tokens, completion_tokens, completion_string
    
    
    
class GeminiClient(LLMClient):
    def __init__(self, model_id: str):
        from google import genai
        from google.genai import types
        
        self.types = types
        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        self.model_id = model_id

    def generate(self, prompt: str, instruction: str, multi_edge: bool):
        full_prompt = instruction + "\n\n" + prompt

        for attempt in range(10):
            try:
                is_pro_model = "pro" in self.model_id.lower()
                print("MODEL_ID DEBUG:", repr(self.model_id), "is_pro:", is_pro_model)
                if is_pro_model:
                    config = self.types.GenerateContentConfig(
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        max_output_tokens=MAX_EDGE_TOKENS,
                        thinking_config=self.types.ThinkingConfig(thinking_budget=512),
                    )
                else:
                    config = self.types.GenerateContentConfig(
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        max_output_tokens=MAX_EDGE_TOKENS,
                        thinking_config=self.types.ThinkingConfig(thinking_budget=0),
                    )
                    
                    
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=full_prompt,
                    config=config,
                )       
                    
                    



                #print("RAW GEMINI:", repr(response.text))
                #print("USAGE:", response.usage_metadata)
                
                
                completion_string = (response.text or "").strip()
                
                completion_string = completion_string.replace("```json", "")
                completion_string = completion_string.replace("```", "")
                completion_string = completion_string.strip()
                
                
                completion_string = " ".join(completion_string.split())
                
                if not completion_string.strip():
                    LOGGER.warning("Empty response, retrying...")
                    continue

                usage = getattr(response, "usage_metadata", None)
                total_tokens = getattr(usage, "total_token_count", 0) if usage else 0
                completion_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

                LOGGER.info(f"Using model {self.model_id}")
                return total_tokens or 0, completion_tokens or 0, completion_string

            except Exception as e:
                LOGGER.warning(f"Gemini attempt {attempt + 1} failed: {e}")
                if attempt == 9:
                    return 0, 0, ""
                time.sleep(10)

        return 0, 0, ""
    
    
class DeepSeekAzureClient(LLMClient):
        def __init__(self, model_id: str):
            from openai import OpenAI

            self.client = OpenAI(
                api_key=os.environ["DEEPSEEK_AZURE_API_KEY"],
                base_url=os.environ["DEEPSEEK_AZURE_ENDPOINT"],
            )
            self.model_id = model_id

        def generate(self, prompt: str, instruction: str, multi_edge: bool):
            stop_token = "\n\n" if multi_edge else "\n"


            result = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_EDGE_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                stop=stop_token,
            )

            total_tokens = result.usage.total_tokens if result.usage else 0
            completion_tokens = result.usage.completion_tokens if result.usage else 0
            completion_string = result.choices[0].message.content or ""

            LOGGER.info(f"Using model {self.model_id}")
            return total_tokens, completion_tokens, completion_string
        
        
        
        
class OpenRouterClient(LLMClient):
    def __init__(self, model_id: str):
        from openai import OpenAI
        import os

        self.client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
        self.model_id = model_id

    def generate(self, prompt: str, instruction: str, multi_edge: bool):
        stop_token = "\n\n" if multi_edge else "\n"

        result = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_EDGE_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            stop=stop_token,
        )

        total_tokens = result.usage.total_tokens if result.usage else 0
        completion_tokens = result.usage.completion_tokens if result.usage else 0
        completion_string = result.choices[0].message.content or ""

        LOGGER.info(f"Using OpenRouter model {self.model_id}")
        return total_tokens, completion_tokens, completion_string
    
    
class QwenClient(LLMClient):
    def __init__(self, model_id: str):
        from openai import OpenAI
        import os

        self.client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
        self.model_id = model_id

    def generate(self, prompt: str, instruction: str, multi_edge: bool):
        import time

        stop_token = "\n\n" if multi_edge else "\n"

        for attempt in range(5):
            try:
                result = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=180,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    stop=stop_token,
                )

                # Handle broken / empty responses
                if not result or not result.choices:
                    LOGGER.warning(f"Qwen attempt {attempt+1}: no choices returned, retrying...")
                    time.sleep(0.2)
                    continue

                total_tokens = result.usage.total_tokens if result.usage else 0
                completion_tokens = result.usage.completion_tokens if result.usage else 0

                completion_string = result.choices[0].message.content or ""

                # Handle empty output
                if not completion_string.strip():
                    LOGGER.warning(f"Qwen attempt {attempt+1}: empty output, retrying...")
                    time.sleep(0.2)
                    continue

                # Cleanup (same as other clients)
                completion_string = completion_string.replace("```json", "")
                completion_string = completion_string.replace("```", "")
                completion_string = completion_string.strip()
                completion_string = " ".join(completion_string.split())

                LOGGER.info(f"Using Qwen model {self.model_id}")
                return total_tokens, completion_tokens, completion_string

            except Exception as e:
                LOGGER.warning(f"Qwen attempt {attempt+1} failed: {e}")
                time.sleep(0.5)

        LOGGER.error("Qwen failed after all retries, returning empty result")
        return 0, 0, ""
    
    
    
    
    