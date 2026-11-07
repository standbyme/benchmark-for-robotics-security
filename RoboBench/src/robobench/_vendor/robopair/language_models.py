import openai
import time
from typing import Dict, List
from robobench.providers import ProviderContentFiltered, openai_client, openai_model


class LanguageModel:
    def __init__(self, model_name):
        self.model_name = openai_model(model_name)
    
    def batched_generate(self, prompts_list, max_n_tokens, temperature):
        """Generates responses for a batch of prompts using a language model."""
        raise NotImplementedError

class GPT(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20

    def __init__(self, model_name, client=None):
        self.model_name = openai_model(model_name)
        self.client = client

    def generate(self, conv: List[Dict], 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''

        output = self.API_ERROR_OUTPUT
        last_error = None
        client = self.client or openai_client()
        for _ in range(self.API_MAX_RETRY):
            try:
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=conv,
                    max_completion_tokens = max_n_tokens,
                    temperature = temperature,
                    top_p = top_p,
                    # request_timeout = self.API_TIMEOUT,
                )
                choice = completion.choices[0]
                if getattr(choice, "finish_reason", None) == "content_filter":
                    raise ProviderContentFiltered("Azure OpenAI filtered the response")
                output = choice.message.content
                break
            except ProviderContentFiltered:
                raise
            except (openai.BadRequestError, openai.AuthenticationError,
                    openai.NotFoundError, openai.PermissionDeniedError):
                raise
            except Exception as e:
                if getattr(e, "code", None) in {"insufficient_quota", "project_spend_limit_exceeded"}:
                    raise RuntimeError(f"RoboPAIR model {self.model_name}: provider quota or spend limit exhausted") from e
                last_error = e
                print(e)
        
            time.sleep(1)
        if output == self.API_ERROR_OUTPUT and last_error is not None:
            raise RuntimeError(f"RoboPAIR model {self.model_name} failed after retries") from last_error
        return output
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]
