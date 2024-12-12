"""
LLM wrapper for LLMs running on ColossalCloud Platform

Usage:

os.environ['URL'] = ""
os.environ['HOST'] = ""

gen_config = {
        'max_new_tokens': 100,
    #     'top_k': 2,
        'top_p': 0.9,
        'temperature': 0.5,
        'repetition_penalty': 2,
    }

llm = ColossalCloudLLM(n=1)
llm.set_auth_config()
resp = llm(prompt='What do you call a three-ton kangaroo?', **gen_config)
print(resp)  # super-heavyweight awesome-natured yawning Australian creature!

"""

import json
from typing import Any, Mapping

import requests
from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env


class ColossalCloudLLM(LLM):
    """
    A custom LLM class that integrates LLMs running on the ColossalCloud Platform

    """

    n: int
    gen_config: dict = None
    auth_config: dict = None
    valid_gen_para: list = ["max_new_tokens", "top_k", "top_p", "temperature", "repetition_penalty"]

    def __init__(self, gen_config=None, **kwargs):
        """
        Args:
            gen_config: config for generation,
                max_new_tokens: 50 by default
                top_k: (1, vocab_size)
                top_p: (0, 1) if not None
                temperature: (0, inf) if not None
                repetition_penalty: (1, inf) if not None
        """
        super(ColossalCloudLLM, self).__init__(**kwargs)
        if gen_config is None:
            self.gen_config = {"max_new_tokens": 50}
        else:
            assert "max_new_tokens" in gen_config, "max_new_tokens is a compulsory key in the gen config"
            self.gen_config = gen_config

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}

    @property
    def _llm_type(self) -> str:
        return "ColossalCloudLLM"

    def set_auth_config(self, **kwargs):
        url = get_from_dict_or_env(kwargs, "url", "URL")
        host = get_from_dict_or_env(kwargs, "host", "HOST")

        auth_config = {}
        auth_config["endpoint"] = url
        auth_config["Host"] = host
        self.auth_config = auth_config

    def _call(self, prompt: str, stop=None, **kwargs: Any) -> str:
        """
        Args:
            prompt: The prompt to pass into the model.
            stop: A list of strings to stop generation when encountered

        Returns:
            The string generated by the model
        """
        # Update the generation arguments
        for key, value in kwargs.items():
            if key not in self.valid_gen_para:
                raise KeyError(
                    f"Invalid generation parameter: '{key}'. Valid keys are: {', '.join(self.valid_gen_para)}"
                )
            if key in self.gen_config:
                self.gen_config[key] = value

        resp_text = self.text_completion(prompt, self.gen_config, self.auth_config)
        # TODO: This may cause excessive tokens count
        if stop is not None:
            for stopping_words in stop:
                if stopping_words in resp_text:
                    resp_text = resp_text.split(stopping_words)[0]
        return resp_text

    def text_completion(self, prompt, gen_config, auth_config):
        # Required Parameters
        endpoint = auth_config.pop("endpoint")
        max_new_tokens = gen_config.pop("max_new_tokens")
        # Optional Parameters
        optional_params = ["top_k", "top_p", "temperature", "repetition_penalty"]  # Self.optional
        gen_config = {key: gen_config[key] for key in optional_params if key in gen_config}
        # Define the data payload
        data = {"max_new_tokens": max_new_tokens, "history": [{"instruction": prompt, "response": ""}], **gen_config}
        headers = {"Content-Type": "application/json", **auth_config}  # 'Host',
        # Make the POST request
        response = requests.post(endpoint, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # raise error if return code is not 200(success)
        # Check the response
        return response.text