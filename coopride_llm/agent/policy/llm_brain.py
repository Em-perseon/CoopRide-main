"""
LLM Brain for CoopRide Weight Optimization

Multi-provider LLM client (OpenAI / Zhipu AI) with Jinja2 template rendering.
Interface aligned with llm4jssp's LLMBrain.
"""

import os
import re
import json
import time
import traceback

import jinja2


class LLMBrain:
    """
    LLM interaction layer for weight optimization.

    Supports:
    - OpenAI API (GPT-4o-mini, etc.)
    - Zhipu AI API (GLM-4.7, etc.)
    - Jinja2 template rendering for prompt construction
    - JSON weight parsing from LLM responses
    """

    def __init__(self, template_path, model_name, api_key=None, base_url=None,
                 temperature=0.7, max_tokens=12800):
        """
        Parameters
        ----------
        template_path : str
            Path to Jinja2 template file.
        model_name : str
            LLM model name (e.g., 'gpt-4o-mini', 'glm-4.7').
        api_key : str or None
            API key. If None, reads from environment variable.
        base_url : str or None
            API base URL. If None, uses default.
        temperature : float
            Sampling temperature.
        max_tokens : int
            Maximum response tokens.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Determine provider from model name
        if model_name.startswith('glm'):
            self.provider = 'zhipu'
        else:
            self.provider = 'openai'

        # Load Jinja2 template
        template_dir = os.path.dirname(os.path.abspath(template_path))
        template_name = os.path.basename(template_path)
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            undefined=jinja2.StrictUndefined
        )
        self.template = self.jinja_env.get_template(template_name)

    def render_prompt(self, **kwargs):
        """Render Jinja2 template with given variables."""
        return self.template.render(**kwargs)

    def call_llm(self, prompt, system_prompt=None):
        """
        Call LLM API and return raw response text.

        Parameters
        ----------
        prompt : str
            User prompt text.
        system_prompt : str or None
            System prompt. If None, uses default.

        Returns
        -------
        dict : {response: str, api_time: float, success: bool}
        """
        if system_prompt is None:
            system_prompt = (
                "You are a professional ride-hailing dispatch system optimization expert, "
                "skilled in analyzing and optimizing dispatch weight parameters through "
                "iterative exploration and exploitation."
            )

        start_time = time.time()

        if self.provider == 'openai':
            result = self._call_openai(prompt, system_prompt)
        elif self.provider == 'zhipu':
            result = self._call_zhipu(prompt, system_prompt)
        else:
            result = {'response': 'Unknown provider', 'success': False}

        result['api_time'] = time.time() - start_time
        return result

    def _call_openai(self, prompt, system_prompt):
        """Call OpenAI-compatible API."""
        try:
            from openai import OpenAI

            api_key = self.api_key or os.environ.get('OPENAI_API_KEY', '')
            base_url = self.base_url or os.environ.get('OPENAI_BASE_URL', None)

            client_kwargs = {'api_key': api_key}
            if base_url:
                client_kwargs['base_url'] = base_url

            client = OpenAI(**client_kwargs)

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return {
                'response': response.choices[0].message.content,
                'success': True
            }
        except Exception as e:
            error_msg = "{}: {}".format(type(e).__name__, str(e))
            print("[ERROR] OpenAI API call failed: {}".format(error_msg))
            traceback.print_exc()
            return {'response': error_msg, 'success': False}

    def _call_zhipu(self, prompt, system_prompt):
        """Call Zhipu AI API."""
        try:
            from zai import ZhipuAiClient

            api_key = self.api_key or os.environ.get('ZHIPU_API_KEY', '')
            client = ZhipuAiClient(api_key=api_key)

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )

            return {
                'response': response.choices[0].message.content,
                'success': True
            }
        except Exception as e:
            error_msg = "{}: {}".format(type(e).__name__, str(e))
            print("[ERROR] Zhipu API call failed: {}".format(error_msg))
            traceback.print_exc()
            return {'response': error_msg, 'success': False}

    def parse_weights(self, response_text, num_params=10, weight_range=(-5.0, 5.0)):
        """
        Parse suggested weights from LLM response.

        Looks for JSON format: {"suggested_weights": [w0, w1, ..., w9]}

        Parameters
        ----------
        response_text : str
        num_params : int
        weight_range : tuple (min, max)

        Returns
        -------
        list or None : parsed weight list, or None if parsing fails
        """
        try:
            if "suggested_weights" not in response_text:
                return None

            json_pattern = r'\{[^}]*"suggested_weights"\s*:\s*\[[^\]]+\][^}]*\}'
            json_match = re.search(json_pattern, response_text, re.DOTALL)

            if not json_match:
                return None

            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            weights = parsed["suggested_weights"]

            if len(weights) != num_params:
                print("[WARNING] Expected {} weights, got {}".format(num_params, len(weights)))
                return None

            # Clamp to valid range
            low, high = weight_range
            weights = [max(low, min(high, float(w))) for w in weights]
            return weights

        except Exception as e:
            print("[WARNING] Failed to parse weights: {}".format(e))
            return None

    def parse_params_line(self, response_text, num_params=10, weight_range=(-5.0, 5.0)):
        """
        Parse 'params[0]: val, params[1]: val, ...' format (llm4jssp compatible).

        Parameters
        ----------
        response_text : str
        num_params : int
        weight_range : tuple (min, max)

        Returns
        -------
        list or None
        """
        try:
            pattern = r'params\[\d+\]\s*:\s*([-+]?\d*\.?\d+)'
            matches = re.findall(pattern, response_text)
            if len(matches) == num_params:
                low, high = weight_range
                return [max(low, min(high, float(v))) for v in matches]
        except Exception:
            pass
        return None
