from pathlib import Path
from typing import Union, List, Dict
import logging
import requests
from datetime import datetime
import json
import openai

class Client(object):
    def __init__(self,
                 model_name: str = "gpt-4-1106-preview-nlp",
                 api_key=DEFAULT_API_KEY,
                 stop = [')(*&^%$#@!)'],
                 url: str = API_URL,
                 ):
        self.api_key = api_key
        self.model_name = model_name
        self.url = url
        self.stop = stop
        # if logdir is None:
        #     logdir = ROOT / 'data/llm_logs'
        # self.logdir = Path(logdir)

    def __call__(self, *args, **kwargs):
        return self.complete(*args, **kwargs)

    def complete(self, messages: List[Dict],
                 content_only=True,
                 stream=False,
                 **kwargs) -> Union[List[str], Dict, requests.Response]:
        """包括拿到response之后的处理

        :param messages: List[Dict]
        :param content_only: bool, 是否只提取文本；
            if True，返回 List[str]，否则在非流式下返回Dict
        :param stream: bool, 是否流式输出；
            if True，返回生成器（Response），需要额外解码
        :param kwargs:
        :return:
        """

        resp = self._complete(messages, stream=stream, **kwargs)
        if stream:
            return resp  # setting ``stream=True`` gets the response (Generator)
        # resp = resp.json()
        if isinstance(resp, str):
            resp = json.loads(resp)
        if content_only:
            if 'choices' in resp:
                choices = resp['choices']
                return [x['message'].get('content', 'NULL') for x in choices]
            # return ['[RESPONSE ERROR]']
        return resp

    def _complete(self,
                  messages: List[Dict],
                  n=1,
                  max_tokens=3000,
                  temperature: float = 0.0,
                #   top_p: float = 1.0,
                  **kwargs) -> requests.Response:
        """只负责请求api，返回response，不做后处理

        :param messages:
        :param content_only:
        :param n: the number of candidate model generates
        :param max_tokens: max number tokens of the completion (prompt tokens are not included)
        :param kwargs:
        :return:
        """
        openai.api_key = self.api_key
        openai.api_base = self.url
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            n=n, 
            temperature=temperature,
            # top_p = top_p,
            max_tokens=max_tokens,
            stop = self.stop,
            **kwargs
        )
        return response


def __test__(response_only=False):
    system_msg = "You are a helpful Assistant."
    user_msg = "Tell me something you know that you think others don't."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    llm = Client(model_name='gpt-4-1106-preview-nlp')
    if response_only:
        resp = llm._complete(messages)
        logging.info(resp)
        # logging.info(resp.text)
        # logging.info(resp.json())
    else:
        res = llm.complete(messages, content_only=True)
        logging.info(res)

        # choices = res['choices']
        # res = [x['message'].get('content', 'NULL') for x in choices]
        # logging.info(res)
