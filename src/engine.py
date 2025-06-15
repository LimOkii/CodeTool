import re
import copy
import execnet


class PythonExecNet:

    def __init__(self, headers):
        self.headers = headers
    
    @staticmethod
    def judge(response):
        if response == None:
            return 0

        if type(response) == str:
            for e in ['Function executing from']:
                if e.lower() in response.lower():
                    return 0
        return 1

    def run(self, code):
        pattern = r"```python(.*?)```"
        matches = re.findall(pattern, code, re.DOTALL)
        tmp = copy.deepcopy(code) if matches == [] else matches[0]
        tmp = tmp.replace('```python', '').replace('[Python code]', '').replace('```','').replace('``','')
        tmp = '\n'.join(['\t' + e for e in tmp.split('\n')])
        exec_code = f"""import sys
from io import StringIO
import requests
import json

old_stdout = sys.stdout
redirected_output = StringIO()
sys.stdout = redirected_output

headers = {self.headers}

try:
{tmp}
except Exception as e:
    error = ' Exception Type: '+ type(e).__name__
    error += ', Exception Value: ' + str(e)
    channel.send((str(error), False))
else:
    sys.stdout = old_stdout
    output = redirected_output.getvalue()
    channel.send((output, True))
"""
        try:
            gw = execnet.makegateway()
            channel = gw.remote_exec(exec_code)
            result, state = channel.receive()
        except Exception as e:
            state = False
            result = e

        result = str(result).strip()

        if state == False:
            return result, 1
        if state == True and PythonExecNet.judge(result) == 0:
            return result, 2

        return result, 0


