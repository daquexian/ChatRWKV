########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

print('\nHere are some demos for RWKV-4-World models (https://huggingface.co/BlinkDL/rwkv-4-world)\n')

import os, re

os.environ['RWKV_JIT_ON'] = '1' #### set these before import RWKV
os.environ["RWKV_CUDA_ON"] = '0' #### set to '1' to compile CUDA kernel (10x faster) - requires c++ compiler & cuda libraries

from rwkv.model import RWKV #### pip install rwkv --upgrade
from rwkv.utils import PIPELINE, PIPELINE_ARGS

MODEL_FILE = '/fsx/BlinkDL/HF-MODEL/rwkv-4-world/RWKV-4-World-7B-v1-20230626-ctx4096'

model = RWKV(model=MODEL_FILE, strategy='cuda fp16')
pipeline = PIPELINE(model, "rwkv_vocab_v20230424") #### vocab for rwkv-4-world models

########################################################################################################
#
#
#
print('\n#### Demo 1: free generation ####\n')
#
#
#
########################################################################################################

ctx = "Assistant: Sure! Here is a Python function to find Elon Musk's current location:"
print(ctx, end='')

def my_print(s):
    print(s, end='', flush=True)

args = PIPELINE_ARGS(temperature = 1.5, top_p = 0.3, top_k = 0, # top_k = 0 -> ignore top_k
                     alpha_frequency = 0.2, # frequency penalty - see https://platform.openai.com/docs/api-reference/parameter-details
                     alpha_presence = 0.2,  # presence penalty - see https://platform.openai.com/docs/api-reference/parameter-details
                     token_ban = [],        # ban the generation of some tokens
                     token_stop = [],       # stop generation at these tokens
                     chunk_len = 256)       # split input into chunks to save VRAM (shorter -> less VRAM, but slower)

pipeline.generate(ctx, token_count=200, args=args, callback=my_print)
print('\n')

########################################################################################################
#
#
#
print('\n#### Demo 2: single-round Q & A ####\n')
#
#
#
########################################################################################################

def my_qa_generator(ctx):
    out_tokens = []
    out_len = 0
    out_str = ''
    occurrence = {}
    state = None
    for i in range(999):

        if i == 0:
            out, state = pipeline.model.forward(pipeline.encode(ctx), state)
        else:
            out, state = pipeline.model.forward([token], state)

        for n in occurrence: out[n] -= (0.4 + occurrence[n] * 0.4) #### higher repetition penalty because of lower top_p here
        
        token = pipeline.sample_logits(out, temperature=1.0, top_p=0.2) #### sample the next token

        if token == 0: break #### exit at token [0] = <|endoftext|>
        
        out_tokens += [token]

        for n in occurrence: occurrence[n] *= 0.996 #### decay repetition penalty
        occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)
        
        tmp = pipeline.decode(out_tokens[out_len:])
        if ('\ufffd' not in tmp) and (not tmp.endswith('\n')): #### print() only when out_str is valid utf-8 and not end with \n
            out_str += tmp
            print(tmp, end = '', flush = True)
            out_len = i + 1    
        elif '\n\n' in tmp: #### exit at '\n\n'
            tmp = tmp.rstrip()
            out_str += tmp
            print(tmp, end = '', flush = True)
            break
    return out_str.strip()

for question in ['Why is there something instead of nothing?', '我捡到了一只小猫，该怎样喂它']:

    chat_rounds = ['User: hi',
    'Assistant: Hi. I am your assistant and I will provide expert full response in full details.',
    'User: ' + re.sub(r'\n{2,}', '\n', question).strip().replace('\r\n','\n'), #### replace all \n\n and \r\n by \n
    'Assistant:'] #### dont add space after this final ":"

    print('\n\n'.join(chat_rounds[-2:]), end = '')

    my_qa_generator('\n\n'.join(chat_rounds))
    print('\n' + '=' * 80)
