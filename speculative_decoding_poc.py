import os
import time
import random
import torch
import torch.nn.functional as F
os.environ['RWKV_JIT_ON'] = '0'
os.environ['RWKV_CUDA_ON'] = '0'
current_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(f'{current_path}/rwkv_pip_package/src')
from rwkv.model import RWKV
from rwkv.utils import PIPELINE

ASSISTANT_MODEL_NAME = '/data/user/cangshui/tianchao/pth_models/RWKV-4-World-CHNtuned-0.1B-v1-20230617-ctx4096.pth'
assistant_model = RWKV(model=ASSISTANT_MODEL_NAME, strategy='cpu fp32', verbose=False)
MAIN_MODEL_NAME = '/data/user/cangshui/tianchao/pth_models/RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096.pth'
# MAIN_MODEL_NAME = '/data/user/cangshui/tianchao/pth_models/RWKV-4-World-CHNtuned-0.1B-v1-20230617-ctx4096.pth'
main_model = RWKV(model=MAIN_MODEL_NAME, strategy='cpu fp32', verbose=False)
pipeline = PIPELINE(assistant_model, "rwkv_vocab_v20230424")


assistant_states = []

input_text = '蔡徐坤拿出了他的篮球，'
input_ids = pipeline.encode(input_text)
if len(input_ids) > 1:
    _, assistant_state = assistant_model(input_ids[:-1], None)
    _, main_state = main_model(input_ids[:-1], None)
else:
    assistant_state = None
    main_state = None
result = input_text
all_history_ids = input_ids[:]
input_ids = input_ids[-1:]
print(input_text, end='', flush=True)

def color_red(text):
    return f'\033[91m{text}\033[0m'

def color_green(text):
    return f'\033[32m{text}\033[0m'

def color_yellow(text):
    return f'\033[34m{text}\033[0m'

start_time = time.time()
new_tokens = []
color_id = 0
while len(new_tokens) < 100:
    SPECULATIVE_LEN = 4
    assistant_probs = []
    speculative_ids = []
    main_model_input_ids = input_ids[:]
    # assistant_probs[0] means the probs of first completion (after reading previous tokens)
    # assistant_state[0] means the state after reading previous tokens
    for i in range(SPECULATIVE_LEN):
        # NOTE: state list will be modified in-place
        assistant_logits, assistant_state = assistant_model(input_ids, assistant_state[:] if assistant_state else None)
        assistant_probs.append(F.softmax(assistant_logits, dim=-1))
        assistant_states.append(assistant_state)
        token = pipeline.sample_logits(
            assistant_probs[-1],
            top_p=1,
            skip_softmax=True,
        )
        speculative_ids.append(token)
        main_model_input_ids.append(token)
        input_ids = [token]

    speculative_text = result + pipeline.decode(speculative_ids)

    main_logits, main_states = main_model(torch.tensor(main_model_input_ids), main_state, full_output=True)
    main_probs = F.softmax(main_logits, dim=-1)

    for i in range(SPECULATIVE_LEN):
        rand = random.random()
        candidate_id = speculative_ids[i]
        assistant_prob = assistant_probs[i][candidate_id]
        main_prob = main_probs[i][candidate_id]
        accept = rand <= main_prob / assistant_prob
        if not accept:
            accept_len = i
            break
    else:
        accept_len = SPECULATIVE_LEN
        _, assistant_state = assistant_model(input_ids, assistant_state)
        assistant_states.append(assistant_state)

    if accept_len < SPECULATIVE_LEN:
        # max(0, ..)
        probs = torch.relu(main_probs[accept_len] - assistant_probs[accept_len])
        probs /= probs.sum()
        token = pipeline.sample_logits(
            probs,
            top_p=1,
            skip_softmax=True,
        )
    else:
        token = pipeline.sample_logits(
            main_probs[accept_len],
            top_p=1,
            skip_softmax=True,
        )
    input_ids = [token]
    _new_tokens = speculative_ids[:accept_len] + [token]
    new_tokens.extend(_new_tokens)
    new_text = pipeline.decode(_new_tokens)
    result += new_text
    if color_id == 0:
        print(color_yellow(new_text), end='', flush=True)
    else:
        print(color_red(new_text), end='', flush=True)
    color_id = 1 - color_id
    all_history_ids.extend(_new_tokens)
    main_state = [s[accept_len] for s in main_states]
    assistant_state = assistant_states[accept_len]
    assistant_states.clear()

print('')
