# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import os
import time
import random
import argparse

import torch

from tokenizers import Tokenizer
from transformers.models.progen.modeling_progen import ProGenForCausalLM



########################################################################
# util


class print_time:
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        print(self.desc)
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f'{self.desc} took {time.time()-self.t:.02f}s')


def set_env():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def set_seed(seed, deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic



########################################################################
# model


def create_model(ckpt, fp16=True):
    if fp16:
        return ProGenForCausalLM.from_pretrained(ckpt, revision='float16', torch_dtype=torch.float16, low_cpu_mem_usage=True)
    else:
        return ProGenForCausalLM.from_pretrained(ckpt)


def create_tokenizer_custom(file):
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())


########################################################################
# sample

def sample(
    device,
    model,
    tokenizer,
    context,
    pad_token_id,
    num_return_sequences=1,
    temp=0.2,
    top_p=0.95,
    max_length_sample=128,
    max_length=2048
):

    input_ids = tokenizer(
        context,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt',
    ).input_ids

    input_ids_len = input_ids.shape[1]
    assert input_ids_len < max_length

    with torch.no_grad():
        input_ids = input_ids.to(device)
        tokens = model.generate(
            input_ids,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            temperature=temp,
            max_length=input_ids_len + max_length_sample,
            top_p=top_p,
            pad_token_id=pad_token_id,
            use_cache=True,
        )
        text = tokenizer.batch_decode(tokens[:, input_ids_len:, ...])

    return text



########################################################################
# main


def main():

    # (0) constants

    models_151M = [ '151M-BFD30-Uniref90' ]
    models_754M = [ '754M-BFD30-Uniref90', '754M-OASu85', '754M-BFD30-Uniref90++' ]
    models_2B = [ '2B-BFD30-Uniref90', '2B-BFD90-Uniref90+' ]
    models_6B = [ '6B-BFD30-Uniref90', '6B-BFD30-Uniref90++' ]
    models = models_151M + models_754M + models_2B + models_6B


    # (1) params

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=models, default='754M-BFD30-Uniref90++')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--p', type=float, default=0.95)
    parser.add_argument('--t', type=float, default=0.2)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--fp16', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--context', type=str, default='1')
    args = parser.parse_args()


    # (2) preamble

    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)

    device = torch.device(args.device)
    if device.type == 'cpu':
        args.fp16 = False

    ckpt = f'./checkpoints/{args.model}'


    # (3) load

    with print_time('loading parameters'):
        model = create_model(ckpt=ckpt, fp16=args.fp16).to(device)


    with print_time('loading tokenizer'):
        tokenizer = create_tokenizer_custom(file='tokenizer.json')


    # (4) sample

    with print_time('sampling'):
        sample = sample(device=device, model=model, tokenizer=tokenizer, context=args.context, pad_token_id=args.pad, num_return_sequences=args.batch_size, temp=args.t, top_p=args.p, max_length_sample=args.max_length)[0]

        print('=' * 100)
        print(sample)
        print('=' * 100)
        print(args.context+sample)
        print('=' * 100)



if __name__ == '__main__':
    main()
    print('done.')
