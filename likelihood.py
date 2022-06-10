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
from models.progen.modeling_progen import ProGenForCausalLM



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


def sample(device, model, tokenizer, context, max_length, num_return_sequences, top_p, temp, pad_token_id):

    with torch.no_grad():
        input_ids = torch.tensor(tokenizer.encode(context).ids).view([1, -1]).to(device)
        tokens_batch = model.generate(input_ids, do_sample=True, temperature=temp, max_length=max_length, top_p=top_p, num_return_sequences=num_return_sequences, pad_token_id=pad_token_id)
        as_lists = lambda batch: [batch[i, ...].detach().cpu().numpy().tolist() for i in range(batch.shape[0])]
        return tokenizer.decode_batch(as_lists(tokens_batch))


########################################################################
# likelihood

def cross_entropy(logits, target, ignore_index, reduction='mean'):
    return torch.nn.functional.cross_entropy(input=logits, target=target, weight=None, size_average=None, ignore_index=ignore_index, reduce=None, reduction=reduction)


def cross_entropy_with_shift(logits, target, ignore_index, reduction):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_target = target[..., 1:].contiguous()
    return cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_target.view(-1), ignore_index=ignore_index, reduction=reduction)


def log_likelihood(logits, target, ignore_index, reduction='mean'):
    with torch.no_grad():
        return -cross_entropy_with_shift(logits=logits, target=target, ignore_index=ignore_index, reduction=reduction)


def log_likelihood_custom_1(logits, target, ignore_index, reduction='mean'):
    with torch.no_grad():
        return -torch.nn.functional.nll_loss(input=torch.log_softmax(logits, dim=1), target=target, reduction=reduction, ignore_index=ignore_index)


def log_likelihood_custom_2(logits, target, ignore_index, reduction='mean'):
    with torch.no_grad():
        assert len(target.shape) == 1
        assert logits.shape[0] == target.shape[0]
        log_likelihood = 0.0
        n = (logits != ignore_index).long().sum()
        for i in range(logits.shape[0]):
            if target[i] != ignore_index:
                log_likelihood += torch.log_softmax(logits, dim=1)[i, target[i]] / (1. if reduction == 'sum' else n)
        return log_likelihood


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
    parser.add_argument('--model', type=str, choices=models, default='2B-BFD30-Uniref90')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--fp16', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--context', type=str, default='1')
    args = parser.parse_args()


    # (2) preamble

    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)

    device = torch.device(args.device)
    ckpt = f'./checkpoints/{args.model}'


    # (3) load

    with print_time('loading parameters'):
        model = create_model(ckpt=ckpt, fp16=args.fp16).to(device)


    with print_time('loading tokenizer'):
        tokenizer = create_tokenizer_custom(file='tokenizer.json')
        pad_token_id = tokenizer.encode('<|pad|>').ids[0]


    # (4) log likelihood

    def ll(tokens, f=log_likelihood, ignore_index=pad_token_id, reduction='mean'):
        with torch.cuda.amp.autocast():
            input_ids = torch.tensor(tokenizer.encode(tokens).ids).to(device)
            logits = model(input_ids, labels=input_ids).logits
            return f(logits=logits, target=input_ids, ignore_index=ignore_index, reduction=reduction)


    # (5) sanity

    with print_time('sanity log-likelihood'):

        observation = '2PAQGRARLAAHYGTGRIGREVTVDERCRNLDRLEPSWELLRLLDDMGFIEGQNGLRRYVAEVFALDEPYDMTWRLRSLDEPHEVNAIEFAAPHERVYATLSERFFPDSVERDLRELVTRSLVEVDLGDPFTPPFVNSVYELRGASRRWVGVVRDVLAPDVLPCDATIRVLADAGTRAATRGLREILDTESGRVCVLGLHAALDAIADDRNEVSTSVAVADLEQCVALREAIRQITPRGAISVLVKGPLRTSGMRAQIAAVVHLRAKSSHLLPGGTDVVTFGAREFAIRSAANERKVVASMRLLALPGFAERSLCGLARPGVGRGRWEPAINVSVAADRDQIDLRVMGADVGDASVIFLKRDFRKLTEEFWRTHTDVPIEREDVSAQRTEPDNRWRWLVPCDDLVAPRLTVVPPRSVGHGM1'

        ll_0 = ll(observation, f=log_likelihood, reduction='mean')
        ll_1 = ll(observation, f=log_likelihood_custom_1, reduction='mean')
        ll_2 = ll(observation, f=log_likelihood_custom_2, reduction='mean')

        print(f'll_0={ll_0}')
        print(f'll_1={ll_1}')
        print(f'll_2={ll_2}')

        assert abs(ll_0 - ll_1) < 1e-2
        assert abs(ll_0 - ll_2) < 1e-2


    # (6) sanity

    with print_time('sanity model'):

        alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        observation_data = '2PAQGRARLAAHYGTGRIGREVTVDERCRNLDRLEPSWELLRLLDDMGFIEGQNGLRRYVAEVFALDEPYDMTWRLRSLDEPHEVNAIEFAAPHERVYATLSERFFPDSVERDLRELVTRSLVEVDLGDPFTPPFVNSVYELRGASRRWVGVVRDVLAPDVLPCDATIRVLADAGTRAATRGLREILDTESGRVCVLGLHAALDAIADDRNEVSTSVAVADLEQCVALREAIRQITPRGAISVLVKGPLRTSGMRAQIAAVVHLRAKSSHLLPGGTDVVTFGAREFAIRSAANERKVVASMRLLALPGFAERSLCGLARPGVGRGRWEPAINVSVAADRDQIDLRVMGADVGDASVIFLKRDFRKLTEEFWRTHTDVPIEREDVSAQRTEPDNRWRWLVPCDDLVAPRLTVVPPRSVGHGM1'
        observation_random = '2' + ''.join([random.choice(alphabet) for _ in range(len(observation_data)-2)]) + '1'
        observation_perturb = observation_random[:64] + observation_data[len(observation_random[:64]):]

        print(observation_data)
        print(observation_perturb)
        print(observation_random)

        assert observation_data != observation_perturb

        ll_observation_data = ll(observation_data)
        ll_observation_random = ll(observation_random)
        ll_observation_perturb = ll(observation_perturb)

        print(f'll_observation_data={ll_observation_data}')
        print(f'll_observation_random={ll_observation_random}')
        print(f'll_observation_perturb={ll_observation_perturb}')

        assert ll_observation_data > ll_observation_random
        assert ll_observation_data > ll_observation_perturb


    # (7) likelihood

    with print_time('log-likelihood'):

        ll_sum = ll(tokens=args.context, reduction='sum')
        ll_mean = ll(tokens=args.context, reduction='mean')

        print(f'll_sum={ll_sum}')
        print(f'll_mean={ll_mean}')


if __name__ == '__main__':
    main()
    print('done.')
