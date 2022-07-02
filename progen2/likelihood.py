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

def cross_entropy(logits, target, reduction='mean'):
    return torch.nn.functional.cross_entropy(input=logits, target=target, weight=None, size_average=None, reduce=None, reduction=reduction)


def log_likelihood(logits, target, reduction='mean'):
    return -cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), reduction=reduction)


def log_likelihood_custom_1(logits, target, reduction='mean'):
    return -torch.nn.functional.nll_loss(input=torch.log_softmax(logits, dim=1), target=target, reduction=reduction)


def log_likelihood_custom_2(logits, target, reduction='mean'):
    assert len(target.shape) == 1
    assert logits.shape[0] == target.shape[0]

    log_likelihood = 0.0
    n = logits.shape[0]
    for i in range(n):
        log_likelihood += torch.log_softmax(logits, dim=1)[i, target[i]] / (1. if reduction == 'sum' else n)
    return log_likelihood


########################################################################
# main


def main():

    # (0) constants

    models_151M = [ 'progen2-small' ]
    models_754M = [ 'progen2-medium', 'progen2-oas', 'progen2-base' ]
    models_2B = [ 'progen2-large', 'progen2-BFD90' ]
    models_6B = [ 'progen2-xlarge' ]
    models = models_151M + models_754M + models_2B + models_6B


    # (1) params

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=models, default='progen2-large')
    # parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--fp16', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--context', type=str, default='1MGHGVSRPPVVTLRPAVLDDCPVLWRWRNDPETRQASVDEREIPVDTHTRWFEETLKRFDRKLFIVSADGVDAGMVRLDIQDRDAAVSVNIAPEWRGRGVGPRALGCLSREAFGPLALLRMSAVVKRENAASRIAFERAGFTVVDTGGPLLHSSKARLHVVAAIQARMGSTRLPGKVLVSIAGRPTIQRIAERLAVCQELDAVAVSTSVENRDDAIADLAAHLGLVCVRGSETDLIERLGRTAARTGADALVRITADCPLVDPALVDRVVGVWRRSAGRLEYVSNVFPPTFPDGLDVEVLSRTVLERLDREVSDPFFRESLTAYVREHPAAFEIANVEHPEDLSRLRWTMDYPEDLAFVEAVYRRLGNQGEIFGMDDLLRLLEWSPELRDLNRCREDVTVERGIRGTGYHAALRARGQAP2')
    parser.add_argument('--sanity', default=True, type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()


    # (2) preamble

    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)

    device = torch.device(args.device)
    ckpt = f'./checkpoints/{args.model}'

    if device.type == 'cpu':
        args.fp16 = False


    # (3) load

    with print_time('loading parameters'):
        model = create_model(ckpt=ckpt, fp16=args.fp16).to(device)


    with print_time('loading tokenizer'):
        tokenizer = create_tokenizer_custom(file='tokenizer.json')


    # (4) log likelihood

    def ll(tokens, f=log_likelihood, reduction='mean'):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                target = torch.tensor(tokenizer.encode(tokens).ids).to(device)
                logits = model(target, labels=target).logits

                # shift
                logits = logits[:-1, ...]
                target = target[1:]

                # remove terminals
                bos_token, eos_token = 3, 4
                if target[-1] in [bos_token, eos_token]:
                    logits = logits[:-1, ...]
                    target = target[:-1]

                assert (target == bos_token).sum() == 0
                assert (target == eos_token).sum() == 0

                # remove unused logits
                first_token, last_token = 5, 29
                logits = logits[:, first_token:(last_token+1)]
                target = target - first_token

                assert logits.shape[1] == (last_token - first_token + 1)
 
                return f(logits=logits, target=target, reduction=reduction).item()



    if args.sanity:

        # (5) sanity

        with print_time('sanity log-likelihood'):

            x_data = '2PAQGRARLAAHYGTGRIGREVTVDERCRNLDRLEPSWELLRLLDDMGFIEGQNGLRRYVAEVFALDEPYDMTWRLRSLDEPHEVNAIEFAAPHERVYATLSERFFPDSVERDLRELVTRSLVEVDLGDPFTPPFVNSVYELRGASRRWVGVVRDVLAPDVLPCDATIRVLADAGTRAATRGLREILDTESGRVCVLGLHAALDAIADDRNEVSTSVAVADLEQCVALREAIRQITPRGAISVLVKGPLRTSGMRAQIAAVVHLRAKSSHLLPGGTDVVTFGAREFAIRSAANERKVVASMRLLALPGFAERSLCGLARPGVGRGRWEPAINVSVAADRDQIDLRVMGADVGDASVIFLKRDFRKLTEEFWRTHTDVPIEREDVSAQRTEPDNRWRWLVPCDDLVAPRLTVVPPRSVGHGM1'

            ll_0 = ll(x_data, f=log_likelihood, reduction='mean')
            ll_1 = ll(x_data, f=log_likelihood_custom_1, reduction='mean')
            ll_2 = ll(x_data, f=log_likelihood_custom_2, reduction='mean')

            print(f'll_0={ll_0}')
            print(f'll_1={ll_1}')
            print(f'll_2={ll_2}')

            assert abs(ll_0 - ll_1) < 1e-2
            assert abs(ll_0 - ll_2) < 1e-2


        # (6) sanity

        with print_time('sanity model'):

            alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
            x_data = '2PAQGRARLAAHYGTGRIGREVTVDERCRNLDRLEPSWELLRLLDDMGFIEGQNGLRRYVAEVFALDEPYDMTWRLRSLDEPHEVNAIEFAAPHERVYATLSERFFPDSVERDLRELVTRSLVEVDLGDPFTPPFVNSVYELRGASRRWVGVVRDVLAPDVLPCDATIRVLADAGTRAATRGLREILDTESGRVCVLGLHAALDAIADDRNEVSTSVAVADLEQCVALREAIRQITPRGAISVLVKGPLRTSGMRAQIAAVVHLRAKSSHLLPGGTDVVTFGAREFAIRSAANERKVVASMRLLALPGFAERSLCGLARPGVGRGRWEPAINVSVAADRDQIDLRVMGADVGDASVIFLKRDFRKLTEEFWRTHTDVPIEREDVSAQRTEPDNRWRWLVPCDDLVAPRLTVVPPRSVGHGM1'
            x_random = '2' + ''.join([random.choice(alphabet) for _ in range(len(x_data)-2)]) + '1'
            x_perturb = x_random[:64] + x_data[len(x_random[:64]):]

            print(x_data)
            print(x_perturb)
            print(x_random)

            assert x_data != x_perturb

            ll_x_data = ll(x_data)
            ll_x_random = ll(x_random)
            ll_x_perturb = ll(x_perturb)

            print(f'll_x_data={ll_x_data}')
            print(f'll_x_random={ll_x_random}')
            print(f'll_x_perturb={ll_x_perturb}')

            assert ll_x_data > ll_x_random
            assert ll_x_data > ll_x_perturb


    # (7) likelihood

    with print_time('log-likelihood (left-to-right)'):

        ll_sum = ll(tokens=args.context, reduction='sum')
        ll_mean = ll(tokens=args.context, reduction='mean')

        print(f'll_sum={ll_sum}')
        print(f'll_mean={ll_mean}')


    # (8) likelihood

    with print_time('log-likelihood (left-to-right, right-to-left)'):

        reverse = lambda s: s[::-1]

        ll_lr_sum = ll(tokens=args.context, reduction='sum')
        ll_rl_sum = ll(tokens=reverse(args.context), reduction='sum')

        ll_lr_mean = ll(tokens=args.context, reduction='mean')
        ll_rl_mean = ll(tokens=reverse(args.context), reduction='mean')

        ll_sum = .5 * (ll_lr_sum + ll_rl_sum)
        ll_mean = .5 * (ll_lr_mean + ll_rl_mean)

        print(f'll_sum={(ll_sum)}')
        print(f'll_mean={ll_mean}')



if __name__ == '__main__':
    main()
    print('done.')
