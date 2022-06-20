# ProGen2
Official release of the **ProGen2** models (`151M`, `764M`, `2.7B`, `6.4B`) for **Protein Engineering**.

## Models

| Model  | Checkpoint |
| ------ | ---------- |
| progen2-small	   | https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-small.tar.gz |
| progen2-medium   | https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-medium.tar.gz |
| progen2-oas	     | https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-oas.tar.gz |
| progen2-base     | https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-base.tar.gz |
| progen2-large    | https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-large.tar.gz |
| progen2-BFD90    | https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-BFD90.tar.gz |
| progen2-xlarge   | https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-xlarge.tar.gz |

## Setup
```sh
# code
git clone https://github.com/salesforce/progen
cd progen/progen2

# checkpoint
model=progen2-large
wget -P checkpoints/${model} https://storage.googleapis.com/sfr-progen-research/checkpoints/${model}.tar.gz
tar -xvf checkpoints/${model}/${model}.tar.gz -C checkpoints/${model}/

# venv
python3.8 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip setuptools
pip3 install -r requirements.txt

# sample
python3 sample.py --model ${model} --t 0.8 --p 0.9 --max-length 1024 --num-samples 2 --context "1" 
python3 sample.py --model ${model} --t 0.8 --p 0.9 --max-length 1024 --num-samples 2 --context "1" --device "cpu" --fp16 false

# log-likelihood
python3 likelihood.py --model ${model} --context "2PAQGRARLAAHYGTGRIGREVTVDERCRNLDRLEPSWELLRLLDDMGFIEGQNGLRRYVAEVFALDEPYDMTWRLRSLDEPHEVNAIEFAAPHERVYATLSERFFPDSVERDLRELVTRSLVEVDLGDPFTPPFVNSVYELRGASRRWVGVVRDVLAPDVLPCDATIRVLADAGTRAATRGLREILDTESGRVCVLGLHAALDAIADDRNEVSTSVAVADLEQCVALREAIRQITPRGAISVLVKGPLRTSGMRAQIAAVVHLRAKSSHLLPGGTDVVTFGAREFAIRSAANERKVVASMRLLALPGFAERSLCGLARPGVGRGRWEPAINVSVAADRDQIDLRVMGADVGDASVIFLKRDFRKLTEEFWRTHTDVPIEREDVSAQRTEPDNRWRWLVPCDDLVAPRLTVVPPRSVGHGM1"
```

## Citation
If you find our code or paper useful, please cite:
```bibtex
@article{,
  title={},
  author={},
  journal={arXiv preprint},
  year={2022}
}
```

## License
Our code and models are BSD-3 licensed. See LICENSE.txt for details.
