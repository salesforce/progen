# ProGen
Official release of the **ProGen** models (`151M`, `764M`, `2.7B`, `6.4B`) for **Protein Synthesis**.

## Models

| Model  | Checkpoint |
| ------ | ---------- |
| 151M-BFD30-Uniref90	   | https://storage.googleapis.com/sfr-progen-research/checkpoints/151M-BFD30-Uniref90.tar.gz |
| 754M-BFD30-Uniref90	   | https://storage.googleapis.com/sfr-progen-research/checkpoints/754M-BFD30-Uniref90.tar.gz |
| 754M-OASu85	           | https://storage.googleapis.com/sfr-progen-research/checkpoints/754M-OASu85.tar.gz |
| 754M-BFD30-Uniref90++	 | https://storage.googleapis.com/sfr-progen-research/checkpoints/754M-BFD30-Uniref90++.tar.gz |
| 2B-BFD30-Uniref90      | https://storage.googleapis.com/sfr-progen-research/checkpoints/2B-BFD30-Uniref90.tar.gz |
| 2B-BFD90-Uniref90+     | https://storage.googleapis.com/sfr-progen-research/checkpoints/2B-BFD90-Uniref90+.tar.gz |
| 6B-BFD30-Uniref90	     | https://storage.googleapis.com/sfr-progen-research/checkpoints/6B-BFD30-Uniref90.tar.gz |
| 6B-BFD30-Uniref90++    | https://storage.googleapis.com/sfr-progen-research/checkpoints/6B-BFD30-Uniref90++.tar.gz |

## Setup
```sh
# clone
git clone https://github.com/salesforce/progen
cd progen

# transfer
model=2B-BFD30-Uniref90
wget -P checkpoints/${model} https://storage.googleapis.com/sfr-progen-research/checkpoints/${model}.tar.gz
tar -xvf checkpoints/${model}/${model}.tar.gz -C checkpoints/${model}/

# venv
python3.8 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip setuptools
pip3 install -r requirements.txt

# sample
python3 sample.py --model ${model} --t 0.8 --p 0.9 --max-length 1024 --context "1"
python3 sample.py --model ${model} --t 0.8 --p 0.9 --max-length 1024 --context "1" --device "cpu" --fp16 false

# log-likelihood
python3 likelihood.py --model ${model} --context "2PAQGRARLAAHYGTGRIGREVTVDERCRNLDRLEPSWELLRLLDDMGFIEGQNGLRRYVAEVFALDEPYDMTWRLRSLDEPHEVNAIEFAAPHERVYATLSERFFPDSVERDLRELVTRSLVEVDLGDPFTPPFVNSVYELRGASRRWVGVVRDVLAPDVLPCDATIRVLADAGTRAATRGLREILDTESGRVCVLGLHAALDAIADDRNEVSTSVAVADLEQCVALREAIRQITPRGAISVLVKGPLRTSGMRAQIAAVVHLRAKSSHLLPGGTDVVTFGAREFAIRSAANERKVVASMRLLALPGFAERSLCGLARPGVGRGRWEPAINVSVAADRDQIDLRVMGADVGDASVIFLKRDFRKLTEEFWRTHTDVPIEREDVSAQRTEPDNRWRWLVPCDDLVAPRLTVVPPRSVGHGM1" --device "cpu" --fp16 false
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
Our code is BSD-3 licensed. See LICENSE.txt for details.