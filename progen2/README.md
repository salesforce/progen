# ProGen2
Official release of the **ProGen2** models (`151M`, `764M`, `2.7B`, `6.4B`) for **Protein Engineering**.

## Models

| Model | Size | Checkpoint |
| ------ | ------ | ---------- |
| progen2-small	   | `151M` | https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-small.tar.gz |
| progen2-medium   | `764M` | https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-medium.tar.gz |
| progen2-oas	     | `764M` | https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-oas.tar.gz |
| progen2-base     | `764M` | https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-base.tar.gz |
| progen2-large    | `2.7B` |  https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-large.tar.gz |
| progen2-BFD90    | `2.7B` | https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-BFD90.tar.gz |
| progen2-xlarge   | `6.4B` | https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-xlarge.tar.gz |

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
python3 likelihood.py --model ${model} --context "1MGHGVSRPPVVTLRPAVLDDCPVLWRWRNDPETRQASVDEREIPVDTHTRWFEETLKRFDRKLFIVSADGVDAGMVRLDIQDRDAAVSVNIAPEWRGRGVGPRALGCLSREAFGPLALLRMSAVVKRENAASRIAFERAGFTVVDTGGPLLHSSKARLHVVAAIQARMGSTRLPGKVLVSIAGRPTIQRIAERLAVCQELDAVAVSTSVENRDDAIADLAAHLGLVCVRGSETDLIERLGRTAARTGADALVRITADCPLVDPALVDRVVGVWRRSAGRLEYVSNVFPPTFPDGLDVEVLSRTVLERLDREVSDPFFRESLTAYVREHPAAFEIANVEHPEDLSRLRWTMDYPEDLAFVEAVYRRLGNQGEIFGMDDLLRLLEWSPELRDLNRCREDVTVERGIRGTGYHAALRARGQAP2"
```

## Citation
If you find our code or paper useful, please cite:
```bibtex
@article{ProGen2,
  title={ProGen2: Exploring the Boundaries of Protein Language Models},
  author={Nijkamp, Erik and Ruffolo, Jeffrey and Weinstein, Eli N. and Naik, Nikhil and Madani, Ali},
  journal={arXiv},
  year={2022}
}
```

## License
Our code and models are BSD-3 licensed. See LICENSE.txt for details.

## Ethics
Predicting the fitness of a protein sequence and capturing the distribution of natural proteins for generative purposes could be a powerful tool for protein design. If our technique or a future iteration thereof is adopted broadly, care should be taken in terms of the end use-cases of these designed samples and downstream effects to ensure safe, non-nefarious, and ethical applications. For projects in any domain, active oversight during project initiation, experimental optimization, and deployment phases should be put in place to ensure safe usage and limitation of unintended harmful effects.
