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
git clone https://github.com/salesforce/progen
cd progen

# transfer checkpoint
wget -P checkpoints/6B-BFD30-Uniref90++ https://storage.googleapis.com/sfr-progen-research/checkpoints/6B-BFD30-Uniref90++.tar.gz && tar -xvf checkpoints/6B-BFD30-Uniref90++/6B-BFD30-Uniref90++.tar.gz -C checkpoints/6B-BFD30-Uniref90++/

# create virtual environment
python3.8 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip setuptools
pip3 install -r requirements.txt

# sample from the model
python3 sample.py --model 6B-BFD30-Uniref90++ --max-length 256 --context "1"
python3 sample.py --model 6B-BFD30-Uniref90++ --max-length 256 --context "1" --device "cpu" --fp16 false
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