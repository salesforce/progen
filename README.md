# ProGen
Official release of the **ProGen** models (`151M`, `764M`, `2.7B`, `6.4B`) for **Protein Synthesis**.

## Models

| Model  | Checkpoint |
| ------------- | ------------- |
| 151M-BFD30-Uniref90	  |  |
| 754M-BFD30-Uniref90	  |  |
| 754M-OASu85	          |  |
| 754M-BFD30-Uniref90++	  |  |
| 2B-BFD30-Uniref90       |  |
| 2B-BFD90-Uniref90+      |  |
| 6B-BFD30-Uniref90	      |  |
| 6B-BFD30-Uniref90++     |  |

## Setup
```sh
git clone https://github.com/salesforce/progen
cd progen

# transfer relevant checkpoints
wget -P checkpoints https://storage.googleapis.com/sfr-progen-research/checkpoints/6B-BFD30-Uniref90++.tar.gz && tar -xvf checkpoints/6B-BFD30-Uniref90++.tar.gz -C checkpoints/

# create a virtual environment with requirements
python3.8 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip setuptools
pip3 install -r requirements.txt

# sample from the model with an arbitrary context
python3 sample.py --model 6B-BFD30-Uniref90++ --context "1"
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