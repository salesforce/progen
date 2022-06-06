# ProGen
Official release of the **ProGen** models (`764M`, `2.8B`, `6.4B`) for **Protein Synthesis**.


## Setup
```sh
git clone https://github.com/salesforce/progen
cd progen

# transfer relevant checkpoints
wget -P checkpoints https://storage.googleapis.com/sfr-codegen-research/checkpoints/codegen-350M-mono.tar.gz && tar -xvf checkpoints/codegen-350M-mono.tar.gz -C checkpoints/

# create a virtual environment with requirements
python3.8 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip setuptools
pip3 install -r requirements.txt

# sample from the model with an arbitrary context
python3 -m jaxformer.hf.sample --model codegen-350M-mono --context "def hello_world():"
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