.ONESHELL:
SHELL=/bin/bash
ENV_NAME=plenoctree
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

test:
	$(CONDA_ACTIVATE) plenoctree
	echo $$(which python)

install:
	conda env update -f environment.yml
	$(CONDA_ACTIVATE) ${ENV_NAME}
	pip install svox==0.2.28

house:
	python -m nerf_sh.train \
	  --train_dir ckpts/house \
	  --config nerf_sh/config/tt \
	  --data_dir data/house
