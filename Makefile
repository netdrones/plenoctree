.ONESHELL:
SHELL=/bin/bash
ENV_NAME=plenoctree
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

install:
	conda env update -f environment.yml
	$(CONDA_ACTIVATE) ${ENV_NAME}
	pip install svox==0.2.28
	pip install --upgrade jax jaxlib==0.1.69+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html

house:
	if [ ! -d data ]; then mkdir data && gsutil -m cp -r gs://lucas.netdron.es/house data; fi
	python utils/generate_poses.py data/house
	python utils/data_loader.py data/house
	python -m nerf_sh.train \
	  --train_dir ckpts/house \
	  --config nerf_sh/config/tt \
	  --data_dir data/house
