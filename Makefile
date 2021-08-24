.ONESHELL:
SHELL=/bin/bash
ENV_NAME=plenoctree
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

install:
	conda env update -f environment.yml
	$(CONDA_ACTIVATE) ${ENV_NAME}
	pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
	pip install svox==0.2.28
	pip install --upgrade jax jaxlib==0.1.69+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html

house:
	if [ ! -d data/coffee ]
	then
		if [ ! -d data ]; then mkdir data; fi
		gsutil -m cp -r gs://lucas.netdron.es/coffee data
	fi
	sh +x bin/train.sh -i house

coffee:
	if [ ! -d data/coffee ]
	then
		if [ ! -d data ]; then mkdir data; fi
		gsutil -m cp -r gs://lucas.netdron.es/coffee data
	fi
	sh +x bin/train.sh -i coffee
