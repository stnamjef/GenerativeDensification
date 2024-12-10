FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y update --no-install-recommends \
	&& apt-get -y install --no-install-recommends \
    tmux \
    lsof \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
	&& apt-get autoremove -y \
	&& apt-get clean -y \
	&& export DEBIAN_FRONTEND=dialog

COPY ./requirements.txt ./
RUN echo "Installing pip packages..." \
	&& pip3 --no-cache-dir install -r ./requirements.txt \
	&& rm ./requirements.txt

RUN echo "Installing torch-scatter, torch_geometric..." \
    && pip3 --no-cache-dir install torch-scatter==2.1.2 torch-geometric==2.5.3  -f https://data.pyg.org/whl/torch-2.1.2+cu118.html

RUN echo "Installing xformers..." \
    && pip3 install xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu118

RUN echo "Installing flash-attention..." \
    && pip3 --no-cache-dir install flash-attn --no-build-isolation

ENTRYPOINT ["/bin/bash"]