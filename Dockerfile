FROM debian:latest
ENV USER=rdtreader

RUN apt-get update --fix-missing \
 && apt-get install --yes locales \
 && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
 && locale-gen \
 && apt-get install --yes \
  ca-certificates \
  curl \
  git git-lfs \
  libgl1-mesa-glx \
  wget

RUN useradd --create-home --shell /bin/bash ${USER}
ENV HOME=/home/${USER}
ENV CONDA=${HOME}/miniconda3

COPY . ${HOME}/rdt-reader
RUN chown -R ${USER}:${USER} ${HOME}

USER ${USER}
WORKDIR ${HOME}

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ${HOME}/install-miniconda.sh
RUN /bin/bash ${HOME}/install-miniconda.sh -b -p ${CONDA}

RUN echo ". ${CONDA}/etc/profile.d/conda.sh" >> ${HOME}/.bash_profile \
 && cat ${HOME}/.bash_profile

SHELL [ "/bin/bash", "--login", "-c" ]
WORKDIR ${HOME}/rdt-reader

RUN conda create --name rdt-reader --file "$HOME/rdt-reader/spec-file_linux.txt" python=3.6
RUN echo "Set up environment based on $HOME/rdt-reader/spec-file_linux.txt"

CMD conda activate rdt-reader \
 && python flasker.py
