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
  wget \
  vim 
 
RUN useradd --create-home --shell /bin/bash ${USER}
ENV HOME=/home/${USER}

ENV CONDA=${HOME}/miniconda3
RUN  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
      -O ${HOME}/install-miniconda.sh \
 && /bin/bash ${HOME}/install-miniconda.sh -b -p ${CONDA}

COPY . ${HOME}/rdt-reader
RUN chown -R ${USER}:${USER} ${HOME}/rdt-reader

USER ${USER}
WORKDIR ${HOME}
USER root

RUN echo ". ${CONDA}/etc/profile.d/conda.sh" >> ${HOME}/.bash_profile \
 && cat ${HOME}/.bash_profile

SHELL [ "/bin/bash", "--login", "-c" ]
WORKDIR ${HOME}/rdt-reader

#we use conda env create by reading a yml file. we can do this using a text file as well.
#please discuss if you want to use that instead.
#RUN conda create --name rdt-reader --file "${HOME}/rdt-reader/spec-file_linux_nogpu.txt" python=3.6
RUN conda env create --file "${HOME}/rdt-reader/rdtEnv.yml" python=3.6
CMD conda activate rdt-reader 





 
