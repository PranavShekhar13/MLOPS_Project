FROM centos

RUN yum install sudo

RUN yum install wget

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda.sh

RUN bash /opt/miniconda.sh -b -p /opt/conda
RUN rm -rf /opt/miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

RUN conda install python3

RUN conda install tensorflow 

RUN conda install numpy 

RUN conda install keras 

RUN conda install matplotlib 

RUN conda install pillow 

RUN conda install opencv-python 

RUN conda install pandas 

RUN mkdir /model_test

WORKDIR /model_test

COPY / /model_test

CMD ["python", "code.py"]
