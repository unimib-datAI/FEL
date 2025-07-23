FROM tensorflow/tensorflow:2.11.0-gpu-jupyter

COPY requirements.txt requirements.txt

RUN /usr/bin/python3 -m pip install --upgrade pip ipykernel

RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt

RUN apt-get update \
    && apt-get install wget python3.8-tk  -y \
    && wget -P /usr/local/lib/python3.8/dist-packages/aif360/data/raw/compas/ https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv \
    && wget -P /usr/local/lib/python3.8/dist-packages/aif360/data/raw/german/ https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data \
    && wget -P /usr/local/lib/python3.8/dist-packages/aif360/data/raw/german/ https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc \
    && wget -P /usr/local/lib/python3.8/dist-packages/aif360/data/raw/adult https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data \
    && wget -P /usr/local/lib/python3.8/dist-packages/aif360/data/raw/adult https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test \
    && wget -P /usr/local/lib/python3.8/dist-packages/aif360/data/raw/adult https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names 

RUN useradd -u 1000 -d /home/developer -m -k /etc/skel developer