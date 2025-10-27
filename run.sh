#/bin/bash
mkdir -p /root/.insightface/models
export INSIGHTFACE_ROOT=/root/.insightface/models
wget https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip -O /root/.insightface/models/antelopev2.zip

unzip /root/.insightface/models/antelopev2.zip -d /root/.insightface/models/

rm /root/.insightface/models/antelopev2.zip

python embedding-arcface.py
