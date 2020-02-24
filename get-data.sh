#!/usr/bin/env bash

mkdir -p data/anli/
mkdir -p data/anlg/
mkdir -p comet-model/
mkdir -p comet-vocab/

wget https://storage.googleapis.com/ai2-mosaic/public/abductive-commonsense-reasoning-iclr2020/anli.zip -P data/ -nc
wget https://storage.googleapis.com/ai2-mosaic/public/abductive-commonsense-reasoning-iclr2020/anlg.zip -P data/ -nc
unzip data/anli.zip -d data/
unzip data/anlg.zip -d data/
rm data/anli.zip
rm data/anlg.zip

wget https://storage.googleapis.com/ai2-mosaic/public/comet/atomic_pretrained_model.th -P comet-model/ -nc
wget https://storage.googleapis.com/ai2-mosaic/public/comet/vocabulary/encoder_bpe_40000.json -P comet-vocab/ -nc
wget https://storage.googleapis.com/ai2-mosaic/public/comet/vocabulary/vocab_40000.bpe -P comet-vocab/ -nc

wget https://storage.googleapis.com/ai2-mosaic/public/abductive-commonsense-reasoning-iclr2020/models.zip -P . -nc
unzip models.zip -d .
rm models.zip