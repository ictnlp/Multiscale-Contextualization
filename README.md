# Multiscale-Contextualization
## Description
Codes and scripts for ACL2024 Findings paper "Integrating Multi-scale Contextualized Information for Byte-based Neural Machine Translation".

## Changes to fairseq files
 - add the file `multiscale_transformer.py` to fairseq/models
 - add the file `multiscale_attention.py` to fairseq/modules
 - add the file `transformer_multiscale_layer.py` to fairseq/modules
 - change fairseq/modules/__init__.py to import the added packages

## How to use it?
Add pathes to the scripts in the `scripts` folder, and run them.
