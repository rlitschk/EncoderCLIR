# Evaluating Multilingual Text Encoders for Unsupervised CLIR
Robert Litschko, Ivan Vulić, Simone Paolo Ponzetto, Goran Glavaš. [Evaluating Multilingual Text Encoders for Unsupervised CLIR](**https://arxiv.org/abs/2101.08370**). arXiv preprint arXiv:2101.08370

## Installation instructions

We recommend installing the dependencies with Anaconda as follows:

````
conda create --name clir python=3.7
conda activate clir
conda install faiss-gpu cudatoolkit=10.0 -c pytorch

pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```` 

*(Optional) Install [LASER](**https://github.com/facebookresearch/LASER**) and set the `LASER` environment variable, then set the `LASER_EMB` variable in [`config.py`]((../src/config.py)). You need to manually adjust the sequence length in `LASER_HOME/source/embed.py`.*    

## Resources

- We run our experiments on [CLEF](**http://catalog.elra.info/en-us/repository/browse/ELRA-E0008/**) for document-level retrieval and samples of [Europarl](**https://madata.bib.uni-mannheim.de/360/9/europarl.tar.gz**) for sentence-level retrieval. Download and extract the files into `PROJECT_ROOT/data/corpus`.
- Download the following embeddings and place them into `PROJECT_ROOT/data/embedding_spaces`:

 Model | XLM | mBERT  
--- | --- | --- 
 ISO | [L1](**https://madata.bib.uni-mannheim.de/361/3/xlm_iso_layer_1.tar.gz**) | [L0](**https://madata.bib.uni-mannheim.de/361/1/mbert_iso_layer_0.tar.gz**)
 AOC | [L12](**https://madata.bib.uni-mannheim.de/361/4/xlm_aoc_layer_12.tar.gz**), [L15](**https://madata.bib.uni-mannheim.de/361/5/xlm_aoc_layer_15.tar.gz**) | [L9](**https://madata.bib.uni-mannheim.de/361/2/mbert_aoc_layer_9.tar.gz**)

- We further make the following cross-lingual word embedding spaces available:

Type | CLWE Space  
--- | --- 
Supervised | [CCA](**https://madata.bib.uni-mannheim.de/360/3/cca.tar.gz**), [proc](**https://madata.bib.uni-mannheim.de/360/2/proc.tar.gz**), [procB](**https://madata.bib.uni-mannheim.de/360/7/procb.tar.gz*), [RCSLS](**https://madata.bib.uni-mannheim.de/360/6/rcsls.tar.gz**)
Unsupervised | [VecMap](**https://madata.bib.uni-mannheim.de/360/8/vecmap.tar.gz**), [Muse](**https://madata.bib.uni-mannheim.de/360/1/muse.tar.gz**), [ICP](**https://madata.bib.uni-mannheim.de/360/4/icp.tar.gz**)

- Optional: Download model checkpoints [here](**https://madata.bib.uni-mannheim.de/361/6/checkpoints.tar.gz**) and extract into `PROJECT_ROOT/data/checkpoints`.
- Optional: Download reference files for significance tests [here](****) and extract into `PROJECT_ROOT/data/ttest-references`.


## Example usage

We provide the following two scripts for running multilingual encoder experiments and cross-lingual word embedding experiments respectively:
```bash
python run_CLWE_exp.py     # Run baseline/CLWE/AOC/ISO experiments
  --dataset                # One of: europarl,clef
  --emb_spaces             # Zero or more: cca,proc,procb,rcsls,icp,muse,vecmap,xlm_aoc,mbert_aoc,xlm_iso,mbert_iso
  --retrieval_models       # One or more: IDF-SUM,TbT-QT
  --baselines              # Zero or more: unigram,gtranslate
  --lang_pairs             # One or more: enfi,enit,ende,enru,defi,deit,deru,fiit,firu

python run_ENC_exp.py     # Run SEMB or similarity-specialized sentence encoder experiments
  --processes             # Number of processes (for parallel idf calculations), e.g. 10
  --gpu                   # Cuda device
  --name                  # Abitrary experiment name, results stored in PROJECT_HOME/results/{name}/
  --dataset               # One of: europarl, clef
  --encoder               # One of: mbert,xlm,laser,labse,muse,distil_mbert,distil_xlmr,distil_muse
  --lang_pairs            # One or more: enfi,enit,enru,ende,defi,deit,deru,fiit,firu
  --maxlen                # 1 <= max_sequence_length =< 512
```

## Citing
If you use this repository please consider citing our paper: 
```bibtex
@inproceedings{litschko2021encoderclir,
 author = {Litschko, Robert and Vuli{\'c}, Ivan, and Ponzetto, Simone Paolo, and Glava{\v{s}}, Goran},
 booktitle = {Proceedings of ECIR},
 title = {Evaluating Multilingual Text Encoders for Unsupervised Cross-Lingual Retrieval},
 year = {2021}
}
```