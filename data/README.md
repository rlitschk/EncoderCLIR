The data directory is expected as follows:
````
data/
├── checkpoints
├── corpus
│         ├── clef
│         └── europarl
├── embedding_spaces
│         ├── cca
│         ├── icp
│         ├── mbert_aoc_layer_9
│         ├── mbert_iso_layer_0
│         ├── muse
│         ├── proc
│         ├── procb
│         ├── rcsls
│         ├── vecmap
│         ├── xlm_aoc_layer_12
│         ├── xlm_aoc_layer_15
│         └── xlm_iso_layer_1
├── gtranslated_clef_queries
│         ├── de_translated_to_fi.pickle
│         ├── de_translated_to_it.pickle
│         ├── de_translated_to_ru.pickle
│         ├── en_translated_to_de.pickle
│         ├── en_translated_to_fi.pickle
│         ├── en_translated_to_it.pickle
│         ├── en_translated_to_ru.pickle
│         ├── fi_translated_to_it.pickle
│         └── fi_translated_to_ru.pickle
└── ttest-references
          ├── clef-precision-values
          └── europarl-precision-values
````

- The folder `ttest-references` is optional.
- We do not provide [CLEF](**http://catalog.elra.info/en-us/repository/browse/ELRA-E0008/**) corpus (`data/clef`) or translated CLEF queries (`gtranslated_clef_queries/`)
  - The contents of `clef/` is expetec to match the structure shown [here](**https://github.com/rlitschk/UnsupCLIR**).
- All other resources can be downloaded [here](**https://madata.bib.uni-mannheim.de/360/**) and [here](**https://madata.bib.uni-mannheim.de/361/**).