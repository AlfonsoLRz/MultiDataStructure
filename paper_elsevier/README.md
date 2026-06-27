# Elsevier Paper Draft

This folder contains an Elsevier-style manuscript draft for the MultiDataStructure project.

## Files

- `main.tex` -- main `elsarticle` manuscript wrapper.
- `sections/introduction.tex` -- current introduction draft.
- `sections/methodology.tex` -- current methodology draft.
- `references.bib` -- starter bibliography, including the two style/reference papers supplied for this draft.

## Build

From this folder:

```powershell
latexmk -pdf main.tex
```

The manuscript expects the Elsevier `elsarticle` class and `elsarticle-num-names` bibliography style to be installed in the local TeX distribution. If MiKTeX prompts to install missing packages, allow `elsarticle`.

## Notes For The Next Pass

- Replace `Author Name`, affiliation, and target journal once known.
- Add the evaluation section only after deciding which result table should be the canonical one.
- Keep discovery and final-measurement claims separate; the methodology already makes that distinction.
- The supplied writing references were used for structure and tone: the Nature Communications NLOS paper for its model-to-objective exposition, and CuRast for its direct contribution/limitation style.
