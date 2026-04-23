# Data Pipeline for Converting Inference Anchoring Theory Annotations into Bipolar Argument Structures

This repository contains documentation on a project that forms part of a broader PhD research effort titled "Identifying the Stance of Argumentative Opinions in Political Discourse", conducted under the HYBRIDS Project within the Horizon Europe framework.

The primary contributor and point of contact for this repository is Siddharth Bhargava (sbhargava@fbk.eu).

---

## Overview

The project involves developing an end-to-end data processing pipeline for systematic adaptation of IAT-based (Inference Anchoring Theory) dialogical corpora into simplified bipolar argument structures (BAS), facilitating consistent benchmarking and computational modeling for Argument Structure Prediction (see [ArgStrPrediction](https://github.com/The-obsrvr/ArgStrPrediction) for more details.)

###### add Example

---

## Data Pipeline

We use [AIFdb](https://corpora.aifdb.org/), a large repository of dialogical argument mining corpora annotated under Inference Anchoring Theory (IAT) by different research teams, and represented in the Argument Interchange Format  (AIF).

For detailed documentation on how the IAT annotations are processed into Bipolar Argument Structures refer to our Data Pipeline repository here: [IAT-BAS-Data-Pipeline](https://github.com/The-obsrvr/IAT-BAS-Data-Pipeline). 

### Collection 

The data is sourced from the AIFdb.

### Restructuring IAT to BAS

Argument units are defined as the direct utterances made by a speaker that demonstrate a dialogical speech act defined under the IAT framework. These are marked as "L" (locution nodes) in the raw annotations. Supporting relations are defined between two inference nodes (I) that are directionally-related through an "inferential" (RA) node. Similarly, attacking relations are defined between two inference nodes (I) that are directionally-related through an "conflicting" (CA) nodes.

We only retain argument units that participate in some supporting or attacking relation (as a source or as a target). The list of the argument units and relations (support and attack) forms the bipolar argument structures. 

The code is extendable to also include the "rephrasing" nodes (MA), though they have not been considered within the scope of this work.

---

## Implementation

### Execution Command

The command is run in the docker environment as follows:
```bash
$ python src/dataprocessing.py 
```

---

### Exploratory Data Analysis: A Brief Report

Currently in jupyter notebook format in the src folder.

###### Add stats as image


---

## Acknowledgements

This research work has received funding from the European Union's Horizon Europe research and innovation programme under the Marie Skłodowska-Curie Grant Agreement No. 101073351. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or European Research Executive Agency (REA). Neither the European Union nor the granting authority can be held responsible for them.

---

## Citation 

tbd
