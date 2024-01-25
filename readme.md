# The CATCR Model

The CATCR model comprises two components: CATCR-D and CATCR-G. CATCR-D is a discriminative model that predicts epitope-CDR3-beta pairs, while CATCR-G is a generative model designed to generate CDR3-beta sequences that bind to a given epitope.

## Environment Requirements:

- Python  3.9.18
- PyTorch  2.1.0
- CUDA 12.2

## Training and Testing Data

The train and test data used for this model can be downloaded from [https://pan.baidu.com/s/1q_yAdiEoH0bvPDXlbr66Pw?pwd=catr].
The pre-trained model is available at [https://pan.baidu.com/s/1jMYhVIV4M1rAq20HGgqAHw?pwd=catr].

The internal test data includes "seen" epitopes with "unseen" CDR3 sequences.
The external test data includes both "unseen" epitopes and CDR3 sequences.
The PDB data was generated by OpenAI (a PyTorch version of AlphaFold2).

## Data Preprocessing Tools

We provide a data processing tool to preprocess the sequence and PDB data in DataTransferTools.py.

- The PDB2PositionFrame module converts .pdb files into Pandas DataFrames.
- The RepDistanceAndSequenceMatrix module generates training data for a given sequence.
- Note: A segment-based coding method is recommended. The correspondence between peptide sequences and their codes is listed in node_index.csv.

## Training and Testing Procedures

- The CATCR model consists of three modules: the discriminator (CATCR-D), the residue contact matrix transformer (RCMT), and the generative model (CATCR-G).
- Train CATCR-D using SupTrain.py.
- Train the RCMT using StructureTransformer.py.
- Train CATCR-G using e2t_Generator.py.
- Note: in the training of RCMT and CATCR-G, a pretrained epitope encoder is recommonded. Our pretrained models were provided at [https://pan.baidu.com/s/1jMYhVIV4M1rAq20HGgqAHw?pwd=catr], or you can train the encoder from beginning.


## Demo

- We provided demo for CATCR-D and CATCR-G. The demo for CATCR-D is `D_test.py`, and demo from CATCR-G is `BeamSearch.py`
- To run the demo programmes, you can download the demodata from [https://pan.baidu.com/s/1nTzZLeauK9v0wHngg0NtgQ?pwd=catr], the pre-trained model from [https://pan.baidu.com/s/1jMYhVIV4M1rAq20HGgqAHw?pwd=catr], and replace the corresponding folders in the CTTCR folder.
