# Overview Pytorch CNN
A modular library that allows you to perform rapid mulitclass classification provided a dataset

## Guide
### Image Dataset
Format your dataset like so:
    - data/datasetname:
        - test
            - classname_1
                - sample_1.jpg / png 
                - etc..
            - classname_2
                - sample_1.jpg
                - etc..
        - train
            - classname_1
                - sample_1.jpg / png 
                - etc..
            - classname_2
                - sample_1.jpg
                - etc..

### Text Dataset
Format dataset into csv file category column & text column
Symbol level tokenization is used as a default


Then add a config.yaml file under configs (sample config provided under configs) and run command ```python main.py train -m='config_file_name'```
Begin training using hyperparameters specified in configs folder with outputs in logs



1) Preprocess csv file
3) Clean data
2) Class balance
4) Create tokenizer from remaining dataset
3) Split csv into train test