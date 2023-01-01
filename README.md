# Overview Pytorch CNN
A modular library that allows you to perform rapid mulitclass classification provided a dataset

## Guide
### Image Dataset
- Format your dataset like so:
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
- Run command ```python main.py train -t=image -m=<config_file_name (without the .yaml extension)>```
- To test model with cifar10 dataset run with command ```python main.py train -t=text -m=cifar_config``` dataset will automaticall download and run


### Text Dataset
- Currently only supports using 1D CNNs with character level tokenization (ie: poor final testset accuracy)
- Setup: Format dataset into csv file category column & text column symbol level tokenization is used as a default word level tokenization coming soon.
- Then add a config.yaml file under configs (sample config provided under configs) and run command ```python main.py train -t='text' -m='<config_file_name> (without .yaml)'```


