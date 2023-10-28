## Requirements

Python 3.7

Pytorch 1.9.0


## Installation
You can replace the last command from the bottom to install
[pytorch](https://pytorch.org/get-started/previous-versions/) 
based on your CUDA version.
```
git clone https://github.com/Gait3D/CDGNet-Parsing.git
cd CDGNet-Parsing
conda create -n py37torch190 python=3.7
conda activate py37torch190
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
pip install tqdm opencv-python
```


## Download Model

[**model_best.pth**](https://xxx)


## Input Structure

Please put your dataset folder and make them follow this structure:

   ```
    |-- INPUT_PATH
       |-- name1.jpg
       |-- name2.jpg
       |-- ...
       |-- namex.jpg
        
   ```

## Configurations
In the **run_inference.sh**, you should change the following four parameters:

(1) Modify the input path
```
INPUT_PATH='/your/path/to/input'
```

(2) Modify the model path
```
SNAPSHOT_FROM='/your/path/to/model_best.pth'
```

(3) Modify the output path
```
OUTPUT_PATH='/your/path/to/output'
```

(4) Output the visual results (Optional)
```
VIS='yes'
```


## Usage
When you have finished the above configurations, run the following command:
```bash
sh run_inference.sh
```


## Output Structure
   ```
   |-- OUTPUT_PATH
      |--Pred_label_results
         |-- name1.jpg
         |-- name2.jpg
         |-- ...
         |-- namex.jpg

      |--Pred_parsing_results
         |-- name1.jpg
         |-- name2.jpg
         |-- ...
         |-- namex.jpg
      
   ```


## Acknowledge
[CDGNet](https://github.com/tjpulkl/CDGNet)





