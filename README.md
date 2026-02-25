SD3_Reward
## Data Prepare
 ### download HPDv3

 ### Download ImageRewardDB
 ```bash
mkdir  ImageRewardDB
hf download zai-org/ImageRewardDB --repo-type dataset --local-dir /zjk_nas/zhiyi/data/ImageRewardDB
seq 1 32 | xargs -I {} unzip ImageRewardDB/images/train/train_{}.zip -d ImageRewardDB/images/train/train_{}
seq 1 2 | xargs -I {} unzip ImageRewardDB/images/test/test_{}.zip -d ImageRewardDB/images/test/test_{}
python utils/generate_imagerewarddb.py 
 ```
### Download HPDv2
```bash
hf download ymhao/HPDv2 --include  test.tar.gz test.json --repo-type dataset --local-dir ./HPDv2
tar -xvf ./HPDv2/test.tar.gz
python utils/generate_hpdv2_test.py 
```
### Download Pick-a-Pic v2
```bash
hf download liuhuohuo2/pick-a-pic-v2 --repo-type dataset --local-dir ../Pickapic
```
### Generate train.json
```bash
python utils/concat_datasets.py
```
 ## Model Prepare

 ### HPSv3 
 ```bash
 # on another server
 pip install hpsv3 gradio
 git clone https://github.com/MizzenAI/HPSv3.git
 cd HPSv3
python gradio_demo/demo.py 
 ```