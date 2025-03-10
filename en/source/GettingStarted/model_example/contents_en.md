### Model Example

Here, we take the LLaMA model in the example as an example to quickly start the model operation.
1. Refer to host_example.yaml and env_example.sh, modify the IP address, conda environment, and other information to generate host.yaml and env.sh files, and specify the file path in train_hetu.sh.
2. Execute create_web_dataset.sh in data_utils to download the dataset, and modify the variables ROOT_FOLDER, JSON_FILE, JSON_KEY, VOCAB_FILE, MERGE_FILE in train_hetu.sh to the path of the dataset.
3. More model and distributed training configurations can be adjusted in train_hetu.sh.
4. Run bash scripts/train_hetu.sh


