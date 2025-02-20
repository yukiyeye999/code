1. Prepare the dataset, data_preprocessing.py:

2. Organize the data (save the data as npy to speed up federated training) and amplitude spectrum of local clients as following structure:
   ``` 
     ├── dataset
        ├── client1
           ├── data_npy
               ├── sample1.npy, sample2.npy, xxxx
           ├── freq_amp_npy
               ├── amp_sample1.npy, amp_sample2.npy, xxxx
        ├── clientxxx
        ├── clientxxx
   ```
3. Train the federated learning model :
   ```shell
   python train.py
   ```

4. Acknowledgement
Some of the code is adapted from https://github.com/liuquande/FedDG-ELCFS. Thank them very much for their code.

Details coming soon