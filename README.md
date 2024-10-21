
# How to Run the Code for DAN-BPE

This document provides instructions on how to run the code for the DAN-BPE assignment. Follow the steps below to ensure everything is set up correctly.

## Step 1: Confirm Setup

To ensure that your setup is working correctly, run the Bag of Words (BOW) model. Open your terminal and execute the following command:

```bash
python main.py --model BOW
```

If there are no errors, your environment is set up correctly.

## Step 2: Implement the Deep Averaging Network (DAN)

### Part 1a: DAN Implementation

To run your implementation of the Deep Averaging Network (DAN), execute the following command in your terminal:

```bash
python main1.py --model DAN
```
You can change hyperparameters in main1.py file.

### Part 1b: Randomly Initialized Embeddings

To modify your model to use randomly initialized embeddings, run:

```bash
python main2.py --model DAN
```

## Step 3: Implement Byte Pair Encoding (BPE)

To train the Byte Pair Encoding (BPE) model with subword tokenization, run the following command:

```bash
python main3.py --model DAN
```
You can change vocabulary size in main3.py file.
Here, dan_dev_accuracy.png and dan_train_accuracy.png are outputs of part 1a. While dan_dev_accuracy-random.png and dan_train_accuracy-random.png are outputs of part 1b.


## Note

Make sure you are running these commands in the terminal from the directory where your assignment files are located. Each script corresponds to different parts of the assignment and should be run in the order outlined above to ensure a smooth progression through the tasks.
