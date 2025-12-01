# Fake News Detection using BERT

A high-accuracy Binary Classification system leveraging Transfer Learning with the BERT (Bidirectional Encoder Representations from Transformers) architecture to distinguish between reliable news and misinformation.

---

## Overview

In the age of digital information, the spread of fake news has become a critical societal issue. This project utilizes advanced Natural Language Processing techniques to automate fake news detection.

By fine-tuning a pre-trained BERT Base Cased model from TensorFlow Hub, this application analyzes semantic context and linguistic patterns (including capitalization nuances) to classify news articles with near-perfect accuracy.

## Key Features

- Uses the `bert_en_cased_L-12_H-768_A-12` architecture.
- Case-sensitive analysis to capture capitalization patterns that often indicate sensationalism.
- Achieves more than 99% accuracy on the testing dataset.
- Includes an interactive CLI for real-time headline classification.
- Scalable data preprocessing pipeline using TensorFlow Text.

---

## Tech Stack

- Python 3.10+
- TensorFlow, Keras, TensorFlow Hub
- Pandas, NumPy
- Matplotlib, Seaborn
- Model: BERT (Base Cased)

---

## Dataset

The model was trained on the ISOT Fake News Dataset (or similar aggregated datasets) with approximately 45,000 news articles.

- True News: collected from reliable sources such as Reuters.
- Fake News: collected from flagged unreliable websites.
- Data Split:
  - Training: 80% (~36,000 samples)
  - Testing: 20% (~9,000 samples)

The dataset is balanced to avoid bias.

---

## Model Architecture

The system applies Transfer Learning using a pre-trained BERT Cased model.

1. Input Layer: raw text.
2. Preprocessing: `bert_en_cased_preprocess` (tokenization, masking, segmentation).
3. Encoder: `bert_en_cased` transformer with 12 layers and 768 hidden units.
4. Dropout: rate of 0.1.
5. Output Layer: sigmoid-activated dense layer for binary classification.

### Model Summary

```python
Layer (type)                Output Shape              Param #   
=================================================================
text (InputLayer)           [(None,)]                 0         
preprocessing (KerasLayer)  (None, 128)               0         
BERT_encoder (KerasLayer)   (None, 768)               108,310,273    
dropout (Dropout)           (None, 768)               0         
classifier (Dense)          (None, 1)                 769       
=================================================================
Total params: 108,311,042
```

---

## Results & Performance

The model was trained for 3 epochs using Adam optimizer (`lr=3e-5`) and Binary Crossentropy loss.

| Metric | Score |
|-------|--------|
| Training Accuracy | 99.95% |
| Validation Accuracy | 99.97% |
| Test Precision | 1.00 |
| Test Recall | 1.00 |
| Test F1-Score | 1.00 |

### Confusion Matrix

The model achieved near-perfect separation of Fake vs Real classes with almost no False Positives or False Negatives.

---

## Directory Structure

```text
.
├── dataset/                
│   ├── Fake.csv
│   └── True.csv
├── images/                 
│   ├── confusion_matrix.png
│   └── dataset_distribution.png
├── notebook/               
│   └── Fake_News_Detection_BERT.ipynb
├── saved_model/            
│   ├── assets/
│   ├── variables/
│   └── saved_model.pb
├── .gitignore              
└── requirements.txt        
```

---

## Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/fake-news-bert.git
cd fake-news-bert
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

```bash
jupyter notebook notebook/Fake_News_Detection_BERT.ipynb
```

### 4. Interactive Prediction

```text
Enter News Headline: "Alien spaceship lands in Times Square!"
RESULT: FAKE NEWS
CONFIDENCE: 99.8%
```


---

## Author
**Nguyen Phuong Vu**

- LinkedIn: [Vu Nguyen](https://www.linkedin.com/in/vu-nguyen-454889335/)
- GitHub: - [itsvnvr](https://github.com/itsvnvr)
- Email: iamvuphuong2005@gmail.com

---

This project demonstrates how modern Transformer-based architectures can achieve exceptional accuracy in text classification tasks.

