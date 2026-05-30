# TruthLens AI: Fake News Detection System

TruthLens AI is a complete engineering mini project that classifies news headlines or articles as **Real News** or **Fake News** using Natural Language Processing and Machine Learning.

The project uses **TF-IDF Vectorization** and **Logistic Regression**, so it is fast to train, simple to explain, and suitable for Google Colab or local execution.

## Features

- News headline or article input
- NLP preprocessing:
  - Lowercasing
  - Punctuation removal
  - Tokenization
  - Stopword removal
- TF-IDF feature extraction
- Logistic Regression classifier
- Real/Fake prediction
- Confidence score
- News credibility percentage
- Model test accuracy display
- Modern dark Streamlit UI
- Fake news warning banner
- Sample news examples
- Prediction history
- Downloadable prediction report

## Project Structure

```text
TruthLens_AI/
│
├── app.py
├── train_model.py
├── model.pkl
├── vectorizer.pkl
├── metrics.json
├── prediction_history.csv
├── requirements.txt
├── README.md
├── dataset/
│   ├── Fake.csv
│   └── True.csv
└── report_content.txt
```

`model.pkl`, `vectorizer.pkl`, `metrics.json`, and `prediction_history.csv` are generated after running the project.

## Dataset

Use the **Fake and Real News Dataset** from Kaggle.

Expected files:

```text
dataset/Fake.csv
dataset/True.csv
```

Download the dataset from Kaggle, extract it, and place `Fake.csv` and `True.csv` inside the `dataset` folder.

The training script includes a very small demo fallback dataset only to test whether the code runs. For real results, always train with the Kaggle dataset.

## Run Locally

### 1. Open the project folder

```bash
cd TruthLens_AI
```

### 2. Create and activate a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add dataset files

Place Kaggle files here:

```text
TruthLens_AI/dataset/Fake.csv
TruthLens_AI/dataset/True.csv
```

### 5. Train the model

```bash
python train_model.py
```

This creates:

```text
model.pkl
vectorizer.pkl
metrics.json
```

### 6. Run the Streamlit app

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Run in Google Colab

### 1. Upload the project

Upload the `TruthLens_AI` folder to Colab or Google Drive.

### 2. Install dependencies

```python
!pip install -r requirements.txt
```

### 3. Add Kaggle dataset files

Upload `Fake.csv` and `True.csv` into:

```text
TruthLens_AI/dataset/
```

If your Colab notebook is inside `TruthLens_AI`, the files should be:

```text
dataset/Fake.csv
dataset/True.csv
```

### 4. Train the model

```python
!python train_model.py
```

### 5. Run Streamlit in Colab

Streamlit needs a tunnel in Colab. One common option is `localtunnel`.

```python
!npm install -g localtunnel
!streamlit run app.py & npx localtunnel --port 8501
```

Open the generated public URL.

## File Explanation

### `train_model.py`

Loads the Kaggle dataset, labels fake news as `0` and real news as `1`, preprocesses text using NLTK, converts text into TF-IDF features, trains Logistic Regression, evaluates accuracy, and saves the model files.

Run it with:

```bash
python train_model.py
```

### `app.py`

Starts the Streamlit web application. It loads the saved model and vectorizer, accepts user input, preprocesses the text, predicts Real or Fake, shows confidence and credibility scores, stores analysis history, and creates a downloadable report.

Run it with:

```bash
streamlit run app.py
```

### `requirements.txt`

Contains all Python dependencies required for the project.

### `report_content.txt`

Contains ready-to-use academic mini project report content:

- Abstract
- Introduction
- Problem Statement
- Objectives
- Methodology
- System Architecture
- Advantages
- Applications
- Future Scope
- Conclusion

## Model Details

- Algorithm: Logistic Regression
- Feature extraction: TF-IDF Vectorization
- NLP library: NLTK
- Dataset split: 80% training, 20% testing
- Labels:
  - `0`: Fake News
  - `1`: Real News

## Important Note

TruthLens AI provides an automated prediction based on text patterns learned from the training dataset. It should be used as a support tool, not as the final authority on whether a news article is true. Always verify important information from trusted sources.
