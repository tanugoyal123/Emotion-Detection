# Emotion Classification using Machine Learning

This project focuses on building a robust emotion classifier using textual data. The model is trained to recognize and classify various emotions expressed in text, leveraging powerful machine learning techniques. A Streamlit app is also developed to demonstrate the model in action.

---

## 📌 Dataset

The dataset used for this project contains labeled emotional text data.

🔗 **Dataset Link**: *https://www.kaggle.com/datasets/nelgiriyewithana/emotions*

---

## 🚀 Workflow / Approach

### 1. **Data Preprocessing**

- **Data Cleaning**:
  - Removal of duplicates
  - Data balancing

- **Text Cleaning**:
  - Punctuation and emoji removal using **NeatText**
  - Stopword removal and Lemmatization using **NLTK**

- **Train-Test Split**

- **Vectorization**:
  - TF-IDF Vectorizer

### 2. **Model Development**

Two machine learning algorithms were trained and compared:

- **Support Vector Classifier (SVC)** with different regularization parameters:
  - `C = 0.1` *(best performing)*
  - `C = 1.0`
- **Multinomial Naïve Bayes**

### 3. **Evaluation Metrics**

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### 4. **Model Saving**
- The best model is saved for deployment in a web application.

### 5. **Deployment**
- The trained model is deployed using **Streamlit** for user interaction and emotion prediction.

---

## 📊 Results (Best Model: SVC Classifier with C=0.1)

- **Training Accuracy**: 93.7% (No Overfitting)
- **Testing Accuracy**: 91.5%
- **Precision**: 91.7%
- **Recall**: 91.5%
- **F1 Score**: 91.5%
---

## 🛠️ Technologies Used

- Python
- Scikit-learn
- NLTK
- NeatText
- Pandas, NumPy
- Streamlit

---

## 📦 How to Run

1. Clone this repository.
2. Install dependencies from `requirements.txt`.
3. Run the preprocessing and model training script.
4. Launch the Streamlit app using:
   ```bash
   streamlit run app.py
