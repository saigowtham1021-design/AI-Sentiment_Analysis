# 🤖 AI Sentiment Analysis
## deployed website
https://ai-sentiment-analyzer1.streamlit.app/
A Machine Learning-based Sentiment Analysis web application that predicts whether a given text expresses **Positive**, **Neutral**, or **Negative** sentiment.

The project uses **TF-IDF Vectorization**, **Linear Support Vector Machine (LinearSVC)**, and **Streamlit** for deployment.

---

## 🚀 Features

- Text preprocessing
- TF-IDF vectorization
- Linear Support Vector Machine (LinearSVC)
- Hyperparameter tuning using GridSearchCV
- Interactive Streamlit web application
- Real-time sentiment prediction
- Clean and responsive user interface

---

## 📊 Dataset

- **Dataset:** Twitter Sentiment Dataset (`Twitter_Data.csv`)
- **Classes:**
  - 😊 Positive (1)
  - 😐 Neutral (0)
  - 😞 Negative (-1)

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib
- Seaborn

---

## 📈 Machine Learning Workflow

1. Import Dataset
2. Data Cleaning
3. Handle Missing Values
4. Text Preprocessing
5. Train-Test Split
6. TF-IDF Vectorization
7. Model Training
8. Hyperparameter Tuning using GridSearchCV
9. Model Evaluation
10. Model Deployment using Streamlit

---

## 🤖 Models Compared

- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine (LinearSVC)

### ✅ Best Model

- **LinearSVC**
- **TF-IDF Vectorizer**
- **GridSearchCV**

---

## 📊 Model Performance

| Metric        | Value      |
|---------------|------------|
| Test Accuracy | **92.78%** |
| Vectorizer    | TF-IDF     |
| Classifier    | LinearSVC  |

---

## 📁 Project Structure

```
AI-Sentiment-Analysis/
│
├── app.py
├── sentiment_analysis.py
├── sentiment_analysis_model.pkl
├── tfidf_vectorizer.pkl
├── Twitter_Data.csv
├── requirements.txt
├── README.md
```

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/saigowtham1021-design/AI-Sentiment-Analysis.git
```

Move into the project

```bash
cd AI-Sentiment-Analysis
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 💻 Sample Inputs

Positive

```
I absolutely loved this movie because the story was amazing and the acting was outstanding.
```

Neutral

```
The meeting starts at 10 AM tomorrow.
```

Negative

```
I didn't like this movie because the story was boring and the acting was disappointing.
```



---

## 👨‍💻 Author

**Gullapudi Shanmuka Ramana Sai Gowtham**

- GitHub: https://github.com/saigowtham1021-design

---

## ⭐ If you like this project

Please give this repository a ⭐ on GitHub.
