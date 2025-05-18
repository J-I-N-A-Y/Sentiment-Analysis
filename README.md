# Sentiment-Analysis

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*:JINAY SHAH

*INTERN ID*:CT04DL198

*DOMAIN*:DATA ANALYTICS

*DURATION*:4 WEEKS

*MENTOR*:NEELA SANTHOSH

## DESCRIPTION

This Python script implements a **sentiment analysis pipeline** on the IMDB movie review dataset using **Natural Language Processing (NLP)** and **machine learning techniques**. The project classifies user reviews as either *positive* or *negative*, combining text preprocessing, feature extraction, logistic regression, and performance evaluation into a single workflow.

---

### **1. Importing Libraries**

The script begins by importing essential libraries for data handling (`pandas`, `numpy`), visualization (`matplotlib`, `seaborn`), text processing (`nltk`), and machine learning (`scikit-learn`). NLTK is used for natural language operations such as tokenization, stopword removal, and lemmatization.

---

### **2. Downloading NLTK Resources**

Before processing text, the necessary NLTK resources are downloaded:

* `'punkt'` for tokenizing words,
* `'stopwords'` for filtering out common English words,
* `'wordnet'` for lemmatization.

These resources are critical for converting raw text into a format suitable for machine learning.

---

### **3. Loading the Dataset**

The dataset is loaded from a CSV file (`"IMDB _Dataset.csv"`), which contains 50,000 movie reviews and their corresponding sentiments (positive or negative). Sentiment values are mapped to binary format: 1 for positive and 0 for negative, which simplifies classification.

---

### **4. Text Preprocessing**

The `preprocess_text()` function performs the following operations:

* **Lowercasing** all text,
* **Tokenizing** sentences into words,
* **Removing stopwords** and non-alphabetic tokens,
* **Lemmatizing** words to their base form.

Each review is cleaned and stored in a new column `cleaned_review`. This step is crucial to reduce noise and improve the accuracy of the model by standardizing textual data.

---

### **5. TF-IDF Vectorization**

The cleaned text data is transformed into numerical form using **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization. It converts each document into a vector of 5000 features, capturing the importance of words in relation to all documents. This matrix (`X`) serves as the input for the classification model.

---

### **6. Train-Test Split**

The dataset is split into training and testing sets using an 80-20 split. This allows the model to learn from a portion of the data and be evaluated on unseen examples, ensuring a reliable estimate of performance.

---

### **7. Model Training**

A **Logistic Regression** classifier is trained on the TF-IDF vectors. Logistic regression is effective for binary classification problems like sentiment analysis. The model is trained with a high iteration cap (`max_iter=1000`) to ensure convergence.

---

### **8. Prediction & Evaluation**

Predictions are made on the test set. Evaluation metrics include:

* **Accuracy**: The percentage of correct predictions.
* **Classification Report**: Shows precision, recall, and F1-score.
* **Confusion Matrix**: Visualized using `seaborn`, this shows how well the model distinguishes between classes.

---

### **9. Insights**

A bar plot displays the distribution of positive and negative reviews in the dataset. This helps check for class imbalance, which can impact model performance.

---

### **10. Testing New Input**

Finally, the script accepts a new review string, processes it using the same preprocessing and vectorization steps, and predicts its sentiment using the trained model. The output is printed as either “Positive” or “Negative”.

## OUTPUT

Accuracy: 88.62%


Classification Report:

              precision    recall  f1-score   support

           0       0.90      0.87      0.88      4961
           1       0.88      0.90      0.89      5039

    accuracy                           0.89     10000
    
   macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000

![image](https://github.com/user-attachments/assets/5f28da9d-6a93-41e4-8557-0e478ffa74de)

![image](https://github.com/user-attachments/assets/fff3a133-f1e4-43ea-b8cf-f4c5e74d4a8d)


Sentiment for the review: Positive
