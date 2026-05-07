# 📊 Customer Data Processing & Recommendation System

## 🚀 Overview

This project is a **Python-based data processing pipeline** that loads customer data from a JSON file, cleans and processes it, performs basic analysis, and generates simple product recommendations.

It simulates real-world tasks performed in **Data Analysis, Backend Development, and Machine Learning preprocessing**.

---

## 🛠️ Features

* 📂 Load customer data from a JSON file
* 🧹 Clean and normalize ratings (e.g., "one" → 1)
* ⚠️ Handle missing values
* 🔁 Remove duplicate users
* 📈 Calculate:

  * Average rating
  * Percentage of poor ratings
* 🤖 Generate product recommendations based on user ratings

---

## 📁 Project Structure

```
project/
│── data.json
│── main.py
│── README.md
```

---

## 📌 How It Works

### 1. Load Data

Reads customer data from a JSON file.

### 2. Data Cleaning

* Converts text ratings into numeric values
* Standardizes input format

### 3. Handle Missing Values

Ensures missing fields (like age) are handled properly.

### 4. Remove Duplicates

Eliminates duplicate users based on name and age.

### 5. Data Analysis

* Computes average rating
* Calculates percentage of users with low ratings

### 6. Recommendation System

* ⭐ Rating ≥ 4 → **Apple**
* ⭐ Rating < 4 → **Samsung**

---

## ▶️ How to Run

1. Make sure you have Python installed (Python 3.x)

2. Create a `data.json` file:

```json
{
  "customers": [
    {"name": "John", "age": 25, "rating": "five"},
    {"name": "Alice", "age": 30, "rating": 3},
    {"name": "Bob", "age": null, "rating": "two"}
  ]
}
```

3. Run the script:

```bash
python main.py
```

---

## 📊 Sample Output

```
Average Rating: 3.33
Poor Rating %: 33.33
[{'name': 'John', 'brand': 'Apple'}, {'name': 'Alice', 'brand': 'Samsung'}]
```

---

## 🧠 Concepts Used

* Python Functions
* JSON Handling
* Data Cleaning Techniques
* Exception Handling
* Sets & Lists
* Basic Data Analysis

---

## 🔮 Future Improvements

* Use **Pandas** for efficient data processing
* Add **CSV support**
* Build a **CLI interface**
* Add **data visualization (matplotlib)**
* Improve recommendation logic

---

## 👨‍💻 Author

**Satyam Sagar**
BSc IT Student | Aspiring Developer

---

## ⭐ Contribute

Feel free to fork this repo, improve it, and submit a pull request!

---

## 📌 Note

This project is built for **learning purposes** and demonstrates how real-world data pipelines work at a basic level.
