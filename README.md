# ML Project

# 🥗 Nutritional Value Predictor

A Machine Learning web application that predicts the **Energy content (kCal)** of fast-food items based on their nutritional composition, built using Linear Regression and deployed with Streamlit.

## 🌐 Live Demo

🔗 [Click here to open the app](https://nutritionalvaluepridiction.streamlit.app/)

## 📌 Project Overview

This project is part of an ML course assignment focused on predicting nutritional values of fast-food items. The app takes 7 nutritional inputs from the user and instantly predicts the **Energy in kCal** using a trained Linear Regression model.

**Supported food chains:**
🍔 McDonald's · 🍕 Pizza Hut · 🍗 KFC · 🍩 Domino's · ☕ Starbucks · 🍔 Burger King

## 🧠 Machine Learning Details

| Model | R² Score |
|---|---|
| **Linear Regression** | **97%** ✅ Best |
| Random Forest | 93% |
| Decision Tree | 90% |

**Best Model:** Linear Regression (Normal Equation)

**Energy Formula (Atwater Factors):**
Energy = (Carbs × 4) + (Protein × 4) + (Fat × 9) + (Fiber × 2)

## 📊 Features Used

| Feature | Unit |
|---|---|
| Carbohydrates | g |
| Protein | g |
| Dietary Fiber | g |
| Sugar | g |
| Total Fat | g |
| Saturated Fat | g |
| Sodium | mg |

**Target Variable:** Energy (kCal)

## 🔄 ML Pipeline
Data Collection
↓
Data Cleaning & Preprocessing
↓
Outlier Removal
↓
Feature Selection
↓
Model Training (Linear Regression, Decision Tree, Random Forest)
↓
Model Evaluation (R² Score, MAE, RMSE)
↓
Best Model Selected → Linear Regression (97%)
↓
Deployment on Streamlit

## 🖥️ App Features

- 🔢 Enter 7 nutritional values via input fields
- ⚡ Instant kCal prediction on button click
- 🎯 Color-coded calorie category:
  - 🟢 Low Calorie (< 200 kCal)
  - 🟡 Moderate Calorie (200–400 kCal)
  - 🟠 High Calorie (400–700 kCal)
  - 🔴 Very High Calorie (> 700 kCal)
- 📊 Daily intake % bar (based on 2000 kCal diet)
- 📋 Macro summary grid (Carbs, Protein, Fat, Sodium)

## 🗂️ Repository Structure
nutritional-value-predictor/
├── nutritional_app.py     ← Main Streamlit application
├── requirements.txt       ← Python dependencies
└── README.md              ← This file

## ⚙️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Programming language |
| NumPy | Model training (Normal Equation) |
| Streamlit | Web app deployment |
| Linear Regression | Prediction model |

## 📄 Dataset

- **Source:** Fast-food nutritional dataset
- **Items:** 500+ food items
- **Chains:** McDonald's, Pizza Hut, KFC, Domino's, Starbucks, Burger King
- **Features:** 7 nutritional attributes
- **Target:** Energy (kCal)

