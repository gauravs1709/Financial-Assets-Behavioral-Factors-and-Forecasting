# Behavioural and Sentiment-Based Stock Market Forecasting

This repository contains the implementation of a short-term stock market forecasting framework that integrates **behavioural finance indicators**, **financial news sentiment**, and **machine learning models** to predict the daily closing price of the **S&P 500 index**.

The project was developed as part of a Master’s dissertation in **Data Science** at **Aston University**.

---

## Project Overview

Traditional market forecasting models rely heavily on historical price data and technical indicators. However, financial markets are also driven by **investor psychology**, **behavioural biases**, and **sentiment derived from news**.

This project proposes a unified forecasting pipeline that:
- Engineers behavioural market indicators (returns, volatility, moving averages, z-scores)
- Extracts sentiment from financial news using **FinBERT** and **TextBlob**
- Combines market and sentiment features into a single dataset
- Trains and evaluates machine learning regression models for short-term forecasting

Two supervised learning models are evaluated:
- **Linear Regression** (interpretable baseline)
- **Random Forest Regression** (non-linear ensemble model)

---

## Key Features

- Behavioural finance indicators (VIX, EMA, daily returns, volatility)
- News-based sentiment analysis using:
  - **FinBERT** (financial-domain transformer)
  - **TextBlob** (polarity and subjectivity)
- Time-aware train/test split to avoid data leakage
- Model evaluation using **MAE**, **RMSE**, and **R²**
- Visual analytics and dashboard-ready outputs

---

## Repository Structure

