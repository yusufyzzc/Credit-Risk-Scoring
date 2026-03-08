# Credit Risk Scoring ML Project

This project is a machine learning powered web application that predicts the risk of credit default based on basic financial features. It is packaged in a Docker and integrated with GitHub Actions.

## Live Demo

**https://credit-risk-scoring.onrender.com**

## Features

- Risk scoring using machine learning model
- RESTful API endpoints
- Docker container support
- CI/CD pipeline integration

## Technologies

- Python 3.9
- Flask
- scikit-learn
- Docker
- GitHub Actions

## CI/CD Pipeline

The project includes automated test and build processes with GitHub Actions. On every push and pull request:
- Dependencies are installed
- Tests are run

---

## 📄 Resume Highlights

### Credit Risk Scoring — Python · Flask · scikit-learn · Docker · GitHub Actions

- Engineered an end-to-end ML web application with Flask and scikit-learn that predicts credit default risk in real time, deployed live on Render with a REST API serving JSON predictions
- Built a machine learning pipeline combining `StandardScaler` normalization and Logistic Regression, trained on 1,000+ records across 5 financial features (income, debt ratio, open loans, late payments), achieving clean probability-based risk scoring
- Containerized the full application with Docker and automated CI/CD using GitHub Actions, enabling zero-touch deployments on every push to main
- Designed RESTful `/predict` and `/health` endpoints with Flask-CORS support, enabling seamless cross-origin frontend integration and production-grade service monitoring

---

### Screen OCR Translator — Electron.js · Tesseract.js · Google Translate API · Python

- Built a cross-platform Electron.js desktop app that captures any screen region, extracts text via Tesseract.js OCR, and translates it into 100+ languages using the Google Translate API — all triggered with a single global shortcut
- Implemented system-wide keyboard shortcuts (`Ctrl+Alt+T/C/S`) that function in the background without requiring window focus, enabling seamless screen capture and clipboard copy while the user works in any other application
- Architected a dual-process Electron IPC bridge between the main and renderer processes to asynchronously invoke a Python OCR backend, parse JSON output, and update the UI with extracted and translated text in real time
- Engineered a silent capture mode that suppresses UI interruption during screen selection, then automatically restores the window and surfaces a translated result notification upon completion

