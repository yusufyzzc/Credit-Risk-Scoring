# Credit Risk Scoring ML Project

This project is a **machine learning-powered API** that predicts the risk of credit default based on basic financial features. It is packaged in a **Docker container** and integrated with **GitHub Actions CI/CD** for a DevOps workflow. Designed for deployment on Railway.
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


## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/credit-risk-scoring.git
cd credit-risk-scoring
```

2. Install backend dependencies:
```bash
cd backend
pip install -r requierement.txt
```

3. Run with Docker:
```bash
docker build -t credit-risk-backend .
docker run -p 5000:5000 credit-risk-backend
```

## 📝 API Usage

### Calculate Risk Score

```bash
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{
    "age": 35,
    "income": 50000,
    "debt": 10000,
    "openLoans": 2,
    "latePayments": 1
}'
```

Example response:
```json
{
    "risk_percent": 25.5
}
```


## 🔄 CI/CD Pipeline

The project includes automated test and build processes with GitHub Actions. On every push and pull request:
- Dependencies are installed
- Tests are run
- Docker image is built
