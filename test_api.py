import requests
import json

def test_api():
    url = "http://localhost:5000/predict"
    
    # Test data
    test_data = {
        "age": 35,
        "income": 5000,
        "debt": 0.3,
        "openLoans": 2,
        "latePayments": 0
    }
    
    try:
        print("🔄 Testing API...")
        response = requests.post(url, json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Risk: {result['risk_percent']:.2f}%")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running! Start it first with:")
        print("   python app.py")
        print("   or")
        print("   waitress-serve --host=0.0.0.0 --port=5000 app:app")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_api()
