"""
Quick Test Script for Simplified Enhanced Service
"""
import requests
import json

def test_service():
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Simplified Enhanced ML Service")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        "COVID-19 vaccines contain microchips",
        "Coffee reduces heart disease risk", 
        "The Earth is flat",
        "Exercise improves health",
        "5G causes coronavirus"
    ]
    
    # Test health
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Service is running")
            health_data = response.json()
            print(f"Version: {health_data.get('version', 'Unknown')}")
        else:
            print("❌ Service not responding")
            return
    except:
        print("❌ Cannot connect to service")
        print("💡 Make sure to run: python simple_enhanced_app.py")
        return
    
    # Test verification
    for i, claim in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {claim}")
        
        try:
            response = requests.post(
                f"{base_url}/verify",
                json={"text": claim},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   Verdict: {result['label']} ({result['confidence']}%)")
                print(f"   Explanation: {result['explanation'][:60]}...")
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Testing complete!")

if __name__ == "__main__":
    test_service()