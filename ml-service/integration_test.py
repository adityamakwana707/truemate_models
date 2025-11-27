"""
Integration Test for TruthMate Frontend-Backend Connection
Test the Next.js to ML service integration
"""
import requests
import json
import time
import subprocess
import sys
import os

def test_ml_service_integration():
    """Test the ML service integration with frontend"""
    
    print("🧪 TruthMate Integration Test")
    print("=" * 50)
    
    # Check if ML service is running
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            print("✅ ML Service is running on port 5000")
            service_info = response.json()
            print(f"   Status: {service_info.get('status')}")
            print(f"   Models Loaded: {service_info.get('models_loaded')}")
        else:
            print(f"⚠️ ML Service returned status {response.status_code}")
    except requests.exceptions.RequestException:
        print("❌ ML Service not running on port 5000")
        print("   Start with: python production_sota_service.py")
        return False
    
    # Test verification endpoint
    print("\n🔍 Testing claim verification...")
    
    test_claims = [
        "COVID-19 vaccines are safe and effective",
        "The Earth is flat",
        "Exercise improves heart health",
        "5G causes coronavirus"
    ]
    
    for claim in test_claims:
        try:
            response = requests.post(
                "http://localhost:5000/verify",
                json={"claim": claim},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n📊 Claim: {claim}")
                print(f"   Verdict: {result.get('verdict')}")
                print(f"   Confidence: {result.get('confidence_score'):.0%}")
                print(f"   Processing Time: {result.get('processing_time')}s")
                print(f"   ✅ Success")
            else:
                print(f"   ❌ Error: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    # Test Next.js API integration
    print("\n🌐 Testing Next.js API integration...")
    
    try:
        # Check if Next.js is running
        next_response = requests.get("http://localhost:3000/api/verify", timeout=5)
        print("✅ Next.js API is accessible")
        
        # Test verification through Next.js
        test_payload = {
            "text": "Regular exercise is beneficial for health"
        }
        
        api_response = requests.post(
            "http://localhost:3000/api/verify",
            json=test_payload,
            timeout=30
        )
        
        if api_response.status_code == 200:
            result = api_response.json()
            print(f"✅ End-to-end test successful!")
            print(f"   Verdict: {result.get('verdict')}")
            print(f"   Confidence: {result.get('confidence')}%")
        else:
            print(f"⚠️ Next.js API returned status {api_response.status_code}")
            
    except requests.exceptions.RequestException:
        print("❌ Next.js not running on port 3000")
        print("   Start with: npm run dev")
    
    print(f"\n🎯 Integration Status Summary:")
    print(f"   ML Service (Port 5000): {'✅ Running' if check_port(5000) else '❌ Not running'}")
    print(f"   Next.js App (Port 3000): {'✅ Running' if check_port(3000) else '❌ Not running'}")
    
    return True

def check_port(port):
    """Check if a port is in use"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        return True
    except:
        try:
            response = requests.get(f"http://localhost:{port}", timeout=2)
            return True
        except:
            return False

def test_model_performance():
    """Test model performance with various claims"""
    
    print(f"\n🧠 Model Performance Test")
    print("=" * 30)
    
    performance_claims = [
        {"claim": "Vaccines cause autism", "expected": "False"},
        {"claim": "Exercise improves cardiovascular health", "expected": "True"},
        {"claim": "Climate change is primarily caused by humans", "expected": "True"},
        {"claim": "The Earth is 6000 years old", "expected": "False"},
        {"claim": "Moderate coffee consumption may reduce heart disease risk", "expected": "True"},
        {"claim": "All natural remedies are safer than pharmaceuticals", "expected": "Misleading"}
    ]
    
    correct_predictions = 0
    total_tests = len(performance_claims)
    
    for test in performance_claims:
        try:
            response = requests.post(
                "http://localhost:5000/verify",
                json={"claim": test["claim"]},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                predicted = result.get('verdict')
                expected = test["expected"]
                
                is_correct = predicted == expected
                if is_correct:
                    correct_predictions += 1
                
                status = "✅" if is_correct else "❌"
                print(f"{status} {test['claim'][:50]}...")
                print(f"   Expected: {expected}, Got: {predicted}")
                
        except Exception as e:
            print(f"❌ Error testing claim: {e}")
    
    accuracy = (correct_predictions / total_tests) * 100
    print(f"\n📊 Model Accuracy: {correct_predictions}/{total_tests} ({accuracy:.1f}%)")
    
    if accuracy >= 70:
        print("🎉 Model performance is good!")
    elif accuracy >= 50:
        print("⚠️ Model performance is acceptable but could be improved")
    else:
        print("🔧 Model needs improvement - consider retraining")

def show_integration_guide():
    """Show integration setup guide"""
    
    print(f"\n📚 TruthMate Setup Guide")
    print("=" * 30)
    
    print("1. 🚀 Start ML Service:")
    print("   cd ml-service")
    print("   python production_sota_service.py")
    print()
    
    print("2. 🌐 Start Next.js App:")
    print("   npm run dev")
    print()
    
    print("3. 🧪 Test Integration:")
    print("   python integration_test.py")
    print()
    
    print("4. 📊 Monitor Service:")
    print("   python monitoring_app.py")
    print("   Visit: http://localhost:8080/dashboard")
    print()
    
    print("🔗 Endpoints:")
    print("   Frontend: http://localhost:3000")
    print("   ML API: http://localhost:5000")
    print("   Monitoring: http://localhost:8080")

def main():
    """Main test execution"""
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--guide":
            show_integration_guide()
            return
        elif sys.argv[1] == "--performance":
            test_model_performance()
            return
    
    # Run full integration test
    success = test_ml_service_integration()
    
    if success:
        test_model_performance()
    
    show_integration_guide()

if __name__ == "__main__":
    main()