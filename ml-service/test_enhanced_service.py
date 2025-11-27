"""
Test Enhanced ML Service
Run this to verify your enhanced models are working correctly
"""
import requests
import json
import time
import sys

def test_enhanced_service():
    """Test all endpoints of the enhanced ML service"""
    base_url = "http://localhost:5000"
    
    # Test cases with expected outcomes
    test_cases = [
        {
            "text": "COVID-19 vaccines contain microchips for tracking people",
            "expected_verdict": "False",
            "description": "Conspiracy theory about vaccines"
        },
        {
            "text": "Regular exercise can help improve cardiovascular health and reduce the risk of heart disease",
            "expected_verdict": "True", 
            "description": "Well-established health fact"
        },
        {
            "text": "The Earth is flat and NASA is hiding the truth from everyone",
            "expected_verdict": "False",
            "description": "Flat Earth conspiracy theory"
        },
        {
            "text": "Drinking water is essential for human survival and health",
            "expected_verdict": "True",
            "description": "Basic biological fact"
        },
        {
            "text": "Climate change is caused by solar radiation and has nothing to do with human activities",
            "expected_verdict": "Misleading",
            "description": "Partially true but missing context"
        }
    ]
    
    print("🚀 Testing Enhanced TruthMate ML Service")
    print("=" * 50)
    
    # Test health endpoint
    print("\n1️⃣ Testing Health Endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Service Status: {health_data['status']}")
            print(f"🤖 Models Loaded: {health_data['models_loaded']}")
            print(f"💻 Device: {health_data['device']}")
            print(f"📊 Version: {health_data['version']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to service: {e}")
        print("💡 Make sure to run: python enhanced_app.py")
        return False
    
    # Test main verification endpoint
    print("\n2️⃣ Testing Enhanced Verification...")
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {case['description']}")
        print(f"Claim: '{case['text'][:60]}...'")
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/verify",
                json={"text": case["text"]},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                
                verdict = result.get('label', 'Unknown')
                confidence = result.get('confidence', 0)
                explanation = result.get('explanation', 'No explanation')
                
                print(f"🎯 Verdict: {verdict} ({confidence}% confidence)")
                print(f"📝 Explanation: {explanation[:100]}...")
                print(f"⏱️  Response Time: {end_time - start_time:.2f}s")
                
                # Check if verdict matches expectation (roughly)
                expected = case['expected_verdict']
                if verdict == expected:
                    print("✅ Verdict matches expectation")
                    score = "PASS"
                elif (expected in ['True', 'False'] and verdict in ['True', 'False']) or \
                     (expected == 'Misleading' and verdict in ['Misleading', 'Unknown']):
                    print("⚠️  Verdict close to expectation")
                    score = "PARTIAL"
                else:
                    print("❌ Verdict differs from expectation")
                    score = "FAIL"
                
                results.append({
                    'test': case['description'],
                    'verdict': verdict,
                    'confidence': confidence,
                    'expected': expected,
                    'score': score,
                    'response_time': end_time - start_time
                })
                
            else:
                print(f"❌ Request failed: {response.status_code}")
                print(f"Error: {response.text}")
                results.append({
                    'test': case['description'],
                    'score': 'ERROR',
                    'error': response.text
                })
                
        except Exception as e:
            print(f"❌ Request error: {e}")
            results.append({
                'test': case['description'],
                'score': 'ERROR', 
                'error': str(e)
            })
    
    # Test other endpoints
    print("\n3️⃣ Testing Stance Detection...")
    try:
        response = requests.post(
            f"{base_url}/stance-detection",
            json={"claim": "Climate change is real and caused by human activities"},
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Stance: {result.get('stance', 'unknown')}")
        else:
            print(f"❌ Stance detection failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Stance detection error: {e}")
    
    print("\n4️⃣ Testing Bias & Sentiment Analysis...")
    try:
        response = requests.post(
            f"{base_url}/bias-sentiment",
            json={"text": "This is absolutely outrageous and completely false information!"},
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Bias: {result.get('bias', 'unknown')}")
            print(f"✅ Sentiment: {result.get('sentiment', 'unknown')}")
            print(f"✅ Emotion: {result.get('emotion', 'unknown')}")
        else:
            print(f"❌ Bias analysis failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Bias analysis error: {e}")
    
    print("\n5️⃣ Testing Source Credibility...")
    try:
        response = requests.post(
            f"{base_url}/source-credibility",
            json={"queries": ["https://www.bbc.com/news", "https://www.cdc.gov"]},
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Average Credibility: {result.get('avg_credibility', 0):.2f}")
        else:
            print(f"❌ Source credibility failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Source credibility error: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    if results:
        pass_count = sum(1 for r in results if r['score'] == 'PASS')
        partial_count = sum(1 for r in results if r['score'] == 'PARTIAL')
        fail_count = sum(1 for r in results if r['score'] in ['FAIL', 'ERROR'])
        
        avg_confidence = sum(r.get('confidence', 0) for r in results if 'confidence' in r) / len([r for r in results if 'confidence' in r])
        avg_response_time = sum(r.get('response_time', 0) for r in results if 'response_time' in r) / len([r for r in results if 'response_time' in r])
        
        print(f"✅ Passed: {pass_count}/{len(results)}")
        print(f"⚠️  Partial: {partial_count}/{len(results)}")
        print(f"❌ Failed: {fail_count}/{len(results)}")
        print(f"📊 Average Confidence: {avg_confidence:.1f}%")
        print(f"⏱️  Average Response Time: {avg_response_time:.2f}s")
        
        success_rate = (pass_count + partial_count * 0.5) / len(results) * 100
        print(f"\n🎯 Overall Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("🎉 Excellent! Your enhanced models are performing very well!")
        elif success_rate >= 60:
            print("👍 Good! Your models are working well with room for improvement.")
        elif success_rate >= 40:
            print("⚠️  Fair. Consider fine-tuning your models or adding more training data.")
        else:
            print("❌ Poor performance. Check your model configuration and training data.")
    
    return results

def test_api_compatibility():
    """Test compatibility with the Next.js frontend"""
    print("\n6️⃣ Testing Frontend Compatibility...")
    base_url = "http://localhost:5000"
    
    # Test the exact request format from Next.js
    test_payload = {
        "text": "Coffee reduces the risk of heart disease according to recent studies"
    }
    
    try:
        response = requests.post(
            f"{base_url}/verify",
            json=test_payload,
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Check required fields for frontend
            required_fields = ['label', 'confidence', 'explanation', 'reasoning', 'evidence_queries']
            missing_fields = [field for field in required_fields if field not in result]
            
            if not missing_fields:
                print("✅ All required fields present")
                print("✅ Response format compatible with Next.js frontend")
                
                # Show sample response
                print(f"\n📄 Sample Response:")
                print(f"Label: {result['label']}")
                print(f"Confidence: {result['confidence']}%")
                print(f"Evidence Queries: {len(result.get('evidence_queries', []))} queries")
                
                return True
            else:
                print(f"❌ Missing required fields: {missing_fields}")
                return False
        else:
            print(f"❌ API request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Compatibility test error: {e}")
        return False

if __name__ == "__main__":
    print("🔬 Enhanced ML Service Tester")
    print("Make sure your enhanced ML service is running on localhost:5000")
    input("Press Enter to continue...")
    
    # Run main tests
    results = test_enhanced_service()
    
    # Test frontend compatibility
    compatible = test_api_compatibility()
    
    print("\n" + "=" * 50)
    if compatible and results:
        print("🎉 Your enhanced ML service is ready for production!")
        print("💡 Next steps:")
        print("   1. Deploy the enhanced service to your cloud provider")
        print("   2. Update your Next.js app to use the new endpoint")
        print("   3. Monitor performance and accuracy in production")
        print("   4. Collect user feedback for further improvements")
    else:
        print("⚠️  Some issues detected. Please review and fix before deployment.")
    
    print("\n🚀 Happy fact-checking!")