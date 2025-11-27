"""
Quick Test Script for TruthMate ML Services
Test all services and models to ensure they're working properly
"""
import requests
import json
import time
import subprocess
import sys
import os
from typing import Dict, List

def test_api_service(base_url: str, service_name: str) -> Dict:
    """Test an API service with sample requests"""
    print(f"\n🔍 Testing {service_name} at {base_url}")
    
    test_cases = [
        {
            'claim': 'COVID-19 vaccines are safe and effective.',
            'expected_verdict': 'True'
        },
        {
            'claim': 'The Earth is flat and NASA is lying to us.',
            'expected_verdict': 'False'
        },
        {
            'claim': '5G networks cause coronavirus infections.',
            'expected_verdict': 'False'
        },
        {
            'claim': 'Climate change might be partially caused by natural factors.',
            'expected_verdict': 'Misleading'
        }
    ]
    
    results = {
        'service_name': service_name,
        'base_url': base_url,
        'status': 'unknown',
        'test_results': [],
        'average_response_time': 0,
        'errors': []
    }
    
    try:
        # Test health endpoint
        print("  ⚡ Testing health endpoint...")
        health_response = requests.get(f"{base_url}/health", timeout=10)
        
        if health_response.status_code != 200:
            results['status'] = 'unhealthy'
            results['errors'].append(f"Health check failed: {health_response.status_code}")
            return results
        
        print("  ✅ Health check passed")
        
        # Test verification endpoint
        response_times = []
        
        for i, test_case in enumerate(test_cases):
            print(f"  🧪 Test case {i+1}: {test_case['claim'][:50]}...")
            
            start_time = time.time()
            
            try:
                verify_response = requests.post(
                    f"{base_url}/verify",
                    json={
                        'claim': test_case['claim'],
                        'analyze_sources': True,
                        'get_explanation': True
                    },
                    timeout=30
                )
                
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                if verify_response.status_code == 200:
                    result_data = verify_response.json()
                    
                    test_result = {
                        'claim': test_case['claim'],
                        'expected_verdict': test_case['expected_verdict'],
                        'actual_verdict': result_data.get('verdict', 'Unknown'),
                        'confidence_score': result_data.get('confidence_score', 0),
                        'response_time': response_time,
                        'status': 'success',
                        'explanation': result_data.get('explanation', '')[:100] + '...' if result_data.get('explanation') else ''
                    }
                    
                    print(f"    📊 Verdict: {result_data.get('verdict', 'Unknown')} (confidence: {result_data.get('confidence_score', 0):.3f})")
                    print(f"    ⏱️ Response time: {response_time:.2f}s")
                    
                else:
                    test_result = {
                        'claim': test_case['claim'],
                        'expected_verdict': test_case['expected_verdict'],
                        'status': 'error',
                        'error': f"HTTP {verify_response.status_code}",
                        'response_time': response_time
                    }
                    print(f"    ❌ Error: HTTP {verify_response.status_code}")
                
                results['test_results'].append(test_result)
                
            except Exception as e:
                response_time = time.time() - start_time
                test_result = {
                    'claim': test_case['claim'],
                    'expected_verdict': test_case['expected_verdict'],
                    'status': 'error',
                    'error': str(e),
                    'response_time': response_time
                }
                results['test_results'].append(test_result)
                results['errors'].append(f"Test case {i+1}: {str(e)}")
                print(f"    ❌ Exception: {str(e)}")
        
        # Calculate average response time
        if response_times:
            results['average_response_time'] = sum(response_times) / len(response_times)
        
        # Determine overall status
        successful_tests = len([t for t in results['test_results'] if t.get('status') == 'success'])
        if successful_tests == len(test_cases):
            results['status'] = 'healthy'
            print(f"  ✅ All {successful_tests} tests passed!")
        elif successful_tests > 0:
            results['status'] = 'partial'
            print(f"  ⚠️ {successful_tests}/{len(test_cases)} tests passed")
        else:
            results['status'] = 'failed'
            print(f"  ❌ All tests failed")
        
    except Exception as e:
        results['status'] = 'error'
        results['errors'].append(f"Service test failed: {str(e)}")
        print(f"  ❌ Service test failed: {str(e)}")
    
    return results

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔧 Checking dependencies...")
    
    required_packages = [
        'flask', 'requests', 'transformers', 'torch', 
        'scikit-learn', 'numpy', 'pandas'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package} (missing)")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        install = input("Install missing packages? (y/n): ")
        if install.lower() == 'y':
            subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_packages)
    else:
        print("✅ All dependencies are installed!")

def run_quick_benchmark():
    """Run a quick performance benchmark"""
    print("\n🚀 Running quick performance benchmark...")
    
    services_to_test = [
        {'name': 'SOTA Service', 'url': 'http://localhost:5000'},
        {'name': 'Enhanced Service', 'url': 'http://localhost:5001'},
    ]
    
    benchmark_results = {}
    
    for service_config in services_to_test:
        try:
            result = test_api_service(service_config['url'], service_config['name'])
            benchmark_results[service_config['name']] = result
        except Exception as e:
            print(f"❌ Failed to test {service_config['name']}: {e}")
            benchmark_results[service_config['name']] = {'status': 'error', 'error': str(e)}
    
    return benchmark_results

def generate_test_report(results: Dict):
    """Generate a comprehensive test report"""
    print("\n📊 TEST REPORT")
    print("=" * 50)
    
    for service_name, service_results in results.items():
        print(f"\n🔍 {service_name}")
        print("-" * 30)
        
        status = service_results.get('status', 'unknown')
        status_emoji = {
            'healthy': '✅',
            'partial': '⚠️',
            'failed': '❌',
            'error': '💥',
            'unknown': '❓'
        }.get(status, '❓')
        
        print(f"Status: {status_emoji} {status.upper()}")
        
        if 'base_url' in service_results:
            print(f"URL: {service_results['base_url']}")
        
        if 'average_response_time' in service_results:
            print(f"Avg Response Time: {service_results['average_response_time']:.3f}s")
        
        # Test results summary
        if 'test_results' in service_results:
            successful = len([t for t in service_results['test_results'] if t.get('status') == 'success'])
            total = len(service_results['test_results'])
            print(f"Tests Passed: {successful}/{total}")
            
            # Show failed tests
            failed_tests = [t for t in service_results['test_results'] if t.get('status') != 'success']
            if failed_tests:
                print("Failed Tests:")
                for test in failed_tests:
                    print(f"  - {test['claim'][:50]}... ({test.get('error', 'unknown error')})")
        
        # Show errors
        if service_results.get('errors'):
            print("Errors:")
            for error in service_results['errors']:
                print(f"  - {error}")
    
    # Overall summary
    print(f"\n📈 SUMMARY")
    print("-" * 20)
    
    healthy_services = len([s for s in results.values() if s.get('status') == 'healthy'])
    total_services = len(results)
    
    print(f"Healthy Services: {healthy_services}/{total_services}")
    
    if healthy_services == total_services:
        print("🎉 All services are working perfectly!")
    elif healthy_services > 0:
        print("⚠️ Some services need attention")
    else:
        print("🚨 Critical: No services are working properly")

def main():
    """Main test execution"""
    print("🧪 TruthMate ML Services Test Suite")
    print("=" * 50)
    
    # Check dependencies
    check_dependencies()
    
    # Test individual service endpoints
    print(f"\n🔗 Testing individual services...")
    
    # Quick test of SOTA service
    print("Testing if SOTA service is accessible...")
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            print("✅ SOTA service is running")
        else:
            print(f"⚠️ SOTA service returned status {response.status_code}")
    except:
        print("❌ SOTA service is not accessible - make sure to start it first")
        print("   Run: python sota_app.py")
    
    # Quick test of enhanced service  
    print("Testing if Enhanced service is accessible...")
    try:
        response = requests.get("http://localhost:5001/health", timeout=5)
        if response.status_code == 200:
            print("✅ Enhanced service is running")
        else:
            print(f"⚠️ Enhanced service returned status {response.status_code}")
    except:
        print("❌ Enhanced service is not accessible")
        print("   Run: python simple_enhanced_app.py")
    
    # Ask user what to test
    print(f"\nWhat would you like to test?")
    print("1. Run full benchmark (requires services to be running)")
    print("2. Test SOTA service only")
    print("3. Test Enhanced service only") 
    print("4. Quick dependency check only")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        results = run_quick_benchmark()
        generate_test_report(results)
    elif choice == '2':
        result = test_api_service('http://localhost:5000', 'SOTA Service')
        generate_test_report({'SOTA Service': result})
    elif choice == '3':
        result = test_api_service('http://localhost:5001', 'Enhanced Service')
        generate_test_report({'Enhanced Service': result})
    elif choice == '4':
        print("✅ Dependency check completed")
    elif choice == '5':
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()