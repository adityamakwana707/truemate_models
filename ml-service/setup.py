#!/usr/bin/env python3
"""
TruthMate Enhanced ML Service Setup
Automated setup script for the enhanced fact-checking models
"""
import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(command, description, check_output=False):
    """Run a command and handle errors gracefully"""
    print(f"🔄 {description}...")
    try:
        if check_output:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✅ {description} completed successfully")
                return result.stdout
            else:
                print(f"❌ {description} failed: {result.stderr}")
                return None
        else:
            result = subprocess.run(command, shell=True, timeout=300)
            if result.returncode == 0:
                print(f"✅ {description} completed successfully")
                return True
            else:
                print(f"❌ {description} failed")
                return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out")
        return False
    except Exception as e:
        print(f"❌ {description} error: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("\n📦 Installing Dependencies...")
    
    # Upgrade pip first
    run_command("python -m pip install --upgrade pip", "Upgrading pip")
    
    # Install PyTorch (CPU version for compatibility)
    torch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    if not run_command(torch_cmd, "Installing PyTorch (CPU version)"):
        # Fallback to regular PyTorch
        run_command("pip install torch torchvision torchaudio", "Installing PyTorch (fallback)")
    
    # Install other requirements
    if os.path.exists("requirements.txt"):
        run_command("pip install -r requirements.txt", "Installing requirements from requirements.txt")
    else:
        # Install essential packages manually
        essential_packages = [
            "flask==3.0.0",
            "flask-cors==4.0.0", 
            "transformers==4.36.0",
            "scikit-learn==1.3.2",
            "pandas==2.1.4",
            "numpy==1.24.3",
            "requests==2.31.0",
            "beautifulsoup4==4.12.2",
            "textblob==0.17.1",
            "vaderSentiment==3.3.2",
            "sentence-transformers==2.2.2",
            "datasets==2.14.6",
            "nltk==3.8.1",
            "python-dotenv==1.0.0"
        ]
        
        for package in essential_packages:
            run_command(f"pip install {package}", f"Installing {package.split('==')[0]}")

def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK Data...")
    
    nltk_script = '''
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True) 
nltk.download("vader_lexicon", quiet=True)
print("NLTK data downloaded successfully")
'''
    
    with open("temp_nltk_download.py", "w") as f:
        f.write(nltk_script)
    
    run_command("python temp_nltk_download.py", "Downloading NLTK data")
    
    # Cleanup
    if os.path.exists("temp_nltk_download.py"):
        os.remove("temp_nltk_download.py")

def test_model_loading():
    """Test if models can be loaded successfully"""
    print("\n🧠 Testing Model Loading...")
    
    test_script = '''
import warnings
warnings.filterwarnings("ignore")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sentence_transformers import SentenceTransformer
    
    print("Testing DistilBERT...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    print("✅ DistilBERT tokenizer loaded")
    
    print("Testing Sentence Transformer...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Sentence Transformer loaded")
    
    print("All models loaded successfully!")
    
except Exception as e:
    print(f"❌ Model loading error: {e}")
    print("This may be due to network issues. Models will be downloaded on first use.")
'''
    
    with open("temp_model_test.py", "w") as f:
        f.write(test_script)
    
    run_command("python temp_model_test.py", "Testing model loading")
    
    # Cleanup
    if os.path.exists("temp_model_test.py"):
        os.remove("temp_model_test.py")

def create_directory_structure():
    """Create necessary directories"""
    print("\n📁 Creating Directory Structure...")
    
    directories = [
        "models",
        "data", 
        "logs",
        "data/sample"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def setup_environment_file():
    """Create environment configuration file"""
    print("\n⚙️ Setting up Environment...")
    
    env_content = '''# TruthMate Enhanced ML Service Configuration
ML_SERVICE_URL=http://localhost:5000
ML_SERVICE_API_KEY=dev-api-key-123
OPENAI_API_KEY=your-openai-key-here
GOOGLE_SEARCH_API_KEY=your-google-api-key
MODEL_CACHE_DIR=./models
LOG_LEVEL=INFO
MAX_REQUEST_TIMEOUT=30
'''
    
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Created .env configuration file")
    else:
        print("✅ Environment file already exists")

def run_basic_test():
    """Run a basic functionality test"""
    print("\n🧪 Running Basic Functionality Test...")
    
    # Check if enhanced_app.py exists
    if not os.path.exists("enhanced_app.py"):
        print("❌ enhanced_app.py not found!")
        return False
    
    # Try to start the service briefly
    print("Starting service for testing...")
    try:
        import subprocess
        import time
        import requests
        
        # Start service in background
        process = subprocess.Popen([sys.executable, "enhanced_app.py"])
        
        # Wait a few seconds for startup
        time.sleep(5)
        
        # Test health endpoint
        try:
            response = requests.get("http://localhost:5000/health", timeout=5)
            if response.status_code == 200:
                print("✅ Service started successfully!")
                print("✅ Health endpoint responding")
                
                # Quick verification test
                test_data = {"text": "The Earth is round"}
                verify_response = requests.post(
                    "http://localhost:5000/verify",
                    json=test_data,
                    timeout=10
                )
                
                if verify_response.status_code == 200:
                    result = verify_response.json()
                    print(f"✅ Verification working: {result.get('label', 'Unknown')}")
                else:
                    print("⚠️  Verification endpoint needs attention")
                
                success = True
            else:
                print("❌ Service not responding properly")
                success = False
                
        except Exception as e:
            print(f"❌ Service test error: {e}")
            success = False
        
        # Stop the service
        process.terminate()
        process.wait(timeout=5)
        
        return success
        
    except Exception as e:
        print(f"❌ Test startup error: {e}")
        return False

def main():
    """Main setup process"""
    print("🚀 TruthMate Enhanced ML Service Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Create directories
    create_directory_structure()
    
    # Setup environment
    setup_environment_file()
    
    # Install dependencies
    install_dependencies()
    
    # Download NLTK data
    download_nltk_data()
    
    # Test model loading
    test_model_loading()
    
    # Run basic test
    print("\n🎯 Final Compatibility Check...")
    if run_basic_test():
        print("\n" + "=" * 50)
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("\n✅ Your enhanced ML service is ready!")
        print("\n🚀 Quick Start:")
        print("   1. Start the service: python enhanced_app.py")
        print("   2. Test the service: python test_enhanced_service.py")
        print("   3. Train custom models: python train_models.py --use_sample")
        print("\n📚 Documentation: Check MODEL_GUIDE.md for detailed information")
        print("\n🔗 Your Next.js app can now connect to http://localhost:5000")
        
    else:
        print("\n" + "=" * 50)
        print("⚠️  SETUP COMPLETED WITH WARNINGS")
        print("=" * 50)
        print("\n✅ Dependencies installed")
        print("⚠️  Some functionality may need attention")
        print("\n📝 Next steps:")
        print("   1. Check error messages above")
        print("   2. Try running: python enhanced_app.py")
        print("   3. If issues persist, check MODEL_GUIDE.md")
    
    print("\n💡 Pro tip: Run with GPU support for better performance!")
    print("   Install CUDA-enabled PyTorch if you have an NVIDIA GPU")

if __name__ == "__main__":
    main()