"""
Quick Start Script for FakeNewsNet Training
Run this to train your models on the FakeNewsNet dataset
"""
import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def download_spacy_model():
    """Download spaCy English model"""
    print("🔄 Downloading spaCy English model...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ spaCy model downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️ spaCy model download failed: {e}")
        print("   Training will continue with reduced NLP features")
        return False

def run_training():
    """Run the FakeNewsNet training"""
    print("🚀 Starting FakeNewsNet training...")
    try:
        subprocess.check_call([sys.executable, "fakenews_net_trainer.py"])
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return False

def main():
    """Main function"""
    print("="*80)
    print("🤖 FakeNewsNet Training Quick Start for TruthMate")
    print("="*80)
    
    # Check if we're in the ml-service directory
    if not Path("ultimate_working_service.py").exists():
        print("❌ Please run this script from the ml-service directory")
        return
    
    # Step 1: Install requirements
    if not install_requirements():
        return
    
    # Step 2: Download spaCy model
    download_spacy_model()
    
    # Step 3: Run training
    if run_training():
        print("\\n🎉 FakeNewsNet training completed!")
        print("\\n📋 Next Steps:")
        print("1. Check training_report.json for model performance")
        print("2. Run the integration script to update your service")
        print("3. Test the improved models with your TruthMate platform")
        print("\\n✅ Your fact-checking models are now enhanced with FakeNewsNet data!")
    else:
        print("\\n❌ Training failed. Check fakenews_training.log for details")

if __name__ == "__main__":
    main()