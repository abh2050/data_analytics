#!/usr/bin/env python3
"""
Simple script to check if OpenAI API is available and working.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_openai_api():
    """Check if OpenAI API is available and working"""
    
    print("🔍 Checking OpenAI API Availability...")
    print("=" * 50)
    
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable is not set")
        print("💡 Please set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("   or add it to your .env file")
        return False
    
    # Mask the API key for display
    masked_key = api_key[:8] + "*" * (len(api_key) - 12) + api_key[-4:] if len(api_key) > 12 else "***masked***"
    print(f"✅ OPENAI_API_KEY found: {masked_key}")
    
    # Try to import OpenAI
    try:
        from langchain_openai import ChatOpenAI
        print("✅ langchain_openai imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import langchain_openai: {e}")
        print("💡 Try: pip install langchain-openai")
        return False
    
    # Try to create a ChatOpenAI instance
    try:
        llm = ChatOpenAI(temperature=0, model="gpt-4.1")
        print("✅ ChatOpenAI instance created successfully")
    except Exception as e:
        print(f"❌ Failed to create ChatOpenAI instance: {e}")
        return False
    
    # Try a simple API call
    print("\n🧪 Testing API connectivity...")
    try:
        from langchain_core.messages import HumanMessage
        
        response = llm.invoke([HumanMessage(content="Hello! Just testing the API. Please respond with 'API test successful'.")])
        print(f"✅ API call successful!")
        print(f"📝 Response: {response.content}")
        
        # Test bind_tools method (needed for LangGraph)
        try:
            bound_llm = llm.bind_tools([])
            print("✅ bind_tools method works correctly")
        except Exception as e:
            print(f"❌ bind_tools method failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ API call failed: {e}")
        
        # Check if it's an authentication error
        if "401" in str(e) or "authentication" in str(e).lower() or "api key" in str(e).lower():
            print("💡 This looks like an authentication error. Please check:")
            print("   - Your API key is correct")
            print("   - Your API key has sufficient credits/quota")
            print("   - Your API key has the necessary permissions")
        
        return False

def check_environment_file():
    """Check if .env file exists and has the required settings"""
    env_file = Path(".env")
    
    print(f"\n📁 Checking .env file...")
    
    if env_file.exists():
        print(f"✅ .env file found at: {env_file.absolute()}")
        
        # Read and check contents (without exposing sensitive data)
        try:
            with open(env_file, 'r') as f:
                content = f.read()
                
            if "OPENAI_API_KEY" in content:
                print("✅ OPENAI_API_KEY found in .env file")
            else:
                print("⚠️  OPENAI_API_KEY not found in .env file")
                print("💡 Add this line to your .env file:")
                print("   OPENAI_API_KEY=your-api-key-here")
                
        except Exception as e:
            print(f"❌ Error reading .env file: {e}")
    else:
        print("⚠️  .env file not found")
        print("💡 Create a .env file with:")
        print("   OPENAI_API_KEY=your-api-key-here")

if __name__ == "__main__":
    print("OpenAI API Availability Checker")
    print("=" * 50)
    
    # Check environment file
    check_environment_file()
    
    # Check API
    api_available = check_openai_api()
    
    print("\n" + "=" * 50)
    if api_available:
        print("🎉 OpenAI API is available and working!")
        print("✅ Your system is ready to run the data analytics agent")
    else:
        print("❌ OpenAI API is not available or not working properly")
        print("🔧 Please fix the issues above before running the application")
    
    sys.exit(0 if api_available else 1)