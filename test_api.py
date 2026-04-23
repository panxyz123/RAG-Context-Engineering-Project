# find available models
# import requests

# api_key = open("/home/yup031/Project1/api-key.txt").read().strip() # 使用你的 DSMLP 路径 [cite: 32]
# base_url = "https://tritonai-api.ucsd.edu/v1/models"

# headers = {"Authorization": f"Bearer {api_key}"}
# response = requests.get(base_url, headers=headers)

# if response.status_code == 200:
#     models = response.json().get('data', [])
#     print("Your accessible models:")
#     for m in models:
#         print(f"- {m['id']}")
# else:
#     print(f"Error fetching models: {response.status_code}, {response.text}")
    
def test_openai_connection():
    """Test OpenAI API connection."""
    print("\n" + "="*70)
    print("6. Testing OpenAI API Connection")
    print("="*70)
    
    try:
        from openai import OpenAI
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = open("/home/yup031/Project1/api-key.txt").read().strip()
        base_url = "https://tritonai-api.ucsd.edu"
        
        # Simple initialization - works with OpenAI, TritonAI, and other compatible APIs
        client = OpenAI(api_key=api_key, base_url=base_url)

        
        # Test with a simple completion
        print("Sending test request to API...")
        response = client.chat.completions.create(
            model="mistral.mistral-large-3-675b-instruct",
            messages=[{"role": "user", "content": "Say 'test successful'"}],
            max_tokens=10
        )
        
        print("✓ Successfully connected to OpenAI API")
        print(f"  Response: {response.choices[0].message.content[:50]}...")
        return True
        
    except Exception as e:
        print(f"✗ OpenAI connection test failed: {e}")
        print("  Note: Verify API key and base URL in .env")
        return False


test_openai_connection()


