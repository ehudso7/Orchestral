#!/usr/bin/env python3
"""Test signup with the user's password that was failing."""

import requests
import json
import random

BASE_URL = 'http://localhost:8000'

# Test with the exact password that was failing
signup_data = {
    'email': f'test{random.randint(10000, 99999)}@example.com',
    'password': 'Bumbawt@3113',
    'full_name': 'Test User',
    'company': 'Test Company'
}

print(f"Testing signup with password: {signup_data['password']}")
print(f"Email: {signup_data['email']}")

try:
    response = requests.post(f'{BASE_URL}/auth/signup', json=signup_data)

    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print("✓ Signup successful!")
        print(f"User ID: {result['user']['id']}")
        print(f"API Key: {result['user'].get('api_key', 'N/A')}")
        print(f"Access Token: {result['access_token'][:50]}...")
    else:
        print("✗ Signup failed!")
        print(f"Response: {response.text}")

except Exception as e:
    print(f"✗ Error: {e}")