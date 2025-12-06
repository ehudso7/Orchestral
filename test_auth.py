#!/usr/bin/env python3
"""Test authentication endpoints."""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_signup():
    """Test user signup."""
    print("\n=== Testing Signup ===")
    data = {
        "email": "test@example.com",
        "password": "TestPass123!",
        "full_name": "Test User",
        "company": "Test Corp"
    }

    try:
        response = requests.post(f"{BASE_URL}/auth/signup", json=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error: {e}")

    return None

def test_login(email, password):
    """Test user login."""
    print("\n=== Testing Login ===")
    data = {
        "email": email,
        "password": password
    }

    try:
        response = requests.post(f"{BASE_URL}/auth/login", json=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error: {e}")

    return None

def test_billing_config():
    """Test billing configuration endpoint."""
    print("\n=== Testing Billing Config ===")

    try:
        response = requests.get(f"{BASE_URL}/billing/config")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        return response.json() if response.status_code == 200 else None
    except Exception as e:
        print(f"Error: {e}")

    return None

def test_pricing():
    """Test pricing endpoint."""
    print("\n=== Testing Pricing ===")

    try:
        response = requests.get(f"{BASE_URL}/v1/pricing")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Plans found: {len(data.get('plans', []))}")
            for plan in data.get('plans', []):
                print(f"  - {plan['name']}: ${plan['price_monthly']}/mo")
        else:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Test signup
    signup_result = test_signup()

    # Test login with the same credentials
    if signup_result or True:  # Try login even if signup fails (user might exist)
        login_result = test_login("test@example.com", "TestPass123!")

    # Test billing configuration
    test_billing_config()

    # Test pricing
    test_pricing()

    print("\n=== All tests complete ===")