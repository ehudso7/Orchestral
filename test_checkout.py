#!/usr/bin/env python3
"""Test Stripe checkout flow."""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_checkout():
    """Test the complete checkout flow."""
    print("=== Testing Checkout Flow ===\n")

    # First, signup/login to get API key
    login_data = {
        "email": "test@example.com",
        "password": "TestPass123!"
    }

    login_response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
    if login_response.status_code != 200:
        print("Login failed!")
        return

    login_result = login_response.json()
    # API key is only provided on signup, not login
    # For testing, we'll use the key from a fresh signup
    api_key = login_result["user"].get("api_key_id")
    if not api_key:
        # Try to get the API key from a fresh signup
        import random
        signup_data = {
            "email": f"test{random.randint(1000, 9999)}@example.com",
            "password": "TestPass123!",
            "full_name": "Test User",
            "company": "Test Corp"
        }
        signup_response = requests.post(f"{BASE_URL}/auth/signup", json=signup_data)
        if signup_response.status_code == 200:
            signup_result = signup_response.json()
            api_key = signup_result["user"]["api_key"]
            login_result = signup_result
        else:
            print("Could not get API key")
            return
    token = login_result["access_token"]

    print(f"Logged in successfully")
    print(f"API Key: {api_key}")
    print(f"Token: {token[:50]}...\n")

    # Create a customer
    print("Creating Stripe customer...")
    customer_data = {
        "email": login_result["user"]["email"],
        "name": login_result["user"]["full_name"]
    }

    customer_response = requests.post(
        f"{BASE_URL}/billing/customers",
        json=customer_data,
        headers={"X-Api-Key": api_key}
    )

    print(f"Customer creation status: {customer_response.status_code}")
    if customer_response.status_code == 200:
        customer_result = customer_response.json()
        print(f"Customer ID: {customer_result.get('customer_id')}\n")
    else:
        print(f"Response: {customer_response.text}\n")

    # Create checkout session
    print("Creating checkout session...")
    checkout_data = {
        "price_id": "price_1QRoGhIv0l4EBRxBNqmvRJXC",  # Starter plan
        "success_url": f"{BASE_URL}/dashboard?success=true",
        "cancel_url": f"{BASE_URL}/#pricing",
        "trial_days": 14
    }

    checkout_response = requests.post(
        f"{BASE_URL}/billing/checkout",
        json=checkout_data,
        headers={"X-Api-Key": api_key}
    )

    print(f"Checkout creation status: {checkout_response.status_code}")
    if checkout_response.status_code == 200:
        checkout_result = checkout_response.json()
        print(f"Checkout URL: {checkout_result.get('checkout_url')}")
        print(f"Session ID: {checkout_result.get('session_id')}")
    else:
        print(f"Error response: {json.dumps(checkout_response.json(), indent=2)}")

    print("\n=== Checkout test complete ===")

if __name__ == "__main__":
    test_checkout()