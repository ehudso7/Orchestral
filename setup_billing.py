#!/usr/bin/env python3
"""
Setup script to initialize billing and authentication for Orchestral.
Run this after deploying to set up initial configuration.
"""

import os
import sys
import secrets
from pathlib import Path


def generate_secure_key():
    """Generate a secure random key."""
    return secrets.token_hex(32)


def main():
    """Generate and display required environment variables."""
    print("=" * 60)
    print("ORCHESTRAL BILLING SETUP")
    print("=" * 60)
    print("\nGenerated secure keys for your deployment:")
    print("\n# Add these to your Vercel Environment Variables:")
    print("# (Settings -> Environment Variables)\n")

    # Generate secure keys
    api_key_secret = generate_secure_key()
    admin_key = secrets.token_urlsafe(32)

    print(f"ORCHESTRAL_BILLING_API_KEY_SECRET={api_key_secret}")
    print(f"ORCHESTRAL_SERVER_ADMIN_API_KEY={admin_key}")

    print("\n# Required Stripe keys (from your Stripe Dashboard):")
    print("STRIPE_SECRET_KEY=sk_live_...")
    print("STRIPE_PUBLISHABLE_KEY=pk_live_...")
    print("STRIPE_WEBHOOK_SECRET=whsec_...")

    print("\n# Optional but recommended:")
    print("REDIS_URL=redis://your-redis-instance")
    print("ORCHESTRAL_SERVER_CORS_ORIGINS=[\"*\"]")

    print("\n# OpenAI API key (required for AI features):")
    print("OPENAI_API_KEY=sk-...")

    print("\n" + "=" * 60)
    print("IMPORTANT NOTES:")
    print("=" * 60)
    print("1. Copy ALL the above environment variables")
    print("2. Go to your Vercel project settings")
    print("3. Navigate to Settings -> Environment Variables")
    print("4. Add each variable (paste the whole line)")
    print("5. Redeploy your application")
    print("\nFor Stripe keys:")
    print("1. Go to https://dashboard.stripe.com/apikeys")
    print("2. Copy your publishable and secret keys")
    print("3. Set up webhook endpoint: https://your-app.vercel.app/billing/webhook")
    print("4. Copy the webhook signing secret")

    # Create a .env.example file
    env_example = Path(".env.example")
    with open(env_example, "w") as f:
        f.write(f"""# Orchestral Environment Variables
# Copy this to .env and fill in your values

# Required for authentication and API key management
ORCHESTRAL_BILLING_API_KEY_SECRET={api_key_secret}
ORCHESTRAL_SERVER_ADMIN_API_KEY={admin_key}

# Stripe Configuration (required for billing)
STRIPE_SECRET_KEY=sk_live_your_key_here
STRIPE_PUBLISHABLE_KEY=pk_live_your_key_here
STRIPE_WEBHOOK_SECRET=whsec_your_secret_here

# OpenAI API (required for AI features)
OPENAI_API_KEY=sk-your_openai_key_here

# Optional: Redis for caching and distributed features
# REDIS_URL=redis://localhost:6379

# Optional: Other AI providers
# ANTHROPIC_API_KEY=sk-ant-your_key_here
# GOOGLE_API_KEY=your_google_key_here
""")
    print(f"\n✅ Created .env.example file for reference")

    print("\n" + "=" * 60)
    print("Need help? Check the documentation:")
    print("https://github.com/ehudso7/Orchestral/blob/main/DEPLOYMENT_STATUS.md")
    print("=" * 60)


if __name__ == "__main__":
    main()