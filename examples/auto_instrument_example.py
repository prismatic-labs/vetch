"""Example: Auto-instrumentation with vetch.

This example demonstrates how to use vetch.instrument() for automatic
tracking of all LLM calls without explicit context managers.

Usage:
    export GOOGLE_API_KEY=your_api_key_here
    python auto_instrument_example.py
"""

from __future__ import annotations

import os

import vetch

# Example 1: Basic auto-instrumentation
print("=" * 60)
print("Example 1: Basic Auto-Instrumentation")
print("=" * 60)

# Call once at startup - all subsequent LLM calls are tracked
vetch.instrument(
    region="us-central1",
    tags={
        "service": "auto-instrument-demo",
        "version": "1.0.0",
        "env": os.getenv("ENVIRONMENT", "dev"),
    },
)

print("✓ Vetch instrumentation enabled")
print()

# Example 2: Google GenAI (if available)
try:
    import google.genai as genai

    print("=" * 60)
    print("Example 2: Google GenAI (auto-tracked)")
    print("=" * 60)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠ GOOGLE_API_KEY not set, skipping GenAI example")
        print()
    else:
        client = genai.Client(api_key=api_key)

        # This call is automatically tracked by vetch!
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents="Say 'Hello' in one word",
        )

        print(f"Response: {response.text}")
        print("✓ Event automatically logged with energy/cost/carbon")
        print()

        # Embeddings are also tracked
        print("=" * 60)
        print("Example 3: Embeddings (auto-tracked)")
        print("=" * 60)

        embeddings_response = client.models.embed_content(
            model="text-embedding-004",
            contents=["Hello world", "Goodbye world"],
        )

        print(f"Generated {len(embeddings_response.embeddings)} embeddings")
        print("✓ Embedding event automatically logged")
        print()

except ImportError:
    print("⚠ google-genai not installed")
    print("  Install with: pip install vetch[genai]")
    print()

# Example 4: OpenAI (if available)
try:
    import openai

    print("=" * 60)
    print("Example 4: OpenAI (auto-tracked)")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠ OPENAI_API_KEY not set, skipping OpenAI example")
        print()
    else:
        client = openai.OpenAI(api_key=api_key)

        # This call is automatically tracked by vetch!
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'Hello' in one word"}],
        )

        print(f"Response: {response.choices[0].message.content}")
        print("✓ Event automatically logged with energy/cost/carbon")
        print()

except ImportError:
    print("⚠ openai not installed")
    print("  Install with: pip install vetch[openai]")
    print()

# Example 5: Anthropic (if available)
try:
    import anthropic

    print("=" * 60)
    print("Example 5: Anthropic (auto-tracked)")
    print("=" * 60)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠ ANTHROPIC_API_KEY not set, skipping Anthropic example")
        print()
    else:
        client = anthropic.Anthropic(api_key=api_key)

        # This call is automatically tracked by vetch!
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say 'Hello' in one word"}],
        )

        print(f"Response: {response.content[0].text}")
        print("✓ Event automatically logged with energy/cost/carbon")
        print()

except ImportError:
    print("⚠ anthropic not installed")
    print("  Install with: pip install anthropic")
    print()

print("=" * 60)
print("All examples complete!")
print("=" * 60)
print()
print("Note: Check stderr for JSON event logs showing cost, energy, and carbon.")
print("Events are emitted automatically without explicit context managers.")
