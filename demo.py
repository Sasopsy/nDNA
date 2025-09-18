#!/usr/bin/env python3
"""
Quick Demo of LLM Metrics Library

This script demonstrates the basic functionality of the LLM metrics library
with a small model and minimal data for quick testing.
"""

import sys
import os

# Add library to path
sys.path.insert(0, os.path.dirname(__file__))

def main():
    """Run a quick demo of the library."""
    print("🚀 LLM Metrics Library - Quick Demo")
    print("=" * 50)
    
    try:
        # Import the high-level API
        from ndna import calculate_all_metrics
        
        print("✓ Library imported successfully!")
        print("\n📊 Calculating metrics for GPT-2 with custom texts...")
        
        # Define some test texts
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "The integral of x squared from 0 to 1 equals one third.",
        ]
        
        # Calculate all metrics (small sample for demo)
        results = calculate_all_metrics(
            model_name="gpt2",
            dataset_source=test_texts,
            max_samples=10,  # Very small for demo
            batch_size=1,
            device="auto"
        )
        
        print("✅ Calculations completed!")
        print("\n📈 Results Summary:")
        print("-" * 30)
        
        # Display results
        if "parameter_strain" in results:
            total_strain = results["parameter_strain"]["total_strain"]
            print(f"🔥 Total Parameter Strain: {total_strain:.3e}")
        
        if "fisher_rao" in results:
            total_length = results["fisher_rao"]["total_length"]
            print(f"📏 Fisher-Rao Length: {total_length:.4f} radians")
        
        if "semantic_efficiency" in results:
            mean_efficiency = results["semantic_efficiency"]["mean_efficiency"]
            print(f"⚡ Mean Semantic Efficiency: {mean_efficiency:.4f}")
        
        if "spectral_curvature" in results:
            if "overall_mean" in results["spectral_curvature"]:
                mean_curvature = results["spectral_curvature"]["overall_mean"]
                print(f"🌀 Mean Spectral Curvature: {mean_curvature:.6e}")
        
        print("\n🎉 Demo completed successfully!")
        print("\n💡 Tips:")
        print("- Use describe_results() methods for interpretation guides")
        print("- Try different models: 'distilgpt2', 'microsoft/DialoGPT-small'")
        print("- Use larger max_samples for more reliable results")
        print("- Run examples/basic_usage.py for more detailed examples")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("This might be due to:")
        print("- Missing model files (will download on first use)")
        print("- Insufficient memory (try device='cpu' and smaller batch_size)")
        print("- Network issues (for model downloading)")

if __name__ == "__main__":
    main()
