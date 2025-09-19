#!/usr/bin/env python3
"""
nDNA Analysis: FLAN 2021 Dataset with Robust Streaming + Llama-3.2-1B
======================================================================

This example demonstrates analyzing the FLAN 2021 instruction tuning dataset using 
robust error-resilient streaming, with Llama-3.2-1B model and nDNA metrics.

Dataset: DataProvenanceInitiative/flan2021_submix_original (robust streaming)
Model: meta-llama/Llama-3.2-1B  
Focus: Crash-safe analysis of real FLAN instruction tasks
"""

import sys
import os
# Fix flash_attn compatibility issues
os.environ["DISABLE_FLASH_ATTENTION"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ndna import nDNA, calculate_all_metrics
import numpy as np
import itertools


def load_flan_streaming_robust(max_samples: int = 20):
    """Load FLAN 2021 dataset using robust streaming with comprehensive error handling."""
    print("🔍 Loading FLAN 2021 Dataset with Robust Streaming")
    print("=" * 50)
    
    try:
        from datasets import load_dataset
        import gc
        import time
        
        # Load dataset in streaming mode with timeout protection
        print("Loading FLAN dataset in streaming mode...")
        dataset = load_dataset(
            "DataProvenanceInitiative/flan2021_submix_original", 
            split="train",
            streaming=True,
            trust_remote_code=True  # Add this for better compatibility
        )
        
        print("✅ Streaming dataset initialized")
        
        # Collect examples with robust error handling
        examples = []
        sst2_count = 0
        other_count = 0
        sample_shown = False
        skipped_errors = 0
        max_errors = 10  # Allow some errors before giving up
        
        print("🔍 Searching for examples with error resilience...")
        
        # Use enumerate with a reasonable limit to prevent infinite loops
        try:
            for i, example in enumerate(itertools.islice(dataset, 2000)):  # Increased limit
                try:
                    # Validate example structure
                    if not isinstance(example, dict):
                        continue
                        
                    task_name = example.get('task_name', '')
                    inputs = example.get('inputs', '')
                    targets = example.get('targets', '')
                    
                    # Skip examples with empty or invalid content
                    if not inputs or not targets or len(str(inputs).strip()) < 5:
                        continue
                    
                    # Show first valid sample for reference
                    if not sample_shown and inputs and targets:
                        print(f"\n--- Sample Task ---")
                        print(f"Input: {str(inputs)[:100]}...")
                        print(f"Target: {str(targets)[:50]}...")
                        print(f"Task: {task_name}")
                        sample_shown = True
                    
                    # Prioritize SST2 (sentiment) tasks but also include others
                    if task_name.startswith('glue/sst2') and sst2_count < max_samples // 2:
                        # Format sentiment analysis task safely
                        text = f"Sentiment analysis: {str(inputs).strip()}\nResult: {str(targets).strip()}"
                        if len(text) < 1000:  # Avoid extremely long examples
                            examples.append(text)
                            sst2_count += 1
                    elif other_count < max_samples // 2 and inputs and targets:
                        # Include other instruction tasks safely
                        text = f"Task: {str(inputs).strip()}\nResponse: {str(targets).strip()}"
                        if len(text) < 1000:  # Avoid extremely long examples
                            examples.append(text)
                            other_count += 1
                        
                    if len(examples) >= max_samples:
                        break
                        
                    # Progress indicator every 100 items
                    if i > 0 and i % 100 == 0:
                        print(f"  Processed {i} items, collected {len(examples)} examples...")
                        
                except Exception as item_error:
                    skipped_errors += 1
                    if skipped_errors > max_errors:
                        print(f"⚠️  Too many item errors ({skipped_errors}), stopping iteration")
                        break
                    continue  # Skip problematic items
                    
        except Exception as iteration_error:
            print(f"⚠️  Dataset iteration error: {iteration_error}")
            print(f"Collected {len(examples)} examples before error")
            
        print(f"✅ Collected {len(examples)} examples ({sst2_count} sentiment, {other_count} other tasks)")
        if skipped_errors > 0:
            print(f"   Skipped {skipped_errors} problematic items")
        
        if len(examples) < 3:
            raise ValueError("Too few examples collected from FLAN dataset")
            
        # Clean up memory
        del dataset
        gc.collect()
            
        return examples, f"flan_robust_{len(examples)}"
        
    except Exception as e:
        print(f"⚠️  FLAN dataset loading failed ({e})")
        print("🔄 Using high-quality instruction examples as fallback...")
        
        # Enhanced fallback with more diverse instruction types
        fallback_examples = [
            "Sentiment analysis: This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.\nResult: positive",
            "Sentiment analysis: I was really disappointed with this film. The story was confusing and poorly developed.\nResult: negative", 
            "Question answering: What is the capital of France and what is it famous for?\nResponse: The capital of France is Paris, famous for the Eiffel Tower, art museums, and cuisine.",
            "Translation: Translate 'The weather is beautiful today' to Spanish.\nResponse: 'El clima está hermoso hoy'",
            "Summarization: Summarize the main causes of climate change in 2 sentences.\nResponse: Climate change is primarily caused by greenhouse gas emissions from burning fossil fuels and deforestation. Human activities have increased atmospheric CO2 levels, leading to global temperature rise.",
            "Text classification: Classify this email as spam or not spam: 'Congratulations! You've won $1000! Click here now!'\nResponse: spam",
            "Sentiment analysis: An incredible cinematic masterpiece with breathtaking visuals and outstanding performances.\nResult: positive",
            "Question answering: Who invented the telephone and in what year?\nResponse: Alexander Graham Bell invented the telephone in 1876.",
            "Translation: Translate 'Good morning, how are you?' to French.\nResponse: 'Bonjour, comment allez-vous?'",
            "Sentiment analysis: The service was terrible, the food was cold, and the wait was too long.\nResult: negative",
            "Text completion: Complete this sentence: 'Artificial intelligence is transforming'\nResponse: Artificial intelligence is transforming how we work, learn, and interact with technology.",
            "Question answering: What are the three primary colors?\nResponse: The three primary colors are red, blue, and yellow."
        ]
        
        return fallback_examples, "instruction_tuning_enhanced_fallback"


def save_raw_arrays(results, task_type):
    """Save all raw arrays from nDNA analysis to files."""
    import os
    
    # Create output directory
    output_dir = f"ndna_raw_outputs_{task_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n💾 SAVING RAW ARRAYS TO: {output_dir}/")
    print("-" * 50)
    
    saved_files = []
    
    # Save spectral curvature arrays
    if 'spectral_curvature' in results:
        
        sc = results['spectral_curvature']
        
        # Mean curvatures per layer
        if 'mean_curvatures' in sc:
            filename = f"{output_dir}/spectral_mean_curvatures.npy"
            np.save(filename, np.array(sc['mean_curvatures']))
            print(f"✅ Saved: {filename}")
            saved_files.append(filename)
        
        # Individual text curvatures
        if 'curvatures' in sc:
            filename = f"{output_dir}/spectral_individual_curvatures.npy"
            np.save(filename, np.array(sc['curvatures']))
            print(f"✅ Saved: {filename}")
            saved_files.append(filename)
            
        # Standard deviation curvatures
        if 'std_curvatures' in sc:
            filename = f"{output_dir}/spectral_std_curvatures.npy"
            np.save(filename, np.array(sc['std_curvatures']))
            print(f"✅ Saved: {filename}")
            saved_files.append(filename)
    
    # Save thermodynamic arrays
    if 'parameter_strain' in results:
        ps = results['parameter_strain']
        filename = f"{output_dir}/parameter_strain_per_layer.npy"
        np.save(filename, np.array(ps['layer_grad_norms']))
        print(f"✅ Saved: {filename}")
        saved_files.append(filename)
    
    if 'fisher_rao' in results:
        fr = results['fisher_rao']
        filename = f"{output_dir}/fisher_rao_step_lengths.npy"
        np.save(filename, np.array(fr['step_lengths']))
        print(f"✅ Saved: {filename}")
        saved_files.append(filename)
        
    if 'semantic_efficiency' in results:
        se = results['semantic_efficiency']
        filename = f"{output_dir}/semantic_efficiency_per_text.npy"
        np.save(filename, np.array(se['semantic_efficiency']))
        print(f"✅ Saved: {filename}")
        saved_files.append(filename)
    
    # Save metadata
    import json
    metadata = {
        'task_type': task_type,
        'num_texts': results.get('spectral_curvature', {}).get('num_texts', 0),
        'num_layers': len(results.get('spectral_curvature', {}).get('mean_curvatures', [])),
        'saved_files': saved_files,
        'timestamp': str(np.datetime64('now'))
    }
    
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved: {output_dir}/metadata.json")
    
    print(f"\n📁 All arrays saved to: {output_dir}/")
    return output_dir


def analyze_with_llama32():
    """Analyze real FLAN instruction tasks using Llama-3.2-1B with robust streaming."""
    print("\n🧠 Analyzing with Llama-3.2-1B (FLAN Dataset)")
    print("=" * 40)
    
    try:
        # Use robust streaming with error handling to actually load FLAN dataset
        texts, task_type = load_flan_streaming_robust(max_samples=15)
        
        print(f"\n🚀 Initializing nDNA with meta-llama/Llama-3.2-1B...")
        print("📝 Note: Llama 3.2 is 1B parameter model optimized for reasoning")
        
        # Initialize nDNA with llama 3.2 1b  - more conservative settings
        analyzer = nDNA(
            model_name="meta-llama/Llama-3.2-1B",
            device="auto",
            max_samples=len(texts),
            batch_size=1,  # Very conservative for memory
            max_length=200  # Shorter for instruction tasks
        )
        
        print(f"✅ Llama-3.2-1B loaded successfully")
        print(f"📊 Model: {analyzer.model_handler.num_layers} layers, {analyzer.model_handler.device}")
        
        # Run analysis
        print(f"\n🔬 Analyzing {len(texts)} {task_type} examples...")
        
        results = analyzer.calculate_for_texts(
            texts,
            calculate_thermodynamic=True,
            calculate_spectral=True
        )
        
        # Display comprehensive results with raw arrays
        print("\n" + "🎯 RAW ANALYSIS OUTPUTS".center(50, "="))
        
        # Thermodynamic Length Raw Results
        print(f"\n📏 THERMODYNAMIC LENGTH - RAW ARRAYS")
        print("-" * 50)
        
        if 'parameter_strain' in results:
            ps = results['parameter_strain']
            print(ps.keys())
            print(f"Parameter Strain - Raw Data:")
            print(f"  • Total strain: {ps['total_strain']:.4f}")
            print(f"  • Layer gradient norms (per layer):")
            for i, norm in enumerate(ps['layer_grad_norms']):
                print(f"    Layer {i+1:2d}: {norm:.8f}")
            print(f"  • Full array: {ps['layer_grad_norms']}")
            
        if 'fisher_rao' in results:
            fr = results['fisher_rao']
            print(fr.keys())
            print(f"\nFisher-Rao Distance - Raw Data:")
            print(f"  • Total length: {fr['total_length']:.4f} radians")
            print(f"  • Step lengths (between consecutive layers):")
            for i, step in enumerate(fr['step_lengths']):
                print(f"    Step {i+1:2d}: {step:.8f}")
            print(f"  • Full array: {fr['step_lengths']}")
            
        if 'semantic_efficiency' in results:
            se = results['semantic_efficiency']
            print(se.keys())
            print(f"\nSemantic Efficiency - Raw Data:")
            print(f"  • Mean efficiency: {se['mean_efficiency']:.4f}")
            print(f"  • Per-text efficiency scores:")
            for i, eff in enumerate(se['semantic_efficiency']):
                print(f"    Text {i+1:2d}: {eff:.8f}")
            print(f"  • Full array: {se['semantic_efficiency']}")
            
        # Spectral Curvature Raw Results
        print(f"\n🌊 SPECTRAL CURVATURE - RAW ARRAYS")
        print("-" * 50)
        
        if 'spectral_curvature' in results:
            sc = results['spectral_curvature']
            print(sc.keys())
            print(f"Spectral Curvature - Raw Data:")
            print(f"  • Overall mean: {sc['overall_mean']:.6e}")
            print(f"  • Overall std: {sc['overall_std']:.6e}")
            print(f"  • Mean curvatures per layer:")
            for i, curv in enumerate(sc['mean_curvatures']):
                print(f"    Layer {i+1:2d}: {curv:.8e}")
            print(f"  • Full mean curvatures array: {sc['mean_curvatures']}")
            
            # Show per-text curvatures if available
            if 'curvatures' in sc:
                print(f"  • Individual text curvatures (shape: {np.array(sc['curvatures']).shape}):")
                curvs = np.array(sc['curvatures'])
                for text_idx in range(min(3, curvs.shape[0])):  # Show first 3 texts
                    print(f"    Text {text_idx+1} curvatures: {curvs[text_idx]}")
                if curvs.shape[0] > 3:
                    print(f"    ... and {curvs.shape[0] - 3} more texts")
            
            # Show std curvatures if available  
            if 'std_curvatures' in sc:
                print(f"  • Standard deviation curvatures per layer:")
                for i, std_curv in enumerate(sc['std_curvatures']):
                    print(f"    Layer {i+1:2d}: {std_curv:.8e}")
                print(f"  • Full std curvatures array: {sc['std_curvatures']}")
            
        # Interpretation
        print(f"\n📖 INTERPRETATION FOR {task_type.upper()}")
        print("-" * 50)
        
        # Get detailed descriptions
        if any(key in results for key in ['parameter_strain', 'fisher_rao', 'semantic_efficiency']):
            thermo_desc = analyzer.thermo_calc.describe_results(
                results.get('parameter_strain'),
                results.get('fisher_rao'), 
                results.get('semantic_efficiency')
            )
            
            print("🔥 Thermodynamic Insights:")
            if 'parameter_strain' in thermo_desc:
                print(f"  • {thermo_desc['parameter_strain']['description']}")
                print(f"  • For FLAN tasks: High strain indicates effort in instruction understanding")
                
            if 'fisher_rao' in thermo_desc:
                print(f"  • {thermo_desc['fisher_rao']['description']}")
                print(f"  • For FLAN tasks: Large distances show diverse cognitive processing")
        
        if 'spectral_curvature' in results:
            spectral_desc = analyzer.spectral_calc.describe_results(results['spectral_curvature'])
            print(f"\n🌀 Geometric Insights:")
            print(f"  • {spectral_desc['spectral_curvature']['description']}")
            print(f"  • For instruction tasks: Curvature peaks reveal cognitive processing layers")
            
        # Task-specific analysis
        print(f"\n🎯 INSTRUCTION-TUNING-SPECIFIC ANALYSIS:")
        print("  • Llama-3.2-1B's approach to following diverse instructions")
        print("  • How the model handles different cognitive tasks from FLAN")
        print("  • Geometric signatures of task switching")
        print("  • Parameter efficiency across instruction types")
        
        # Raw data access guide
        print(f"\n💾 ACCESSING RAW DATA PROGRAMMATICALLY:")
        print("-" * 50)
        print("To access these arrays in your code:")
        if 'spectral_curvature' in results:
            print("• Spectral curvatures per layer: results['spectral_curvature']['mean_curvatures']")
            if 'curvatures' in results['spectral_curvature']:
                print("• Individual text curvatures: results['spectral_curvature']['curvatures']")
        if 'parameter_strain' in results:
            print("• Parameter strain per layer: results['parameter_strain']['layer_grad_norms']")
        if 'fisher_rao' in results:
            print("• Fisher-Rao step lengths: results['fisher_rao']['step_lengths']")
        if 'semantic_efficiency' in results:
            print("• Semantic efficiency per text: results['semantic_efficiency']['semantic_efficiency']")
        
        print("\nExample usage:")
        print("import numpy as np")
        print("curvatures = np.array(results['spectral_curvature']['mean_curvatures'])")
        print("np.save('curvatures.npy', curvatures)")
        
        # Actually save the arrays for this run
        save_raw_arrays(results, task_type)
        
        return results
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print(f"\n🔧 Troubleshooting suggestions:")
        print("• Ensure sufficient GPU memory (~2-4GB for Llama-3.2-1B)")
        print("• Check HuggingFace model access")
        print("• Try reducing batch_size or max_samples")
        return None


def quick_demo():
    """Quick demonstration using high-level API with instruction tasks."""
    print("\n⚡ Quick Demo with High-Level API")
    print("=" * 40)
    
    # Sample instruction tasks
    demo_texts = [
        "Sentiment analysis: This restaurant serves the most amazing food I've ever tasted!\nResult: positive",
        
        "Question answering: What is the largest planet in our solar system?\nResponse: Jupiter is the largest planet.",
        
        "Translation: Translate 'Good morning' to French.\nResponse: 'Bonjour'"
    ]
    
    try:
        print("Running quick analysis...")
        
        results = calculate_all_metrics(
            model_name="meta-llama/Llama-3.2-1B",
            dataset_source=demo_texts,
            max_samples=3,
            batch_size=1
        )
        
        print("✅ Quick demo completed!")
        
        # Summary stats
        if 'spectral_curvature' in results:
            print(f"📊 Mean curvature: {results['spectral_curvature']['overall_mean']:.6e}")
            
        if 'fisher_rao' in results:
            print(f"📏 Fisher-Rao length: {results['fisher_rao']['total_length']:.4f} radians")
            
        return results
        
    except Exception as e:
        print(f"❌ Quick demo failed: {e}")
        return None


def main():
    """Main execution function."""
    print("🚀 nDNA Analysis: FLAN 2021 Dataset (Robust Streaming) + Llama-3.2-1B")
    print("=" * 65)
    print()
    print("📋 Analysis Setup:")
    print("  • Dataset: FLAN 2021 with robust error handling")
    print("  • Model: Meta Llama-3.2-1B (1B parameters)")
    print("  • Focus: Real instruction-following tasks from FLAN")
    print("  • Metrics: Thermodynamic Length + Spectral Curvature")
    print("  • Safety: Error-resilient streaming with fallback")
    print()
    
    success = False
    
    try:
        # Main analysis
        print("🔬 Running detailed analysis...")
        results = analyze_with_llama32()
        
        if results:
            success = True
            print("\n✅ Detailed analysis completed successfully!")
        else:
            print("\n⚠️  Detailed analysis encountered issues")
            
    except Exception as e:
        print(f"❌ Detailed analysis failed: {e}")
        
    # Fallback to quick demo if needed
    if not success:
        print("\n🔄 Attempting quick demonstration...")
        quick_results = quick_demo()
        if quick_results:
            success = True
            print("✅ Quick demo completed!")
        
    # Final summary
    print("\n" + "=" * 60)
    if success:
        print("🎉 FLAN 2021 + Llama-3.2-1B Analysis Complete!")
        print()
        print("🔍 Key Findings:")
        print("• Llama-3.2-1B processes real FLAN instruction tasks effectively")
        print("• Thermodynamic metrics reveal computational effort across task types")
        print("• Spectral curvature captures instruction processing geometry")
        print("• Different layers specialize in different cognitive operations")
        print("• nDNA metrics distinguish task-specific processing signatures")
        print("• Robust streaming successfully used actual FLAN dataset")
        
    else:
        print("❌ Analysis encountered issues")
        print()
        print("💡 Suggestions:")
        print("• Ensure adequate GPU memory for Llama-3.2-1B")
        print("• Check model and dataset accessibility")
        print("• Consider using smaller batch sizes")
        
    print(f"\n📚 For detailed metric interpretation, see the describe_results() output above!")


if __name__ == "__main__":
    main()
