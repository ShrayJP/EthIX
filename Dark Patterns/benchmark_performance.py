
import time
import psutil
import os
import torch
import numpy as np
from static_detection_module import TextPatternDetector

def measure_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Returns MB

def benchmark():
    print("="*60)
    print("ETHIX - PERFORMANCE BENCHMARK")
    print("="*60)
    
    print(f"Initial Memory Usage: {measure_memory():.2f} MB")
    
    # Measure Model Loading Time
    start_time = time.time()
    print("\nLoading Model...")
    detector = TextPatternDetector()
    load_time = time.time() - start_time
    print(f"Model Load Time: {load_time:.2f} seconds")
    print(f"Memory Usage After Load: {measure_memory():.2f} MB")
    
    # Test Data
    texts = [
        "Hurry! Only 2 items left in stock.",
        "Join 10,000 satisfied customers today.",
        "No thanks, I hate saving money.",
        "You must agree to the terms to continue.",
        "This is a normal sentence with no dark pattern.",
        "Warning: This offer expires in 5 minutes.",
        "Exclusive deal for new members only.",
        "Don't miss out on this amazing opportunity.",
        "I prefer to pay full price for my order.",
        "Required field: Email address.",
        "That man will die in 3 minutes."
    ] * 5  # 50 samples total
    
    print(f"\nRunning Inference Benchmark ({len(texts)} samples)...")
    
    # Measure Inference Time
    latencies = []
    
    # Warmup
    detector.analyze_text("Warmup text")
    
    start_total = time.time()
    
    for text in texts:
        t0 = time.time()
        detector.analyze_text(text)
        latencies.append(time.time() - t0)
        
    total_time = time.time() - start_total
    
    avg_latency = np.mean(latencies) * 1000  # ms
    p95_latency = np.percentile(latencies, 95) * 1000 # ms
    throughput = len(texts) / total_time
    
    print("\n" + "-"*30)
    print("RESULTS")
    print("-"*-30)
    print(f"Total Time:       {total_time:.2f} s")
    print(f"Average Latency:  {avg_latency:.2f} ms per text")
    print(f"P95 Latency:      {p95_latency:.2f} ms")
    print(f"Throughput:       {throughput:.2f} texts/sec")
    print("-"*-30)
    
    # Qualitative Check
    if avg_latency > 500:
        print("\nWARNING: Latency is high (>500ms). Real-time analysis might be sluggish.")
    else:
        print("\nPerformance looks good for real-time analysis.")

if __name__ == "__main__":
    benchmark()
