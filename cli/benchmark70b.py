import requests
import json
import time


def benchmark_ollama(model_name, prompt, num_iterations=5):
    """
    Benchmarks an Ollama model by calculating the average tokens/second
    and the Time to First Token (TTFT).
    """
    url = "http://localhost:11434/api/generate"
    total_tokens = 0
    total_time = 0
    total_ttft_msec = 0  # <--- CHANGED: Initialize total TTFT in msec

    print(f"Benchmarking {model_name} with prompt: '{prompt[:40]}...'")

    for i in range(num_iterations):
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False  # Critical for getting the final JSON object with all stats
        }

        # We don't need to manually time the request because Ollama provides
        # the component timings in its response.
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()

            response_json = response.json()

            # --- Key Metrics from Ollama API Response (all in nanoseconds) ---
            prompt_eval_duration_ns = response_json.get('prompt_eval_duration', 0)
            eval_duration_ns = response_json.get('eval_duration', 0)
            eval_count = response_json.get('eval_count', 0)  # Number of output tokens
            # -----------------------------------------------------------------

            # 1. Calculate Time to First Token (TTFT)
            # TTFT is the time to load the model + time to process the prompt.
            # In the non-streaming response, this is the prompt_eval_duration.
            ttft_seconds = prompt_eval_duration_ns / 1e9

            # --- ADDED: Convert TTFT to Milliseconds ---
            ttft_msec = ttft_seconds * 1000

            # 2. Calculate Token Generation Rate (Tokens/s)
            generation_time_seconds = eval_duration_ns / 1e9

            if generation_time_seconds > 0:
                tokens_per_second = eval_count / generation_time_seconds
            else:
                tokens_per_second = 0

            # Accumulate totals for the average
            total_tokens += eval_count
            total_time += generation_time_seconds
            total_ttft_msec += ttft_msec  # <--- CHANGED: Accumulate msec value

            # --- CHANGED: Print TTFT in msec for each iteration ---
            print(f"Iteration {i + 1}: TTFT = {ttft_msec:.2f} msec | Speed = {tokens_per_second:.2f} tokens/s")

        except requests.exceptions.RequestException as e:
            print(f"Error: Could not connect to Ollama. Is it running? Details: {e}")
            return

    # Calculate final averages
    avg_tokens_per_second = total_tokens / total_time
    avg_ttft_msec = total_ttft_msec / num_iterations  # <--- CHANGED: Calculate average in msec

    print("-" * 50)
    print(f"--- BENCHMARK RESULTS for {model_name} ---")
    # --- CHANGED: Print final average TTFT in msec ---
    print(f"Average Time to First Token (TTFT): {avg_ttft_msec:.2f} milliseconds")
    print(f"Average Token Generation Speed: {avg_tokens_per_second:.2f} tokens/s")
    print("-" * 50)


if __name__ == "__main__":
    # Use 'python3 benchmark.py' on Ubuntu 22.04 if 'python' command is not available.

    # Use a long, complex prompt for a meaningful benchmark
    benchmark_prompt = "Explain the fundamental principles of quantum computing, including superposition and entanglement, and discuss the primary challenges in building a large-scale quantum computer."

    # Update this model name to the one you are testing (e.g., "qwen3:32b", "gemma3:7b", or "qwen3:32b-force-gpu")
    benchmark_ollama("llama3:70b", benchmark_prompt, num_iterations=5)
benchmark70b.py

benchmark70b.py.На
экране.