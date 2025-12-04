"""
Load testing suite using Locust.

Features:
- HTTP endpoint load testing
- LLM inference stress testing
- RAG pipeline benchmarking
- Concurrent user simulation
- Performance metrics collection
"""

from locust import HttpUser, task, between, events
from locust.env import Environment
from locust.stats import stats_printer
import json
import random
import time
from typing import List, Dict


# ============================================================================
# Test Data
# ============================================================================

SAMPLE_PROMPTS = [
    "What is machine learning?",
    "Explain quantum computing in simple terms.",
    "How does neural network training work?",
    "What are the benefits of cloud computing?",
    "Describe the process of photosynthesis.",
    "What is the capital of France?",
    "Explain Einstein's theory of relativity.",
    "How do vaccines work?",
    "What causes climate change?",
    "Describe the water cycle.",
]

SAMPLE_QUERIES = [
    "artificial intelligence applications",
    "machine learning algorithms",
    "deep learning frameworks",
    "natural language processing",
    "computer vision techniques",
    "reinforcement learning",
    "neural network architectures",
    "data science tools",
    "big data processing",
    "cloud infrastructure",
]


# ============================================================================
# Locust Users
# ============================================================================

class LLMInferenceUser(HttpUser):
    """
    Simulate LLM inference requests.
    
    Tests:
    - Text generation endpoint
    - Chat completion endpoint
    - Streaming responses
    """
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    @task(3)
    def generate_text(self):
        """Test text generation endpoint."""
        prompt = random.choice(SAMPLE_PROMPTS)
        
        payload = {
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.7,
        }
        
        with self.client.post(
            "/api/llm/generate",
            json=payload,
            catch_response=True,
            name="LLM Generate"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "text" in data:
                        response.success()
                    else:
                        response.failure("Missing text in response")
                except Exception as e:
                    response.failure(f"Invalid JSON: {e}")
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(2)
    def chat_completion(self):
        """Test chat completion endpoint."""
        message = random.choice(SAMPLE_PROMPTS)
        
        payload = {
            "messages": [
                {"role": "user", "content": message}
            ],
            "max_tokens": 100,
        }
        
        with self.client.post(
            "/api/llm/chat",
            json=payload,
            catch_response=True,
            name="LLM Chat"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Test health endpoint."""
        with self.client.get(
            "/api/llm/health",
            catch_response=True,
            name="Health Check"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")


class RAGQueryUser(HttpUser):
    """
    Simulate RAG query requests.
    
    Tests:
    - RAG query endpoint
    - Document retrieval
    - Context-aware generation
    """
    
    wait_time = between(1, 3)
    
    @task(5)
    def rag_query(self):
        """Test RAG query endpoint."""
        query = random.choice(SAMPLE_QUERIES)
        
        payload = {
            "query": query,
            "max_results": 5,
            "max_tokens": 150,
        }
        
        with self.client.post(
            "/api/rag/query",
            json=payload,
            catch_response=True,
            name="RAG Query"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "answer" in data and "sources" in data:
                        response.success()
                    else:
                        response.failure("Missing answer or sources")
                except Exception as e:
                    response.failure(f"Invalid JSON: {e}")
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(2)
    def vector_search(self):
        """Test vector search endpoint."""
        query = random.choice(SAMPLE_QUERIES)
        
        payload = {
            "query": query,
            "limit": 10,
        }
        
        with self.client.post(
            "/api/vector/search",
            json=payload,
            catch_response=True,
            name="Vector Search"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")


class EmbeddingUser(HttpUser):
    """
    Simulate embedding requests.
    
    Tests:
    - Text embedding endpoint
    - Batch embedding
    """
    
    wait_time = between(0.5, 2)
    
    @task(4)
    def embed_text(self):
        """Test text embedding endpoint."""
        text = random.choice(SAMPLE_PROMPTS)
        
        payload = {
            "text": text,
        }
        
        with self.client.post(
            "/api/embeddings/embed",
            json=payload,
            catch_response=True,
            name="Embed Text"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "embedding" in data and isinstance(data["embedding"], list):
                        response.success()
                    else:
                        response.failure("Invalid embedding format")
                except Exception as e:
                    response.failure(f"Invalid JSON: {e}")
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(2)
    def batch_embed(self):
        """Test batch embedding endpoint."""
        texts = random.sample(SAMPLE_PROMPTS, k=5)
        
        payload = {
            "texts": texts,
        }
        
        with self.client.post(
            "/api/embeddings/batch",
            json=payload,
            catch_response=True,
            name="Batch Embed"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "embeddings" in data and len(data["embeddings"]) == len(texts):
                        response.success()
                    else:
                        response.failure("Invalid batch embedding response")
                except Exception as e:
                    response.failure(f"Invalid JSON: {e}")
            else:
                response.failure(f"Status {response.status_code}")


class MixedWorkloadUser(HttpUser):
    """
    Simulate mixed workload.
    
    Combines:
    - LLM inference
    - RAG queries
    - Embeddings
    """
    
    wait_time = between(1, 4)
    
    @task(3)
    def llm_generate(self):
        """LLM generation."""
        LLMInferenceUser.generate_text(self)
    
    @task(2)
    def rag_query(self):
        """RAG query."""
        RAGQueryUser.rag_query(self)
    
    @task(1)
    def embed_text(self):
        """Text embedding."""
        EmbeddingUser.embed_text(self)


# ============================================================================
# Load Test Scenarios
# ============================================================================

class LoadTestScenarios:
    """Predefined load test scenarios."""
    
    @staticmethod
    def light_load():
        """Light load: 10 users."""
        return {
            "users": 10,
            "spawn_rate": 2,
            "run_time": "5m",
            "user_classes": [MixedWorkloadUser],
        }
    
    @staticmethod
    def moderate_load():
        """Moderate load: 50 users."""
        return {
            "users": 50,
            "spawn_rate": 5,
            "run_time": "10m",
            "user_classes": [MixedWorkloadUser],
        }
    
    @staticmethod
    def heavy_load():
        """Heavy load: 100 users."""
        return {
            "users": 100,
            "spawn_rate": 10,
            "run_time": "15m",
            "user_classes": [MixedWorkloadUser],
        }
    
    @staticmethod
    def stress_test():
        """Stress test: 200 users."""
        return {
            "users": 200,
            "spawn_rate": 20,
            "run_time": "20m",
            "user_classes": [MixedWorkloadUser],
        }
    
    @staticmethod
    def spike_test():
        """Spike test: Rapid increase to 150 users."""
        return {
            "users": 150,
            "spawn_rate": 50,
            "run_time": "10m",
            "user_classes": [MixedWorkloadUser],
        }
    
    @staticmethod
    def endurance_test():
        """Endurance test: 30 users for 1 hour."""
        return {
            "users": 30,
            "spawn_rate": 5,
            "run_time": "1h",
            "user_classes": [MixedWorkloadUser],
        }


# ============================================================================
# Custom Event Handlers
# ============================================================================

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Track custom metrics on each request."""
    if exception:
        print(f"Request failed: {name} - {exception}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts."""
    print("🚀 Load test starting...")
    print(f"Target: {environment.host}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops."""
    print("\n✅ Load test complete!")
    print("\n📊 Summary Statistics:")
    print(f"Total requests: {environment.stats.total.num_requests}")
    print(f"Total failures: {environment.stats.total.num_failures}")
    print(f"Average response time: {environment.stats.total.avg_response_time:.2f}ms")
    print(f"Min response time: {environment.stats.total.min_response_time:.2f}ms")
    print(f"Max response time: {environment.stats.total.max_response_time:.2f}ms")
    print(f"RPS: {environment.stats.total.current_rps:.2f}")


# ============================================================================
# Run Load Test Programmatically
# ============================================================================

def run_load_test(
    host: str,
    scenario: str = "moderate",
    headless: bool = True
):
    """
    Run load test programmatically.
    
    Args:
        host: Target host URL
        scenario: Test scenario (light, moderate, heavy, stress, spike, endurance)
        headless: Run without web UI
    
    Example:
        run_load_test("http://localhost:8000", scenario="moderate")
    """
    from locust.env import Environment
    from locust.log import setup_logging
    
    # Setup logging
    setup_logging("INFO")
    
    # Get scenario config
    scenarios = {
        "light": LoadTestScenarios.light_load(),
        "moderate": LoadTestScenarios.moderate_load(),
        "heavy": LoadTestScenarios.heavy_load(),
        "stress": LoadTestScenarios.stress_test(),
        "spike": LoadTestScenarios.spike_test(),
        "endurance": LoadTestScenarios.endurance_test(),
    }
    
    config = scenarios.get(scenario, LoadTestScenarios.moderate_load())
    
    # Create environment
    env = Environment(
        user_classes=config["user_classes"],
        host=host,
    )
    
    if headless:
        # Start test
        env.create_local_runner()
        env.runner.start(config["users"], spawn_rate=config["spawn_rate"])
        
        # Run for specified time
        import time as time_module
        run_time = config["run_time"]
        
        # Parse run time
        if run_time.endswith("m"):
            seconds = int(run_time[:-1]) * 60
        elif run_time.endswith("h"):
            seconds = int(run_time[:-1]) * 3600
        else:
            seconds = int(run_time)
        
        time_module.sleep(seconds)
        
        # Stop test
        env.runner.quit()
        
        # Print final stats
        env.stats.print_stats()
    else:
        # Start web UI
        env.create_web_ui("0.0.0.0", 8089)
        env.web_ui.start()
        
        print(f"Web UI started: http://localhost:8089")
        print("Press Ctrl+C to stop")
        
        try:
            env.web_ui.greenlet.join()
        except KeyboardInterrupt:
            print("\nStopping...")


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run load tests")
    parser.add_argument("--host", default="http://localhost:8000", help="Target host")
    parser.add_argument(
        "--scenario",
        default="moderate",
        choices=["light", "moderate", "heavy", "stress", "spike", "endurance"],
        help="Test scenario"
    )
    parser.add_argument("--headless", action="store_true", help="Run without web UI")
    
    args = parser.parse_args()
    
    run_load_test(args.host, args.scenario, args.headless)
