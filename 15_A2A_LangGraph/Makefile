.PHONY: help install server test-server test-client demo studio clean check-env

# Default target - show help
help:
	@echo "🤖 A2A LangGraph Agent - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install/sync all dependencies"
	@echo "  make check-env      Verify environment variables are set"
	@echo ""
	@echo "Running Servers:"
	@echo "  make server         Start the A2A server (http://localhost:10000)"
	@echo "  make studio         Start LangGraph Studio (http://localhost:2024)"
	@echo ""
	@echo "Testing:"
	@echo "  make test-server    Run original instructor's test client"
	@echo "  make test-client    Run client agent test suite"
	@echo "  make test-personas  Run CrewAI persona agents (Advanced Build)"
	@echo "  make demo           Run interactive client demo"
	@echo "  make demo-quick     Run quick demo (3 predefined queries)"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean          Clean up cache files and __pycache__"
	@echo "  make help           Show this help message"
	@echo ""
	@echo "💡 Tip: Run 'make server' in one terminal and 'make studio' in another"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	uv sync

# Check environment variables
check-env:
	@echo "🔍 Checking environment variables..."
	uv run python check_env.py

# Start the A2A server
server:
	@echo "🚀 Starting A2A server at http://localhost:10000"
	@echo "Press Ctrl+C to stop"
	uv run python -m app

# Start LangGraph Studio
studio:
	@echo "🎨 Starting LangGraph Studio..."
	@echo "Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024"
	@echo "API Server: http://localhost:2024"
	@echo "Press Ctrl+C to stop"
	uv run langgraph dev --allow-blocking

# Run original instructor's test client
test-server:
	@echo "🧪 Running original test client..."
	uv run python app/test_client.py

# Run client agent test suite
test-client:
	@echo "🧪 Running client agent test suite..."
	@echo "⚠️  Make sure the A2A server is running (make server)"
	uv run python app/test_client_agent.py

# Run interactive demo
demo:
	@echo "🎮 Starting interactive client demo..."
	@echo "⚠️  Make sure the A2A server is running (make server)"
	uv run python run_client_demo.py

# Run quick demo
demo-quick:
	@echo "⚡ Running quick demo (3 queries)..."
	@echo "⚠️  Make sure the A2A server is running (make server)"
	uv run python run_client_demo.py --quick

# Run CrewAI persona agents (Advanced Build)
test-personas:
	@echo "🎭 Running CrewAI persona agents..."
	@echo "⚠️  Make sure the A2A server is running (make server)"
	uv run python app/test_persona_agents.py

# Clean up cache files
clean:
	@echo "🧹 Cleaning up cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "✨ Cleanup complete!"
