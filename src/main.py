#!/usr/bin/env python3
"""AI Ops agent CLI entry point."""
import sys

from .agents import LogAnalyzerAgent
from .config import Config


def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("AI Ops Kubernetes Assistant")
    print("=" * 60)
    print("\nCapabilities:")
    print("  - Diagnose Kubernetes cluster state")
    print("  - Inspect pods, events, workloads, and pod logs")
    print("  - Recommend and execute safe remediations")
    print("  - Maintain conversation context")
    print("\nCommands:")
    print("  'quit' or 'exit' - Exit the program")
    print("  'clear' - Clear conversation history")
    print("  'help' - Show available commands")
    print("=" * 60)
    print()


def print_help():
    """Print help information"""
    print("\nAvailable commands:")
    print("  quit/exit  - Exit the program")
    print("  clear      - Clear conversation history")
    print("  help       - Show this help message")
    print("\nExample questions:")
    print("  - Show current kubectl context and namespaces")
    print("  - Find pods for service nexus in namespace nexus")
    print("  - Get recent warning/error events in production")
    print("  - Show pod logs for last 30 minutes")
    print("  - Which pods are crash looping?")
    print()


def main():
    """Main interactive loop"""
    try:
        # Validate configuration
        Config.validate()
        
        # Print banner
        print_banner()
        
        # Initialize agent
        print("Initializing AI agent...")
        agent = LogAnalyzerAgent()
        print("Ready!\n")
        
        # Main loop
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                # Handle empty input
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    agent.clear_history()
                    print("\nConversation history cleared.\n")
                    continue
                
                if user_input.lower() == 'help':
                    print_help()
                    continue
                
                # Process query
                response = agent.process_query(user_input)
                print(f"\nAgent: {response}\n")
            
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit.\n")
                continue
            except EOFError:
                print("\n\nGoodbye!")
                break
    
    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
        print("Please check your .env file and ensure the API key for your chosen provider is set.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()