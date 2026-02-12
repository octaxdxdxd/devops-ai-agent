#!/usr/bin/env python3
"""
AI Logging Agent
Main entry point for the interactive agent
"""
import sys

from .agents import LogAnalyzerAgent
from .config import Config


def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("AI Log Analyzer")
    print("=" * 60)
    print("\nCapabilities:")
    print("  - Read and analyze log files")
    print("  - Answer questions about errors and patterns")
    print("  - Maintain conversation context")
    print("\nCommands:")
    print("  'quit' or 'exit' - Exit the program")
    print("  'clear' - Clear conversation history")
    print("  'help' - Show available commands")
    print("=" * 60)
    print()


def main():
    """Main interactive loop"""
    try:
        # Validate configuration
        Config.validate()
        
        # Print banner
        print_banner()
        
        # Initialize agent
        agent = LogAnalyzerAgent()
        
        # Interactive loop
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit']:
                    print("\nGoodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    agent = LogAnalyzerAgent()
                    print("\nConversation history cleared.")
                    continue
                
                if user_input.lower() == 'help':
                    print_help()
                    continue
                
                # Process query
                print("\nAgent:", end=" ")
                response = agent.process_query(user_input)
                print(response)
            
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit.")
                continue
            except EOFError:
                print("\n\nGoodbye!")
                break
    
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()