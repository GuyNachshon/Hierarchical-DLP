"""
Main entry point for agentic data generation.
"""

import asyncio
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv
from config import AgenticConfig, create_demo_config, create_production_config, create_hrm_dlp_config
from fixed_enhanced_coordinator import FixedEnhancedAgenticDataGenerator

load_dotenv()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Agentic Data Generator for DLP Training Data")

    parser.add_argument("--output-dir", type=str, default="data/dlp_agentic",
                        help="Output directory for generated dataset")

    parser.add_argument("--train-size", type=int, default=2000,
                        help="Number of training examples")

    parser.add_argument("--val-size", type=int, default=400,
                        help="Number of validation examples")

    parser.add_argument("--test-size", type=int, default=400,
                        help="Number of test examples")

    parser.add_argument("--demo", action="store_true",
                        help="Use demo configuration (small dataset, concurrent processing)")

    parser.add_argument("--production", action="store_true",
                        help="Use production configuration (large dataset, batch processing)")
    
    parser.add_argument("--hrm-dlp", action="store_true",
                        help="Use HRM-DLP optimized configuration (GPT-5, enhanced prompts, DLP-specific)")

    parser.add_argument("--concurrent-agents", type=int, default=10,
                        help="Maximum concurrent agents")

    parser.add_argument("--disable-batch", action="store_true",
                        help="Disable batch API and use concurrent processing")

    parser.add_argument("--no-auto-retrieve", action="store_true",
                        help="Submit batches but do not wait for results (operator will retrieve later)")

    parser.add_argument("--sequential-splits", action="store_true",
                        help="Process splits sequentially instead of submitting in parallel")

    parser.add_argument("--clear-state", action="store_true",
                        help="Clear previous state and start fresh")

    parser.add_argument("--min-quality", type=float, default=0.7,
                        help="Minimum quality score threshold")

    parser.add_argument("--resume-session", type=str,
                        help="Resume from existing session ID")

    parser.add_argument("--list-sessions", action="store_true",
                        help="List all available sessions and exit")

    parser.add_argument("--cleanup-session", type=str, default=None,
                        help="Clean up specific session ID")

    return parser.parse_args()


def create_config_from_args(args) -> AgenticConfig:
    """Create configuration from command line arguments."""
    if args.demo:
        config = create_demo_config()
    elif args.production:
        config = create_production_config()
    elif args.hrm_dlp:
        config = create_hrm_dlp_config()
    else:
        config = AgenticConfig()

    # Override with command line arguments
    config.output_dir = args.output_dir
    config.train_size = args.train_size
    config.val_size = args.val_size
    config.test_size = args.test_size
    config.max_concurrent_agents = args.concurrent_agents
    config.min_quality_score = args.min_quality

    if args.disable_batch:
        config.enable_batch_api = False
    
    # Behavior controls
    if args.no_auto_retrieve:
        config.auto_retrieve_batches = False
    if args.sequential_splits:
        config.submit_splits_concurrently = False

    # Update checkpoint directory based on output directory
    config.checkpoint_dir = f"{config.output_dir}/.checkpoints"

    return config


def check_environment():
    """Check if required environment variables are set."""
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    missing_keys = []

    for key in required_keys:
        import os
        if not os.getenv(key):
            missing_keys.append(key)

    if missing_keys:
        print("⚠️  Warning: Missing API keys:")
        for key in missing_keys:
            print(f"   - {key}")
        print("   Some models may not be available.")
        print()


async def main():
    """Main entry point."""
    args = parse_arguments()

    print("🔧 Fixed Agentic Data Generator v2.2")
    print("🎯 One batch per split • Consistent model per batch")
    print("=" * 60)

    # Check environment
    check_environment()

    # Create configuration
    config = create_config_from_args(args)

    # Handle special commands
    if args.list_sessions:
        generator = FixedEnhancedAgenticDataGenerator(config)
        sessions = generator.list_all_sessions()
        if sessions:
            print("📋 Available sessions:")
            for session in sessions:
                duration = f"{session['duration']:.1f}s"
                print(f"  {session['session_id']} - {session['status']} ({duration})")
        else:
            print("📋 No sessions found")
        return

    if args.cleanup_session:
        generator = FixedEnhancedAgenticDataGenerator(config)
        await generator.cleanup_session(cleanup_files=True)
        print(f"🗑️  Session {args.cleanup_session} cleaned up")
        return

    print(f"📊 Fixed Enhanced Configuration:")
    print(f"   Output: {config.output_dir}")
    print(f"   Dataset size: {config.train_size + config.val_size + config.test_size:,}")
    print(f"   Batch strategy: One batch per split (max 140K entries)")
    print(f"   Model consistency: Enforced per batch")
    print(f"   Batch API: {'Enabled' if config.enable_batch_api else 'Disabled'}")
    print(f"   Concurrent agents: {config.max_concurrent_agents}")
    print()

    # Clear state if requested
    if args.clear_state:
        from .state_manager import StateManager
        state_manager = StateManager(config)
        state_manager.clear_state()

    # Create and run fixed enhanced generator
    generator = FixedEnhancedAgenticDataGenerator(config)

    try:
        if args.resume_session:
            success = await generator.resume_from_session(args.resume_session)
            if not success:
                sys.exit(1)
        else:
            await generator.generate_dataset()

        print()
        print("🎉 Enhanced generation completed successfully!")
        print(f"📁 Dataset saved to: {config.output_dir}")

        # Show session info
        session_info = generator.get_session_info()
        if session_info:
            print(f"📋 Session: {session_info['session_id']}")

    except KeyboardInterrupt:
        print("\\n⚠️  Generation interrupted by user")
        print("💾 Progress has been saved and can be resumed with --resume-session")
        session_info = generator.get_session_info()
        if session_info:
            print(f"📋 Resume with: --resume-session {session_info['session_id']}")
        # sys.exit(1)

    except Exception as e:
        print(f"\\n❌ Generation failed: {e}")
        print("💾 Progress has been saved and can be resumed")
        session_info = generator.get_session_info()
        if session_info:
            print(f"📋 Resume with: --resume-session {session_info['session_id']}")
        # sys.exit(1)

    finally:
        # Clean up session resources
        await generator.cleanup_session(cleanup_files=False)


def run():
    """Entry point for command line usage."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\n👋 Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    run()
