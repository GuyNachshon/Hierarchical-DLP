# Enhanced Agentic Data Generator

A robust, modular system for generating high-quality DLP (Data Loss Prevention) training datasets using a 3-tier agentic architecture with advanced batch management.

## 🎯 Key Features

- **One Batch Per Split**: Each dataset split (train/val/test) becomes exactly one batch for optimal efficiency
- **Consistent Model Usage**: Each batch uses exactly one model for all requests - no mixing
- **140K Entry Limit**: Automatic handling of large datasets with smart batch division
- **Run Session Isolation**: Complete isolation between generation runs with unique session IDs
- **Robust Recovery**: Automatic pause/resume on connection failures with user intervention prompts
- **Split-Aware Processing**: Independent generation and recovery of train/val/test splits

## 🚀 Quick Start

### Prerequisites

1. **API Keys** (Required):
```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

2. **Navigate to Directory**:
```bash
cd /path/to/HRM-DLP/HRM/scripts
```

### Basic Usage

**Demo Run (Recommended First Test):**
```bash
python agentic_data_generator.py --demo
```
- Creates 140 total examples (100 train + 20 val + 20 test)
- Uses concurrent processing for speed
- Perfect for testing the system

**Default Run:**
```bash
python agentic_data_generator.py
```
- Creates 2,800 total examples (2000 train + 400 val + 400 test)
- Uses batch APIs when available

**Production Run:**
```bash
python agentic_data_generator.py --production
```
- Creates 70,000 total examples (60K train + 5K val + 5K test)
- Optimized for large-scale generation

## 📋 Command Line Options

### Dataset Configuration
```bash
--train-size 2000           # Number of training examples
--val-size 400              # Number of validation examples  
--test-size 400             # Number of test examples
--output-dir data/my_output # Output directory
```

### Generation Presets
```bash
--demo                      # Small dataset, fast processing
--production               # Large dataset, batch optimized
```

### Processing Options
```bash
--concurrent-agents 10      # Max concurrent agents
--disable-batch            # Force concurrent processing (no batch API)
--min-quality 0.7          # Minimum quality score threshold
```

### Session Management
```bash
--list-sessions            # List all available sessions
--resume-session <id>      # Resume interrupted session
--cleanup-session <id>     # Clean up specific session
--clear-state             # Clear all previous state
```

### Examples

**Custom Dataset:**
```bash
python agentic_data_generator.py \
  --train-size 5000 \
  --val-size 1000 \
  --test-size 1000 \
  --output-dir data/custom_dlp
```

**Large Production Dataset:**
```bash
python agentic_data_generator.py \
  --production \
  --output-dir data/production_dlp \
  --concurrent-agents 20
```

## 🏗️ Architecture

### 3-Tier Agentic System

#### **Tier 1: Manager Agent**
- Controls dataset balance and quality
- Creates generation plans for each split
- Manages overall orchestration

#### **Tier 2: Specialized Domain Agents**
- **Legal Agent**: Contracts, NDAs, legal correspondence
- **Finance Agent**: Payment processing, financial records  
- **HR Agent**: Employee data, benefits, personnel files
- **Security Agent**: Incident reports, access logs
- **Casual Agent**: Personal emails in corporate context
- **Clean Business Agent**: Standard business communications
- **Obfuscation Specialist**: Advanced obfuscation techniques

#### **Tier 3: Conversational Agent**
- Multi-turn conversation simulation
- Context-aware follow-up messages
- Realistic thread progression

### Fixed Batch Management

#### **One Batch Per Split**
```
Train Split (2,000 examples) → 1 Batch using anthropic/claude-3-sonnet
Val Split   (400 examples)   → 1 Batch using openai/gpt-4
Test Split  (400 examples)   → 1 Batch using anthropic/claude-3-sonnet
```

#### **140K Entry Limit**
- Splits ≤140K entries: **Single batch**
- Splits >140K entries: **Multiple batches** (rare, only for very large datasets)
- No arbitrary splitting within normal-sized splits

#### **Consistent Model Usage**
- Each batch uses exactly **one model** for all requests
- Model selected based on request characteristics and agent types
- No mixing of models within a single batch

## 📊 Session Management

### Run Sessions
Each generation run gets a unique session ID:
```
run_20250823_181400_a1b2c3d4
```

### Session Directory Structure
```
data/runs/run_20250823_181400_a1b2c3d4/
├── session.json                          # Session metadata
├── batch_inputs/                         # Input files for each batch
│   ├── train_batch_001_anthropic_claude-3-sonnet_input.jsonl
│   ├── val_batch_001_openai_gpt-4_input.jsonl
│   └── test_batch_001_anthropic_claude-3-sonnet_input.jsonl
├── split_outputs/                        # Raw outputs before final processing
│   ├── train_examples.jsonl
│   ├── val_examples.jsonl
│   └── test_examples.jsonl
└── checkpoints/                          # Recovery checkpoints
    └── batches/
        ├── active_batches.json
        └── completed_batches.json
```

### Session Commands

**List All Sessions:**
```bash
python agentic_data_generator.py --list-sessions
```
Output:
```
📋 Available sessions:
  run_20250823_181400_a1b2c3d4 - completed (1247.3s)
  run_20250823_194500_e5f6g7h8 - interrupted (45.2s)
```

**Resume Interrupted Session:**
```bash
python agentic_data_generator.py --resume-session run_20250823_194500_e5f6g7h8
```

**Clean Up Old Session:**
```bash
python agentic_data_generator.py --cleanup-session run_20250823_181400_a1b2c3d4
```

## 🛡️ Error Handling & Recovery

### Automatic Connection Monitoring
The system continuously monitors:
- OpenAI API connectivity
- Anthropic API connectivity  
- General internet connectivity

### Graceful Failure Handling
When connection issues occur:
```
🚨 Connection failed: openai_api (Rate limited (HTTP 429))
🤔 User intervention required. Operations are paused.

Options:
  1. Resume all operations
  2. Resume selective splits
  3. Check connection status  
  4. Abort and exit
  5. Show recovery information

Enter choice (1-5):
```

### Recovery Options

**Option 1: Resume All Operations**
- Continues all paused splits
- Waits for connections to recover

**Option 2: Resume Selective Splits**
```
Active splits:
  1. train (1,250/2,000 examples)
  2. val (0/400 examples)

Enter split numbers to resume (comma-separated), or 'all': 1
```

**Option 3: Check Connection Status**
```
🔌 Connection Status:
  🟢 internet_connectivity: healthy
    Response time: 0.15s
  🟡 openai_api: degraded  
    Error: Rate limited (HTTP 429)
    Response time: 5.23s
  🟢 anthropic_api: healthy
    Response time: 0.89s
```

### Manual Recovery
**Graceful Shutdown:**
```bash
# Use Ctrl+C for graceful shutdown
^C
⚠️  Generation interrupted by user
💾 Progress has been saved and can be resumed with --resume-session
📋 Resume with: --resume-session run_20250823_194500_e5f6g7h8
```

## 📁 Output Format

### Final Dataset Files
```
data/dlp_agentic/
├── train.jsonl                    # Training examples
├── val.jsonl                      # Validation examples
├── test.jsonl                     # Test examples
├── generation_stats.json          # Comprehensive statistics
├── train_metadata.json            # Training batch metadata
├── val_metadata.json              # Validation batch metadata
└── test_metadata.json             # Test batch metadata
```

### Example Output Record
```json
{
  "channel": "email",
  "user": {
    "role": "LEGAL",
    "dept": "CORP", 
    "seniority": "SENIOR"
  },
  "recipients": ["external@competitor.com"],
  "subject": "Confidential Contract Terms",
  "body": "Please find attached the NDA for Project Phoenix...",
  "labels": {
    "sensitivity": 1,
    "exposure": 1,
    "context": 1
  },
  "spans": [
    {
      "type": "NDA_TERM",
      "start": 45,
      "end": 58,
      "text": "Project Phoenix"
    }
  ]
}
```

### Metadata Files
```json
{
  "split_name": "train",
  "total_examples": 2000,
  "provider": "anthropic", 
  "model": "claude-3-sonnet-20240229",
  "batch_count": 1,
  "agent_distribution": {
    "clean_business": 1000,
    "casual": 500,
    "legal": 160,
    "finance": 160,
    "hr": 80,
    "security": 60,
    "obfuscation": 40
  }
}
```

### Statistics File
```json
{
  "session": {
    "session_id": "run_20250823_181400_a1b2c3d4",
    "duration_seconds": 1247.3,
    "version": "v2.2_fixed"
  },
  "generation": {
    "total_examples": 2800,
    "split_breakdown": {
      "train": 2000,
      "val": 400,
      "test": 400
    }
  },
  "batch_processing": {
    "strategy": "fixed_one_batch_per_split",
    "max_batch_size": 140000,
    "total_batches_created": 3,
    "model_consistency": "enforced"
  }
}
```

## ⚙️ Configuration

### Agent Distribution (Default)
- Clean Business: 50%
- Casual: 25%  
- Legal: 8%
- Finance: 8%
- HR: 4%
- Security: 3%
- Obfuscation: 2%

### Risk Level Distribution (Default)
- No Risk: 40%
- Low Risk: 45%
- Medium Risk: 10% 
- High Risk: 4%
- Obfuscated: 1%

### Model Selection Strategy
- **Legal/Finance/Security**: Prefer Anthropic Claude (better reasoning)
- **Casual/Clean Business**: Prefer OpenAI GPT (faster, cost-effective)
- **Default**: Anthropic Claude (robustness)

## 🐛 Troubleshooting

### Common Issues

**Missing API Keys:**
```
⚠️  Warning: Missing API keys:
   - OPENAI_API_KEY
   Some models may not be available.
```
**Solution**: Export your API keys before running.

**Import Errors:**
```
ImportError: No module named 'anthropic'
```
**Solution**: Install required dependencies:
```bash
pip install anthropic openai aiohttp
```

**Permission Errors:**
```
PermissionError: [Errno 13] Permission denied: 'data/dlp_agentic'
```
**Solution**: Ensure write permissions to output directory.

**Rate Limiting:**
```
🚨 Connection failed: openai_api (Rate limited (HTTP 429))
```
**Solution**: The system will automatically pause and wait. Choose option 1 to resume when ready.

### Debug Mode
For detailed logging, check the session statistics and batch input files to understand what was sent to each model.

## 🔄 Migration from Legacy System

The legacy system had two critical issues that are now fixed:

### Before (Broken)
```python
# Issue 1: Arbitrary batch splitting
for batch_start in range(0, len(requests), batch_size=50):
    # Created many unnecessary small batches

# Issue 2: Different model per request  
for request in requests:
    provider, model = choose_model()  # Different model each time!
```

### After (Fixed)
```python  
# Fix 1: One batch per split
if len(requests) <= 140000:
    # Single batch for entire split
    
# Fix 2: One model per batch
provider, model = choose_model_for_batch(requests)  # One model for all!
for request in requests:
    # All requests use the same model
```

### Upgrading
The new system is backwards compatible. Simply run:
```bash
python agentic_data_generator.py
```

Old command-line arguments still work, but now use the fixed batch processing logic.

## 📈 Performance

### Efficiency Improvements
- **Fewer API Calls**: One batch per split vs. many small batches
- **Better Rate Limiting**: Consistent model usage reduces switching overhead
- **Cost Optimization**: Batch API utilization where possible
- **Faster Processing**: No unnecessary batch fragmentation

### Typical Generation Times
- **Demo (140 examples)**: ~2-5 minutes
- **Default (2,800 examples)**: ~15-30 minutes  
- **Production (70K examples)**: ~2-4 hours

*Times vary based on API response times and selected models.*

## 🤝 Contributing

### File Structure
```
agentic_data_generator/
├── README.md                              # This file
├── __init__.py                            # Module exports
├── main.py                               # CLI entry point
├── config.py                             # Configuration classes
├── fixed_enhanced_coordinator.py         # Main coordinator
├── agents/                               # Agent implementations
├── batch/                                # Batch management
│   ├── fixed_batch_processor.py         # Model-consistent processing
│   ├── fixed_split_batch_coordinator.py # One-batch-per-split logic
│   └── ...
├── data/                                 # Data utilities
└── utils/                                # Helper functions
```

### Adding New Agents
1. Create agent class in `agents/domain_agents.py`
2. Add to agent distribution in `config.py`
3. Update agent initialization in coordinator

### Customizing Model Selection
Modify `_choose_consistent_model_for_batch()` in `FixedSplitBatchCoordinator` to implement custom logic based on request characteristics.

## 📜 License

Part of the HRM-DLP project. See main project for license information.

---

**Version**: 2.2 (Fixed Batch Management)  
**Last Updated**: August 2025

For issues or questions, check the main project documentation or create an issue in the project repository.