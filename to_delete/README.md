# Deprecated Files

This directory contains files that were deprecated during the codebase reorganization.

## What's Here

### Deprecated Source Structure (`src/`)
- Original scattered source code before consolidation
- Superseded by the new `hrm_core/` and `dlp/` structure

### Deprecated Scripts
- `evaluate.py` - Old evaluation script (replaced by `evaluate_unified.py`)
- `evaluate_dlp.py` - DLP-specific evaluation (integrated into unified script)
- `train_dlp.py` - Simplified DLP training (replaced by unified `train.py`)

### Deprecated Documentation
- `HRM DLP Project Overview.md` - Old project overview (info now in CLAUDE.md)
- `README_DLP.md` - Old DLP readme (consolidated into main docs)
- `AGENTS.md` - Agent-related documentation (outdated)
- `GEMINI.md` - Gemini-specific docs (no longer relevant)
- `data-set-generation.md` - Data generation docs (info now in CLAUDE.md)
- `training-changes.md` - Training change log (outdated)

### Deprecated Batch Files
- Various `.json` files from batch API experiments
- `batch_outputs.txt` - Old batch processing outputs
- `retrieve-batches.py` - Batch retrieval script (functionality moved to main scripts)

## Removal Schedule

These files are preserved temporarily in case any references need to be recovered. 
After confirming the new unified structure is working correctly, this directory can be safely deleted.

## Migration Notes

All functionality from these deprecated files has been:
1. **Consolidated** - Multiple scattered implementations merged into single, clean modules
2. **Modernized** - Updated to use current best practices and patterns  
3. **Tested** - Covered by the new comprehensive test suite
4. **Documented** - Properly documented in the updated CLAUDE.md

The new structure provides:
- Single source of truth for each component
- Unified entry points for training and evaluation  
- Comprehensive testing framework
- Clear separation of concerns
- Maintainable architecture