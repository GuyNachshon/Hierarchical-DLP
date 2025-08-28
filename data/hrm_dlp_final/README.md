# HRM-DLP Training Dataset (CORRECTED)
Generated: 2025-08-28 with rich attachment metadata

## Final Training Data (2653 examples)
- `train.jsonl`: 1854 training examples
- `val.jsonl`: 400 validation examples  
- `test.jsonl`: 399 test examples

## Rich Attachment Format
Each example contains proper attachment objects with:
- `name`: Filename with extension
- `size`: File size in bytes
- `mime_type`: Proper MIME type
- `content_summary`: Detailed content description
- `sensitivity_indicators`: Array of sensitivity types

## Quality
- ✅ Rich attachment metadata (not simple strings)
- ✅ Proper DLP format with spans and labels
- ✅ High-quality examples from successful batch generation
- ✅ Ready for HRM-DLP model training

Total: 2653 examples ready for training!
