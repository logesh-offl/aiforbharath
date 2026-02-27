# aiforbharath

Garbage detection system using Amazon Nova Pro vision model via AWS Bedrock.

## Features

- Advanced garbage detection and classification
- Support for multiple image formats (JPEG, PNG, WebP, AVIF, etc.)
- Detailed analysis with confidence scores
- JSON output for easy integration

## Main Components

- `nova.py` - Main garbage detection pipeline using Amazon Nova Pro
- `gem.py` - Alternative implementation
- `scripts/` - Utility scripts for validation and UI
- `vis_agent/` - Visual agent implementation

## Setup

1. Install dependencies:
```bash
pip install boto3 pillow requests
```

2. Configure AWS credentials with Bedrock access

3. Run the garbage detection:
```bash
python nova.py
```
