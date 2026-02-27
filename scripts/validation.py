#!/usr/bin/env python3
"""
Excel Validation Script for Service Requirements
Validates Partner Responses against Requirement Descriptions using AWS Bedrock Claude 4.5 Haiku

Usage:
    python validation.py <excel_file.xlsx> [--output results.xlsx]

Example:
    python validation.py "Amazon EKS Self-Assessment.xlsx" --output validation_results.xlsx

Requirements:
    - Install dependencies: pip install -r requirements.txt
    - Configure AWS credentials with profile: hackathon_access-302263040839
    - Excel file must contain a sheet named "Service Requirements"
    - The sheet should have columns: ID, Requirement Description, and project/customer columns
      with "met?" and "Partner response" sub-columns
"""

import sys
import json
import pandas as pd
import boto3
from typing import Dict, List, Tuple, Optional
import argparse
from botocore.exceptions import ClientError
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


# AWS Configuration
AWS_PROFILE = "hackathon_access-302263040839"
AWS_REGION = "us-east-1"
BEDROCK_MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

# Few-shot examples for AWS competency evaluation
FEW_SHOT_EXAMPLES = """The following examples are AWS-approved competency responses.
They demonstrate how AWS accepts evidence, not which AWS service is used.
Learn the acceptance pattern and apply the same evaluation logic to any AWS service
(EKS, Lambda, EC2, S3, etc.).
This ensures:
- EKS examples work for Lambda sheets
- The model learns evaluation behavior

FEW-SHOT REFERENCE (AWS-APPROVED EKS COMPETENCY EXAMPLES)

Example 1:
Requirement:
Amazon EKS represents majority of the workloads deployed for the solution.

Partner Response:
Amazon EKS Workload Overview:
Amazon EKS manages the majority of application workloads. Core services and microservices are containerized and orchestrated using Amazon EKS, ensuring scalability, resilience, and centralized cluster management.

AWS Verdict:
Meets Requirement

Example 2:
Requirement:
Infrastructure changes to Amazon EKS clusters are automated and managed using Infrastructure as Code.

Partner Response:
Tool Used: Terraform (Standardized IaC Tool)
All Amazon EKS infrastructure components such as clusters, node groups, networking, and IAM roles are provisioned and updated using Terraform, ensuring repeatable and controlled deployments.

AWS Verdict:
Meets Requirement

Example 3:
Requirement:
Changes to applications/workloads running on Amazon EKS are automated through CI/CD pipelines.

Partner Response:
API Version and Deployment Process:
Application workloads are deployed using Kubernetes manifests through CI/CD pipelines. Updates are automatically applied using controlled rollout strategies to ensure minimal downtime.

AWS Verdict:
Meets Requirement

Example 4:
Requirement:
Partner chooses appropriate isolation of resources for multi-tenant workloads on Amazon EKS.

Partner Response:
The application deployed on Amazon EKS follows namespace-based isolation. Each tenant workload is deployed in a separate namespace with dedicated resource quotas and network policies.

AWS Verdict:
Meets Requirement

Example 5:
Requirement:
Partner follows soft multi-tenancy best practices on Amazon EKS.

Partner Response:
Client uses control sets such as Kubernetes namespaces, IAM roles for service accounts, and network policies to ensure logical isolation while sharing the same EKS cluster.

AWS Verdict:
Meets Requirement

Example 6:
Requirement:
Amazon EKS clusters are configured with appropriate networking and security controls.

Partner Response:
Networking Configuration:
Amazon EKS clusters utilize VPC-based networking with security groups and network ACLs. Pod-to-pod communication is secured using network policies, and all traffic is encrypted in transit using TLS.

AWS Verdict:
Meets Requirement

Example 7:
Requirement:
Partner implements monitoring and observability for Amazon EKS workloads.

Partner Response:
Monitoring Setup:
Amazon CloudWatch Container Insights is configured for cluster and pod-level metrics. Application logs are aggregated using Amazon CloudWatch Logs, and custom dashboards provide visibility into workload performance and health.

AWS Verdict:
Meets Requirement

Example 8:
Requirement:
Partner demonstrates disaster recovery and backup strategies for Amazon EKS.

Partner Response:
Backup and DR Strategy:
Kubernetes manifests and configuration are stored in version control. Cluster state is backed up using Velero, and disaster recovery procedures are documented and tested quarterly to ensure rapid recovery capabilities.

AWS Verdict:
Meets Requirement"""


class RequirementValidator:
    """Validates partner responses against requirement descriptions using AWS Bedrock"""
    
    def __init__(self, aws_profile: str, aws_region: str, model_id: str):
        """Initialize the validator with AWS credentials"""
        self.aws_profile = aws_profile
        self.aws_region = aws_region
        self.model_id = model_id
        self.bedrock_runtime = None
        self._initialize_bedrock_client()
    
    def _initialize_bedrock_client(self):
        """Initialize AWS Bedrock runtime client"""
        try:
            session = boto3.Session(profile_name=self.aws_profile, region_name=self.aws_region)
            self.bedrock_runtime = session.client('bedrock-runtime')
            print(f"✓ Successfully initialized Bedrock client with profile: {self.aws_profile}")
        except Exception as e:
            print(f"✗ Error initializing Bedrock client: {e}")
            print("Please ensure AWS credentials are configured correctly.")
            sys.exit(1)
    
    def validate_response(
        self, 
        requirement_id: str, 
        requirement_description: str, 
        partner_response: str
    ) -> Dict[str, any]:
        """
        Evaluate partner response alignment with requirement using AWS competency reviewer perspective
        
        Args:
            requirement_id: The ID of the requirement
            requirement_description: The requirement description text
            partner_response: The partner's response to evaluate
            
        Returns:
            Dictionary with 'overall_alignment_score', 'evidence_similarity', 'missing', 'recommendation'
        """
        if pd.isna(partner_response) or not str(partner_response).strip():
            return {
                'overall_alignment_score': 0.0,
                'evidence_similarity': 'Low',
                'missing': 'Partner response is empty or missing',
                'recommendation': 'Provide a detailed partner response addressing the requirement'
            }
        
        # Construct the prompt for Claude
        prompt = f"""{FEW_SHOT_EXAMPLES}

---

TASK INSTRUCTION:
Assume the Partner Response has already been reviewed and approved by AWS for competency achievement.
Your task is NOT to re-judge correctness.
Your task is to determine whether the Partner Response is aligned in intent and evidence with the requirement,
based on the AWS-approved examples above.
Evaluate using an AWS competency reviewer perspective.
Indirect or architectural evidence is acceptable.

REQUIREMENT ID: {requirement_id}

REQUIREMENT DESCRIPTION:
{requirement_description}

PARTNER RESPONSE:
{partner_response}

Evaluate the alignment between the Partner Response and the Requirement Description using the AWS-approved examples as reference.

STRICT RULES:
For "missing":
- Only list information that would improve clarity for AWS reviewers.
- Do NOT list items that are implied or commonly accepted by AWS.
- If nothing meaningful is missing, return "None".

For "recommendation":
- Provide ONE concise, actionable improvement only.
- Do NOT restate the requirement.
- Do NOT suggest unnecessary AWS services.
- If no improvement is needed, return "No action required".

Provide your assessment in the following JSON format:
{{
    "overall_alignment_score": 0.0,
    "evidence_similarity": "High | Medium | Low",
    "missing": "Short text or None",
    "recommendation": "Short actionable sentence"
}}

Important:
- overall_alignment_score should be a float between 0.0 and 1.0
- evidence_similarity must be exactly one of: "High", "Medium", or "Low"
- missing should be brief or "None"
- recommendation should be one actionable sentence or "No action required"
- If the response is generally aligned with AWS-approved examples, do not treat missing explicit keywords as intent mismatch.

Respond ONLY with valid JSON, no additional text."""

        # Retry configuration for throttling
        max_retries = 5
        base_delay = 1  # Start with 1 second delay
        response = None
        
        for attempt in range(max_retries):
            try:
                # Prepare the request body for Claude
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2048,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
                
                # Invoke the model
                response = self.bedrock_runtime.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(body)
                )
                
                # Parse the response
                response_body = json.loads(response['body'].read())
                content = response_body['content'][0]['text']
                
                # Extract JSON from the response
                # Try to find JSON in the response (Claude might add markdown formatting)
                content = content.strip()
                if content.startswith('```json'):
                    content = content[7:]
                if content.startswith('```'):
                    content = content[3:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()
                
                result = json.loads(content)
                
                # Extract and validate fields
                score = float(result.get('overall_alignment_score', 0.0))
                evidence_similarity = result.get('evidence_similarity', 'Low')
                missing = result.get('missing', 'None')
                recommendation = result.get('recommendation', 'No action required')
                
                # Validate evidence_similarity
                if evidence_similarity not in ['High', 'Medium', 'Low']:
                    evidence_similarity = 'Low'
                
                # Ensure score is in valid range
                score = max(0.0, min(1.0, score))
                
                return {
                    'overall_alignment_score': score,
                    'evidence_similarity': evidence_similarity,
                    'missing': missing if missing else 'None',
                    'recommendation': recommendation if recommendation else 'No action required'
                }
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                
                # Handle throttling with exponential backoff
                if error_code == 'ThrottlingException' or error_code == 'TooManyRequestsException':
                    if attempt < max_retries - 1:
                        # Calculate exponential backoff: 1s, 2s, 4s, 8s, 16s
                        delay = base_delay * (2 ** attempt)
                        print(f"  ⚠ Throttling detected for ID {requirement_id}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(delay)
                        continue
                    else:
                        # Final attempt failed
                        print(f"✗ AWS Bedrock throttling error for ID {requirement_id} after {max_retries} attempts")
                        return {
                            'overall_alignment_score': 0.0,
                            'evidence_similarity': 'Low',
                            'missing': f'Throttling error after {max_retries} retries',
                            'recommendation': 'Please retry later or reduce concurrent requests'
                        }
                else:
                    # Non-throttling error, don't retry
                    error_message = e.response['Error']['Message']
                    print(f"✗ AWS Bedrock error for ID {requirement_id}: {error_code} - {error_message}")
                    return {
                        'overall_alignment_score': 0.0,
                        'evidence_similarity': 'Low',
                        'missing': f'AWS Error: {error_code}',
                        'recommendation': 'Please check AWS credentials and permissions'
                    }
            except json.JSONDecodeError as e:
                # JSON parsing error - don't retry, return error
                print(f"Warning: Could not parse JSON response for ID {requirement_id}: {e}")
                return {
                    'overall_alignment_score': 0.0,
                    'evidence_similarity': 'Low',
                    'missing': 'Error parsing AI response',
                    'recommendation': 'Please review the response manually'
                }
            except Exception as e:
                # Other unexpected errors - don't retry
                print(f"✗ Unexpected error validating ID {requirement_id}: {e}")
                return {
                    'overall_alignment_score': 0.0,
                    'evidence_similarity': 'Low',
                    'missing': f'Error: {str(e)}',
                    'recommendation': 'Please review manually'
                }
        
        # Should not reach here, but handle just in case
        return {
            'overall_alignment_score': 0.0,
            'evidence_similarity': 'Low',
            'missing': 'Unexpected error: No response after retries',
            'recommendation': 'Please review manually'
        }


def derive_intent_match(score: float) -> str:
    """
    Derive intent_match from overall_alignment_score using threshold logic
    
    Args:
        score: Overall alignment score (0.0 to 1.0)
        
    Returns:
        "Yes", "Partial", or "No"
    """
    if score >= 0.75:
        return "Yes"
    elif score >= 0.20:
        return "Partial"
    else:
        return "No"


def calculate_optimal_workers(total_tasks: int) -> int:
    """
    Set max workers to 20 for maximum throughput.
    Throttling is handled via retry logic with exponential backoff.
    
    Args:
        total_tasks: Total number of validation tasks to process (not used, kept for compatibility)
        
    Returns:
        Fixed number of workers: 20
    """
    return 20


def sanitize_excel_sheet_name(name: str) -> str:
    """
    Sanitize a string to be used as an Excel sheet name.
    Removes invalid characters and truncates to Excel's 31 character limit.
    
    Excel does not allow these characters in sheet names: : \ / ? * [ ]
    
    Args:
        name: Original sheet name string
        
    Returns:
        Sanitized sheet name safe for Excel
    """
    # Replace invalid Excel characters
    invalid_chars = [':', '\\', '/', '?', '*', '[', ']']
    sanitized = name
    for char in invalid_chars:
        sanitized = sanitized.replace(char, '_')
    
    # Truncate to Excel's 31 character limit
    sanitized = sanitized[:31]
    
    # Remove trailing spaces/underscores that might cause issues
    sanitized = sanitized.rstrip(' _')
    
    # Ensure it's not empty (fallback)
    if not sanitized:
        sanitized = "Sheet1"
    
    return sanitized


def get_column_name(col) -> str:
    """Extract column name from single value or tuple (handles multi-level headers)"""
    if isinstance(col, tuple):
        # Join tuple elements, filtering out NaN/None
        return ' '.join(str(x) for x in col if pd.notna(x) and str(x).strip())
    return str(col)


def parse_excel_structure(df: pd.DataFrame, extracted_project_columns: Optional[Dict[str, int]] = None) -> Tuple[str, str, Dict[str, int]]:
    """
    Parse the Excel structure to identify:
    - ID column
    - Requirement Description column
    - Project/Customer columns with their Partner Response column indices
    
    Handles both single-level and multi-level headers.
    
    Args:
        df: DataFrame to parse
        extracted_project_columns: Optional pre-extracted project columns mapping (project_name -> column_index)
                                  from multi-row header extraction
    
    Returns:
        Tuple of (id_column, req_desc_column, project_columns_dict)
        where project_columns_dict maps project_name -> column_index (for Partner Response columns only)
    """
    columns = df.columns.tolist()
    
    # If we already have extracted project columns from multi-row headers, use them
    if extracted_project_columns:
        print("  Using extracted project columns from multi-row headers")
        
        # For multi-row headers, DataFrame columns are "Column_0", "Column_1", etc.
        # We need to find ID and Requirement Description by index, not name
        # Typically: Column 0 = ID, Column 1 = Requirement Description
        # But we should check the actual data to be sure
        
        # Default to column indices 0 and 1 (most common pattern)
        id_column = 0
        req_desc_column = 1
        
        # Try to verify by checking column names or first row data
        # Since columns are now "Column_0", "Column_1", etc., we use indices
        if len(columns) > 0:
            # Check if first column looks like ID (short values, alphanumeric)
            if len(df) > 0:
                sample_id = str(df.iloc[0, 0]) if len(df) > 0 else ""
                # If it doesn't look like an ID, try to find it
                # For now, assume standard pattern: ID in column 0, Description in column 1
                pass
        
        print(f"    ID column: Index {id_column}")
        print(f"    Requirement Description column: Index {req_desc_column}")
        
        return id_column, req_desc_column, extracted_project_columns
    
    # Debug: Print all columns
    print(f"\n  Analyzing {len(columns)} columns:")
    for i, col in enumerate(columns[:10]):  # Print first 10 columns
        print(f"    Column {i}: {get_column_name(col)}")
    if len(columns) > 10:
        print(f"    ... and {len(columns) - 10} more columns")
    
    # Find ID column (usually contains 'ID' or 'id')
    id_column = None
    for col in columns:
        col_name = get_column_name(col).lower().strip()
        # Look for ID column - should be short and contain 'id'
        if 'id' in col_name and len(col_name) < 20 and 'project' not in col_name and 'customer' not in col_name:
            id_column = col
            break
    
    if not id_column:
        # Try first column if it looks like an ID column (short, alphanumeric)
        first_col_name = get_column_name(columns[0]).strip()
        if len(first_col_name) < 30 and not 'requirement' in first_col_name.lower():
            id_column = columns[0]
            print(f"  Using first column as ID: {get_column_name(id_column)}")
        else:
            # Try second column
            if len(columns) > 1:
                second_col_name = get_column_name(columns[1]).strip()
                if len(second_col_name) < 30:
                    id_column = columns[1]
                    print(f"  Using second column as ID: {get_column_name(id_column)}")
                else:
                    id_column = columns[0]
                    print(f"Warning: No ID column found, using first column: {get_column_name(id_column)}")
            else:
                id_column = columns[0]
                print(f"Warning: No ID column found, using first column: {get_column_name(id_column)}")
    
    # Find Requirement Description column
    req_desc_column = None
    for col in columns:
        col_name = get_column_name(col).lower().strip()
        if 'requirement' in col_name and 'description' in col_name:
            req_desc_column = col
            break
    
    if not req_desc_column:
        print("  Checking alternative patterns for Requirement Description...")
        # Try alternative patterns
        for col in columns:
            if col == id_column:
                continue
            col_name = get_column_name(col).lower().strip()
            if 'requirement' in col_name:
                req_desc_column = col
                break
            elif 'description' in col_name and len(col_name) > 10:  # Longer description column
                req_desc_column = col
                break
        
        if not req_desc_column:
            # Use the column after ID if available
            id_index = columns.index(id_column) if id_column in columns else -1
            if id_index >= 0 and id_index + 1 < len(columns):
                req_desc_column = columns[id_index + 1]
                print(f"  Using column after ID as Requirement Description: {get_column_name(req_desc_column)}")
            elif len(columns) > 1:
                req_desc_column = columns[1]
                print(f"  Using second column as Requirement Description: {get_column_name(req_desc_column)}")
            else:
                print("  Warning: Could not find Requirement Description column")
    
    # Identify project/customer columns (single-row header case)
    # Find all Partner Response columns and map them to project names
    project_columns = {}
    
    # Find all Partner Response columns (skip Met? columns)
    for i, col in enumerate(columns):
        col_name = get_column_name(col).lower().strip()
        
        # Skip ID and Requirement Description columns
        if col == id_column or col == req_desc_column:
            continue
        
        # Skip "Met?" columns
        if ('met' in col_name or 'meet' in col_name) and 'partner' not in col_name:
            continue
        
        # Find Partner Response columns
        if 'partner' in col_name and 'response' in col_name:
            # Look backward to find project name
            project_name = None
            
            # Check previous columns for project name
            for j in range(i - 1, -1, -1):
                prev_col = columns[j]
                prev_col_name = get_column_name(prev_col).lower().strip()
                
                # Skip ID, Requirement Description, and Met? columns
                if prev_col == id_column or prev_col == req_desc_column:
                    break
                if 'met' in prev_col_name or 'meet' in prev_col_name:
                    continue
                if 'partner' in prev_col_name and 'response' in prev_col_name:
                    continue
                
                # Found potential project name
                project_name = get_column_name(prev_col).strip()
                break
            
            # If no project name found, use column index as fallback
            if not project_name:
                project_name = f"Project_{i}"
            
            # Map project name to column index
            project_columns[project_name] = i
            print(f"  ✓ Found project/customer: {project_name} -> Column {i} (Partner Response)")
    
    return id_column, req_desc_column, project_columns


def extract_project_columns_from_multi_row_headers(
    df: pd.DataFrame, 
    header_row_idx: int, 
    excel_path: str, 
    sheet_name: str
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Extract project information from multi-row headers WITHOUT combining them.
    
    Handles cases where Excel has grouped headers like:
    Row 1: ID | Requirement Description | SellerGeni #1 | SuperPe #2
    Row 2:    |                         | Met? | Partner Response | Met? | Partner Response
    
    This function:
    - Keeps headers separate (does NOT combine)
    - Extracts ONLY Partner Response columns
    - Maps project name (from row 1) to column index (from row 2 where value == "Partner Response")
    
    Args:
        df: DataFrame that may have multi-row headers
        header_row_idx: The detected header row index
        excel_path: Path to the Excel file
        sheet_name: Name of the sheet
        
    Returns:
        Tuple of (DataFrame with data starting from row 2, project_columns_dict)
        where project_columns_dict maps project_name -> column_index
    """
    # Check if we have "Unnamed" columns (indicator of multi-row headers)
    has_unnamed = any('unnamed' in str(col).lower() for col in df.columns)
    
    if not has_unnamed:
        # No multi-row headers detected, return as-is
        return df, {}
    
    print("  Detected multi-row headers (found 'Unnamed' columns)")
    print("  Extracting project columns from separate header rows...")
    
    try:
        # Read the raw Excel to get both header rows
        excel_file = pd.ExcelFile(excel_path)
        # Read first few rows without header to see raw data
        df_raw = pd.read_excel(excel_file, sheet_name=sheet_name, header=None, nrows=header_row_idx + 3)
        
        if len(df_raw) < header_row_idx + 2:
            print("  Warning: Not enough rows to extract headers")
            return df, {}
        
        # Get header row (row 1) and sub-header row (row 2)
        header_row_1 = df_raw.iloc[header_row_idx].values
        header_row_2 = df_raw.iloc[header_row_idx + 1].values if len(df_raw) > header_row_idx + 1 else [None] * len(header_row_1)
        
        # Ensure both arrays have the same length
        max_len = max(len(header_row_1), len(header_row_2), len(df.columns))
        header_row_1 = list(header_row_1) + [None] * (max_len - len(header_row_1))
        header_row_2 = list(header_row_2) + [None] * (max_len - len(header_row_2))
        
        # Extract project columns mapping
        # Format: project_name -> column_index (only for Partner Response columns)
        project_columns = {}
        current_project = None
        
        for i in range(max_len):
            row1_val = header_row_1[i] if i < len(header_row_1) else None
            row2_val = header_row_2[i] if i < len(header_row_2) else None
            
            # Update current project name from row 1
            if pd.notna(row1_val) and str(row1_val).strip():
                current_project = str(row1_val).strip()
            
            # Check row 2 for column role
            if pd.notna(row2_val):
                row2_val_str = str(row2_val).strip()
                
                # Skip "Met?" columns
                if 'met' in row2_val_str.lower() or 'meet' in row2_val_str.lower():
                    continue
                
                # Keep only "Partner Response" columns
                if 'partner' in row2_val_str.lower() and 'response' in row2_val_str.lower():
                    if current_project:
                        project_columns[current_project] = i
                        print(f"    ✓ Found project: {current_project} -> Column {i} (Partner Response)")
        
        # Re-read the data starting from after both header rows
        df_data = pd.read_excel(excel_file, sheet_name=sheet_name, header=None, 
                               skiprows=header_row_idx + 2, nrows=None)
        
        # Use simple column names (Column_0, Column_1, etc.) since we're tracking by index
        df_data.columns = [f"Column_{i}" for i in range(len(df_data.columns))]
        
        print(f"  ✓ Extracted {len(project_columns)} project(s) with Partner Response columns")
        
        return df_data, project_columns
        
    except Exception as e:
        print(f"  Warning: Error during header extraction: {e}")
        import traceback
        traceback.print_exc()
        return df, {}


def reconstruct_multi_row_headers(df: pd.DataFrame, header_row_idx: int, excel_path: str, sheet_name: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Wrapper function that calls extract_project_columns_from_multi_row_headers.
    Maintains backward compatibility while using new extraction logic.
    """
    return extract_project_columns_from_multi_row_headers(df, header_row_idx, excel_path, sheet_name)


def detect_header_row(excel_path: str, sheet_name: str, max_rows_to_check: int = 10) -> int:
    """
    Detect which row contains the actual column headers by looking for:
    - ID column
    - Requirement Description column
    - Partner Response columns
    
    Returns:
        Row index (0-based) that should be used as header, or 0 if not found
    """
    excel_file = pd.ExcelFile(excel_path)
    
    for header_row in range(max_rows_to_check):
        try:
            # Try reading with this row as header
            df_test = pd.read_excel(excel_file, sheet_name=sheet_name, header=header_row, nrows=5)
            
            if df_test.empty:
                continue
            
            columns = df_test.columns.tolist()
            found_id = False
            found_req_desc = False
            found_partner_response = False
            
            for col in columns:
                col_name = get_column_name(col).lower()
                # Check for ID column
                if 'id' in col_name and len(col_name) < 50:  # Short column name suggests it's a header
                    found_id = True
                # Check for Requirement Description
                if 'requirement' in col_name and 'description' in col_name:
                    found_req_desc = True
                # Check for Partner Response
                if 'partner' in col_name and 'response' in col_name:
                    found_partner_response = True
            
            # If we found key columns, this is likely the header row
            if found_id or (found_req_desc and found_partner_response):
                print(f"  Detected header row at index {header_row} (row {header_row + 1} in Excel)")
                return header_row
                
        except Exception:
            continue
    
    print(f"  Warning: Could not detect header row, using row 0 (first row)")
    return 0


def process_excel_file(excel_path: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Read and process the Excel file, extracting only Service Requirements sheet
    
    Args:
        excel_path: Path to the Excel file
        
    Returns:
        Tuple of (DataFrame containing Service Requirements sheet, project_columns_dict)
        where project_columns_dict maps project_name -> column_index for Partner Response columns
    """
    try:
        # Read all sheets first to check availability
        excel_file = pd.ExcelFile(excel_path)
        available_sheets = excel_file.sheet_names
        print(f"Available sheets: {available_sheets}")
        
        # Check for Service Requirements sheet
        if 'Service Requirements' not in available_sheets:
            print("✗ Error: 'Service Requirements' sheet not found in the Excel file.")
            print(f"Available sheets: {available_sheets}")
            sys.exit(1)
        
        # Detect the correct header row
        print(f"\nDetecting header row in 'Service Requirements' sheet...")
        header_row = detect_header_row(excel_path, 'Service Requirements')
        
        # Read Service Requirements sheet with detected header row
        print(f"\nReading 'Service Requirements' sheet...")
        df = pd.read_excel(excel_file, sheet_name='Service Requirements', header=header_row)
        
        # Check for multi-row headers and extract project columns
        has_unnamed = any('unnamed' in str(col).lower() for col in df.columns)
        project_columns = {}
        
        if has_unnamed:
            print(f"\nDetected potential multi-row headers, extracting project columns...")
            df, project_columns = reconstruct_multi_row_headers(df, header_row, excel_path, 'Service Requirements')
        else:
            print(f"✓ Single-row headers detected")
        
        print(f"✓ Loaded {len(df)} rows from 'Service Requirements' sheet")
        
        # Optionally read Common Requirements (for reference, not processed yet)
        if 'Common Requirements' in available_sheets:
            print(f"✓ Found 'Common Requirements' sheet (not processed in this version)")
        
        return df, project_columns
        
    except FileNotFoundError:
        print(f"✗ Error: File not found: {excel_path}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error reading Excel file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main function to process Excel file and validate requirements"""
    parser = argparse.ArgumentParser(
        description='Validate Partner Responses against Requirement Descriptions using AWS Bedrock'
    )
    parser.add_argument(
        'excel_file',
        type=str,
        help='Path to the Excel file (.xlsx) containing Service Requirements'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='validation_results.xlsx',
        help='Output Excel file path for results (default: validation_results.xlsx)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Excel Requirement Validation Tool")
    print("Using AWS Bedrock Claude 4.5 Haiku")
    print("=" * 80)
    
    # Process Excel file
    df, extracted_project_columns = process_excel_file(args.excel_file)
    
    # Parse structure
    print("\n" + "=" * 80)
    print("Parsing Excel structure...")
    print("=" * 80)
    id_column, req_desc_column, project_columns = parse_excel_structure(df, extracted_project_columns)
    
    print(f"\nIdentified columns:")
    print(f"  - ID Column: {id_column}")
    print(f"  - Requirement Description Column: {req_desc_column}")
    print(f"  - Projects/Customers found: {len(project_columns)}")
    
    if not project_columns:
        print("\n✗ Error: No project/customer columns found with Partner Response columns.")
        print("Please ensure the Excel file has columns following the pattern:")
        print("  Project/Customer Name | met? | Partner response")
        sys.exit(1)
    
    # Initialize validator
    print("\n" + "=" * 80)
    print("Initializing AWS Bedrock client...")
    print("=" * 80)
    validator = RequirementValidator(AWS_PROFILE, AWS_REGION, BEDROCK_MODEL_ID)
    
    # Prepare validation tasks
    print("\n" + "=" * 80)
    print("Preparing validation tasks...")
    print("=" * 80)
    
    validation_tasks = []
    for idx, row in df.iterrows():
        # Get requirement ID and description using column names/indices
        if isinstance(id_column, int):
            requirement_id = row.iloc[id_column] if id_column < len(row) else None
        else:
            requirement_id = row[id_column] if id_column in row.index else None
        
        if isinstance(req_desc_column, int):
            requirement_description = row.iloc[req_desc_column] if req_desc_column < len(row) else None
        else:
            requirement_description = row[req_desc_column] if req_desc_column in row.index else None
        
        # Skip if requirement description is empty
        if pd.isna(requirement_description) or not str(requirement_description).strip():
            print(f"⚠ Skipping ID {requirement_id}: Requirement description is empty")
            continue
        
        # Skip if requirement ID is empty
        if pd.isna(requirement_id) or not str(requirement_id).strip():
            print(f"⚠ Skipping row {idx + 1}: Requirement ID is empty")
            continue
        
        # Iterate over project columns (project_name -> column_index)
        for project_name, column_index in project_columns.items():
            # Get partner response using column index
            try:
                if column_index < len(row):
                    partner_response = row.iloc[column_index]
                else:
                    partner_response = None
            except (IndexError, KeyError):
                print(f"⚠ Warning: Column index {column_index} not found, skipping")
                continue
            
            validation_tasks.append({
                'requirement_id': str(requirement_id),
                'requirement_description': str(requirement_description),
                'project_name': project_name,
                'partner_response': str(partner_response) if partner_response is not None else ""
            })
    
    total_validations = len(validation_tasks)
    print(f"✓ Prepared {total_validations} validation tasks")
    
    # Set workers to 20 for maximum throughput (throttling handled via retry logic)
    max_workers = calculate_optimal_workers(total_validations)
    print(f"Using {max_workers} parallel workers (fixed at 20 for maximum throughput, throttling handled with retry)")
    
    # Process validations in parallel
    print("\n" + "=" * 80)
    print("Processing requirements (parallel)...")
    print("=" * 80)
    
    results = []
    completed = 0
    lock = threading.Lock()
    
    def process_validation(task):
        """Process a single validation task"""
        try:
            validation_result = validator.validate_response(
                requirement_id=task['requirement_id'],
                requirement_description=task['requirement_description'],
                partner_response=task['partner_response']
            )
            
            # Derive intent_match from score
            intent_match = derive_intent_match(validation_result['overall_alignment_score'])
            
            result = {
                'Requirement ID': task['requirement_id'],
                'Project/Customer': task['project_name'],
                'intent_match': intent_match,
                'evidence_similarity': validation_result['evidence_similarity'],
                'missing': validation_result['missing'],
                'recommendation': validation_result['recommendation']
            }
            
            with lock:
                completed = len(results) + 1
                results.append(result)
                print(f"  [{completed}/{total_validations}] Completed: {task['requirement_id']} - {task['project_name']}")
            
            return result
        except Exception as e:
            print(f"  ✗ Error processing {task['requirement_id']} - {task['project_name']}: {e}")
            return {
                'Requirement ID': task['requirement_id'],
                'Project/Customer': task['project_name'],
                'intent_match': 'No',
                'evidence_similarity': 'Low',
                'missing': f'Processing error: {str(e)}',
                'recommendation': 'Please review manually'
            }
    
    # Execute validations in parallel with retry logic
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_validation, task): task for task in validation_tasks}
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                task = futures[future]
                print(f"  ✗ Failed task {task['requirement_id']}: {e}")
    
    # Group results by project and save to Excel
    print("\n" + "=" * 80)
    print("Saving results...")
    print("=" * 80)
    
    # Group results by project
    projects_data = {}
    for result in results:
        project_name = result['Project/Customer']
        if project_name not in projects_data:
            projects_data[project_name] = []
        projects_data[project_name].append(result)
    
    # Create Excel writer with multiple sheets
    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        used_sheet_names = {}  # Track used sheet names to ensure uniqueness
        
        for project_name, project_results in projects_data.items():
            # Create DataFrame for this project
            df_project = pd.DataFrame(project_results)
            
            # Select and order columns as specified
            df_output = df_project[['Requirement ID', 'intent_match', 'evidence_similarity', 'missing', 'recommendation']].copy()
            
            # Sort by Requirement ID
            df_output = df_output.sort_values('Requirement ID')
            
            # Sanitize sheet name for Excel (remove invalid characters and truncate)
            base_sheet_name = sanitize_excel_sheet_name(project_name)
            
            # Ensure uniqueness by appending number if needed
            sheet_name = base_sheet_name
            counter = 1
            while sheet_name in used_sheet_names:
                # Append counter, but keep within 31 char limit
                suffix = f"_{counter}"
                max_base_len = 31 - len(suffix)
                sheet_name = sanitize_excel_sheet_name(project_name[:max_base_len]) + suffix
                counter += 1
            
            used_sheet_names[sheet_name] = True
            df_output.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"✓ Saved {len(df_output)} results for: {project_name} (sheet: {sheet_name})")
    
    print(f"\n✓ Results saved to: {args.output}")
    print(f"✓ Total validations completed: {len(results)}")
    print("\n✓ Validation complete!")


if __name__ == "__main__":
    main()

