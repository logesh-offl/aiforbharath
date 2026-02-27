import boto3
import json
import base64
import os
import time
import sys
import re
import concurrent.futures
from botocore.exceptions import ClientError
from qdrant_client import QdrantClient
from qdrant_client.models import SearchRequest
from pypdf import PdfReader

# --- CONFIGURATION ---
AWS_PROFILE = "hackathon_access-302263040839"
AWS_REGION = "us-east-1"
BEDROCK_MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0" 

# --- QDRANT & EMBEDDING CONFIG ---
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Nro76SpmjLw42zBBi-vxbwXZpnxnMX-a1Gat9SXPpew"
QDRANT_ENDPOINT = "https://6d0b685d-5433-42ee-930e-33280233b1f5.us-east-1-1.aws.cloud.qdrant.io"
QDRANT_COLLECTION_NAME = "Variable Repository"
TITAN_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIMENSION = 1024
MAPPING_CONFIDENCE_THRESHOLD = 0.5

# --- ADD THIS ---
VARIABLE_CACHE = {}  # Simple in-memory cache for this run

# OUTPUT FILE PATH
OUTPUT_FILE_PATH = "/Users/hemarubinij/Desktop/wonderlend_client/temp_check_1/output1.json"

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT_TEXT = """
SYSTEM / MAIN PROMPT (Generalized):
You are a rule-engine builder. Your job is to read ANY provided scheme PDF and generate:
1. Variable Repository (all base, derived, grid & aggregate variables)
2. Derived Variable Master (DV Master) (all required formulas)
3. Final WNBP Outputs
   * `final_wnbp_log`
   * `final_wnbp_iss`
4. Slabs (reward qualification tiers based on achievement thresholds)

STRICT REQUIREMENTS
* You must extract every rule, condition, gate, logic, slab, date, T&C from the PDF.
* Do not create separate variables for constant values extracted from the PDF. Use these values directly in formulas where needed (e.g., if ( Premium.premium >= 30000 ) instead of creating a ticket_size_threshold_cp variable).
* Hardcoded values can and should be used directly inside logic/formulas (e.g., if ( Persistency.previous_month_persistency_pct + 0.03 > 0.82 )).
* Use only these objects: `Agent`, `Policy`, `Premium`, `Persistency`
* CRITICAL: All field references MUST use Object.field_name format (e.g., Policy.login_date, Agent.Task_Start_Date, Premium.premium, Persistency.persistency_pct)
* Use nested IF–THEN–ELSE formulas exactly like this syntax:
if ( Policy.field_name = 1 ) then ( X ) else ( if ( Agent.field_name > 5 ) then ( Y ) else ( Z ) )

* Output only JSON with three keys:
{ "variable_repository": [...], "derived_variables": [...], "slabs": [...] }

✅ THE MODEL MUST FOLLOW THIS LOGIC
1️⃣ Extract these from the PDF:
* Contest names
* All login windows
* All issuance windows
* All WNBP/CP/AP/NOP metrics
* All product weightage rules
* All slabs and achievement %
* All FA Active rules
* All persistency gates
* All fallback rules
* All split policy rules
* All policy validity rules
* All cash/travel/white goods payout rules
* All monthly mode rules
* All TDS/settlement rules
* All reward conditions
* All exclusions
* All "only CV allowed" situations
* All hybrid disallowed/mode allowed conditions
* Any grid references
* Achievement % definition
* Anything that can affect logic or WNBP calculation
You must extract 100% of eligibility constraints.

2️⃣ Mandatory Weightage Logic (Always Include)
The model must ALWAYS apply:
A. ProductWeightageLogin Grid
field_credit = lookup from ProductWeightageLoginGrid

B. RnRWeightageIssue Grid
field_credit_pr= lookup from RnRWeightageIssueGrid

C. Fusion Weightage
Fusion must ALWAYS be applied:
fusion_rnr_wt at login stage 
fusion_rnr_wt_pr at issuance stage 
cp_after_fusion_login = Premium.premium * Policy.field_credit * Policy.fusion_rnr_wt 
cp_after_fusion_iss = Premium.premium * Policy.field_credit_pr * Policy.fusion_rnr_wt_pr

D. WNBP Computation
Use separate logic for login and issuance:
Login WNBP:(Weighted WNBP for all login policies)
wnbp_login

Issuance WNBP(Weighted WNBP for all issued policies)
wnbp_issuance

Use constant values directly in formulas. Do not create separate variables for thresholds like ticket sizes.

3️⃣ Variable Repository Requirements
For every policy and agent rule identified, create variables with:
* variable_name
* object (Agent/Policy/Premium/Persistency)

CRITICAL FIELD NAMING RULES:
* Base variables should NOT have Object prefix in variable_name (e.g., "login_date" not "Policy.login_date")
* Derived variables should NOT have Object prefix in variable_name (e.g., "wnbp_login" not "Policy.wnbp_login")
* In formulas/logic, ALWAYS reference as Object.field_name (e.g., Policy.login_date, Agent.Task_Start_Date)
* Object field can be: Agent, Policy, Premium, Persistency

Must include at minimum:
Mandatory variable categories
* Policy-level validity checks
* Ticket size & CP/Term premium (use constants directly in formulas, not as separate variables)
* Split policy check
* Bounce/resubmitted rules
* Hybrid rules
* In-force/premium-realized rules
* Product type/name
* Payment mode rules
* Monthly mode premium limit (use constants directly in formulas)
* Login date & issuance date windows
* Product weightage
* RnR weightage
* Fusion weightage
* WNBP login & issuance
* Agent-level persistency gates
* Renewal fallback gates
* Leakage gate
* FA active counts
* Achievement %
* Slab qualification
Also include any new variables discovered inside the PDF.

DO NOT create separate derived variables for constant thresholds, percentages, or fixed values extracted from the PDF. Use them directly in the formulas with proper Object.field_name references.

4️⃣ DV Master Requirements
For each derived variable, output:
{ "dv_variable": "<name>", "logic": "<nested if formula OR aggregation>" }

CRITICAL: All field references in logic MUST use Object.field_name format.

Aggregation format:
Agg Operator: SUM / COUNT / MAX / MIN 
Agg Variable: Policy.var_name 
Agg Condition: [ Policy.field_name operator value ] AND [ Agent.field_name operator value ]

Formula format:
Nested IF–THEN–ELSE with bracket syntax and Object.field_name references:
if ( Policy.field_name = 1 ) then ( Premium.field_name + 5 ) else ( if ( Agent.field_name > 0 ) then ( X ) else ( Y ) )

Examples of proper field references in formulas:
- if ( Premium.premium >= 30000 ) then ( 1 ) else ( 0 )
- if ( Persistency.previous_month_persistency_pct + 0.03 > 0.82 ) then ( 1 ) else ( 0 )
- [ between ( Policy.login_date , Agent.Task_Start_Date , Agent.Task_End_Date ) ]
- [ NOT [ Policy.PRODUCT_DESCRIPTION_2 in ("ULIP") ] ]
- [ NOT [ Agent.DESIGNATION_CODE = "SELFAP" ] ]
- Premium.premium * Policy.field_credit * Policy.fusion_rnr_wt
- if ( monthdiff ( endofmonth ( monthadd ( Agent.Task_End_Date , -1 ) ) , Agent.Agent_Coding_Date ) < 0 ) then ( 0 ) else ( monthdiff ( endofmonth ( Agent.Task_End_Date ) , Agent.Agent_Coding_Date ) )

Date function examples with Object.field_name:
- endofmonth( monthadd( Agent.Task_End_Date, -1 ) )
- dateadd( Agent.Task_Start_Date, 7 )
- monthdiff( Agent.Task_End_Date, Agent.Agent_Coding_Date )
- [ between ( Policy.login_date , Agent.Task_Start_Date , endofmonth( monthadd( Agent.Task_End_Date, -1 ) ) ) ]

5️⃣ Final Output Variables (Mandatory)
final_wnbp_log = Agent.total_wnbp_login 
final_wnbp_iss = Agent.total_wnbp_issuance

Important: Eligibility must NOT change these totals. Only reward calculation should depend on eligibility.

6️⃣ Slab Generation Requirements (Mandatory)

After generating derived variables, you MUST generate slabs for reward qualification.

Slab Structure:
- Extract ruleset_name from PDF filename (without extension) + "_login" and "_issuance"
- Create slabs based on achievement thresholds found in PDF
- Include all eligibility conditions in ruleset_logic using Object.field_name references
- Use bracket syntax for conditions: [ Object.variable_name operator value ]
- Combine conditions with AND/OR operators

Each slab must include:
- ruleset_name: "<pdf_name>_login" or "<pdf_name>_issuance"
- slab_name: "Slab1", "Slab2", "Slab3", etc.
- slab_precedence: 1, 2, 3, etc. (higher precedence = higher achievement)
- ruleset_logic: Complete eligibility logic with all gates using Object.field_name
- cash_payout: Cash voucher amount from PDF
- gift: Reward description from PDF

Slab Precedence Rules:
- Slab1: Always "Yet To Qualify" (achievement below minimum threshold)
- Higher slabs: Ordered by increasing achievement % requirements
- Include persistency_qualification gate in all qualifying slabs
- Use >= for lower bound, < for upper bound in range conditions

Format ruleset_logic examples with Object.field_name:
- "[ Agent.final_wnbp_log < 100000 ]"
- "[ Agent.achievement_pct_login >= 1.10 ] AND [ Agent.achievement_pct_login < 1.25 ]"
- "[ Agent.achievement_pct_issuance >= 1.25 ] AND [ Agent.persistency_qualification = 1 ]"

✅ OBJECT.FIELD_NAME REFERENCE RULES (CRITICAL):
1. Base variables in variable_repository: variable_name = "login_date", object = "Policy"
2. In all formulas/logic: MUST use Policy.login_date (Object.field_name format)
3. Aggregation variables: Agg Variable: Policy.wnbp_login
4. Aggregation conditions: Agg Condition: [ Policy.login_date >= Agent.Task_Start_Date ]
5. Derived variables referencing other derived variables: Agent.wnbp_final_login (if it's agent-level)
6. Never use just [field_name] - always use [Object.field_name] in formulas
7. Date fields: Agent.Task_Start_Date, Agent.Task_End_Date, Policy.login_date, Policy.issuance_date
8. Premium fields: Premium.premium, Premium.PREMIUM_EXCLUDING_TAX
9. Persistency fields: Persistency.previous_month_persistency_pct, Persistency.sep25_13m_persistency_pct

✅ YOUR FINAL ANSWER MUST BE A SINGLE JSON OBJECT
{ "variable_repository": [...], "derived_variables": [...], "slabs": [...] }

No explanation. No lists. No markdown. No notes.
Only pure JSON.

"""


def print_timing(step_name, start_time):
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n[TIME] {step_name}: {duration:.4f} seconds")
    return end_time


def get_titan_embedding(text, bedrock_runtime):
    """Generates embedding using AWS Titan v2"""
    try:
        body = json.dumps({
            "inputText": text,
            "dimensions": EMBEDDING_DIMENSION,
            "normalize": True
        })
        response = bedrock_runtime.invoke_model(
            modelId=TITAN_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body
        )
        response_body = json.loads(response['body'].read())
        return response_body['embedding']
    except Exception as e:
        print(f"Error generating embedding for '{text}': {e}")
        return None


def search_variable_in_qdrant(query_text, qdrant_client, bedrock_runtime):
    """
    Searches Qdrant using embeddings with multiple fallback methods.
    Returns: (best_match_name, confidence_score)
    """
    # Generate Embedding
    vector = get_titan_embedding(query_text, bedrock_runtime)
    if not vector:
        return None, 0.0
    
    vector_results = []
    try:
        # Method 1: Try query_points with Query object (newer API)
        try:
            from qdrant_client.models import Query
            query_result = qdrant_client.query_points(
                collection_name=QDRANT_COLLECTION_NAME,
                query=Query(
                    vector=vector,
                    top=1
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
                timeout=30
            )
            # Extract points from QueryResult
            if hasattr(query_result, 'points'):
                vector_results = query_result.points
            elif hasattr(query_result, 'result') and hasattr(query_result.result, 'points'):
                vector_results = query_result.result.points
            elif isinstance(query_result, list):
                vector_results = query_result
        except (AttributeError, TypeError, ImportError, ValueError):
            # Method 2: Try query_points with direct vector (alternative format)
            try:
                query_result = qdrant_client.query_points(
                    collection_name=QDRANT_COLLECTION_NAME,
                    query=vector,
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                    timeout=30
                )
                if hasattr(query_result, 'points'):
                    vector_results = query_result.points
                elif isinstance(query_result, list):
                    vector_results = query_result
            except (AttributeError, TypeError):
                # Method 3: Try search_points method
                try:
                    vector_results = qdrant_client.search_points(
                        collection_name=QDRANT_COLLECTION_NAME,
                        query_vector=vector,
                        limit=1,
                        with_payload=True,
                        with_vectors=False,
                        timeout=30
                    )
                except (AttributeError, TypeError) as e:
                    raise Exception(f"All Qdrant query methods failed. Last error: {str(e)}")
        
        if vector_results:
            best_match = vector_results[0]
            # Extract payload and score
            if hasattr(best_match, 'payload') and hasattr(best_match, 'score'):
                payload = best_match.payload
                score = best_match.score
            elif isinstance(best_match, dict):
                payload = best_match.get('payload', best_match)
                score = best_match.get('score', 0.0)
            else:
                return None, 0.0
            
            # Get variable name from payload
            mapped_name = payload.get('variable_display_name', 
                                     payload.get('variable_name', 
                                                payload.get('name', 'Unknown')))
            return mapped_name, score
        
        return None, 0.0
    except Exception as e:
        print(f"Qdrant Search Error: {e}")
        return None, 0.0



def replace_variable_in_text(text, mapping_dict):
    """
    Replaces variables in a string (formulas, conditions).
    """
    if not text or not isinstance(text, str):
        return text
    
    # Sort by length descending to prevent partial replacements
    sorted_vars = sorted(mapping_dict.keys(), key=len, reverse=True)
    result = text
    
    for old_var in sorted_vars:
        new_var = mapping_dict[old_var]
        
        # UPDATED REGEX: Simple word boundary check.
        # We rely on the fact that we added explicit "Domain.Var" keys to mapping_dict
        pattern = r'\b' + re.escape(old_var) + r'\b'
        
        result = re.sub(pattern, new_var, result)
        
    return result


def batch_resolve_variables(variable_names, qdrant_client, bedrock_runtime):
    """
    Resolves a list of variable names using Parallel Execution (Threading).
    Reuses the robust 'search_variable_in_qdrant' logic to ensure compatibility.
    """
    # 1. Check Cache
    unique_vars = list(set(variable_names))
    to_process = [v for v in unique_vars if v not in VARIABLE_CACHE]
    
    if not to_process:
        return {v: VARIABLE_CACHE[v] for v in unique_vars if VARIABLE_CACHE.get(v)}
        
    print(f"Parallel processing {len(to_process)} variables...")

    # 2. Parallel Execution (Strategy 1: Multi-threading)
    # We use the existing 'search_variable_in_qdrant' function which handles 
    # both Embedding and Searching. Running it in threads speeds it up significantly.
    
    # max_workers=5 is safe for AWS Bedrock throttling limits
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        future_to_var = {
            executor.submit(search_variable_in_qdrant, var, qdrant_client, bedrock_runtime): var 
            for var in to_process
        }
        
        for future in concurrent.futures.as_completed(future_to_var):
            var_name = future_to_var[future]
            try:
                mapped_name, score = future.result()
                
                # Update Cache
                if mapped_name and score >= MAPPING_CONFIDENCE_THRESHOLD:
                    VARIABLE_CACHE[var_name] = mapped_name
                else:
                    VARIABLE_CACHE[var_name] = None
                    
            except Exception as e:
                print(f"Processing failed for {var_name}: {e}")
                VARIABLE_CACHE[var_name] = None

    # Return only successful mappings
    return {v: VARIABLE_CACHE[v] for v in unique_vars if VARIABLE_CACHE.get(v)}



def map_and_update_variables(json_data, bedrock_runtime):
    """
    Main logic to map variables via Qdrant and update the JSON structure.
    Maps variable_name from variable_repository and dv_variable from derived_variables.
    """
    print("\n--- Starting Variable Mapping with Qdrant ---")
    
    # 1. Initialize Qdrant Client
    try:
        qdrant_client = QdrantClient(url=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}")
        return json_data
    
    mapping_dict = {}
    
    # --- STEP A: COLLECT ALL VARIABLES FIRST ---
    all_vars_to_map = []
    var_repo_mapping_info = []  # Store (index, original_name, object) for variable_repository
    derived_var_mapping_info = []  # Store (index, original_name) for derived_variables
    
    # Collect variable_name from variable_repository
    if "variable_repository" in json_data and isinstance(json_data["variable_repository"], list):
        for idx, var_obj in enumerate(json_data["variable_repository"]):
            if isinstance(var_obj, dict) and "variable_name" in var_obj:
                var_name = var_obj["variable_name"]
                var_object = var_obj.get("object", "")
                all_vars_to_map.append(var_name)
                var_repo_mapping_info.append((idx, var_name, var_object))
    
    # Collect dv_variable from derived_variables
    if "derived_variables" in json_data and isinstance(json_data["derived_variables"], list):
        for idx, dv_obj in enumerate(json_data["derived_variables"]):
            if isinstance(dv_obj, dict) and "dv_variable" in dv_obj:
                dv_name = dv_obj["dv_variable"]
                all_vars_to_map.append(dv_name)
                derived_var_mapping_info.append((idx, dv_name))
    
    if not all_vars_to_map:
        print("No variables found to map.")
        return json_data
    
    # --- STEP B: BATCH RESOLVE (THE SPEED BOOST) ---
    resolved_map = batch_resolve_variables(all_vars_to_map, qdrant_client, bedrock_runtime)
    
    # --- STEP C: BUILD MAPPING DICT ---
    # Build mapping for variable_repository variables
    for idx, var_name, var_object in var_repo_mapping_info:
        mapped_name = resolved_map.get(var_name)
        if mapped_name:
            print(f"Mapping Variable Repository: {var_name} -> {mapped_name}")
            # Add mappings for both bare name and Object.name format
            mapping_dict[var_name] = mapped_name
            if var_object:
                mapping_dict[f"{var_object}.{var_name}"] = f"{var_object}.{mapped_name}"
    
    # Build mapping for derived_variables
    for idx, dv_name in derived_var_mapping_info:
        mapped_name = resolved_map.get(dv_name)
        if mapped_name:
            print(f"Mapping Derived Variable: {dv_name} -> {mapped_name}")
            # Add mappings for both bare name and Object.name format
            mapping_dict[dv_name] = mapped_name
            # Try to infer object from context (could be Agent, Policy, Premium, Persistency)
            # For now, add mappings for all possible objects
            for obj in ["Agent", "Policy", "Premium", "Persistency"]:
                mapping_dict[f"{obj}.{dv_name}"] = f"{obj}.{mapped_name}"
    
    if not mapping_dict:
        print("No variables mapped.")
        return json_data
    
    print(f"\nApplying mappings to {len(mapping_dict)} variables across the schema...")
    
    # --- STEP D: UPDATE VARIABLE NAMES IN STRUCTURES ---
    
    # Update variable_name in variable_repository
    if "variable_repository" in json_data and isinstance(json_data["variable_repository"], list):
        for idx, var_name, var_object in var_repo_mapping_info:
            mapped_name = resolved_map.get(var_name)
            if mapped_name and mapped_name != var_name:
                json_data["variable_repository"][idx]["variable_name"] = mapped_name
    
    # Update dv_variable in derived_variables
    if "derived_variables" in json_data and isinstance(json_data["derived_variables"], list):
        for idx, dv_name in derived_var_mapping_info:
            mapped_name = resolved_map.get(dv_name)
            if mapped_name and mapped_name != dv_name:
                json_data["derived_variables"][idx]["dv_variable"] = mapped_name
    
    # --- STEP E: REPLACE VARIABLES IN FORMULAS/LOGIC ---
    
    # Replace in variable_repository formulas/logic (if any)
    if "variable_repository" in json_data and isinstance(json_data["variable_repository"], list):
        for var_obj in json_data["variable_repository"]:
            if isinstance(var_obj, dict):
                # Check for any formula/logic fields
                for field in ["formula", "logic", "condition", "default_value"]:
                    if field in var_obj and isinstance(var_obj[field], str):
                        var_obj[field] = replace_variable_in_text(var_obj[field], mapping_dict)
    
    # Replace in derived_variables logic
    if "derived_variables" in json_data and isinstance(json_data["derived_variables"], list):
        for dv_obj in json_data["derived_variables"]:
            if isinstance(dv_obj, dict) and "logic" in dv_obj:
                if isinstance(dv_obj["logic"], str):
                    dv_obj["logic"] = replace_variable_in_text(dv_obj["logic"], mapping_dict)
    
    # Also replace in slabs if they reference variables
    if "slabs" in json_data and isinstance(json_data["slabs"], list):
        for slab in json_data["slabs"]:
            if isinstance(slab, dict):
                if "ruleset_logic" in slab and isinstance(slab["ruleset_logic"], str):
                    slab["ruleset_logic"] = replace_variable_in_text(slab["ruleset_logic"], mapping_dict)
                # Check other fields that might contain formulas
                for field in ["formula", "condition", "logic"]:
                    if field in slab and isinstance(slab[field], str):
                        slab[field] = replace_variable_in_text(slab[field], mapping_dict)
    
    return json_data


def analyze_insurance_pdf(pdf_file_path):
    print(f"--- Starting Analysis for: {pdf_file_path} ---")
    overall_start = time.time()
    current_time = overall_start

    # 1. Initialize Session
    try:
        session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
        bedrock_runtime = session.client('bedrock-runtime')
        current_time = print_timing("AWS Session Initialization", current_time)
    except Exception as e:
        print(f"Error setting up AWS Session: {e}")
        return

    # 2. Read and Extract Text from PDF using PyPDF
    try:
        reader = PdfReader(pdf_file_path)
        extracted_text = ""
        for page in reader.pages:
            extracted_text += page.extract_text() + "\n"
        current_time = print_timing("PDF Read & Text Extraction", current_time)
    except FileNotFoundError:
        print(f"Error: File not found at {pdf_file_path}")
        return
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return

    # 3. Prepare Payload
    messages = [
        {
            "role": "user",
            "content": f"{SYSTEM_PROMPT_TEXT}\n\nHere is the input document content:\n{extracted_text}"
        },
        {
            "role": "assistant",
            "content": "{"
        }
    ]

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8192,
        "messages": messages,
        "temperature": 0.0
    })

    # 4. Invoke Model with Streaming
    print(f"Invoking model (Streaming): {BEDROCK_MODEL_ID}...\n")
    print("-" * 50)
    full_response_text = ""
    
    try:
        response = bedrock_runtime.invoke_model_with_response_stream(
            modelId=BEDROCK_MODEL_ID,
            body=body
        )
        
        stream = response.get('body')
        
        if stream:
            for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    chunk_obj = json.loads(chunk.get('bytes').decode())
                    if 'delta' in chunk_obj and 'text' in chunk_obj['delta']:
                        text_fragment = chunk_obj['delta']['text']
                        sys.stdout.write(text_fragment)
                        sys.stdout.flush()
                        full_response_text += text_fragment

        print("\n" + "-" * 50)
        current_time = print_timing("Model Processing & Streaming", current_time)
        
        # Add "{" back to the start since we pre-filled it in assistant message
        full_response_text = "{" + full_response_text
        
        # 5. Process JSON & Map Variables
        # Extract JSON first
        json_content = full_response_text
        if "```json" in full_response_text:
            try:
                parts = full_response_text.split("```json")
                json_part = parts[1].split("```")[0]
                json_content = json_part.strip()
            except:
                pass 
        
        try:
            json_data = json.loads(json_content)
            
            # --- PERFORM QDRANT MAPPING HERE ---
            mapped_json_data = map_and_update_variables(json_data, bedrock_runtime)
            current_time = print_timing("Qdrant Variable Mapping", current_time)
            
            # Save Mapped JSON
            save_output(mapped_json_data)
            
        except json.JSONDecodeError:
            print("Error: Failed to parse model output as JSON. Saving raw text.")
            save_raw_output(full_response_text)

    except ClientError as e:
        print(f"\nAWS Bedrock API Error: {e}")
    except Exception as e:
        print(f"\nGeneral Error: {e}")
    
    print(f"\n[TOTAL TIME]: {time.time() - overall_start:.4f} seconds")


def save_output(json_data):
    directory = os.path.dirname(OUTPUT_FILE_PATH)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            print(f"Error creating directory: {e}")
            return

    try:
        with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        print(f"✅ Success! Mapped Output saved to: {OUTPUT_FILE_PATH}")
    except Exception as e:
        print(f"Error writing to file: {e}")


def save_raw_output(text_content):
    # Fallback if JSON parsing fails
    try:
        with open(OUTPUT_FILE_PATH.replace(".json", ".txt"), 'w', encoding='utf-8') as f:
            f.write(text_content)
    except:
        pass


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # REPLACE THIS WITH THE PATH TO YOUR INPUT PDF
    PDF_INPUT_PATH = "/Users/hemarubinij/Desktop/wonderlend_client/Wonderlend 2/Sparkling September Contest for ACLH & CLH (DD) - Sep'25.pdf"
    
    if os.path.exists(PDF_INPUT_PATH):
        analyze_insurance_pdf(PDF_INPUT_PATH)
    else:
        print(f"Please update the 'PDF_INPUT_PATH' variable in the script. File not found: {PDF_INPUT_PATH}")
