#Python code for structured data extraction from SAP F110 reports using Google Gemini API with strict schema enforcement and robust error handling. The code is designed to be run in an enterprise environment, with logging, retries, and clean output generation.
import os
import re
import json
import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from google import genai
from google.genai import types

# ===========================================================================
# 1. Logging Configuration
# ===========================================================================
# We create a structured logs directory to track runs and API latency.
# This log file is useful for auditing and debugging inside enterprise runners like UiPath.
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===========================================================================
# 2. Environment Variable Setup
# ===========================================================================
# Load standard environment configurations (.env)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")

# Adjust this threshold if you experience rate limits (TPM) or payload constraints.
# 5 suppliers per chunk is a reliable default for structured financial documents.
MAX_SUPPLIERS_PER_CHUNK = int(os.getenv("MAX_SUPPLIERS_PER_CHUNK", "5"))

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is missing!")

# Initialize the official Google GenAI Client
client = genai.Client(api_key=GEMINI_API_KEY)


# ===========================================================================
# 3. Pydantic Schemas for Strict Structured Schema Generation
# ===========================================================================
# By using Pydantic models with the response_schema config parameter, we force 
# Gemini to constrain its raw output token generation to match our exact schema.
# This prevents parsing failures caused by random markdown or syntax variations.

class DocumentLine(BaseModel):
    """
    Represents an individual document or exception row nested under a supplier.
    Each field contains description tags to help Gemini align visual text coordinates 
    with the accurate schema key.
    """
    busA: Optional[str] = Field(description="Business Area (BusA). Keep as null if not present.")
    company_code: Optional[str] = Field(description="Company Code (CoCd), usually 4 characters.")
    reference: Optional[str] = Field(description="Document reference number, e.g., BOA_DD_022024, 101010826.")
    document_type: Optional[str] = Field(description="Document Type (Type), e.g., RE, KG, KR, AB.")
    document_date: Optional[str] = Field(description="Document Date in DD.MM.YYYY formatting.")
    due_date: Optional[str] = Field(description="Due Date in DD.MM.YYYY formatting.")
    pay_t: Optional[str] = Field(description="Payment Terms (PayT), e.g., B014, B007.")
    pk: Optional[str] = Field(description="Posting Key (PK), e.g., 31, 21, 25.")
    currency: Optional[str] = Field(description="Transaction Currency (Crcy), e.g., SGD, USD.")
    
    # We keep amounts as strings during extraction to preserve custom trailing minus signs.
    # Conversion to float is performed safely during the Python clean-up phase.
    net_amount_fc: Optional[str] = Field(description="Net amount in Foreign Currency (FC). Keep trailing minus if present.")
    net_amount_lc: Optional[str] = Field(description="Net amount in Local Currency (LC). Keep trailing minus if present.")
    
    err: Optional[str] = Field(description="Error / Exception block code (Err), e.g., 016, 003, 099.")
    net: Optional[str] = Field(description="Net indicator value (Net), e.g., 0.")
    confidence_score: float = Field(description="Self-evaluation score (0.0 to 1.0) based on character legibility.")

class SupplierBlock(BaseModel):
    """
    Represents a Supplier block, grouping parent information with nested child line items.
    """
    supplier_number: Optional[str] = Field(description="10-digit ID after '--Supplier' prefix, e.g., '0020001498'.")
    supplier_name: Optional[str] = Field(description="Name of the company/payee listed inside the supplier box header.")
    supplier_address: Optional[str] = Field(description="Combined multiline address strings found inside the supplier box.")
    documents: List[DocumentLine] = Field(description="List of transaction entries, exceptions, or direct payments.")

class ExtractionResult(BaseModel):
    """
    Root structure containing array of extracted suppliers.
    """
    suppliers: List[SupplierBlock]


# ===========================================================================
# 4. Utility Functions & Cleansing Pipelines
# ===========================================================================
def clean_sap_number(val: Optional[str]) -> Optional[float]:
    """
    Parses financial string variations from SAP into floats.
    Handles thousands separators and shifts the trailing negative sign.
    
    Example input: '12,678.79-' -> float: -12678.79
    """
    if not val:
        return None
    val = str(val).strip()
    if not val:
        return None
    
    # Detect the SAP trailing minus sign indicating a negative ledger item
    is_negative = val.endswith('-')
    
    # Clean formatting anomalies (commas and minus signs) to make value parseable
    val = val.replace(',', '').replace('-', '').strip()
    
    try:
        num = float(val)
        return -num if is_negative else num
    except ValueError:
        logger.warning(f"Failed to cast financial string '{val}' to float.")
        return None

def chunk_sap_text(file_content: str, max_suppliers_per_chunk: int) -> List[str]:
    """
    Splits the SAP document by supplier boundaries to avoid splitting transactions.
    
    Uses positive lookahead re.split to isolate segments beginning with the
    pattern '--Supplier [10-digit number]'.
    """
    # Lookahead pattern splits text but retains the '--Supplier 1234567890' header 
    # at the start of each split array element.
    raw_chunks = re.split(r'(?=--Supplier \d{10}-+)', file_content)
    
    chunks = []
    current_chunk = ""
    supplier_count = 0
    
    for block in raw_chunks:
        if "--Supplier" in block:
            supplier_count += 1
            
        # Re-aggregate sub-blocks until we hit our maximum count threshold
        if supplier_count > max_suppliers_per_chunk:
            chunks.append(current_chunk)
            current_chunk = block
            supplier_count = 1  # Reset count to current block
        else:
            current_chunk += block
            
    # Append remaining block
    if current_chunk.strip():
        chunks.append(current_chunk)
        
    return chunks


# ===========================================================================
# 5. Gemini API Handler with Resiliency Retries
# ===========================================================================
# We use exponential backoff via tenacity to absorb temporary 429 Rate Limits
# or transient server issues. This is required for production enterprise environments.
@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def extract_data_with_gemini(text_chunk: str) -> dict:
    """
    Queries Gemini API using strict schema-enforced parameters.
    """
    # System and extraction rules sent along with the chunk.
    # Explaining the alignment of tabular indexes prevents column value shifting.
    prompt = f"""
    You are an expert SAP data extraction assistant.
    Extract the Supplier details and nested document transactions from this SAP F110 segment.

    Rules:
    1. Align document text visually into columns before parsing values (BusA, CoCd, Reference, net_amount_fc, etc.).
    2. Read line items sequentially. If an entry is an Exception, parse the error code in the 'err' property.
    3. Retain trailing negative signs ('-') for net amounts.
    4. Do not invent details. Return null if values are missing or unreadable.
    5. Skip top metadata headers, report legends, or final page summary blocks.

    Data Segment:
    {text_chunk}
    """

    # Structured schema validation config settings
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ExtractionResult,
        temperature=0.0  # Set to 0.0 to limit stylistic variation and hallucination
    )

    logger.info("Executing structured API extraction...")
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=config,
    )
    
    # Load and return parsed dictionary matching ExtractionResult
    return json.loads(response.text)


# ===========================================================================
# 6. Extraction Pipeline Pipeline Execution
# ===========================================================================
def main(input_path: str):
    """
    Main controller orchestration flow.
    Reads data, executes chunking, queries LLM API, flattens output and saves files.
    """
    os.makedirs("output", exist_ok=True)
    
    logger.info(f"Target extraction filepath: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        file_content = f.read()

    # Pre-process text to partition massive reports into reliable, isolated sub-blocks.
    logger.info("Initializing document chunking engine...")
    chunks = chunk_sap_text(file_content, MAX_SUPPLIERS_PER_CHUNK)
    logger.info(f"Document successfully chunked into {len(chunks)} blocks.")

    all_suppliers = []

    # Iterate and extract each chunk sequentially
    for i, chunk in enumerate(chunks):
        # We skip any preamble fragments that do not contain actual Supplier markers
        if "--Supplier" not in chunk:
            logger.info(f"Skipping index {i+1} as it does not contain a supplier boundary.")
            continue
            
        logger.info(f"Processing sequence chunk {i+1}/{len(chunks)}...")
        try:
            result_json = extract_data_with_gemini(chunk)
            all_suppliers.extend(result_json.get("suppliers", []))
        except Exception as e:
            logger.error(f"Critical error processing segment index {i+1} after maximum retries: {e}")

    # =======================================================================
    # 7. Flattening, Cleaning, and Export Processing
    # =======================================================================
    logger.info("Normalizing and flattening JSON dictionary tree into raw relational format...")
    flat_data = []
    
    for supplier in all_suppliers:
        sup_num = supplier.get("supplier_number")
        sup_name = supplier.get("supplier_name")
        sup_addr = supplier.get("supplier_address")
        
        # Unwind nested transaction rows to map 1-to-many relationship rows
        for doc in supplier.get("documents", []):
            flat_data.append({
                "Supplier_Number": sup_num,
                "Supplier_Name": sup_name,
                "Supplier_Address": sup_addr,
                "BusA": doc.get("busA"),
                "Company_Code": doc.get("company_code"),
                "Reference": doc.get("reference"),
                "Document_Type": doc.get("document_type"),
                "Document_Date": doc.get("document_date"),
                "Due_Date": doc.get("due_date"),
                "PayT": doc.get("pay_t"),
                "PK": doc.get("pk"),
                "Currency": doc.get("currency"),
                # Apply deterministic parsing to sanitize raw currency strings to system float values
                "Net_Amount_FC": clean_sap_number(doc.get("net_amount_fc")),
                "Net_Amount_LC": clean_sap_number(doc.get("net_amount_lc")),
                "Err_Code": doc.get("err"),
                "Net_Indicator": doc.get("net"),
                "Confidence_Score": doc.get("confidence_score")
            })

    # Save structured source JSON output file for traceability and validation
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    raw_json_path = f"output/extracted_raw_{timestamp}.json"
    with open(raw_json_path, "w") as f:
        json.dump({"suppliers": all_suppliers}, f, indent=4)
    logger.info(f"Archived structured raw JSON document at: {raw_json_path}")

    # Convert to standard Pandas DataFrame
    df = pd.DataFrame(flat_data)
    
    if not df.empty:
        csv_path = f"output/sap_extraction_{timestamp}.csv"
        excel_path = f"output/sap_extraction_{timestamp}.xlsx"
        
        # Write to system files
        df.to_csv(csv_path, index=False)
        df.to_excel(excel_path, index=False)
        
        logger.info(f"Pipeline executed successfully. Processed {len(df)} total line transactions.")
        logger.info(f"Target Destination (CSV): {csv_path}")
        logger.info(f"Target Destination (Excel): {excel_path}")
    else:
        logger.warning("No tabular rows were recovered. Check input structure or configuration values.")

if __name__ == "__main__":
    # Ensure input file exists before execution
    input_file = "input/input_sample.txt"
    if os.path.exists(input_file):
        main(input_file)
    else:
        logger.error(f"Execution terminated. Source document does not exist at: {input_file}")