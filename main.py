import os
import re
import json
import logging
import concurrent.futures
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
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash-lite")
MAX_SUPPLIERS_PER_CHUNK = int(os.getenv("MAX_SUPPLIERS_PER_CHUNK", "15"))
MAX_CONCURRENT_CALLS = int(os.getenv("MAX_CONCURRENT_CALLS", "5"))

if not GEMINI_API_KEY or GEMINI_API_KEY == "your_api_key_here":
    raise ValueError("GEMINI_API_KEY environment variable is missing or invalid!")

client = genai.Client(api_key=GEMINI_API_KEY)

# ===========================================================================
# 3. Pydantic Schemas for Strict Structured Schema Generation
# ===========================================================================
class DocumentLine(BaseModel):
    busA: Optional[str] = Field(description="Business Area (BusA). Keep as null if not present.")
    company_code: Optional[str] = Field(description="Company Code (CoCd), usually 4 characters.")
    reference: Optional[str] = Field(description="Document reference number, e.g., BOA_DD_022024, 101010826.")
    document_type: Optional[str] = Field(description="Document Type (Type), e.g., RE, KG, KR, AB.")
    document_date: Optional[str] = Field(description="Document Date in DD.MM.YYYY formatting.")
    due_date: Optional[str] = Field(description="Due Date in DD.MM.YYYY formatting.")
    pay_t: Optional[str] = Field(description="Payment Terms (PayT), e.g., B014, B007.")
    pk: Optional[str] = Field(description="Posting Key (PK), e.g., 31, 21, 25.")
    currency: Optional[str] = Field(description="Transaction Currency (Crcy), e.g., SGD, USD.")
    net_amount_fc: Optional[str] = Field(description="Net amount in Foreign Currency (FC). Keep trailing minus if present.")
    net_amount_lc: Optional[str] = Field(description="Net amount in Local Currency (LC). Keep trailing minus if present.")
    err: Optional[str] = Field(description="Error / Exception block code (Err), e.g., 016, 003, 099.")
    net: Optional[str] = Field(description="Net indicator value (Net), e.g., 0.")
    confidence_score: float = Field(description="Self-evaluation score (0.0 to 1.0) based on character legibility.")

class SupplierBlock(BaseModel):
    supplier_number: Optional[str] = Field(description="10-digit ID after '--Supplier' prefix, e.g., '0020001498'.")
    supplier_name: Optional[str] = Field(description="Name of the company/payee listed inside the supplier box header.")
    supplier_address: Optional[str] = Field(description="Combined multiline address strings found inside the supplier box.")
    documents: List[DocumentLine] = Field(description="List of transaction entries, exceptions, or direct payments.")

class ExtractionResult(BaseModel):
    suppliers: List[SupplierBlock]

# ===========================================================================
# 4. Utility Functions & Cleansing Pipelines
# ===========================================================================
def clean_sap_number(val: Optional[str]) -> Optional[float]:
    """Converts SAP financial string (e.g., '12,678.79-') to a standard float (-12678.79)."""
    if not val:
        return None
    val = str(val).strip()
    if not val:
        return None
    
    is_negative = val.endswith('-')
    val = val.replace(',', '').replace('-', '').strip()
    
    try:
        num = float(val)
        return -num if is_negative else num
    except ValueError:
        return None

def chunk_sap_text(file_content: str, max_suppliers: int) -> List[str]:
    """Intelligently splits the SAP text by Supplier blocks to avoid cutting transactions."""
    raw_chunks = re.split(r'(?=--Supplier \d{10}-+)', file_content)
    chunks, current_chunk = [], ""
    supplier_count = 0
    
    for block in raw_chunks:
        if "--Supplier" in block:
            supplier_count += 1
            
        if supplier_count > max_suppliers:
            chunks.append(current_chunk)
            current_chunk = block
            supplier_count = 1
        else:
            current_chunk += block
            
    if current_chunk.strip():
        chunks.append(current_chunk)
        
    return chunks

# ===========================================================================
# 5. Gemini API Handler with Resiliency Retries
# ===========================================================================
@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def extract_data_with_gemini(text_chunk: str) -> dict:
    """Calls the Gemini API and enforces JSON output matching our Pydantic schema."""
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

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ExtractionResult,
        temperature=0.0
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=config,
    )
    
    return json.loads(response.text)

# ===========================================================================
# 6. Extraction Pipeline (MULTITHREADED)
# ===========================================================================
def main(input_path: str):
    os.makedirs("output", exist_ok=True)
    
    logger.info(f"Reading file: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        file_content = f.read()

    logger.info("Chunking document...")
    chunks = chunk_sap_text(file_content, MAX_SUPPLIERS_PER_CHUNK)
    valid_chunks = [c for c in chunks if "--Supplier" in c]
    logger.info(f"Created {len(valid_chunks)} valid chunks.")

    all_suppliers = []

    # Thread worker function
    def process_chunk(payload):
        index, text_chunk = payload
        logger.info(f"Extracting Chunk {index}/{len(valid_chunks)}...")
        try:
            return extract_data_with_gemini(text_chunk)
        except Exception as e:
            logger.error(f"Chunk {index} failed after retries: {e}")
            return None

    # Execute API calls in parallel
    logger.info(f"Starting parallel execution with {MAX_CONCURRENT_CALLS} threads...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CALLS) as executor:
        payloads = [(i + 1, chunk) for i, chunk in enumerate(valid_chunks)]
        results = list(executor.map(process_chunk, payloads))
        
        for res in results:
            if res and "suppliers" in res:
                all_suppliers.extend(res["suppliers"])

    # =======================================================================
    # 7. Flattening, Cleaning, and Exporting
    # =======================================================================
    logger.info("Flattening JSON to tabular format...")
    flat_data = []
    
    for supplier in all_suppliers:
        sup_num = supplier.get("supplier_number")
        sup_name = supplier.get("supplier_name")
        sup_addr = supplier.get("supplier_address")
        
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
                "Net_Amount_FC": clean_sap_number(doc.get("net_amount_fc")),
                "Net_Amount_LC": clean_sap_number(doc.get("net_amount_lc")),
                "Err_Code": doc.get("err"),
                "Net_Indicator": doc.get("net"),
                "Confidence_Score": doc.get("confidence_score")
            })

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Export Raw JSON
    with open(f"output/extracted_raw_{timestamp}.json", "w") as f:
        json.dump({"suppliers": all_suppliers}, f, indent=4)
        
    # Export CSV & Excel
    df = pd.DataFrame(flat_data)
    if not df.empty:
        csv_path = f"output/sap_extraction_{timestamp}.csv"
        excel_path = f"output/sap_extraction_{timestamp}.xlsx"
        df.to_csv(csv_path, index=False)
        df.to_excel(excel_path, index=False)
        
        logger.info(f"Extraction successful! Processed {len(df)} document lines.")
        logger.info(f"Files saved in output/ directory.")
    else:
        logger.warning("No data extracted. Please check the input file.")

if __name__ == "__main__":
    input_file = "input/input_sample.txt"
    if os.path.exists(input_file):
        main(input_file)
    else:
        logger.error(f"File not found: {input_file}. Ensure you created the input folder and added the text file.")