"""
Swans Legal AI — Police Report Verification App
Parses police report PDFs using Claude Vision API,
presents extracted data for human review, then pushes
approved data downstream (Make.com webhook → Clio).
"""

import os
import json
import base64
import re
import requests
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MAKE_WEBHOOK_URL = os.getenv("MAKE_WEBHOOK_URL", "")

EXTRACTION_PROMPT = """You are a legal AI assistant specializing in parsing police accident reports.

Analyze the attached police report document and extract the following fields.
Return ONLY a valid JSON object with these exact keys (no markdown, no explanation):

{
  "client_name": "Full name of the client/victim (LAST, FIRST format → convert to FIRST LAST)",
  "client_first_name": "First name only",
  "client_last_name": "Last name only",
  "client_dob": "Date of birth (MM/DD/YYYY)",
  "client_gender": "Male or Female",
  "client_address": "Full street address",
  "client_phone": "Phone number if available, otherwise empty string",
  "client_email": "Email if available, otherwise empty string",
  "client_vehicle_plate": "Vehicle registration plate number",
  "defendant_name": "Full name of the adverse party (FIRST LAST format)",
  "defendant_first_name": "First name of defendant",
  "defendant_last_name": "Last name of defendant",
  "accident_date": "Date of the accident (MM/DD/YYYY)",
  "accident_location": "Full location/address of the accident",
  "direction_of_travel": "Direction client was traveling (e.g., Northbound)",
  "number_of_injured": "Number of people injured (integer as string, '0' if none)",
  "accident_narrative": "Full narrative/description of what happened from the report",
  "police_report_number": "Report/case number if available"
}

Important rules:
- Convert names from LAST, FIRST to FIRST LAST format
- If a field is not found in the document, use an empty string ""
- For accident_narrative, include the full narrative text from the report
- Dates should be in MM/DD/YYYY format
- Return ONLY the JSON object, nothing else"""


def parse_police_report(pdf_bytes: bytes) -> dict:
    """Use Claude Vision API to extract structured data from a police report PDF."""
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": EXTRACTION_PROMPT,
                    },
                ],
            }
        ],
    )

    text_block = next(b for b in response.content if b.type == "text")
    raw_text = text_block.text.strip()

    # Try to extract JSON from the response
    # Remove markdown code fences if present
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

    return json.loads(raw_text)


def compute_statute_of_limitations(accident_date_str: str) -> str:
    """Compute statute of limitations date (accident date + 8 years for NY PI)."""
    try:
        dt = datetime.strptime(accident_date_str, "%m/%d/%Y")
        sol = dt.replace(year=dt.year + 8)
        return sol.strftime("%m/%d/%Y")
    except (ValueError, TypeError):
        return ""


def get_seasonal_calendly_link() -> str:
    """Return the correct seasonal Calendly link based on current date."""
    month = datetime.now().month
    if 3 <= month <= 8:
        return "https://calendly.com/swans-santiago-p/spring-summer"
    else:
        return "https://calendly.com/swans-santiago-p/winter-autumn"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/parse", methods=["POST"])
def api_parse():
    """Accept PDF upload and return extracted data."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a PDF file"}), 400

    if not ANTHROPIC_API_KEY:
        return jsonify({"error": "ANTHROPIC_API_KEY not configured"}), 500

    try:
        pdf_bytes = file.read()
        extracted = parse_police_report(pdf_bytes)

        # Compute derived fields
        accident_date = extracted.get("accident_date", "")
        sol_date = compute_statute_of_limitations(accident_date)
        extracted["statute_of_limitations_date"] = sol_date
        extracted["calendly_link"] = get_seasonal_calendly_link()

        # Determine claim type based on injuries
        num_injured = extracted.get("number_of_injured", "0")
        if num_injured and int(num_injured) > 0:
            extracted["claim_type"] = "Bodily Injury & Property Damage"
        else:
            extracted["claim_type"] = "Property Damage Only"

        return jsonify({"success": True, "data": extracted})

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Failed to parse AI response as JSON: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Error processing PDF: {str(e)}"}), 500


@app.route("/api/approve", methods=["POST"])
def api_approve():
    """Accept verified data and forward to Make.com webhook."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Add metadata
    data["approved_at"] = datetime.now().isoformat()
    data["approved_by"] = "Paralegal"
    data["calendly_link"] = get_seasonal_calendly_link()

    # If Make.com webhook is configured, forward the data
    if MAKE_WEBHOOK_URL:
        try:
            resp = requests.post(
                MAKE_WEBHOOK_URL,
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            return jsonify({
                "success": True,
                "message": "Data approved and sent to automation workflow",
                "webhook_status": resp.status_code,
            })
        except requests.RequestException as e:
            return jsonify({
                "success": True,
                "message": f"Data approved but webhook failed: {str(e)}",
                "data": data,
            })
    else:
        # No webhook configured — just return the approved data
        return jsonify({
            "success": True,
            "message": "Data approved successfully (no webhook configured — demo mode)",
            "data": data,
        })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
