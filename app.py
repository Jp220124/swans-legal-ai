"""
Swans Legal AI — Police Report Verification App
Parses police report PDFs using Claude Vision API,
presents extracted data for human review, then pushes
approved data downstream (Make.com webhook → Clio).
v2.1 — Includes Matter ID field and filename-based client identification.
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

EXTRACTION_PROMPT = """You are a legal AI assistant for a personal injury law firm. Your job is to parse police accident reports and correctly identify the CLIENT (injured party the firm represents) vs the DEFENDANT (at-fault / adverse party).

CRITICAL — HOW TO IDENTIFY CLIENT vs DEFENDANT:
1. The CLIENT is the person who was STRUCK, HIT, or INJURED by the other party.
2. The DEFENDANT is the person who CAUSED the accident (ran a red light, failed to yield, rear-ended, made an improper turn, etc.).
3. In NYC MV-104 reports: read the narrative carefully — the driver described as causing the collision is the DEFENDANT. The other driver is the CLIENT.
4. Look for fault indicators in the narrative: "Vehicle 1 struck Vehicle 2" means Vehicle 2's driver is likely the CLIENT.
5. If the report header or title contains "X v Y" or "X vs Y", X is the CLIENT and Y is the DEFENDANT.
6. Count ALL injured persons mentioned anywhere in the report (including passengers).

Analyze the attached police report and extract these fields.
Return ONLY a valid JSON object with these exact keys (no markdown, no explanation):

{
  "client_name": "Full name of the CLIENT/injured party (FIRST LAST format)",
  "client_first_name": "CLIENT's first name",
  "client_last_name": "CLIENT's last name",
  "client_dob": "CLIENT's date of birth (MM/DD/YYYY)",
  "client_gender": "CLIENT's gender — Male or Female",
  "client_address": "CLIENT's full street address",
  "client_phone": "CLIENT's phone number if listed, otherwise empty string",
  "client_email": "",
  "client_vehicle_plate": "CLIENT's vehicle registration plate number",
  "defendant_name": "Full name of the DEFENDANT/at-fault party (FIRST LAST format)",
  "defendant_first_name": "DEFENDANT's first name",
  "defendant_last_name": "DEFENDANT's last name",
  "accident_date": "Date of the accident (MM/DD/YYYY)",
  "accident_location": "Full location including street, county/borough, city, and state",
  "direction_of_travel": "Direction the CLIENT's vehicle was traveling (Northbound, Southbound, Eastbound, or Westbound)",
  "number_of_injured": "Total number of injured persons in the accident (integer as string, '0' if none)",
  "accident_narrative": "The COMPLETE narrative/description from the officer's report — include every sentence",
  "police_report_number": "Official report or case number"
}

Rules:
- CLIENT = injured/victim party. DEFENDANT = at-fault/adverse party. Do NOT swap them.
- client_email is ALWAYS an empty string (police reports never contain emails).
- Convert all names from LAST, FIRST to FIRST LAST format.
- For number_of_injured, count every injured person mentioned (drivers + passengers).
- For direction_of_travel, report the CLIENT's travel direction specifically.
- Dates must be in MM/DD/YYYY format.
- If a field is not found, use an empty string "".
- Return ONLY the JSON object, nothing else."""


def parse_police_report(pdf_bytes: bytes, filename: str = "") -> dict:
    """Use Claude Vision API to extract structured data from a police report PDF."""
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

    # Build system message with filename context so Claude reads it FIRST
    system_msg = ""
    if filename:
        system_msg = (
            f"CRITICAL CONTEXT: The uploaded file is named \"{filename}\". "
            "The filename is set by the law firm and ALWAYS names their CLIENT first. "
            "If the filename follows 'FIRSTNAME_LASTNAME_v_FIRSTNAME_LASTNAME', "
            "the FIRST person is the CLIENT and the SECOND is the DEFENDANT. "
            "If the filename contains only ONE person's name (e.g., 'FIRSTNAME_LASTNAME_police_report'), "
            "that person IS the CLIENT — the other driver in the report is the DEFENDANT. "
            "This OVERRIDES any ambiguity in the police report narrative. "
            "You MUST assign the person named in the filename as the CLIENT."
        )

    api_kwargs = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": EXTRACTION_PROMPT,
                    },
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                    },
                ],
            }
        ],
    }
    if system_msg:
        api_kwargs["system"] = system_msg

    response = client.messages.create(**api_kwargs)

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
        try:
            sol = dt.replace(year=dt.year + 8)
        except ValueError:
            # Feb 29 leap year → target year not leap: fall back to Feb 28
            sol = dt.replace(year=dt.year + 8, day=28)
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
        extracted = parse_police_report(pdf_bytes, filename=file.filename or "")

        # Compute derived fields
        accident_date = extracted.get("accident_date", "")
        sol_date = compute_statute_of_limitations(accident_date)
        extracted["statute_of_limitations_date"] = sol_date
        extracted["calendly_link"] = get_seasonal_calendly_link()

        # Determine claim type based on injuries
        num_injured = extracted.get("number_of_injured", "0")
        try:
            has_injuries = num_injured and int(num_injured) > 0
        except (ValueError, TypeError):
            has_injuries = False
        if has_injuries:
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

    # Validate required fields
    matter_id = (data.get("matter_id") or "").strip()
    if not matter_id:
        return jsonify({"error": "Clio Matter ID is required"}), 400
    data["matter_id"] = matter_id

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
                "success": False,
                "message": f"Data approved but webhook delivery failed: {str(e)}",
                "data": data,
            }), 502
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
