from flask import Flask, request, jsonify, render_template
import os
import PyPDF2
import docx
from werkzeug.utils import secure_filename
import socket
import google.generativeai as genai
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Retrieve the Gemini API key from environment variables
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")

# Configure the Gemini API client
genai.configure(api_key=api_key)
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 5000,
}

model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    generation_config=generation_config,
    safety_settings=[
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "HIGH"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "HIGH"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "HIGH"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "HIGH"},
    ],
    system_instruction=(
        "Your name is CurriculumGen, an AI Education assistant and large language model created and trained by J I Technologies with the support from Google to provide assistance to teachers, including Lesson Plan Generation, Summarizing content, Research, help with quiz or test preparations, answer questions and more . "
        "JI Technologies is an emerging tech company aimed at bridging the gap in underserved regions by fostering individuals with practical skills and knowledge needed to thrive in the tech world, and also to aid a surge in Tech produced in Africa. Including Robotics, Electronics, Software development, AI and many more."
        "J I Technologies is founded by Jabez Mwewa, a Computer Engineer."
        "To find out more about Jabez or J I Technologies, you can contact them on their website."
        "You were trained and developed on 13th January 2025."
    ),
)


# Conversation history to maintain context
conversation_history = []


def net_check():
   
    try:
        socket.create_connection(("8.8.8.8", 53))
        return True
    except OSError:
        return False


def extract_table(response_text):
  
    lines = response_text.strip().split("\n")
    table_lines = []
    
    for line in lines:
        # Include lines with | and exclude lines that are too generic or instructional
        if "|" in line and "customize" not in line.lower() and "spreadsheet" not in line.lower():
            table_lines.append(line)

    if not table_lines:
        raise ValueError("No table detected in the response.")

    headers = [header.strip() for header in table_lines[0].split("|")]
    rows = [
        [cell.strip() for cell in row.split("|")]
        for row in table_lines[1:]
    ]
    
    return {"headers": headers, "rows": rows}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_pdf_text(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


def extract_docx_text(file_path):
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])


@app.route("/")
def index():
    """
    Render the main HTML page.
    """
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Handle POST requests from the chat interface and return AI-generated responses.
    """
    data = request.get_json()
    user_input = data.get("user_input", "").strip()

    if not user_input:
        return jsonify({"response": "Input cannot be empty."})

    if not net_check():
        return jsonify({"response": "Network error. Please check your connection."})

    try:
        # Add user input to conversation history
        conversation_history.append({"role": "user", "parts": [user_input + "\n"]})

        # Send the user's input to the AI model
        chat_session = model.start_chat(history=conversation_history)
        response = chat_session.send_message(user_input)
        response_text = response.text.strip()

        # Add the AI response to conversation history
        conversation_history.append({"role": "model", "parts": [response_text + "\n"]})

        # Enhanced table detection logic
        if (
            "timetable" in user_input.lower() 
            or "table form" in user_input.lower()
            or "tabular" in response_text.lower()  # Check AI response for table-like cues
        ):
            try:
                table_data = extract_table(response_text)
                return jsonify({"response": response_text, "table": table_data})
            except Exception:
                # Fallback if table extraction fails
                return jsonify({"response": response_text})

        # Return the plain response
        return jsonify({"response": response_text})

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})


@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"response": "No file part in the request."})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"response": "No file selected."})

    if file and allowed_file(file.filename):
        # Ensure the temporary uploads directory exists
        upload_folder = os.path.join(os.getcwd(), 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        # Save the file temporarily
        filename = secure_filename(file.filename)
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)

        try:
            if filename.endswith('.pdf'):
                file_content = extract_pdf_text(file_path)
            elif filename.endswith('.docx'):
                file_content = extract_docx_text(file_path)
            else:
                return jsonify({"response": "Unsupported file format."})

            # Remove the uploaded file after processing
            os.remove(file_path)

            # Send the file content as input to the AI model
            conversation_history.append({"role": "user", "parts": [file_content + "\n"]})
            chat_session = model.start_chat(history=conversation_history)
            response = chat_session.send_message(file_content)
            response_text = response.text.strip()
            conversation_history.append({"role": "model", "parts": [response_text + "\n"]})

            return jsonify({"response": response_text})
        except Exception as e:
            return jsonify({"response": f"Error processing file: {str(e)}"})
    else:
        return jsonify({"response": "Invalid file type. Only PDF and DOCX are allowed."})


if __name__ == "__main__":

    app.run(debug=True)
