import os
import shutil
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_file
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date
import re
from io import BytesIO

# Imports for PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Imports for Email Sending
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# LangChain + Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Added for the new advocate search feature
import json

# Disable ONNX embeddings in Chroma to avoid a dependency conflict
os.environ["CHROMADB_DEFAULT_EMBEDDING_FUNCTION"] = "null"

load_dotenv()

# ==================== Flask App Setup ====================
app = Flask(__name__)

# User authentication setup
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a-very-secret-key-that-you-should-change')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///user.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# RAG setup
UPLOAD_FOLDER = 'data'
DB_PATH = 'chroma_db'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)

# Email configuration
# Make sure to set these in your .env file
# GMAIL_EMAIL=your_email@gmail.com
# GMAIL_APP_PASSWORD=your_app_password
EMAIL_USER = os.environ.get('GMAIL_EMAIL')
EMAIL_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD')
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 465 # For SSL

# ==================== Database Model ====================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    address = db.Column(db.String(200), nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# ==================== RAG Functions ====================
rag_chain = None
vectorstore = None
llm = Ollama(model="llama3.2")
embedding_model = OllamaEmbeddings(model="llama3.2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)

def process_document_and_ingest_to_db(file_path):
    global rag_chain, vectorstore
    try:
        # Clear existing data
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
        
        # Load the document
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            return False # Unsupported file type

        documents = loader.load()
        docs = text_splitter.split_documents(documents)

        # Ingest into ChromaDB
        vectorstore = Chroma.from_documents(documents=docs, embedding=embedding_model, persist_directory=DB_PATH)

        # Set up the RAG chain
        rag_prompt_template = """
        You are a legal AI assistant. Your purpose is to provide legal information based on the context provided.
        Answer the question as concisely as possible based only on the following context.
        If you cannot find the answer in the context, do not make up an answer. Simply state that you cannot find the answer in the provided document.

        Context: {context}

        Question: {question}
        """
        rag_prompt = PromptTemplate.from_template(rag_prompt_template)
        
        rag_chain = (
            {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        return True
    except Exception as e:
        print(f"Error processing document: {e}")
        return False

# ==================== Search Advocates Function ====================
def search_advocates_data(location, query):
    """
    Simulates searching for advocates and returns a structured list.
    """
    search_results = [
        {"name": "Eman Rahim", "experience": "34 Years", "areas": ["Contracts", "Criminal", "Civil"], "phone": "+91 95854*****"},
        {"name": "Rama Subramanian Ammamuthu", "experience": "20 Years", "areas": ["real estate", "Commercial Law"], "phone": "+91 93456*****"},
        {"name": "Adv. Deepak Chandrakanth", "experience": "N/A", "areas": ["General Services", "Family Law"], "phone": "N/A"},
        {"name": "S.SHANMUGAM", "experience": "N/A", "areas": ["Corporate Law", "Commercial Law"], "phone": "+91 97876 75250", "email": "s.shanmugamadv84@gmail.com"},
        {"name": "S.MOHAMED YUNUS", "experience": "N/A", "areas": ["General Services", "Commercial Law"], "phone": "+91 88072 60676", "email": "smyunus.adv@gmail.com"},
        {"name": "K.SUSEELA", "experience": "N/A", "areas": ["Civil", "Family Law"], "phone": "+91 94455 77779", "email": "s.suseelaadv90@gmail.com"},
        {"name": "Advocate Nithiyanandan", "experience": "N/A", "areas": ["Corporate Law"], "phone": "N/A"},
        {"name": "Advocate A Prabu Armugam", "experience": "N/A", "areas": ["Corporate Law"], "phone": "N/A"}
    ]
    return search_results

# ==================== Routes ====================
@app.route('/')
def welcome():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        address = request.form.get('address')
        
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'error')
            return redirect(url_for('register'))

        new_user = User(username=username, address=address)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please sign in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Signed in successfully!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'error')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/home')
def home():
    if 'username' in session:
        return render_template('home.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/index')
def index():
    if 'username' in session:
        return render_template('index.html')
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
def upload_document():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    try:
        file.save(file_path)
        success = process_document_and_ingest_to_db(file_path)
        if success:
            return jsonify({'message': '✅ Document uploaded & processed successfully!'}), 200
        else:
            return jsonify({'error': 'Failed to process document. Use PDF or DOCX'}), 500
    except Exception as e:
        return jsonify({'error': f"Failed to save file: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def chat_with_ai():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    global rag_chain
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    if rag_chain is None:
        return jsonify({'response': '⚠️ Please upload a legal document first.'}), 200

    try:
        response = rag_chain.invoke(query)

        print(f"💬 Query: {query}")
        print(f"🤖 Response: {response}")

        if not response or response.strip() == "":
            response = "⚠️ No answer generated. Check if the LLM is running."

        return jsonify({'response': response}), 200
    except Exception as e:
        print(f"❌ Chat error: {e}")
        return jsonify({'response': f"❌ An error occurred: {str(e)}"}), 500

@app.route('/extract_clause', methods=['POST'])
def extract_clause():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    global rag_chain
    data = request.json
    clause_type = data.get('clause_type')

    if not clause_type:
        return jsonify({'response': '❌ Please provide a clause type to extract.'}), 400

    if rag_chain is None:
        return jsonify({'response': '⚠️ Please upload a legal document first.'}), 200
    
    try:
        # Construct a specific query for clause extraction
        extract_query = f"Extract the full text of the '{clause_type}' clause from the provided document. If multiple clauses exist, extract all of them. If the clause is not found, state that."
        
        response = rag_chain.invoke(extract_query)
        
        if not response or response.strip() == "":
            response = f"⚠️ The '{clause_type}' clause was not found in the document or the LLM failed to generate a response."
            
        return jsonify({'response': response}), 200
    
    except Exception as e:
        print(f"❌ Extraction error: {e}")
        return jsonify({'response': f"❌ Extraction error: {e}"}), 500

@app.route('/generate_document', methods=['POST'])
def generate_document():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.json
    doc_type = data.get('docType')
    
    if not doc_type:
        return jsonify({'document': '❌ Please provide the document type.'}), 400

    # Define a prompt template for generating a legal document
    prompt_template = """
    You are a legal document drafting AI. Your task is to provide a standardized template or boilerplate for a legal document.
    The template should be clearly structured with placeholders for variable information like names and addresses.
    Do not fill in the personal details provided by the user.

    Please provide a legal boilerplate template for a **{doc_type}**.
    Include the standard sections and clauses that would be found in such a document, using placeholders like [PARTY A NAME], [PARTY A ADDRESS], etc.
    """
    
    formatted_prompt = prompt_template.format(doc_type=doc_type)
    
    # Use the existing LLM to generate the document
    try:
        llm = Ollama(model="llama3.2")
        response = llm.invoke(formatted_prompt)

        # Add a note to the response
        final_document = f"**{doc_type} Boilerplate Template**\n\n{response}\n\n"
        final_document += "---"
        final_document += "\n\nDisclaimer: This is an AI-generated template. It is recommended to consult with a legal professional before using this document. You will need to manually replace the bracketed placeholders with the correct information."

        return jsonify({'document': final_document}), 200

    except Exception as e:
        print(f"❌ Document generation error: {e}")
        return jsonify({'document': f"❌ Failed to generate document: {e}"}), 500

@app.route('/download_pdf')
def download_pdf():
    doc_type = request.args.get('doc_type', 'Legal Document')
    content = request.args.get('content', 'No content available.')
    
    buffer = BytesIO()
    
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    flowables = []
    flowables.append(Paragraph(f"<b>{doc_type}</b>", styles['Title']))
    flowables.append(Paragraph("<br/>", styles['Normal']))
    
    for line in content.split('\n'):
        flowables.append(Paragraph(line, styles['Normal']))
    
    doc.build(flowables)
    
    buffer.seek(0)
    
    return send_file(buffer, as_attachment=True, download_name=f"{doc_type.replace(' ', '_')}.pdf", mimetype='application/pdf')

@app.route('/find_advocates', methods=['POST'])
def find_advocates():
    data = request.json
    location = data.get('location')
    specialty = data.get('specialty')

    if not location:
        return jsonify({'error': 'Please provide a location.'}), 400
    
    query = f"advocates in {location}"
    if specialty:
        query = f"{specialty} {query}"

    advocates = search_advocates_data(location, query)
    
    return jsonify({'advocates': advocates}), 200

# NEW ROUTE: This route drafts the letter and sends it back for review
@app.route('/draft_legal_mail', methods=['POST'])
def draft_legal_mail():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    issue = data.get('issue')
    sender_name = data.get('sender_name')
    sender_address = data.get('sender_address')
    
    if not issue or not sender_name or not sender_address:
        return jsonify({'error': 'Please provide all required information.'}), 400

    # Revised prompt to include sender details and request a formal letter
    drafting_prompt = f"""
    You are a professional business correspondent. Draft a formal and strongly-worded letter of complaint based on the following issue.
    The letter should begin with the sender's details. It should clearly state the problem, the required actions from the recipient, and a reasonable deadline for a response.

    Sender's Name: {sender_name}
    Sender's Address: {sender_address}

    Problem: {issue}
    """
    
    try:
        llm = Ollama(model="llama3.2")
        letter_body = llm.invoke(drafting_prompt)
        
        # Add a clear disclaimer to the generated content
        disclaimer = "\n\n---"
        disclaimer += "\n\nDisclaimer: This is an AI-generated draft. It is for informational purposes only and does not constitute legal advice. Please review the content carefully and consult a legal professional before sending."
        
        final_letter = letter_body + disclaimer
        
        return jsonify({'letter_content': final_letter}), 200

    except Exception as e:
        print(f"❌ Drafting error: {e}")
        return jsonify({'error': f'Failed to draft letter: {str(e)}'}), 500

# MODIFIED ROUTE: This route now only sends the mail
@app.route('/send_legal_mail', methods=['POST'])
def send_legal_mail():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.json
    recipient_email = data.get('recipient_email')
    letter_content = data.get('letter_content')
    
    if not recipient_email or not letter_content:
        return jsonify({'error': 'Email content or recipient is missing.'}), 400

    if not EMAIL_USER or not EMAIL_PASSWORD:
        return jsonify({'error': 'Email credentials not configured on the server.'}), 500
    
    try:
        # Step 1: Construct the email message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = recipient_email
        msg['Subject'] = "URGENT ATTENTION: Formal Complaint Regarding a Problem"
        
        # Attach the body of the legal notice
        msg.attach(MIMEText(letter_content, 'plain'))
        
        # Step 2: Send the email via SMTP
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USER, recipient_email, msg.as_string())
        
        return jsonify({'message': '✅ Your formal letter has been sent successfully!'}), 200

    except Exception as e:
        print(f"❌ Email sending error: {e}")
        return jsonify({'error': f'Failed to send email: {str(e)}'}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)