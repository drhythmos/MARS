import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from PIL import Image
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

import fitz
import os
import uuid
import tempfile

import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence


# Initialize global objects only once to save resources
_blip_processor = None
_blip_model = None
_sr_recognizer = sr.Recognizer()

def get_temp_dir():
    temp_dir = os.path.join(tempfile.gettempdir(), 'multimodal_rag_temp')
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

def handle_uploaded_file(uploaded_file):
    suffix = uploaded_file.name.split(".")[-1]
    temp_dir = get_temp_dir()
    unique_id = str(uuid.uuid4())[:8]
    filename = f"upload_{unique_id}.{suffix}"
    filepath = os.path.join(temp_dir, filename)
    with open(filepath, 'wb') as f:
        f.write(uploaded_file.read())
    return filepath


def init_blip():
    global _blip_processor, _blip_model
    if _blip_processor is None or _blip_model is None:
        _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return _blip_processor, _blip_model


def process_images(image_path):
    # Setup Tesseract paths
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break
    else:
        # OCR engine missing
        return {
            "source_text": "",
            "aux": {"error": "OCR engine not available on this deployment."}
        }
    
    # OCR text extraction
    img = Image.open(image_path)
    ocr_text = pytesseract.image_to_string(img).replace("\x0c", "").strip()

    # Generate caption only for display/UI purposes
    processor, model = init_blip()
    inputs = processor(images=img, return_tensors="pt") # type: ignore
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return {
        "source_text": ocr_text,
        "aux": {
            "caption": caption,
            "has_text": bool(ocr_text),
            "image_path": image_path,
        }
    }


def process_pdf(pdf_path):
    pdf_text = []
    pdf_tables = []
    pdf_images = []

    doc = fitz.open(pdf_path)
    for page in doc:
        text = page.get_text()
        pdf_text.append(text)

        tables = page.find_tables()  # type: ignore
        for t in tables:
            extracted = t.extract()
            clean_rows = ["\t".join(map(str, row)) for row in extracted]
            clean_table = "\n".join(clean_rows)
            pdf_tables.append(clean_table)

    # Extract images separately to aux (not indexed)
    img_folder = os.path.join(get_temp_dir(), 'extracted-images')
    os.makedirs(img_folder, exist_ok=True)

    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)

            uid = uuid.uuid4().hex
            img_name = f"img_{page.number}_{xref}_{uid}.png"
            img_path = os.path.join(img_folder, img_name)

            if pix.n < 5:
                pix.save(img_path)
            else:
                pix = fitz.Pixmap(fitz.csRGB, pix)
                pix.save(img_path)

            del pix
            pdf_images.append({
                "page": page.number,
                "xref": xref,
                "path": img_path
            })

    combined_text = "\n".join(pdf_text).strip()

    return {
        "source_text": combined_text,
        "aux": {
            "tables": pdf_tables,
            "images": pdf_images,
            "page_count": len(doc)
        }
    }


def transcribe_audio(path):
    with sr.AudioFile(path) as source:
        audio_listened = _sr_recognizer.record(source)
        text = _sr_recognizer.recognize_google(audio_listened)  # type: ignore
    return text


def process_audio(audio_path):
    sound = AudioSegment.from_file(audio_path)
    chunks = split_on_silence(
        sound,
        min_silence_len=700,
        silence_thresh=sound.dBFS - 14,
        keep_silence=700,
    )

    folder_name = os.path.join(get_temp_dir(), 'audio-chunks')
    os.makedirs(folder_name, exist_ok=True)

    transcript = ""
    for i, chunk in enumerate(chunks, start=1):
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        chunk.export(chunk_filename, format="wav")
        try:
            text = transcribe_audio(chunk_filename)
        except sr.UnknownValueError:
            continue
        else:
            transcript += f"{text.capitalize()}. "

    return {
        "source_text": transcript.strip(),
        "aux": {
            "chunk_count": len(chunks),
            "audio_path": audio_path
        }
    }


def process_uploaded_file(uploaded_file):
    path = handle_uploaded_file(uploaded_file)
    mime = uploaded_file.type

    if mime.startswith("image/"):
        return process_images(path), uploaded_file.name
    elif mime.startswith("audio/"):
        return process_audio(path), uploaded_file.name
    elif mime == "application/pdf":
        return process_pdf(path), uploaded_file.name
    else:
        raise ValueError(f"Unsupported file type: {mime}")


# Initialize FAISS index and vector store
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embedding_dim = len(embedding_model.embed_query("hello world"))

index = faiss.IndexFlatL2(embedding_dim)
vector_store = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)


def chunk_text(text, chunk_size=800, chunk_overlap=150):
    if not text:
        return []
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap if end - chunk_overlap > start else end

    return chunks


def add_to_vector_store(data_dict, source_name):
    if not isinstance(data_dict, dict):
        raise TypeError(f"Expected dict for data_dict, got {type(data_dict)}")

    combined_text = data_dict.get("source_text", "")
    if not combined_text or not combined_text.strip():
        # Skip indexing if empty
        print(f"Skipping indexing for '{source_name}': no text found.")
        return []

    chunks = chunk_text(combined_text)

    if not chunks:
        print(f"Skipping indexing for '{source_name}': no chunks to index.")
        return []

    docs = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "source": source_name,
                "chunk_id": i,
                # We do NOT include captions or hallucinated data here
            }
        )
        docs.append(doc)

    vector_store.add_documents(docs)
    return docs


def retrieve_from_vector_store():
    return vector_store.as_retriever(search_kwargs={"k": 5})
