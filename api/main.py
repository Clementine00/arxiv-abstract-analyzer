from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import torch

# --- App setup ---
app = FastAPI(title="arXiv Abstract Analyzer")

# --- Category labels (matching your 11 classes) ---
CATEGORY_LABELS = {
    0: "math.AC",
    1: "cs.CV",
    2: "cs.AI",
    3: "cs.SY",
    4: "math.GR",
    5: "cs.CE",
    6: "cs.PL",
    7: "cs.IT",
    8: "cs.DS",
    9: "cs.NE",
    10: "math.ST",
}

# --- Load classifier model at startup ---
tokenizer_cls = AutoTokenizer.from_pretrained("models/classifier")
model_cls = AutoModelForSequenceClassification.from_pretrained("models/classifier")
model_cls.eval()  # Put model in evaluation mode (turns off dropout etc.)

# --- Load summarizer model at startup ---
tokenizer_sum = AutoTokenizer.from_pretrained("models/summarizer")
model_sum = AutoModelForSeq2SeqLM.from_pretrained("models/summarizer")
model_sum.eval()

# --- Request/Response schemas ---
class AbstractRequest(BaseModel):
    abstract: str = Field(..., min_length=50, max_length=5000)

class ClassifyResponse(BaseModel):
    predicted_category: str
    confidence: float
    all_scores: dict

class SummarizeResponse(BaseModel):
    summary: str
    original_length: int
    summary_length: int

class AnalyzeResponse(BaseModel):
    predicted_category: str
    confidence: float
    all_scores: dict
    summary: str
    original_length: int
    summary_length: int

# --- Endpoints ---
@app.get("/")
def root():
    return {"message": "arXiv Abstract Analyzer is running"}

@app.post("/classify", response_model=ClassifyResponse)
def classify_abstract(request: AbstractRequest):
    # Tokenize the input
    inputs = tokenizer_cls(
        request.abstract,
        return_tensors="pt",
        max_length=256,
        truncation=True,
    )

    # Run inference (no_grad = don't track gradients, saves memory)
    with torch.no_grad():
        outputs = model_cls(**inputs)

    # Convert raw scores to probabilities
    probabilities = torch.softmax(outputs.logits, dim=1)[0]

    # Get the top prediction
    predicted_idx = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_idx].item()

    # Build scores dict for all categories
    all_scores = {
        CATEGORY_LABELS[i]: round(probabilities[i].item(), 4)
        for i in range(len(CATEGORY_LABELS))
    }

    return ClassifyResponse(
        predicted_category=CATEGORY_LABELS[predicted_idx],
        confidence=round(confidence, 4),
        all_scores=all_scores,
    )
@app.post("/summarize", response_model=SummarizeResponse)
def summarize_abstract(request: AbstractRequest):
    inputs = tokenizer_sum(
        request.abstract,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    )

    with torch.no_grad():
        summary_ids = model_sum.generate(
            inputs["input_ids"],
            max_length=60,
            min_length=20,
            num_beams=4,
            do_sample=False,
            early_stopping=True,
        )

    summary_text = tokenizer_sum.decode(summary_ids[0], skip_special_tokens=True)

    return SummarizeResponse(
        summary=summary_text,
        original_length=len(request.abstract.split()),
        summary_length=len(summary_text.split()),
    )

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_abstract(request: AbstractRequest):
    # Classification
    cls_inputs = tokenizer_cls(
        request.abstract,
        return_tensors="pt",
        max_length=256,
        truncation=True,
    )

    with torch.no_grad():
        cls_outputs = model_cls(**cls_inputs)

    probabilities = torch.softmax(cls_outputs.logits, dim=1)[0]
    predicted_idx = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_idx].item()
    all_scores = {
        CATEGORY_LABELS[i]: round(probabilities[i].item(), 4)
        for i in range(len(CATEGORY_LABELS))
    }

    # Summarization
    sum_inputs = tokenizer_sum(
        request.abstract,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    )

    with torch.no_grad():
        summary_ids = model_sum.generate(
            sum_inputs["input_ids"],
            max_length=60,
            min_length=20,
            num_beams=4,
            do_sample=False,
            early_stopping=True,
        )

    summary_text = tokenizer_sum.decode(summary_ids[0], skip_special_tokens=True)

    return AnalyzeResponse(
        predicted_category=CATEGORY_LABELS[predicted_idx],
        confidence=round(confidence, 4),
        all_scores=all_scores,
        summary=summary_text,
        original_length=len(request.abstract.split()),
        summary_length=len(summary_text.split()),
    )