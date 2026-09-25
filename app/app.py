
# ============================================================
# CONTEXT-AWARE GRAPH NEURAL NETWORK FOR BLOOD CELL MORPHOLOGY AND
# DISEASE ANALYSIS - FINAL YEAR PROJECT DASHBOARD
# ============================================================

import os
import json
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

from torchvision.models import resnet18

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfgen import canvas


# ============================================================
# Paths and Classes
# ============================================================

PROJECT_TITLE = "Context-Aware Graph Neural Network for Blood Cell Morphology and Disease Analysis"
PROJECT_DISCLAIMER = (
    "Student Final Year Project prototype for academic and educational purposes only. "
    "It is not a certified medical system, has not been clinically validated, and must not be used "
    "for diagnosis or medical decision-making. Anyone using or modifying it does so at their own risk. "
    "The student authors, supervisors, university, and contributors accept no responsibility or liability "
    "for its use or misuse."
)

# Use repository-relative paths by default. Set FYDP_APP_DIR and FYDP_PROJECT_DIR
# when running the dashboard with files stored somewhere else, such as Google Drive.
APP_DIR = Path(os.getenv("FYDP_APP_DIR", Path(__file__).resolve().parent))
PROJECT_DIR = Path(os.getenv("FYDP_PROJECT_DIR", Path(__file__).resolve().parents[1] / "models"))
REPORTS_DIR = APP_DIR / "reports"
os.makedirs(APP_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

CLASS_NAMES_PATH = f"{PROJECT_DIR}/class_names.json"
DEFAULT_CLASS_NAMES = ['BA', 'BNE', 'EO', 'ERB', 'LY', 'MMY', 'MO', 'MY', 'PLT', 'PMY', 'RBC', 'SNE']

if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, 'r') as f:
        CLASS_NAMES = json.load(f)
else:
    CLASS_NAMES = DEFAULT_CLASS_NAMES

NUM_CLASSES = len(CLASS_NAMES)

CELL_FULL_NAMES = {
    'BA': 'Basophil',
    'BNE': 'Band Neutrophil',
    'EO': 'Eosinophil',
    'ERB': 'Erythroblast',
    'LY': 'Lymphocyte',
    'MMY': 'Metamyelocyte',
    'MO': 'Monocyte',
    'MY': 'Myelocyte',
    'PLT': 'Platelet',
    'PMY': 'Promyelocyte',
    'RBC': 'Red Blood Cell',
    'SNE': 'Segmented Neutrophil'
}

CELL_DESCRIPTIONS = {
    'BA': 'A granulocytic white blood cell involved in immune and inflammatory responses.',
    'BNE': 'An immature neutrophil stage often associated with active immune response or infection-related changes.',
    'EO': 'A granulocytic white blood cell commonly linked with allergic response and parasitic infection response.',
    'ERB': 'An immature red blood cell precursor. Its presence may be relevant for erythropoietic activity assessment.',
    'LY': 'A lymphoid white blood cell involved in adaptive immune response.',
    'MMY': 'An immature myeloid precursor cell. Increased presence may support abnormal myeloid activity screening.',
    'MO': 'A large white blood cell involved in immune defense and phagocytosis.',
    'MY': 'An immature myeloid precursor cell. It can be relevant in screening abnormal blood cell maturation.',
    'PLT': 'A platelet fragment involved in clotting and hemostasis.',
    'PMY': 'A very immature myeloid precursor cell. Its detection can be important for leukemia-related screening.',
    'RBC': 'A red blood cell responsible for oxygen transport. It is relevant for anemia-related analysis.',
    'SNE': 'A mature segmented neutrophil involved in immune defense against infection.'
}

DISEASE_ANALYSIS = {
    'RBC': 'The detected cell belongs to the red blood cell lineage. This supports the framework applicability for anemia-related screening when analyzed with RBC morphology, count, and distribution patterns.',
    'ERB': 'The detected cell is an immature red blood cell precursor. This may be relevant for disease-level analysis of abnormal erythropoiesis and anemia-related hematological assessment.',
    'PMY': 'The detected cell is an early immature myeloid precursor. Detection of such immature cells is important for leukemia-related screening and abnormal myeloid maturation analysis.',
    'MY': 'The detected cell is an immature myeloid precursor. Its identification supports disease-level interpretation related to abnormal myeloid development and possible leukemia screening.',
    'MMY': 'The detected cell is a myeloid precursor stage. Increased occurrence of such cells may support screening of abnormal hematological conditions.',
    'BNE': 'The detected cell is an immature neutrophil stage. It can be useful for analyzing immune response and abnormal granulocytic activity.',
    'LY': 'The detected lymphocyte can support white blood cell pattern analysis. Abnormal lymphocyte distribution may be relevant for hematological disease screening.',
    'MO': 'The detected monocyte can support immune-cell distribution analysis and disease-level hematological interpretation.',
    'SNE': 'The detected mature neutrophil supports white blood cell differential analysis, useful for infection or immune-response-related interpretation.',
    'EO': 'The detected eosinophil supports differential blood cell analysis, especially for allergic or parasitic-response-related screening.',
    'BA': 'The detected basophil supports differential white blood cell analysis and immune-response-related interpretation.',
    'PLT': 'The detected platelet supports blood component analysis related to clotting and platelet distribution assessment.'
}

LAST_REPORT_DATA = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Image Processing and Graph Construction
# ============================================================

def segment_cell_mask(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Keep foreground as the cell region.
    if th.mean() > 127:
        th = 255 - th

    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    return th


def segment_nucleus_kmeans(img_bgr, min_ratio=0.01, max_ratio=0.75):
    cell_mask = segment_cell_mask(img_bgr)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    ys, xs = np.where(cell_mask > 0)
    if len(xs) < 80:
        return np.zeros_like(cell_mask)

    feats = np.stack([lab[ys, xs, 1], lab[ys, xs, 2], gray[ys, xs]], axis=1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.2)
    _, labels, centers = cv2.kmeans(feats, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

    # Nucleus is usually the darker cluster.
    nuc_cluster = int(np.argmin(centers[:, 2]))
    nuc_mask = np.zeros_like(cell_mask)
    pick = labels.flatten() == nuc_cluster
    nuc_mask[ys[pick], xs[pick]] = 255

    nuc_mask = cv2.morphologyEx(nuc_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    nuc_mask = cv2.morphologyEx(nuc_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(nuc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final = np.zeros_like(nuc_mask)

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        nucleus_area = cv2.contourArea(c)
        cell_area = float(np.count_nonzero(cell_mask))
        ratio = nucleus_area / (cell_area + 1e-6)
        if min_ratio < ratio < max_ratio:
            cv2.drawContours(final, [c], -1, 255, -1)

    return final


def region_features(img_bgr, mask):
    if mask is None or np.count_nonzero(mask) < 10:
        return np.zeros(12, dtype=np.float32)

    ys, xs = np.where(mask > 0)
    region = img_bgr[ys, xs]
    area = float(len(xs))
    H, W = img_bgr.shape[:2]
    area_norm = area / float(H * W + 1e-6)
    x_mean = float(xs.mean()) / float(W + 1e-6)
    y_mean = float(ys.mean()) / float(H + 1e-6)

    b_mean, g_mean, r_mean = region.mean(axis=0) / 255.0
    b_std, g_std, r_std = region.std(axis=0) / 255.0

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        perim = cv2.arcLength(c, True)
        contour_area = cv2.contourArea(c)
        perim_norm = float(perim) / float(2 * (H + W) + 1e-6)
        circularity = float(4 * np.pi * contour_area / ((perim ** 2) + 1e-6))
    else:
        perim_norm = 0.0
        circularity = 0.0

    return np.array([
        area_norm, x_mean, y_mean,
        b_mean, g_mean, r_mean,
        b_std, g_std, r_std,
        perim_norm, circularity,
        1.0,
    ], dtype=np.float32)


def build_cell_graph_with_masks(img_bgr):
    cell_mask = segment_cell_mask(img_bgr)
    nuc_mask = segment_nucleus_kmeans(img_bgr)

    cyto_mask = None
    if np.count_nonzero(nuc_mask) > 20:
        cyto_mask = cv2.bitwise_and(cell_mask, cv2.bitwise_not(nuc_mask))

    node_feats = []
    node_types = []

    node_feats.append(region_features(img_bgr, cell_mask))
    node_types.append(0)

    if np.count_nonzero(nuc_mask) > 20:
        node_feats.append(region_features(img_bgr, nuc_mask))
        node_types.append(1)
        if cyto_mask is not None and np.count_nonzero(cyto_mask) > 20:
            node_feats.append(region_features(img_bgr, cyto_mask))
            node_types.append(2)

    x = torch.tensor(np.stack(node_feats, axis=0), dtype=torch.float)
    n = x.size(0)
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j:
                edges.append([i, j])

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    graph = Data(x=x, edge_index=edge_index, y=torch.tensor([0], dtype=torch.long))
    graph.node_types = torch.tensor(node_types, dtype=torch.long)
    return graph, cell_mask, nuc_mask


# ============================================================
# Image Tensor Helper for CNN / Hybrid Model
# ============================================================

IMG_SIZE = 224
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def image_rgb_to_tensor(img_rgb, img_size=IMG_SIZE):
    img_rgb = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = img_rgb.astype(np.float32) / 255.0
    img = (img - IMAGE_MEAN) / IMAGE_STD
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float32)


# ============================================================
# Model Definitions Matching the Improved Notebook
# ============================================================

class GCNClassifier(nn.Module):
    def __init__(self, in_dim=12, hidden=96, num_classes=NUM_CLASSES, dropout=0.30):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.dropout = dropout
        self.lin = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        g = global_mean_pool(x, batch)
        return self.lin(g)


class OldGCNClassifier(nn.Module):
    """Fallback architecture for the previous dashboard model best_gnn.pt."""
    def __init__(self, in_dim=12, hidden=64, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        g = global_mean_pool(x, batch)
        out = self.lin(g)
        return out


def build_resnet18_backbone():
    backbone = resnet18(weights=None)
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feat_dim


class CNNClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.35):
        super().__init__()
        self.backbone, feat_dim = build_resnet18_backbone()
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, images):
        feat = self.backbone(images)
        return self.classifier(feat)


class GraphEncoder(nn.Module):
    def __init__(self, in_dim=12, hidden=96, dropout=0.25):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        g = global_mean_pool(x, batch)
        return g


class HybridCNNGNNClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.35):
        super().__init__()
        self.cnn_backbone, cnn_dim = build_resnet18_backbone()
        self.graph_encoder = GraphEncoder(in_dim=12, hidden=96, dropout=0.25)
        self.fusion = nn.Sequential(
            nn.Linear(cnn_dim + 96, 384),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, images, graphs):
        image_feat = self.cnn_backbone(images)
        graph_feat = self.graph_encoder(graphs)
        fused = torch.cat([image_feat, graph_feat], dim=1)
        return self.fusion(fused)


# ============================================================
# Load Best Available Model
# Priority: Hybrid → CNN → Balanced GNN → Old GNN
# ============================================================

def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def load_model_safely(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_best_available_model():
    candidates = [
        ("Hybrid CNN + GNN", "hybrid", f"{PROJECT_DIR}/best_hybrid_cnn_gnn.pt"),
        ("CNN ResNet18", "cnn", f"{PROJECT_DIR}/best_cnn_resnet18.pt"),
        ("Balanced GNN", "gnn", f"{PROJECT_DIR}/best_balanced_gnn.pt"),
        ("Previous Dashboard GNN", "old_gnn", f"{APP_DIR}/best_gnn.pt"),
    ]

    errors = []
    for display_name, model_type, path in candidates:
        if not os.path.exists(path):
            continue
        try:
            if model_type == "hybrid":
                model = HybridCNNGNNClassifier(num_classes=NUM_CLASSES, dropout=0.35)
            elif model_type == "cnn":
                model = CNNClassifier(num_classes=NUM_CLASSES, dropout=0.35)
            elif model_type == "gnn":
                model = GCNClassifier(in_dim=12, hidden=96, num_classes=NUM_CLASSES, dropout=0.30)
            else:
                model = OldGCNClassifier(in_dim=12, hidden=64, num_classes=NUM_CLASSES)

            model = load_model_safely(model, path)
            return model, model_type, display_name, path, []
        except Exception as e:
            errors.append(f"{display_name}: {str(e)}")

    return None, None, "No trained model found", None, errors


model, ACTIVE_MODEL_TYPE, ACTIVE_MODEL_DISPLAY, ACTIVE_MODEL_PATH, MODEL_LOAD_ERRORS = load_best_available_model()

print("Dashboard model status:", ACTIVE_MODEL_DISPLAY)
print("Model type:", ACTIVE_MODEL_TYPE)
print("Model path:", ACTIVE_MODEL_PATH)
print("Device:", device)
if MODEL_LOAD_ERRORS:
    print("Model load warnings:")
    for err in MODEL_LOAD_ERRORS:
        print("-", err)


# ============================================================
# PDF Report Generation
# ============================================================

class ReportFooterCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self.pages = []

    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        page_count = len(self.pages)
        for page in self.pages:
            self.__dict__.update(page)
            self.draw_footer(page_count)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(0.75 * inch, 0.45 * inch, "FYDP Blood Cell Classification Dashboard")
        self.drawRightString(7.5 * inch, 0.45 * inch, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def safe_text(value):
    if value is None:
        return ""
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def generate_pdf_report():
    if not LAST_REPORT_DATA:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = f"{REPORTS_DIR}/blood_cell_report_{timestamp}.pdf"

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.70 * inch,
        bottomMargin=0.70 * inch,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=26,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=10,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12,
    )

    section_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading2"],
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#1D4ED8"),
        spaceBefore=12,
        spaceAfter=7,
    )

    body_style = ParagraphStyle(
        "BodyTextClean",
        parent=styles["BodyText"],
        fontSize=10.2,
        leading=15,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6,
    )

    note_style = ParagraphStyle(
        "NoteStyle",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=6,
    )

    story = []

    pred_class = safe_text(LAST_REPORT_DATA.get("pred_class"))
    cell_name = safe_text(LAST_REPORT_DATA.get("cell_full_name"))
    confidence = float(LAST_REPORT_DATA.get("confidence", 0))
    disease_category = safe_text(LAST_REPORT_DATA.get("disease_category"))
    model_used = safe_text(LAST_REPORT_DATA.get("model_display"))
    image_type = safe_text(LAST_REPORT_DATA.get("image_type"))
    generated_at = safe_text(LAST_REPORT_DATA.get("generated_at"))
    cell_description = safe_text(LAST_REPORT_DATA.get("cell_description"))
    disease_note = safe_text(LAST_REPORT_DATA.get("disease_note"))

    if confidence >= 0.90:
        confidence_level = "High confidence prediction"
        confidence_explanation = (
            "The model prediction is strong because the predicted class has a high probability score. "
            "This indicates that the visual and morphological patterns of the uploaded cell are clearly aligned "
            "with the predicted blood cell class."
        )
    elif confidence >= 0.70:
        confidence_level = "Moderate confidence prediction"
        confidence_explanation = (
            "The model prediction is acceptable, but some visual similarity with other blood cell classes may exist. "
            "The result should be interpreted carefully, especially if the uploaded image quality is low."
        )
    else:
        confidence_level = "Low confidence prediction"
        confidence_explanation = (
            "The model confidence is low. This may happen due to poor image quality, unclear cell boundaries, "
            "overlapping cells, or similarity between multiple blood cell classes. Expert review is recommended."
        )

    story.append(Paragraph(PROJECT_TITLE, title_style))
    story.append(Paragraph(
        "AI Hematology Suite — Hybrid CNN-GNN FYDP Research Prototype",
        subtitle_style
    ))
    story.append(Paragraph(PROJECT_DISCLAIMER, note_style))

    story.append(Paragraph("1. Prediction Summary", section_style))

    summary_data = [
        ["Predicted Class Code", pred_class],
        ["Identified Cell", cell_name],
        ["Prediction Confidence", f"{confidence * 100:.2f}%"],
        ["Confidence Level", confidence_level],
        ["Disease Screening Category", disease_category],
        ["Model Used", model_used],
        ["Image Type", image_type],
        ["Generated At", generated_at],
    ]

    summary_table = Table(summary_data, colWidths=[2.25 * inch, 4.45 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EFF6FF")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1E3A8A")),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#0F172A")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)

    story.append(Paragraph("2. Cell-Level Morphological Interpretation", section_style))
    story.append(Paragraph(cell_description, body_style))
    story.append(Paragraph(
        "The prediction is based on visual and morphological characteristics observed from the uploaded blood cell image. "
        "These characteristics may include cell shape, cytoplasm appearance, nucleus pattern, granules, color distribution, "
        "and other microscopic image features learned by the trained model.",
        body_style
    ))

    story.append(Paragraph("3. Disease-Level Analysis", section_style))
    story.append(Paragraph(disease_note, body_style))
    story.append(Paragraph(
        "At disease level, this system does not depend on a single cell only. A stronger interpretation is achieved when "
        "multiple blood cells are analyzed together and their class distribution patterns are compared. For example, abnormal "
        "red blood cell related patterns may support anemia-related screening, while abnormal immature white blood cell patterns "
        "may support leukemia-related screening.",
        body_style
    ))

    story.append(Paragraph("4. Confidence Interpretation", section_style))
    story.append(Paragraph(confidence_explanation, body_style))

    story.append(Paragraph("5. Top 3 Model Predictions", section_style))
    top3_rows = [["Rank", "Class", "Confidence"]]
    for rank, (cls_name, prob) in enumerate(LAST_REPORT_DATA.get("top3", []), start=1):
        top3_rows.append([str(rank), safe_text(cls_name), f"{prob * 100:.2f}%"])

    top3_table = Table(top3_rows, colWidths=[0.8 * inch, 2.5 * inch, 1.7 * inch])
    top3_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(top3_table)

    story.append(Paragraph("6. Model and Framework Explanation", section_style))
    story.append(Paragraph(
        "The final system uses a Hybrid CNN-GNN approach. The CNN branch extracts deep visual image features such as "
        "texture, color, nucleus shape, cytoplasm pattern, and granules. The GNN branch models morphology-based structural "
        "relationships between important cell components such as the whole cell, nucleus, and cytoplasm. The fused features "
        "are then used for final blood cell classification.",
        body_style
    ))

    story.append(Paragraph("7. Clinical Note and Limitation", section_style))
    story.append(Paragraph(
        "This report is generated by a computational research prototype. It is intended for educational, research, and "
        "decision-support purposes only. It is not a final clinical diagnosis. Reliable disease-level decision making requires "
        "multiple cell images, complete blood count values, laboratory findings, patient history, and expert medical validation.",
        note_style
    ))

    story.append(Paragraph("8. Recommendation", section_style))
    story.append(Paragraph(
        "For stronger disease-level analysis, upload multiple clear cropped blood cell images from the same sample and compare "
        "the distribution of predicted classes. Low-quality, blurred, overlapping, or uncropped images may reduce prediction reliability.",
        body_style
    ))

    doc.build(story, canvasmaker=ReportFooterCanvas)
    return pdf_path


def download_pdf_report():
    pdf_path = generate_pdf_report()
    if pdf_path is None:
        return None, "Please run prediction first, then generate the PDF report."
    return pdf_path, f"PDF report generated successfully and saved in Google Drive:\n{pdf_path}"


# ============================================================
# Prediction Function
# ============================================================

def disease_category_for_class(pred_class):
    if pred_class in ["RBC", "ERB"]:
        return "Anemia-related hematological screening"
    if pred_class in ["PMY", "MY", "MMY", "BNE"]:
        return "Leukemia / abnormal myeloid maturation screening"
    if pred_class in ["LY", "MO", "SNE", "EO", "BA"]:
        return "White blood cell differential and immune-response analysis"
    if pred_class == "PLT":
        return "Platelet-related blood component analysis"
    return "General hematological analysis"


def predict_cell(image_rgb, image_type):
    global LAST_REPORT_DATA

    if image_rgb is None:
        return (
            "Please upload a blood cell image first.",
            {},
            "No prediction available.",
            None
        )

    if model is None:
        message = """
# Model Not Found

No trained model checkpoint was found yet.

Please train at least one model in the improved notebook first. The dashboard will automatically use the best available model in this order:

1. Hybrid CNN + GNN
2. CNN ResNet18
3. Balanced GNN
4. Previous dashboard GNN
"""
        if MODEL_LOAD_ERRORS:
            message += "\nModel loading warnings:\n" + "\n".join([f"- {e}" for e in MODEL_LOAD_ERRORS])
        return message, {}, "Model not loaded.", None

    try:
        img_rgb = np.array(image_rgb)

        if img_rgb.ndim == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)

        if img_rgb.shape[-1] == 4:
            img_rgb = img_rgb[:, :, :3]

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        graph, cell_mask, nuc_mask = build_cell_graph_with_masks(img_bgr)
        graph = Batch.from_data_list([graph]).to(device)

        image_tensor = image_rgb_to_tensor(img_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            if ACTIVE_MODEL_TYPE == "hybrid":
                logits = model(image_tensor, graph)
            elif ACTIVE_MODEL_TYPE == "cnn":
                logits = model(image_tensor)
            else:
                logits = model(graph)

            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        sorted_indices = np.argsort(probs)[::-1]

        cell_full_name = CELL_FULL_NAMES.get(pred_class, pred_class)
        cell_description = CELL_DESCRIPTIONS.get(pred_class, "No cell description available.")
        disease_note = DISEASE_ANALYSIS.get(pred_class, "Disease-level interpretation is not available for this class.")
        disease_category = disease_category_for_class(pred_class)

        LAST_REPORT_DATA = {
            "pred_class": pred_class,
            "cell_full_name": cell_full_name,
            "confidence": confidence,
            "cell_description": cell_description,
            "disease_note": disease_note,
            "disease_category": disease_category,
            "image_type": image_type,
            "top3": [(CLASS_NAMES[int(idx)], float(probs[int(idx)])) for idx in sorted_indices[:3]],
            "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
            "model_display": ACTIVE_MODEL_DISPLAY,
            "model_type": ACTIVE_MODEL_TYPE,
            "model_path": ACTIVE_MODEL_PATH,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        report_markdown = f"""
# Cell Identification Report

## Final Classification

**Predicted Class Code:** `{pred_class}`

**Identified Cell:** **{cell_full_name}**

**Confidence Score:** **{confidence * 100:.2f}%**

**Model Used:** **{ACTIVE_MODEL_DISPLAY}**

---

## Cell-Level Morphological Interpretation

{cell_description}

---

## Disease-Level Analysis

{disease_note}

---

## Disease Screening Category

**{disease_category}**

---

## Framework Relevance

The uploaded blood cell image is processed through the final trained model. The improved notebook keeps the accuracy strategy unchanged and uses the best available trained model for dashboard prediction.

This demonstrates the applicability of the proposed framework for hematological disease screening, including:

- Anemia-related analysis
- Leukemia-related abnormal cell maturation analysis
- White blood cell differential assessment
- General blood cell morphology-based interpretation

---

## Selected Image Type

{image_type}

---

## Clinical Note

This output is a **computational screening interpretation** based on image morphology and trained model prediction. It is **not a clinical diagnosis**.

For reliable disease-level decision making, multiple cell images, cell count distribution, laboratory findings, and expert medical validation are required.
"""

        prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

        top3_text = "Top 3 Predictions\n\n"
        for rank, idx in enumerate(sorted_indices[:3], start=1):
            top3_text += f"{rank}. {CLASS_NAMES[idx]} : {probs[idx] * 100:.2f}%\n"

        cell_mask_rgb = cv2.cvtColor(cell_mask, cv2.COLOR_GRAY2RGB)
        nuc_mask_rgb = cv2.cvtColor(nuc_mask, cv2.COLOR_GRAY2RGB)

        h, w = img_rgb.shape[:2]
        cell_mask_rgb = cv2.resize(cell_mask_rgb, (w, h))
        nuc_mask_rgb = cv2.resize(nuc_mask_rgb, (w, h))

        combined = np.concatenate([img_rgb, cell_mask_rgb, nuc_mask_rgb], axis=1)

        return report_markdown, prob_dict, top3_text, combined

    except Exception as e:
        error_text = f"""
# Prediction Error

{str(e)}

Please try uploading a clear cropped blood cell image.
"""
        return error_text, {}, "Prediction failed.", None


# ============================================================
# Dashboard CSS — same polished style as previous app
# ============================================================

custom_css = """
body {
    background:
        radial-gradient(circle at top left, rgba(37, 99, 235, 0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(14, 165, 233, 0.08), transparent 22%),
        linear-gradient(180deg, #f5f8fc 0%, #eef3f9 100%);
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: #0f172a;
}

.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding-top: 18px !important;
    padding-bottom: 24px !important;
    background: transparent !important;
}

#header-card {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #ffffff 0%, #f8fbff 55%, #f1f7ff 100%);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 28px;
    padding: 30px 32px;
    box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
    margin-bottom: 24px;
}

#header-card::before {
    content: '';
    position: absolute;
    inset: 0 auto auto 0;
    width: 100%;
    height: 5px;
    background: linear-gradient(90deg, #1d4ed8, #0ea5e9, #10b981);
}

.brand-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    border-radius: 999px;
    background: rgba(37, 99, 235, 0.08);
    border: 1px solid rgba(37, 99, 235, 0.12);
    color: #1d4ed8;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 14px;
}

#main-title {
    color: #0f172a;
    font-size: 38px;
    font-weight: 900;
    line-height: 1.12;
    letter-spacing: -0.03em;
    margin-bottom: 8px;
}

#subtitle {
    color: #475569;
    font-size: 16px;
    line-height: 1.65;
    max-width: 920px;
    margin-bottom: 22px;
}

.hero-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin-bottom: 20px;
}

.hero-stat-card {
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 16px 18px;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
}

.hero-stat-label {
    color: #64748b;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}

.hero-stat-value {
    color: #0f172a;
    font-size: 22px;
    font-weight: 900;
    margin-bottom: 4px;
}

.hero-stat-subtitle {
    color: #475569;
    font-size: 13px;
    line-height: 1.5;
}

.card {
    background: rgba(255, 255, 255, 0.88);
    backdrop-filter: blur(6px);
    border-radius: 24px;
    padding: 22px;
    border: 1px solid rgba(226, 232, 240, 0.95);
    box-shadow: 0 16px 40px rgba(15, 23, 42, 0.07);
}

.section-title {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #0f172a;
    font-size: 18px;
    font-weight: 850;
    margin-bottom: 16px;
    letter-spacing: -0.01em;
}

.section-title::before {
    content: '';
    width: 6px;
    height: 22px;
    border-radius: 999px;
    background: linear-gradient(180deg, #2563eb, #0ea5e9);
    display: inline-block;
}

.info-box,
.warning-box,
.model-box {
    border-radius: 18px;
    padding: 16px 18px;
    font-size: 14px;
    line-height: 1.65;
    border: 1px solid transparent;
}

.info-box {
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.07), rgba(14, 165, 233, 0.06));
    border-color: rgba(37, 99, 235, 0.12);
    color: #1e3a8a;
}

.warning-box {
    background: linear-gradient(135deg, rgba(249, 115, 22, 0.08), rgba(251, 191, 36, 0.06));
    border-color: rgba(249, 115, 22, 0.14);
    color: #9a3412;
    margin-top: 16px;
}

.model-box {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(34, 197, 94, 0.06));
    border-color: rgba(22, 163, 74, 0.14);
    color: #14532d;
    margin-top: 16px;
}

.status-heading {
    margin-top: 22px;
    margin-bottom: 14px;
    color: #0f172a;
    font-size: 16px;
    font-weight: 850;
}

.objective-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 12px;
    margin-top: 10px;
}

.objective-card {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 16px 12px;
    text-align: center;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
}

.objective-check {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: rgba(22, 163, 74, 0.10);
    color: #15803d;
    font-size: 18px;
    font-weight: 900;
    margin-bottom: 8px;
}

.objective-title {
    color: #0f172a;
    font-size: 13px;
    font-weight: 800;
    margin-bottom: 4px;
}

.objective-subtitle {
    color: #64748b;
    font-size: 11px;
    line-height: 1.4;
}

button {
    border-radius: 16px !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 60%, #0ea5e9 100%) !important;
    color: white !important;
    border: none !important;
    min-height: 48px !important;
    font-size: 15px !important;
    box-shadow: 0 12px 24px rgba(37, 99, 235, 0.20) !important;
    transition: all 0.25s ease !important;
}

button:hover {
    transform: translateY(-1px);
    box-shadow: 0 16px 28px rgba(37, 99, 235, 0.28) !important;
}

textarea, input {
    border-radius: 16px !important;
}

.footer-text {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    line-height: 1.8;
    margin-top: 20px;
    padding: 8px 0 4px;
}

.class-pill {
    display: inline-block;
    background: linear-gradient(180deg, #ffffff 0%, #f7fafc 100%);
    color: #334155;
    padding: 7px 11px;
    border-radius: 999px;
    margin: 4px;
    font-size: 12px;
    font-weight: 700;
    border: 1px solid #e2e8f0;
}

footer {
    display: none !important;
}

@media (max-width: 1100px) {
    .objective-grid {
        grid-template-columns: repeat(3, minmax(0, 1fr));
    }
}

@media (max-width: 900px) {
    #header-card {
        padding: 24px 22px;
    }

    #main-title {
        font-size: 30px;
    }

    .hero-grid,
    .objective-grid {
        grid-template-columns: 1fr;
    }
}
"""


# ============================================================
# Dashboard Layout
# ============================================================

class_pills_html = "\n".join([f'<span class="class-pill">{c}</span>' for c in CLASS_NAMES])
model_status_html = f"""
<div class="model-box">
    <b>Active Model:</b> {ACTIVE_MODEL_DISPLAY}<br>
    <b>Model Type:</b> {ACTIVE_MODEL_TYPE if ACTIVE_MODEL_TYPE else 'Not loaded yet'}<br>
    <b>Checkpoint:</b> {ACTIVE_MODEL_PATH if ACTIVE_MODEL_PATH else 'Train a model first'}
</div>
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as interface:

    gr.HTML(f"""
    <div id="header-card">
        <div class="brand-badge">AI Hematology Suite</div>
        <div id="main-title">Context-Aware Blood Cell Classification Dashboard</div>
        <div id="subtitle">
            AI-powered blood cell image classification using the improved FYDP pipeline.
            The dashboard keeps the previous professional interface, but now it automatically uses the best available trained model:
            Hybrid CNN + GNN, CNN baseline, balanced GNN, or previous GNN fallback.
        </div>

        <div class="hero-grid">
            <div class="hero-stat-card">
                <div class="hero-stat-label">Model Scope</div>
                <div class="hero-stat-value">{NUM_CLASSES} Blood Cell Classes</div>
                <div class="hero-stat-subtitle">Covers white blood cells, red blood cell lineage, platelet class, and immature myeloid stages.</div>
            </div>

            <div class="hero-stat-card">
                <div class="hero-stat-label">Analysis Pipeline</div>
                <div class="hero-stat-value">Image → CNN/GNN → Report</div>
                <div class="hero-stat-subtitle">Uses deep visual features and morphology graph features depending on the trained checkpoint available.</div>
            </div>

            <div class="hero-stat-card">
                <div class="hero-stat-label">Reporting</div>
                <div class="hero-stat-value">Instant PDF Output</div>
                <div class="hero-stat-subtitle">Generates a professional report containing predicted class, confidence distribution, and disease-level notes.</div>
            </div>
        </div>

        <div class="info-box">
            <b>Recommended:</b> Upload one cropped blood cell image for the most reliable prediction.
            Full microscope images can still be tested, but results may be less reliable because the model was primarily trained on cropped single-cell images.
        </div>

        <div class="warning-box">
            <b>Final Year Project Disclaimer:</b> {PROJECT_DISCLAIMER}
        </div>

        <div class="status-heading">Project Implementation Status</div>

        <div class="objective-grid">
            <div class="objective-card">
                <div class="objective-check">✓</div>
                <div class="objective-title">Balanced Data</div>
                <div class="objective-subtitle">Controlled class balancing and augmentation</div>
            </div>

            <div class="objective-card">
                <div class="objective-check">✓</div>
                <div class="objective-title">CNN Baseline</div>
                <div class="objective-subtitle">Deep visual feature learning</div>
            </div>

            <div class="objective-card">
                <div class="objective-check">✓</div>
                <div class="objective-title">GNN Features</div>
                <div class="objective-subtitle">Morphology graph representation</div>
            </div>

            <div class="objective-card">
                <div class="objective-check">✓</div>
                <div class="objective-title">Hybrid Model</div>
                <div class="objective-subtitle">CNN + GNN fusion support</div>
            </div>

            <div class="objective-card">
                <div class="objective-check">✓</div>
                <div class="objective-title">PDF Report</div>
                <div class="objective-subtitle">Professional downloadable report</div>
            </div>
        </div>
    </div>
    """)

    with gr.Row():

        with gr.Column(scale=1):
            with gr.Group(elem_classes="card"):
                gr.HTML('<div class="section-title">Upload Blood Cell Image</div>')

                input_image = gr.Image(
                    type="pil",
                    label="Blood Cell Image",
                    height=360
                )

                image_type = gr.Radio(
                    choices=[
                        "Cropped cell image - Recommended",
                        "Full image - Experimental"
                    ],
                    value="Cropped cell image - Recommended",
                    label="Image Type"
                )

                predict_btn = gr.Button("Run Prediction")

                gr.HTML("""
                <div class="warning-box">
                    <b>Demo Tip:</b> Use a clear cropped single-cell image for the cleanest and most reliable result.
                </div>
                """)

                gr.HTML(model_status_html)

        with gr.Column(scale=1):
            with gr.Group(elem_classes="card"):
                gr.HTML('<div class="section-title">Cell Identification Report and Disease-Level Interpretation</div>')

                result_output = gr.Markdown(label="Cell Identification Report")

                top3_output = gr.Textbox(label="Top 3 Predictions", lines=6)

                pdf_btn = gr.Button("Download Professional PDF Report")

                pdf_output = gr.File(label="Generated PDF Report - Click the file below to download")

                pdf_status = gr.Textbox(label="PDF Save Status", lines=2)

            with gr.Group(elem_classes="card"):
                gr.HTML('<div class="section-title">Class Probability Distribution</div>')

                prob_output = gr.Label(label="Class Probabilities", num_top_classes=12)

    with gr.Row():
        with gr.Column():
            with gr.Group(elem_classes="card"):
                gr.HTML('<div class="section-title">Image Processing Preview</div>')

                preview_output = gr.Image(
                    label="Original Image | Cell Mask | Nucleus Mask",
                    height=280
                )

    gr.HTML(f"""
    <div class="footer-text">
        <b>Supported Model Classes</b><br>
        {class_pills_html}
    </div>
    """)

    predict_btn.click(
        fn=predict_cell,
        inputs=[input_image, image_type],
        outputs=[result_output, prob_output, top3_output, preview_output]
    )

    pdf_btn.click(
        fn=download_pdf_report,
        inputs=[],
        outputs=[pdf_output, pdf_status]
    )


interface.launch(
    share=True,
    inline=False,
    debug=True
)
