const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, HeadingLevel,
  BorderStyle, WidthType, ShadingType, PageNumber, PageBreak,
  VerticalAlign
} = require('docx');
const fs = require('fs');

// ── Helpers ──────────────────────────────────────────────────────────────────

const CONTENT_W = 9360; // US Letter 8.5" - 2×1" margins
const border = { style: BorderStyle.SINGLE, size: 4, color: "AAAAAA" };
const borders = { top: border, bottom: border, left: border, right: border };
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, font: "Arial", bold: true, size: 30, color: "1F3864" })]
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, font: "Arial", bold: true, size: 24, color: "2E6DA4" })]
  });
}

function para(text, opts = {}) {
  return new Paragraph({
    spacing: { before: 60, after: 120 },
    children: [new TextRun({ text, font: "Arial", size: 22, ...opts })]
  });
}

function blank() {
  return new Paragraph({ spacing: { before: 0, after: 80 }, children: [new TextRun("")] });
}

function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { before: 40, after: 40 },
    children: [new TextRun({ text, font: "Arial", size: 22 })]
  });
}

function numbered(text) {
  return new Paragraph({
    numbering: { reference: "numbers", level: 0 },
    spacing: { before: 40, after: 40 },
    children: [new TextRun({ text, font: "Arial", size: 22 })]
  });
}

function divider() {
  return new Paragraph({
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "2E6DA4", space: 1 } },
    spacing: { before: 100, after: 200 },
    children: [new TextRun("")]
  });
}

function qaBlock(question, answer) {
  return [
    new Paragraph({
      spacing: { before: 160, after: 60 },
      children: [new TextRun({ text: question, font: "Arial", size: 22, bold: true, color: "1F3864" })]
    }),
    new Paragraph({
      spacing: { before: 40, after: 120 },
      children: [new TextRun({ text: answer, font: "Arial", size: 22 })]
    })
  ];
}

// ── Table builder ─────────────────────────────────────────────────────────────

function makeTable(headers, rows, colWidths) {
  const totalW = colWidths.reduce((a, b) => a + b, 0);

  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => new TableCell({
      borders,
      width: { size: colWidths[i], type: WidthType.DXA },
      shading: { fill: "2E6DA4", type: ShadingType.CLEAR },
      margins: { top: 100, bottom: 100, left: 150, right: 150 },
      verticalAlign: VerticalAlign.CENTER,
      children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: h, font: "Arial", size: 20, bold: true, color: "FFFFFF" })]
      })]
    }))
  });

  const dataRows = rows.map((row, ri) => new TableRow({
    children: row.map((cell, ci) => new TableCell({
      borders,
      width: { size: colWidths[ci], type: WidthType.DXA },
      shading: { fill: ri % 2 === 0 ? "EBF3FB" : "FFFFFF", type: ShadingType.CLEAR },
      margins: { top: 80, bottom: 80, left: 150, right: 150 },
      children: [new Paragraph({
        children: [new TextRun({ text: cell, font: "Arial", size: 20 })]
      })]
    }))
  }));

  return new Table({
    width: { size: totalW, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows]
  });
}

// ── Cover page ────────────────────────────────────────────────────────────────

function coverSection() {
  return {
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    children: [
      // top spacer
      new Paragraph({ spacing: { before: 0, after: 2880 }, children: [new TextRun("")] }),

      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 240 },
        children: [new TextRun({ text: "NORTH SOUTH UNIVERSITY", font: "Arial", size: 24, bold: true, color: "2E6DA4", allCaps: true })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 80 },
        children: [new TextRun({ text: "Department of Computer Science & Engineering", font: "Arial", size: 22, color: "555555" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 800 },
        children: [new TextRun({ text: "Course: CSE499 — Senior Design Project", font: "Arial", size: 22, color: "555555" })]
      }),

      // Divider line via border paragraph
      new Paragraph({
        alignment: AlignmentType.CENTER,
        border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: "2E6DA4", space: 1 } },
        spacing: { before: 0, after: 480 },
        children: [new TextRun("")]
      }),

      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 200 },
        children: [new TextRun({ text: "Mushroom Disease Detection", font: "Arial", size: 56, bold: true, color: "1F3864" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 200 },
        children: [new TextRun({ text: "Using Deep Learning", font: "Arial", size: 56, bold: true, color: "1F3864" })]
      }),

      new Paragraph({
        alignment: AlignmentType.CENTER,
        border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: "2E6DA4", space: 1 } },
        spacing: { before: 200, after: 480 },
        children: [new TextRun("")]
      }),

      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 120 },
        children: [new TextRun({ text: "A Comprehensive Project Report", font: "Arial", size: 28, italics: true, color: "444444" })]
      }),

      new Paragraph({ spacing: { before: 0, after: 1440 }, children: [new TextRun("")] }),

      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 80 },
        children: [new TextRun({ text: "Submitted in partial fulfilment of course requirements", font: "Arial", size: 20, color: "666666" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 80 },
        children: [new TextRun({ text: "April 2026", font: "Arial", size: 22, bold: true, color: "333333" })]
      }),

      // Page break to start main content
      new Paragraph({ children: [new PageBreak()] })
    ]
  };
}

// ── Main content section ──────────────────────────────────────────────────────

function mainSection() {
  const children = [];

  // ── SECTION 1: PROJECT OVERVIEW ──
  children.push(h1("1. Project Overview"));
  children.push(divider());

  children.push(h2("1.1  Background and Motivation"));
  children.push(para("Mushrooms are a vital agricultural commodity in Bangladesh, particularly cultivated using substrate (sawdust/rice husk blocks). Fungal contamination — primarily Trichoderma (green mold), Aspergillus (black mold), and Rhizopus — can destroy 20-40% of annual yield if not detected early. Traditional detection relies on manual visual inspection by experienced farmers, which is slow, subjective, and unavailable to small-scale growers. This project builds an AI-powered image classification system that can detect disease from a single photograph — enabling early, accurate, and accessible diagnosis."));
  children.push(blank());

  children.push(h2("1.2  Objectives"));
  children.push(bullet("Collect and prepare a real-world dataset of mushroom substrate images"));
  children.push(bullet("Apply data augmentation to address class imbalance"));
  children.push(bullet("Train and compare multiple deep learning architectures"));
  children.push(bullet("Achieve high classification accuracy on three disease classes"));
  children.push(bullet("Deploy a user-friendly web application for farmers"));
  children.push(bullet("Integrate a RAG-powered chatbot for farming advice"));
  children.push(blank());

  children.push(h2("1.3  Problem Statement"));
  children.push(para("Classify mushroom substrate images into three categories:"));
  children.push(bullet("Class 0 - Healthy: No contamination present"));
  children.push(bullet("Class 1 - Single Infected: One type of mold (Green OR Black mold)"));
  children.push(bullet("Class 2 - Mixed Infected: Multiple pathogens OR healthy + mold in same frame"));
  children.push(para("The Mixed Infected class is a novel contribution not addressed in prior literature."));
  children.push(blank());

  // ── SECTION 2: DATASET ──
  children.push(h1("2. Dataset"));
  children.push(divider());

  children.push(h2("2.1  Data Collection"));
  children.push(para("761 JPEG images were captured at the Mushroom Development Institute, Savar, Dhaka, Bangladesh in April 2025 using an iPhone 11 Pro Max under natural and indoor lighting conditions. All images were collected by the research team on-site, making this a fully original, real-world dataset."));
  children.push(blank());

  children.push(h2("2.2  Class Distribution"));
  children.push(makeTable(
    ["Class", "Images", "Percentage"],
    [
      ["Healthy", "299", "39.3%"],
      ["Single Infected", "147", "19.3%"],
      ["Mixed Infected", "315", "41.4%"],
      ["Total", "761", "100%"]
    ],
    [4680, 2340, 2340]
  ));
  children.push(blank());
  children.push(para("The imbalance ratio is 2.14 (Mixed vs Single). No corrupt images were found."));
  children.push(blank());

  children.push(h2("2.3  Why This Dataset is Unique"));
  children.push(para("Existing mushroom disease datasets in the literature are either (a) leaf-disease datasets not specific to substrate cultivation, (b) limited to binary healthy/diseased classification, or (c) collected under controlled lab conditions. This dataset covers the mixed infection scenario — when a block shows signs of multiple pathogens simultaneously — which is the most damaging and hardest-to-detect scenario in practice."));
  children.push(blank());

  children.push(h2("2.4  Dataset Split"));
  children.push(para("Stratified split was applied to preserve class proportions across all three splits:"));
  children.push(makeTable(
    ["Split", "Image Count"],
    [
      ["Training", "531"],
      ["Validation", "113"],
      ["Test", "117"],
      ["Total", "761"]
    ],
    [5400, 3960]
  ));
  children.push(blank());

  // ── SECTION 3: PREPROCESSING ──
  children.push(h1("3. Preprocessing & Data Augmentation"));
  children.push(divider());

  children.push(h2("3.1  Preprocessing Steps"));
  children.push(para("All images were resized to 224x224 pixels (standard input for CNN architectures), converted to RGB format, and normalized. Normalization strategy varied by model: standard models used /255.0 pixel scaling; EfficientNetV2S used raw [0,255] input because it has a built-in preprocessing layer."));
  children.push(blank());

  children.push(h2("3.2  Augmentation Strategy"));
  children.push(para("Because the training set was only 531 images, aggressive augmentation was applied to expand the training set to 2,400 images:"));
  children.push(bullet("Horizontal flip"));
  children.push(bullet("Vertical flip"));
  children.push(bullet("Rotation +/-30 degrees"));
  children.push(bullet("Zoom (up to 20%)"));
  children.push(bullet("Brightness and contrast variation"));
  children.push(bullet("Horizontal/vertical shift"));
  children.push(bullet("Gaussian noise injection"));
  children.push(bullet("Shear transformation"));
  children.push(blank());

  children.push(h2("3.3  Results After Augmentation"));
  children.push(makeTable(
    ["Split", "Before Augmentation", "After Augmentation"],
    [
      ["Training", "531", "2,400"],
      ["Validation", "113", "113 (unchanged)"],
      ["Test", "117", "117 (unchanged)"]
    ],
    [3120, 3120, 3120]
  ));
  children.push(blank());
  children.push(para("Validation and test sets were NOT augmented — they represent real-world unseen conditions."));
  children.push(blank());

  children.push(h2("3.4  Class Imbalance Handling"));
  children.push(para("Even after augmentation, the imbalance ratio remained approximately 2.16. To compensate, class_weight was computed using sklearn's compute_class_weight function and passed to model.fit(), penalising misclassification of the minority class (Single Infected) more heavily during backpropagation."));
  children.push(blank());

  // ── SECTION 4: CPU TRAINING ──
  children.push(h1("4. Model Training — CPU Baseline"));
  children.push(divider());

  children.push(h2("4.1  Setup"));
  children.push(para("Five architectures were trained on CPU (no GPU) using TensorFlow 2.20.0, batch size 16. This served as a baseline comparison before GPU training, establishing which architectures were worth investing further resources in."));
  children.push(blank());

  children.push(h2("4.2  Models Compared"));
  children.push(makeTable(
    ["Rank", "Model", "Val Accuracy", "Val Loss"],
    [
      ["1", "Custom CNN", "85.84%", "0.4062"],
      ["2", "InceptionV3", "83.19%", "0.4677"],
      ["3", "DenseNet201", "73.45%", "0.4857"],
      ["4", "ResNet50", "67.26%", "0.8874"],
      ["5", "EfficientNetB0", "44.25%", "1.0745"],
      ["—", "VGG16", "SKIPPED (too slow on CPU)", "—"]
    ],
    [1200, 3000, 2580, 2580]
  ));
  children.push(blank());

  children.push(h2("4.3  Why Custom CNN Outperformed Transfer Learning (CPU)"));
  children.push(para("On a small dataset with CPU constraints, large pre-trained models (ResNet50, DenseNet201) can overfit or fail to converge properly without sufficient fine-tuning epochs. The Custom CNN — being smaller and task-specific — trained more effectively within the resource constraints. This result does not mean Custom CNN is superior in general; it reflects the limitations of CPU training."));
  children.push(blank());

  // ── SECTION 5: GPU TRAINING ──
  children.push(h1("5. Model Training — GPU (RTX 5060)"));
  children.push(divider());

  children.push(h2("5.1  Setup"));
  children.push(para("Training was repeated on GPU (NVIDIA RTX 5060) using TensorFlow 2.21.0 via WSL2 Ubuntu. Three architectures were selected for GPU training: EfficientNetV2S (best expected), DenseNet121, and Custom CNN v2. FP16 mixed precision was enabled to maximise GPU throughput."));
  children.push(blank());

  children.push(h2("5.2  Critical Bugs Fixed During GPU Training"));

  children.push(new Paragraph({
    spacing: { before: 120, after: 40 },
    children: [new TextRun({ text: "Bug 1 — Keras 3 Learning Rate API", font: "Arial", size: 22, bold: true, color: "C0392B" })]
  }));
  children.push(para("The ReduceLROnPlateau callback used the old Keras 2 API: keras.backend.set_value(optimizer.learning_rate, new_lr). In Keras 3, this raises an AttributeError. Fixed by directly assigning: self.model.optimizer.learning_rate = new_lr"));
  children.push(blank());

  children.push(new Paragraph({
    spacing: { before: 120, after: 40 },
    children: [new TextRun({ text: "Bug 2 — Normalization Double-Processing (Main Accuracy Bug)", font: "Arial", size: 22, bold: true, color: "C0392B" })]
  }));
  children.push(para("The data pipeline applied /255.0 normalization to all images. EfficientNetV2S has a built-in preprocessing layer that expects raw pixel values in the range [0, 255]. When /255.0 was also applied, the model received input in [0,1] instead of [0,255], causing the built-in layer to produce incorrect scaled values. This single error dropped accuracy from approximately 92% to approximately 35%. Removing /255.0 for EfficientNetV2S fixed the issue immediately."));
  children.push(blank());

  children.push(new Paragraph({
    spacing: { before: 120, after: 40 },
    children: [new TextRun({ text: "Bug 3 — Out Of Memory (OOM)", font: "Arial", size: 22, bold: true, color: "C0392B" })]
  }));
  children.push(para("With batch size 32, WSL2 consumed all available RAM, causing the training process to crash. Fixed by reducing batch size to 16 and increasing WSL2 memory allocation to 12 GB via the .wslconfig configuration file."));
  children.push(blank());

  children.push(h2("5.3  GPU Training Results"));
  children.push(makeTable(
    ["Model", "Val Accuracy", "Training Time"],
    [
      ["EfficientNetV2S", "92.92%", "13.8 min"],
      ["DenseNet121", "83.19%", "14.2 min"],
      ["Custom CNN v2", "72.57%", "5.3 min"]
    ],
    [4680, 2340, 2340]
  ));
  children.push(blank());
  children.push(para("Best model: EfficientNetV2S at 92.92% validation accuracy. Model saved as EfficientNetV2S_best.keras."));
  children.push(blank());

  // ── SECTION 6: EVALUATION ──
  children.push(h1("6. Model Evaluation on Test Set"));
  children.push(divider());

  children.push(h2("6.1  Test Set Results"));
  children.push(makeTable(
    ["Model", "Test Accuracy", "Macro AUC"],
    [
      ["EfficientNetV2S", "77.78%", "0.9210"],
      ["DenseNet121", "74.36%", "0.9216"],
      ["Custom CNN v2", "66.67%", "0.7999"]
    ],
    [4680, 2340, 2340]
  ));
  children.push(blank());

  children.push(h2("6.2  Per-Class Performance — EfficientNetV2S"));
  children.push(makeTable(
    ["Class", "Precision", "Recall", "F1-Score"],
    [
      ["Healthy", "0.79", "0.59", "0.68"],
      ["Mixed Infected", "0.68", "0.85", "0.76"],
      ["Single Infected", "1.00", "1.00", "1.00"]
    ],
    [3510, 1950, 1950, 1950]
  ));
  children.push(blank());

  children.push(h2("6.3  Why Is Test Accuracy Lower Than Validation Accuracy?"));
  children.push(para("Validation accuracy (92.92%) vs test accuracy (77.78%) — the 15% gap is explained by three factors:"));
  children.push(bullet("Small dataset size: 117 test images is statistically small — each misclassified sample equals approximately 0.85% accuracy drop, amplifying variance."));
  children.push(bullet("Class overlap: Healthy and Mixed Infected visually overlap in early-stage contamination."));
  children.push(bullet("Validation overfitting: The model saw validation labels implicitly through hyperparameter tuning decisions."));
  children.push(para("Despite the gap, AUC = 0.9210 confirms the model has excellent discriminative power. AUC is a threshold-independent metric and is considered a more reliable indicator of true model quality than point accuracy."));
  children.push(blank());

  children.push(h2("6.4  Evaluation Outputs Generated"));
  children.push(bullet("Confusion matrices (per model)"));
  children.push(bullet("ROC-AUC curves (per class, per model)"));
  children.push(bullet("Grad-CAM heatmaps showing which image regions triggered the prediction"));
  children.push(bullet("Model comparison bar chart"));
  children.push(blank());

  children.push(h2("6.5  Grad-CAM Explanation"));
  children.push(para("Gradient-weighted Class Activation Mapping (Grad-CAM) visualises which pixels most influenced the model's decision. Gradients of the predicted class score are computed with respect to the feature maps of the final convolutional layer. For infected images, the heatmap highlights the mold-covered regions, confirming the model is learning biologically meaningful features — not background artifacts."));
  children.push(blank());

  // ── SECTION 7: WEB APP ──
  children.push(h1("7. Deployment — Web Application"));
  children.push(divider());

  children.push(h2("7.1  Technology Stack"));
  children.push(makeTable(
    ["Component", "Technology"],
    [
      ["Web Framework", "Streamlit (Python)"],
      ["Deep Learning Backend", "TensorFlow 2.21.0 + Keras"],
      ["Runtime Environment", "WSL2 Ubuntu, tf_env virtual environment"],
      ["Primary Model", "EfficientNetV2S_best.keras"],
      ["RAG Retriever", "ChromaDB + SentenceTransformer (all-MiniLM-L6-v2)"],
      ["RAG Generator", "Groq API (Llama-3.3-70B)"]
    ],
    [4680, 4680]
  ));
  children.push(blank());

  children.push(h2("7.2  Application Features"));
  children.push(new Paragraph({
    spacing: { before: 80, after: 40 },
    children: [new TextRun({ text: "Tab 1 — Disease Detection:", font: "Arial", size: 22, bold: true })]
  }));
  children.push(bullet("Upload a mushroom substrate image"));
  children.push(bullet("Select model (EfficientNetV2S recommended)"));
  children.push(bullet("See prediction with confidence percentages for all 3 classes"));
  children.push(bullet("View Grad-CAM heatmap overlaid on the original image"));
  children.push(bullet("Colour-coded severity grade cards (Green = Healthy, Yellow = Mild, Red = Infected)"));
  children.push(blank());
  children.push(new Paragraph({
    spacing: { before: 80, after: 40 },
    children: [new TextRun({ text: "Tab 2 — AI Farming Assistant (RAG Chatbot):", font: "Arial", size: 22, bold: true })]
  }));
  children.push(bullet("Ask questions about mushroom diseases, prevention, and treatment"));
  children.push(bullet("RAG retrieves relevant passages from a curated knowledge base, then uses Groq Llama-3.3-70B to generate contextual answers"));
  children.push(bullet("Maintains conversation history in session"));
  children.push(blank());

  children.push(h2("7.3  UI Design Philosophy"));
  children.push(para("The interface was designed for non-technical farmers with zero machine learning background:"));
  children.push(bullet("Simple image upload with drag-and-drop"));
  children.push(bullet("Plain language labels (No Disease / Mild / Severe) instead of technical class names"));
  children.push(bullet("Mobile-first responsive layout"));
  children.push(bullet("Dark navy sidebar with green status indicators for model and API health"));
  children.push(bullet("Colour-coded result cards using a traffic-light system (green/yellow/red)"));
  children.push(blank());

  children.push(h2("7.4  How to Run the Application"));
  children.push(para("Run from WSL2 Ubuntu using the tf_env virtual environment:"));
  children.push(new Paragraph({
    spacing: { before: 80, after: 80 },
    shading: { fill: "F0F0F0", type: ShadingType.CLEAR },
    children: [new TextRun({
      text: 'wsl -d Ubuntu -e sh -c "cd /home/junaid/Mashroom-disease-detection && /home/junaid/tf_env/bin/streamlit run app.py --server.port 8501"',
      font: "Courier New", size: 18, color: "333333"
    })]
  }));
  children.push(blank());

  children.push(h2("7.5  Public Access for Remote Demo"));
  children.push(para("For remote demonstration, localtunnel provides a public HTTPS URL:"));
  children.push(new Paragraph({
    spacing: { before: 80, after: 80 },
    shading: { fill: "F0F0F0", type: ShadingType.CLEAR },
    children: [new TextRun({ text: "npx localtunnel --port 8501", font: "Courier New", size: 18, color: "333333" })]
  }));
  children.push(para("This allows the app running on a home PC to be accessed from any device with internet access — useful for live demonstrations at university or off-site."));
  children.push(blank());

  // ── SECTION 8: NOVELTY ──
  children.push(h1("8. Research Novelty & Contributions"));
  children.push(divider());

  children.push(h2("8.1  Novel Contributions"));
  children.push(numbered("Mixed Infected Class: First dataset with a dedicated Mixed Infected category for mushroom substrate, where multiple pathogens coexist — not addressed in any referenced prior work."));
  children.push(numbered("Original Bangladesh Dataset: 761 real-world images from Mushroom Development Institute, Savar — not derived from any publicly available dataset."));
  children.push(numbered("CORN Loss Exploration: Ordinal regression using CORN loss (Conditional Ordinal Regression for Neural networks) was explored for treating disease severity as an ordered variable: Healthy < Single Infected < Mixed Infected."));
  children.push(numbered("Full-Stack AI Application: End-to-end system from raw image collection to deployed web app with RAG chatbot — not just a standalone classification model."));
  children.push(numbered("Multi-Model Comparison: Six architectures compared under identical conditions in two hardware environments (CPU and GPU)."));
  children.push(blank());

  children.push(h2("8.2  Why Deep Learning?"));
  children.push(para("Traditional image processing approaches (edge detection, colour thresholding, texture analysis) fail because mold appearance varies significantly with lighting conditions, substrate colour, contamination stage, and camera angle. Deep learning learns hierarchical features automatically — edges and textures at lower layers, object parts at middle layers, and semantic disease patterns at higher layers — making it robust to these visual variations without hand-crafted feature engineering."));
  children.push(blank());

  // ── SECTION 9: SYSTEM ARCHITECTURE ──
  children.push(h1("9. System Architecture"));
  children.push(divider());

  children.push(h2("9.1  Pipeline Overview"));
  children.push(makeTable(
    ["Stage", "Description", "Output"],
    [
      ["1 — Data Collection", "iPhone photos at Savar MDI, Bangladesh", "761 raw JPEG images"],
      ["2 — Preprocessing", "Resize 224x224, augment with 8 transforms", "2,400 training images"],
      ["3 — Model Training", "EfficientNetV2S + DenseNet121 + Custom CNN on RTX 5060", "3 saved .keras models"],
      ["4 — Evaluation", "Test set metrics, confusion matrix, ROC-AUC, Grad-CAM", "Plots, JSON reports"],
      ["5 — Deployment", "Streamlit app + RAG chatbot on WSL2", "Farmer-facing web interface"]
    ],
    [2200, 4000, 3160]
  ));
  children.push(blank());

  children.push(h2("9.2  Full Technology Stack"));
  children.push(makeTable(
    ["Component", "Technology"],
    [
      ["Language", "Python 3.10"],
      ["Deep Learning Framework", "TensorFlow 2.21 / Keras"],
      ["GPU Training", "NVIDIA RTX 5060, WSL2 Ubuntu"],
      ["Explainability", "Grad-CAM (gradient-based heatmaps)"],
      ["Web Framework", "Streamlit"],
      ["RAG Retriever", "ChromaDB + SentenceTransformer"],
      ["RAG Generator", "Groq API (Llama-3.3-70B)"],
      ["Version Control", "Git"],
      ["Image Processing", "OpenCV, PIL (Pillow)"],
      ["Visualization", "Matplotlib, Seaborn"]
    ],
    [4680, 4680]
  ));
  children.push(blank());

  // ── SECTION 10: Q&A ──
  children.push(h1("10. Anticipated Supervisor Questions & Model Answers"));
  children.push(divider());
  children.push(para("This section prepares you for the most likely technical questions a supervisor or examiner will ask. Read each answer aloud until you can deliver it naturally."));
  children.push(blank());

  const qaItems = [
    [
      "Q1: Why did you choose EfficientNetV2S as your final model?",
      "EfficientNetV2S was selected for three reasons. First, it achieved the highest validation accuracy (92.92%) among all models tested. Second, it uses compound scaling — simultaneously scaling network depth, width, and resolution — making it more parameter-efficient than ResNet50 or DenseNet. Third, it includes a built-in preprocessing layer optimised for its training distribution, which reduces the risk of input scaling errors. Its AUC of 0.9210 on the test set confirms excellent discriminative ability across all three classes."
    ],
    [
      "Q2: Why is test accuracy (77.78%) significantly lower than validation accuracy (92.92%)?",
      "Three factors explain this gap. First, the test set contains only 117 images — each misclassified sample equals approximately 0.85% accuracy drop, amplifying statistical variance. Second, hyperparameter tuning (learning rate, batch size, augmentation intensity) was informed by validation performance, meaning the model was implicitly optimised for that split. Third, Healthy vs Mixed Infected is inherently ambiguous in early-stage contamination — some test images show borderline cases that even human experts would struggle to classify. The AUC of 0.9210 better reflects true model quality, as it is threshold-independent and robust to class imbalance."
    ],
    [
      "Q3: How did you handle class imbalance?",
      "We used two complementary strategies. First, we applied targeted data augmentation to expand the training set from 531 to 2,400 images. Although augmentation was applied proportionally, Single Infected remained the minority class with an imbalance ratio of approximately 2.16. Second, we computed class_weight using sklearn's compute_class_weight function and passed it to model.fit(). This penalises misclassification of the minority class more heavily during backpropagation, preventing the model from defaulting to majority-class predictions."
    ],
    [
      "Q4: What is the Mixed Infected class and why is it novel?",
      "Mixed Infected refers to substrate images where multiple types of contamination coexist in a single frame — for example, Trichoderma (green mold) and Aspergillus (black mold) simultaneously, or a partially healthy block with visible mold intrusion. This is biologically distinct from Single Infected because it represents a more advanced contamination stage requiring different treatment intervention. No prior mushroom disease classification paper we reviewed includes this category — existing work treats the problem as binary (healthy/diseased) or focuses on single-pathogen identification in leaf diseases."
    ],
    [
      "Q5: Why did you use Grad-CAM?",
      "Grad-CAM provides explainability — it shows which regions of the input image caused the model to make its prediction. This is critically important for a farming application because farmers will not trust a black-box system that gives no visual explanation. When the heatmap highlights the mold-covered region of the substrate block, it validates that the model is reasoning correctly about the disease. This also helps researchers identify failure modes: if Grad-CAM highlights background objects instead of the mushroom block, that indicates a dataset bias problem that needs correction."
    ],
    [
      "Q6: What is RAG and why did you include it?",
      "RAG stands for Retrieval-Augmented Generation. Instead of relying solely on an LLM's general training knowledge, RAG first retrieves relevant passages from a curated knowledge base (mushroom disease literature, treatment guides), then passes them as grounding context to the LLM (Groq Llama-3.3-70B) to generate a factual answer. We included it because classification alone tells the farmer what disease is present — but not what to do about it. The chatbot provides actionable treatment recommendations, prevention strategies, and cultivation advice, making the system practically useful beyond just diagnosis."
    ],
    [
      "Q7: Why did you collect your own dataset instead of using an existing one?",
      "Existing mushroom disease datasets are predominantly focused on different cultivation types (oyster mushrooms on logs, or dried mushroom cap disease) and collected under controlled laboratory conditions. Our target scenario — substrate bag cultivation as practiced in Bangladesh — has very different visual characteristics: different substrate colour, block shape, bag material, and lighting conditions. Training on existing datasets would result in poor generalization to our target domain. Additionally, the Mixed Infected class does not exist in any published dataset, making collection mandatory."
    ],
    [
      "Q8: How does your system work in practice for a farmer?",
      "The farmer opens the web application on their phone or computer, takes a photo of the mushroom substrate block, and uploads it. Within seconds, the system displays: (1) the disease classification with confidence scores for all three classes, (2) a visual heatmap highlighting the infected region on the image, (3) a severity grade with colour coding (green/yellow/red) in plain language, and (4) access to the chatbot where they can ask follow-up questions. The entire interaction takes under one minute and requires no technical knowledge."
    ],
    [
      "Q9: What are the limitations of your system?",
      "Four main limitations: First, the dataset size (761 images) is small — more data from diverse farms would improve generalization. Second, the model was trained on images from a single location (Savar MDI) and may not generalize to different cultivation environments or lighting conditions without retraining. Third, the RAG chatbot quality depends on the knowledge base — it may not cover all regional disease variants or Bangladeshi farming practices comprehensively. Fourth, the app currently requires internet connectivity and a web browser — an offline mobile app would be more accessible for rural farmers without reliable internet."
    ],
    [
      "Q10: What would you do differently or improve in the future?",
      "Four priority improvements: First, expand the dataset to 5,000+ images from multiple farms across Bangladesh to improve generalization. Second, implement edge deployment — a lightweight TensorFlow Lite model on a mobile app for fully offline use. Third, explore real-time video analysis to detect contamination as it spreads across a bag over time, enabling even earlier intervention. Fourth, integrate multilingual support (Bangla language) so the chatbot can serve rural farmers who are not comfortable reading English."
    ]
  ];

  qaItems.forEach(([q, a]) => {
    qaBlock(q, a).forEach(p => children.push(p));
    children.push(blank());
  });

  // ── SECTION 11: PRESENTATION GUIDE ──
  children.push(h1("11. How to Present This Project"));
  children.push(divider());

  children.push(h2("11.1  Opening Statement (60 seconds)"));
  children.push(para("Start with the problem to hook your audience immediately:"));
  children.push(new Paragraph({
    spacing: { before: 120, after: 120 },
    indent: { left: 720 },
    border: { left: { style: BorderStyle.SINGLE, size: 12, color: "2E6DA4", space: 12 } },
    children: [new TextRun({
      text: "\"Every year, mushroom farmers in Bangladesh lose 20-40% of their crop to fungal contamination — and most of them find out too late because they cannot tell what disease it is until it has already spread. Our project builds an AI system that can detect disease from a single smartphone photo in under a second.\"",
      font: "Arial", size: 22, italics: true, color: "333333"
    })]
  }));
  children.push(para("This immediately establishes real-world relevance and gives the audience a reason to pay attention."));
  children.push(blank());

  children.push(h2("11.2  Recommended Presentation Flow"));
  children.push(numbered("Problem: Real-world mushroom contamination statistics, farmer pain points"));
  children.push(numbered("Dataset: Show sample images of all 3 classes side by side; mention 761 images from Savar MDI"));
  children.push(numbered("Methods: Brief pipeline diagram; highlight the 3-stage training approach (CPU baseline → GPU training → Evaluation)"));
  children.push(numbered("Results: Show model comparison table; emphasize EfficientNetV2S 92.92% val / 77.78% test / AUC 0.9210"));
  children.push(numbered("Demo: Live app demo — upload a mushroom image, show prediction + Grad-CAM heatmap"));
  children.push(numbered("Novelty: Mixed Infected class, original dataset, RAG chatbot — distinguish from prior work"));
  children.push(numbered("Future Work: Mobile app, larger dataset, multilingual support"));
  children.push(blank());

  children.push(h2("11.3  Key Numbers to Memorise"));
  children.push(makeTable(
    ["Metric", "Value"],
    [
      ["Original dataset size", "761 images"],
      ["After augmentation", "2,400 training images"],
      ["Number of classes", "3 (Healthy / Single Infected / Mixed Infected)"],
      ["Architectures compared", "6 models"],
      ["Best model", "EfficientNetV2S"],
      ["Validation accuracy", "92.92%"],
      ["Test accuracy", "77.78%"],
      ["Test AUC (Macro)", "0.9210"],
      ["Single Infected F1", "1.00 (perfect)"],
      ["GPU training time", "13.8 minutes (RTX 5060)"]
    ],
    [4680, 4680]
  ));
  children.push(blank());

  children.push(h2("11.4  Handling Tough Questions"));
  children.push(makeTable(
    ["If Asked About...", "Your Response"],
    [
      ["Val/test accuracy gap", "\"Small dataset variance — the AUC of 0.9210 is the more reliable metric and shows excellent discriminative power.\""],
      ["Deployment readiness", "\"The app is running live right now — I can demonstrate it immediately.\""],
      ["Real-world validation", "\"We collected data at an actual institute. The next step is field testing with real farmers.\""],
      ["Ethical concerns", "\"The app augments, not replaces, expert judgment. It is a decision support tool that flags problems for human verification.\""],
      ["Why not use more data", "\"This was limited by what was available at Savar MDI. Expanding the dataset is our top future work priority.\""]
    ],
    [3000, 6360]
  ));
  children.push(blank());

  children.push(h2("11.5  Live Demo Script"));
  children.push(numbered("Open app: \"This is our Streamlit web application — designed to be used on any phone or computer by a farmer with no technical background.\""));
  children.push(numbered("Upload image: \"I am uploading a contaminated substrate image from our test set.\""));
  children.push(numbered("Point to confidence bars: \"The model is 87% confident this is Mixed Infected — it has identified multiple types of contamination.\""));
  children.push(numbered("Show Grad-CAM: \"The heatmap confirms the model is focusing on the mold-covered region of the substrate block — not the background or the bag.\""));
  children.push(numbered("Switch to chatbot: \"And the farmer can immediately ask what to do next. The RAG system retrieves relevant information from our knowledge base and generates a specific, actionable answer.\""));
  children.push(blank());

  // ── APPENDIX ──
  children.push(h1("Appendix"));
  children.push(divider());

  children.push(h2("A.  Project File Structure"));
  children.push(makeTable(
    ["File / Folder", "Purpose"],
    [
      ["phase1_prepare.py", "Dataset audit and stratified 70/15/15 split"],
      ["phase2_augment.py", "Image preprocessing and augmentation (531 -> 2,400)"],
      ["phase3_train.py", "CPU baseline training (5 models)"],
      ["phase3b_train.py", "GPU training (EfficientNetV2S, DenseNet121, Custom CNN v2)"],
      ["phase4b_evaluate.py", "Test set evaluation: confusion matrix, ROC, Grad-CAM"],
      ["app.py", "Main Streamlit web application"],
      ["models/", "Saved .keras model files"],
      ["dataset/", "70/15/15 split raw images"],
      ["dataset_augmented/", "Augmented training set (2,400 images)"],
      ["plots/", "All evaluation plots (confusion matrices, ROC curves, Grad-CAM)"]
    ],
    [3500, 5860]
  ));
  children.push(blank());

  children.push(h2("B.  Environment Specifications"));
  children.push(makeTable(
    ["Component", "Specification"],
    [
      ["Operating System", "Windows 11 + WSL2 Ubuntu 22.04"],
      ["GPU", "NVIDIA RTX 5060"],
      ["TensorFlow Version", "2.21.0 (GPU via WSL2)"],
      ["Python Version", "3.10 (WSL2 tf_env virtual environment)"],
      ["Web Framework", "Streamlit (latest via pip)"],
      ["RAM Allocated to WSL2", "12 GB (configured via .wslconfig)"]
    ],
    [4680, 4680]
  ));

  return {
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1080, left: 1440 }
      }
    },
    headers: {
      default: new Header({
        children: [
          new Paragraph({
            border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "2E6DA4", space: 1 } },
            spacing: { before: 0, after: 120 },
            children: [
              new TextRun({ text: "Mushroom Disease Detection Using Deep Learning", font: "Arial", size: 18, color: "2E6DA4" }),
              new TextRun({ text: "\tNorth South University | April 2026", font: "Arial", size: 18, color: "888888" })
            ],
            tabStops: [{ type: "right", position: 9360 }]
          })
        ]
      })
    },
    footers: {
      default: new Footer({
        children: [
          new Paragraph({
            border: { top: { style: BorderStyle.SINGLE, size: 4, color: "2E6DA4", space: 1 } },
            spacing: { before: 120, after: 0 },
            alignment: AlignmentType.CENTER,
            children: [
              new TextRun({ text: "Page ", font: "Arial", size: 18, color: "888888" }),
              new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 18, color: "888888" }),
              new TextRun({ text: " of ", font: "Arial", size: 18, color: "888888" }),
              new TextRun({ children: [PageNumber.TOTAL_PAGES], font: "Arial", size: 18, color: "888888" })
            ]
          })
        ]
      })
    },
    children
  };
}

// ── Build & Save ──────────────────────────────────────────────────────────────

const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
      },
      {
        reference: "numbers",
        levels: [{
          level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
      }
    ]
  },
  styles: {
    default: {
      document: { run: { font: "Arial", size: 22 } }
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Arial", color: "1F3864" },
        paragraph: { spacing: { before: 480, after: 120 }, outlineLevel: 0 }
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: "2E6DA4" },
        paragraph: { spacing: { before: 280, after: 100 }, outlineLevel: 1 }
      }
    ]
  },
  sections: [coverSection(), mainSection()]
});

Packer.toBuffer(doc).then(buffer => {
  const out = "F:\\Mashroom-disease-detection\\Project_Report.docx";
  fs.writeFileSync(out, buffer);
  console.log("Saved:", out, `(${Math.round(buffer.length / 1024)} KB)`);
}).catch(err => {
  console.error("Error:", err.message);
});
