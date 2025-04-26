import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QComboBox,
                               QVBoxLayout, QPushButton, QTextEdit, QLineEdit, QGroupBox, QCheckBox, QHBoxLayout, QScrollArea)

class AutoPipelineUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoPipeline - ML/DL Builder")
        self.setGeometry(100, 100, 800, 900)
        
        scroll = QScrollArea()
        container = QWidget()
        layout = QVBoxLayout()

        # Step 1
        step1 = QGroupBox("Step 1: Task Selection")
        step1_layout = QVBoxLayout()
        self.task_combo = QComboBox()
        self.task_combo.addItems([
            "Image Classification", "Object Detection", "Image Segmentation",
            "Text Classification", "Sentiment Analysis", "NER", "Text Generation",
            "Classification", "Regression", "Clustering",
            "Value-based RL", "Policy-based RL", "Anomaly Detection", "Recommendation Systems"
        ])
        step1_layout.addWidget(self.task_combo)
        step1.setLayout(step1_layout)

        # Step 2
        step2 = QGroupBox("Step 2: Data Path/Type")
        step2_layout = QVBoxLayout()
        self.data_type_combo = QComboBox()
        self.data_type_combo.addItems(["Image", "Text", "Structured", "Audio", "Video", "Medical", "Other"])
        self.data_path_edit = QLineEdit()
        self.data_path_edit.setPlaceholderText("Data Path (Optional)")
        step2_layout.addWidget(self.data_type_combo)
        step2_layout.addWidget(self.data_path_edit)
        step2.setLayout(step2_layout)

        # Step 3
        step3 = QGroupBox("Step 3: Model Selection")
        step3_layout = QVBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(["ResNet", "VGG", "YOLOv5", "BERT", "RoBERTa", "Custom"])
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Pretrained Model Path (Optional)")
        step3_layout.addWidget(self.model_combo)
        step3_layout.addWidget(self.model_path_edit)
        step3.setLayout(step3_layout)

        # Step 4
        step4 = QGroupBox("Step 4: Parameters")
        step4_layout = QVBoxLayout()
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["SGD", "Adam", "RMSprop", "AdamW"])
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["CrossEntropyLoss", "MSELoss", "DiceLoss"])
        self.metric_checkboxes = [QCheckBox(m) for m in ["Accuracy", "Precision", "Recall", "F1 Score"]]
        step4_layout.addWidget(self.optimizer_combo)
        step4_layout.addWidget(self.loss_combo)
        for cb in self.metric_checkboxes:
            step4_layout.addWidget(cb)
        step4.setLayout(step4_layout)

        # Buttons
        self.output_text = QTextEdit()
        self.save_run_btn = QPushButton("Save & Run")
        self.clear_btn = QPushButton("Clear")

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.save_run_btn)
        btn_layout.addWidget(self.clear_btn)

        # Inference Section
        inference = QGroupBox("Inference")
        inference_layout = QVBoxLayout()
        self.infer_model_path = QLineEdit()
        self.infer_model_path.setPlaceholderText("Inference Model Path")
        self.infer_prompt = QLineEdit()
        self.infer_prompt.setPlaceholderText("Prompt")
        self.infer_btn = QPushButton("Run Inference")
        self.infer_output = QTextEdit()
        inference_layout.addWidget(self.infer_model_path)
        inference_layout.addWidget(self.infer_prompt)
        inference_layout.addWidget(self.infer_btn)
        inference_layout.addWidget(self.infer_output)
        inference.setLayout(inference_layout)

        # Assemble layout
        layout.addWidget(step1)
        layout.addWidget(step2)
        layout.addWidget(step3)
        layout.addWidget(step4)
        layout.addLayout(btn_layout)
        layout.addWidget(self.output_text)
        layout.addWidget(inference)

        container.setLayout(layout)
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)

        # Connect buttons
        self.save_run_btn.clicked.connect(self.save_run_pipeline)
        self.clear_btn.clicked.connect(self.clear_output)
        self.infer_btn.clicked.connect(self.run_inference)

    def save_run_pipeline(self):
        metrics_selected = [cb.text() for cb in self.metric_checkboxes if cb.isChecked()]
        text = f"Training: Task={self.task_combo.currentText()}, Model={self.model_combo.currentText()}, Optimizer={self.optimizer_combo.currentText()}, Loss={self.loss_combo.currentText()}, Metrics={metrics_selected}"
        self.output_text.setPlainText(text)

    def clear_output(self):
        self.output_text.clear()

    def run_inference(self):
        text = f"Inference using model {self.infer_model_path.text()} with prompt '{self.infer_prompt.text()}'"
        self.infer_output.setPlainText(text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AutoPipelineUI()
    window.show()
    sys.exit(app.exec())
