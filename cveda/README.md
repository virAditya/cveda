CVEDA — Computer Vision Evaluation and Dataset Auditor

CVEDA is a Python package that scans image datasets, checks annotation quality, finds issues, computes distributions, and builds an automatic PDF audit report.
It is designed for teams that want a reliable way to inspect, clean, and validate datasets before training any computer vision model.

Why CVEDA matters

High quality datasets decide the quality of your model. This tool lets you review a dataset in a structured and automated way.
You receive clear checks, detection of corrupted images, distribution summaries, bounding box diagnostics, and many feature modules that point out issues that would normally take hours of manual inspection.

What CVEDA offers
Automatic checks

Missing annotation detection

Empty annotation file detection

Bounding box sanity checks

Missing classes or unused classes

Split validation for train or validation or test folders

Overlapping bounding boxes

Hard negative sample detection

Dataset statistics

Class counts

Bounding box size statistics

Box area distribution

Spatial heatmaps

Cooccurrence matrix

Sample image summaries

Feature modules

A growing collection of more than thirty feature modules that run independently.
These include annotation confidence checks, image quality checks, visual similarity detection, and spatial insights.

Corrupted image finder

Detects unreadable, broken, zero byte, or partially corrupted images.

PDF report generation

Generates a clear multi page PDF audit report with tables, figures, and summaries.
This helps teams share results quickly without writing any manual documentation.

Installation
pip install cveda


If you want PDF reporting support, install extras:

pip install cveda[report]

Quick start
from cveda.api import CVEDA

auditor = CVEDA(
    dataset_path="path_to_dataset",
    generate_report=True,
    report_path="dataset_report.pdf"
)

result = auditor.run_audit()
print(result.keys())


This scans your dataset, runs all checks, computes all distributions, runs the enabled features, and finally creates a PDF report.

Recommended folder format
dataset/
    images/
    annotations/


Supported annotation formats

COCO JSON

Pascal VOC XML

YOLO TXT

Sample output

The result returned by run_audit() is a dictionary containing

image index

checks

feature outputs

distribution statistics

summary levels

report path

You can serialize it as JSON or feed it into your production tools.

Command line usage
cveda audit --data dataset_folder --report out.pdf


Extra arguments let you enable features, change sampling limits, or disable the PDF.

Configuration

The auditor accepts many parameters that control behavior.
Some commonly used parameters are:

Parameter	Description
dataset_path	Path to dataset root
generate_report	Whether to generate a PDF
report_path	Output PDF path
sample_limit	Limit images for heavy checks
workers	Parallel worker count
enable_features	List of feature names
disable_features	Exclude some features
Performance suggestions

CVEDA reads all images once and stores metadata. You can enable caching for faster repeated runs.
For heavy tasks such as duplicate detection or overlap checks, you can set a sample limit to manage speed.

Extending CVEDA

You can add your own feature modules by placing a Python file inside the features folder with a run method.
CVEDA will automatically discover it and include the result in the final audit.

Project goals

CVEDA aims to help computer vision practitioners trust their training data. This package is built for practical use in real projects and continues to grow with more checks and feature modules.

License

MIT License.

Contributing

Pull requests are welcome. Please open an issue if you want a new feature or if you find incorrect behavior.