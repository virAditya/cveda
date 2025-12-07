"""
Text templates used by the report generator.

Keep templates small and safe for rendering in different environments.
"""

SUMMARY_TEMPLATE = """
Dataset summary

Number of images: {n_images}
Number of classes: {n_classes}
Top problem classes: {problem_classes}
"""
