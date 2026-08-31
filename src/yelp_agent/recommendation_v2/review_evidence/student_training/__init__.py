"""把教师标签转换成学生模型可直接训练和评测的数据。"""

from .dataset_builder import BuildConfig, build_student_dataset, validate_student_dataset

__all__ = ["BuildConfig", "build_student_dataset", "validate_student_dataset"]
