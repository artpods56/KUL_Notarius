from orchestration.assets.export import PandasDataFrameExport
from orchestration.constants import AssetLayer, DataSource
from orchestration.utils import AssetKeyHelper

"""
Asset: [[export.py#eval__export_dataframe__pandas]]
Defined in: src/orchestration/assets/export.py
Resolves to: mrt__huggingface__eval__export_dataframe__pandas
"""
EVAL_EXPORT_DATAFRAME_PANDAS = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.MRT, DataSource.HUGGINGFACE, "eval", "export_dataframe", "pandas"
    ): {"config": PandasDataFrameExport(file_name="schematism_comp.xlsx").model_dump()}
}
