import json
import os
import sys

from decompiler.pipeline.ssa.eval_helper import EvalHelper
from decompiler.pipeline.stage import PipelineStage
from decompiler.task import DecompilerTask
from decompiler.util.decoration import DecoratedCFG 


class SSAEvalExport(PipelineStage):
    """Pipeline stage that exports SSA evaluation metrics to a JSON file."""

    name = "ssa-eval-export"
    dependencies = ["out-of-ssa-translation"]

    def run(self, task: DecompilerTask) -> None:
        """Export SSA evaluation metrics to JSON file specified by HEINZ_PETER env var."""
        eval_helper = EvalHelper(task.graph)
        metrics_dict = eval_helper.to_dict()
        
        output_path = os.environ.get("HEINZ_PETER")
        if output_path:
            with open(output_path, "w") as f:
                json.dump(metrics_dict, f, indent=2, default=str)

        #png_path = os.environ.get("HEINZ_P_DOG")
        #if png_path:
        #    DecoratedCFG.from_cfg(task.graph).export_plot(png_path, "png")
        
        sys.exit(0)
