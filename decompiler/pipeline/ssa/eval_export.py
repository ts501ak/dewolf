import json
import os
import sys

from decompiler.pipeline.ssa.eval_helper import EvalHelper
from decompiler.pipeline.stage import PipelineStage
from decompiler.task import DecompilerTask
import traceback
from decompiler.util.decoration import DecoratedCFG


class SsaEvalExport(PipelineStage):
    """Pipeline stage that exports SSA evaluation metrics to a JSON file."""

    name = "ssa-eval-export"
    dependencies = ["out-of-ssa-translation"]

    def run(self, task: DecompilerTask) -> None:
        """Export SSA evaluation metrics to JSON file specified by HEINZ_PETER env var."""
        try: 
            eval_helper = EvalHelper(task.graph)
            eval_helper._collect_metrics()
            metrics_dict = eval_helper.to_dict()
            print(DecoratedCFG(task.graph).get_ascii(task.graph))
            print(metrics_dict)
            
            output_path = os.environ.get("HEINZ_PETER")
            if output_path:
                with open(output_path, "w") as f:
                    json.dump(metrics_dict, f, indent=2, default=str)
            else:
                self.logger.warning("HEINZ_PETER environment variable not set, skipping export")
            
            sys.exit(0)
        except Exception as e:
            traceback.print_exception(e)

