import json
import math
import traceback
from collections import defaultdict
from typing import Dict, Set

from decompiler.structures.graphs.cfg import BasicBlock, ControlFlowGraph
from decompiler.structures.pseudo.expressions import Constant, Variable
from decompiler.structures.pseudo.instructions import Assignment
from decompiler.structures.pseudo.operations import BinaryOperation, Call, TernaryExpression, UnaryOperation

from collections import defaultdict
from typing import Dict, Set

class LivenessDataflowAnalysis:
    """
    Computes LiveIn and LiveOut sets for a Control Flow Graph out of SSA form
    using standard iterative backward dataflow analysis.
    """
    def __init__(self, cfg: ControlFlowGraph):
        self.cfg = cfg
            
        # Local block sets
        self.use_block: Dict[BasicBlock, Set[str]] = defaultdict(set)
        self.def_block: Dict[BasicBlock, Set[str]] = defaultdict(set)
        
        # Global dataflow sets
        self.live_in: Dict[BasicBlock, Set[str]] = defaultdict(set)
        self.live_out: Dict[BasicBlock, Set[str]] = defaultdict(set)

        self._compute_local_sets()
        self._run_dataflow_iteration()

    def _compute_local_sets(self) -> None:
        """
        Step 1: Compute USE and DEF sets for each block.
        USE = Upward exposed uses (used before defined in this block)
        DEF = Variables defined in this block
        """
        for block in self.cfg: # Assuming self.cfg yields BasicBlocks
            block_use = set()
            block_def = set()

            for instruction in block.instructions:
                # 1. Process Usages FIRST (for instructions like x = x + 1)
                for var in instruction.requirements:
                    if var.name not in block_def:
                        block_use.add(var.name)

                # 2. Process Definitions SECOND
                if isinstance(instruction, Assignment):
                    for var in instruction.definitions:
                        block_def.add(var.name)

            self.use_block[block] = block_use
            self.def_block[block] = block_def

    def _run_dataflow_iteration(self) -> None:
        """
        Step 2: Iterative Fixed-Point Algorithm (Backward Analysis)
        Loops until the IN sets no longer change.
        """
        changed = True
        while changed:
            changed = False
            
            # For dataflow, iterating backwards usually converges faster, 
            # but standard iteration works fine until it hits a fixed point.
            for block in self.cfg.iter_postorder():
                # OUT[B] = Union of IN[S] for all successors S of B
                new_out = set()
                for succ_block in self.cfg.get_successors(block):
                    new_out.update(self.live_in[succ_block])
                
                self.live_out[block] = new_out

                # IN[B] = USE[B] Union (OUT[B] \ DEF[B])
                # Note: 'new_out - self.def_block[block]' means elements in OUT but not in DEF
                new_in = self.use_block[block].union(new_out - self.def_block[block])

                # If the IN set changed, we must keep iterating
                if new_in != self.live_in[block]:
                    self.live_in[block] = new_in
                    changed = True

    def is_live_out(self, block: BasicBlock, var_name: str) -> bool:
        return var_name in self.live_out[block]
        
    def is_live_in(self, block: BasicBlock, var_name: str) -> bool:
        return var_name in self.live_in[block]


class EvalHelper:
    """Helper-Klasse zur Erfassung von Metriken für Out-of-SSA-Translation

    Zählt Variablen NACHher Out-of-SSA - nur nach name (ohne ssa_label)
    Beispiele: x#1 und x#2 werden als "x" gezählt

    Enthält auch Halstead-Metriken:
    - n1 (distinct operators), n2 (distinct operands)
    - N1 (total operators), N2 (total operands)
    - Vocabulary (n = n1 + n2)
    - Length (N = N1 + N2)
    - Volume (V = N * log2(n))
    """

    def __init__(self, cfg: ControlFlowGraph) -> None:
        self.cfg = cfg
        self.num_variables: int = 0
        self.num_copy_assignments: int = 0
        self.variables: Dict[str, Dict] = defaultdict(lambda: {
            "definitions": 0,
            "usages": 0,
            "scopes": 0,
            "max_live_distance": 0,
            "live_ranges": [],
        })

        try:
            self._collect_metrics()
        except:
            traceback.print_exc()

    def _calculate_live_ranges(self) -> None:
        """
        Calculates disjoint [start, end] live intervals per variable using Dataflow Analysis.
        """
        # 1. Run Dataflow Analysis
        dataflow = LivenessDataflowAnalysis(self.cfg)
        
        # 2. Linearize the CFG and map instructions to gap-numbers
        # We use iter_preorder() based on your screenshot to get a flat list of blocks
        
        global_instr_idx = 1
        instr_mapping = {}
        block_ranges = {} 
        
        ordered_blocks = list(self.cfg.iter_preorder())

        for block in ordered_blocks: 
            if not block.instructions:
                continue
                
            start_idx = global_instr_idx
            for instr in block.instructions:
                instr_mapping[instr] = global_instr_idx
                global_instr_idx += 1
                
            # The boundary where this block ends and the next potential block begins.
            # This ensures intervals that span across blocks will mathematically touch.
            end_idx = global_instr_idx 
            block_ranges[block] = (start_idx, end_idx)

        # 3. Backward Analysis to build [start, end] intervals
        for block in ordered_blocks:
            if not block.instructions:
                continue
                
            block_start, block_end = block_ranges[block]
            
            # Variables that survive past this block (LiveOut)
            currently_live = set(dataflow.live_out[block])
            interval_ends = {}
            
            # If a variable is LiveOut, its current interval extends to the block's exit boundary
            for var_name in currently_live:
                interval_ends[var_name] = block_end

            # Traverse backwards through the block
            for instr in reversed(block.instructions):
                idx = instr_mapping[instr]
                
                # DEFs: Variable dies going backwards (This marks the START of a Live Range)
                if isinstance(instr, Assignment):
                    for var in instr.definitions:
                        if var.name in interval_ends:
                            # Close the interval [definition_idx, last_use_idx]
                            start = idx
                            end = interval_ends[var.name]
                            self.variables[var.name].setdefault("live_ranges", []).append([start, end])
                            
                            # Remove from tracking since it's dead above this point
                            del interval_ends[var.name]
                            currently_live.discard(var.name)
                        else:
                            # Dead code: Defined but never used locally or globally.
                            # The range is just the instruction itself.
                            self.variables[var.name].setdefault("live_ranges", []).append([idx, idx])

                # USEs: Variable becomes live going backwards (This marks the END of a new Live Range)
                for var in instr.requirements:
                    if var.name not in interval_ends:
                        interval_ends[var.name] = idx
                        currently_live.add(var.name)

            # Any variables STILL tracked at the top of the block are LiveIn.
            # Their interval spans from the start of the block to their first use inside it.
            for var_name, end_idx in interval_ends.items():
                self.variables[var_name].setdefault("live_ranges", []).append([block_start, end_idx])

        # 4. Clean up and merge adjacent/overlapping intervals across block boundaries
        self._merge_all_live_ranges()

    def _merge_all_live_ranges(self) -> None:
        """
        Helper method to merge touching intervals created by block boundaries,
        calculate the distance for each merged interval, and find the max distance.
        """
        for var_name, data in self.variables.items():
            # Extract the raw [start, end] pairs generated by the backward pass
            raw_ranges = data.get("live_ranges", [])
            if not raw_ranges:
                continue
                
            # 1. Sort intervals by their start time
            sorted_ranges = sorted(raw_ranges, key=lambda r: r[0])
            merged = [sorted_ranges[0]]
            
            # 2. Merge touching or overlapping intervals
            for current_start, current_end in sorted_ranges[1:]:
                last_start, last_end = merged[-1]
                
                if current_start <= last_end:
                    merged[-1] = [last_start, max(last_end, current_end)]
                else:
                    # Disjoint range (reassigned variable)
                    merged.append([current_start, current_end])
                    
            # 3. Calculate distances and the absolute max distance
            final_ranges = []
            max_dist = 0
            
            for start, end in merged:
                dist = end - start
                final_ranges.append({
                    "interval": [start, end],
                    "distance": dist
                })
                
                if dist > max_dist:
                    max_dist = dist
                    
            # 4. Save the enriched data structure back to the variable tracker
            self.variables[var_name]["live_ranges"] = final_ranges
            self.variables[var_name]["max_live_distance"] = max_dist

    def _collect_metrics(self) -> None:
        total_operands =  0
        total_operators = 0
        variables = set()
        constants = set()
        distinct_operators = set()
        var_to_block = defaultdict(set) 


        for block in self.cfg:
            for instruction in block: 
                for expr in instruction.subexpressions():
                    if isinstance(expr, Variable):
                        total_operands += 1
                        variables.add(expr.name)
                        var_to_block[expr.name].add(block.address)

                    elif isinstance(expr, Constant):
                        total_operands += 1
                        constants.add((str(expr.value), str(expr.type)))

                    elif isinstance(expr, (BinaryOperation, UnaryOperation, Call, TernaryExpression)):
                        total_operators += 1
                        distinct_operators.add(expr.operation)


                if isinstance(instruction, Assignment) and isinstance(instruction.destination, Variable):
                    # count definitions
                    self.variables[instruction.destination.name]["definitions"] += 1

                    # count copy assignments
                    if isinstance(instruction.value, Variable):
                        self.num_copy_assignments += 1

                # count usages
                for var in instruction.requirements:
                    self.variables[var.name]["usages"] += 1


        # Setting variable count
        self.num_variables = len(variables)

        # Berechne Halstead Metriken
        n1 = len(distinct_operators)
        n2 = len(variables) + len(constants)
        N1 = total_operators
        N2 = total_operands
        
        # Vocabulary = n1 + n2
        vocabulary = n1 + n2
        # Length = N1 + N2
        length = N1 + N2
        
        # Volume = N * log2(n)
        if vocabulary > 1 and length > 0:
            halstead_volume = length * math.log2(vocabulary)
        else:
            halstead_volume = 0.0
        
        # Difficulty = (n1/2) * (N2/n2)
        if n1 > 0 and n2 > 0:
            halstead_difficulty = (n1 / 2) * (N2 / n2)
        else:
            halstead_difficulty = 0.0
        
        # Effort = Difficulty * Volume
        halstead_effort = halstead_difficulty * halstead_volume
        
        # Bugs = Volume / 3000
        halstead_bugs = halstead_volume / 3000
        
        self.halstead_vocabulary = vocabulary
        self.halstead_length = length
        self.halstead_volume = round(halstead_volume, 2)
        self.halstead_difficulty = round(halstead_difficulty, 2)
        self.halstead_effort = round(halstead_effort, 2)
        self.halstead_bugs = round(halstead_bugs, 6)

        # calculating scopes
        for var_name, blocks in var_to_block.items():
            self.variables[var_name]["scopes"] = len(blocks)

        # calculating live ranges 
        self._calculate_live_ranges()

    def to_dict(self) -> Dict:
        """Gibt alle Metriken als Dict zurück"""
        return {
            "num_variables": self.num_variables,
            "num_copy_assignments": self.num_copy_assignments,
            # Halstead Metriken
            "halstead_vocabulary": self.halstead_vocabulary,
            "halstead_length": self.halstead_length,
            "halstead_volume": self.halstead_volume,
            "halstead_difficulty": self.halstead_difficulty,
            "halstead_effort": self.halstead_effort,
            "halstead_bugs": self.halstead_bugs,
            "variables": {
                k: {
                    "definitions": v["definitions"],
                    "usages": v["usages"],
                    "scopes": v["scopes"],
                    "live_ranges": v["live_ranges"],
                    "max_live_distance": v["max_live_distance"],
                }
                for k, v in self.variables.items()
            },
        }

    def toJSON(self) -> str:
        """Gibt alle Metriken als JSON String zurück"""
        return json.dumps(self.to_dict(), indent=2, default=str)
