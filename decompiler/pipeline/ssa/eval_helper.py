import json
import math
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Union

from decompiler.task import DecompilerTask
from decompiler.structures.graphs.cfg import BasicBlock, ControlFlowGraph
from decompiler.structures.pseudo.expressions import Constant, Variable
from decompiler.structures.pseudo.instructions import Assignment, Branch, Return, Break, Continue
from decompiler.structures.pseudo.operations import BinaryOperation, Call, MemberAccess, Operation, TernaryExpression, UnaryOperation

class ReachingDefinitionsAnalysis:
    """
    Computes Forward Reaching Definitions to map every use to its possible definitions.
    A definition is uniquely identified by: (BasicBlock, instruction_index, variable_name)
    """
    def __init__(self, cfg: ControlFlowGraph, function_parameters: List[Variable]):
        self.cfg = cfg
        self.function_parameters = function_parameters
        
        # Data structure: Tuple[BasicBlock, int, str] -> (block, instr_idx, var_name)
        self.gen: Dict[BasicBlock, Set[Tuple]] = defaultdict(set)
        self.kill: Dict[BasicBlock, Set[Tuple]] = defaultdict(set)
        
        self.reach_in: Dict[BasicBlock, Set[Tuple]] = defaultdict(set)
        self.reach_out: Dict[BasicBlock, Set[Tuple]] = defaultdict(set)

        self.all_defs: Dict[str, Set[Tuple]] = defaultdict(set)
        
        self._compute_local_sets()
        self._run_dataflow_iteration()

    def _compute_local_sets(self) -> None:
        # Map all definitions globally
        for block in self.cfg:
            for idx, instr in enumerate(block.instructions):
                for var in instr.definitions:
                    self.all_defs[var.name].add((block, idx, var.name))

        # Compute GEN and KILL per block
        for block in self.cfg:
            block_gen = set()
            for idx, instr in enumerate(block.instructions):
                for var in instr.definitions:
                    d = (block, idx, var.name)
                    block_gen = {x for x in block_gen if x[2] != var.name }
                    block_gen.add(d)
            self.gen[block] = block_gen

            block_kill = set()
            for d in block_gen:
                var_name = d[2]
                block_kill.update(self.all_defs[var_name] - {d})
            self.kill[block] = block_kill

            self.reach_out[block] = set(block_gen) # Initial guess

    def _run_dataflow_iteration(self) -> None:
        changed = True
        while changed:
            changed = False
            for block in self.cfg.iter_preorder(): # Forward analysis
                new_in = set()

                if block == self.cfg.root:
                    new_in.update({(None, -1, var.name) for var in self.function_parameters})
                for pred in self.cfg.get_predecessors(block):
                    new_in.update(self.reach_out[pred])
                self.reach_in[block] = new_in

                new_out = self.gen[block].union(self.reach_in[block]- self.kill[block])
                if new_out != self.reach_out[block]:
                    self.reach_out[block] = new_out
                    changed = True


class VariableReuseAnalyzer:
    """
    Builds Def-Use webs using Reaching Definitions and networkx.
    Counts connected components per variable to establish a reuse/fragmentation metric.
    """
    def __init__(self, cfg: ControlFlowGraph, function_parameters: List[Variable]):
        self.cfg = cfg
        self.rd = ReachingDefinitionsAnalysis(cfg, function_parameters)
        
        # One graph per variable name
        self.graphs: Dict[str, nx.Graph] = defaultdict(nx.Graph)
        self.reuse_metrics: Dict[str, int] = {}

        self._build_webs()

    def _build_webs(self) -> None:
        for block in self.cfg:
            # Reaching defs at the start of the block
            current_reaching = set(self.rd.reach_in[block])

            for idx, instr in enumerate(block.instructions):
                # Process uses 
                for var in instr.requirements:
                    use_node = ("USE", block, idx, var.name)
                    self.graphs[var.name].add_node(use_node)

                    # Connect all reaching defs of this variable to this use
                    for d in current_reaching:
                        if d[2] == var.name:
                            def_node = ("DEF", d[0], d[1], d[2])
                            self.graphs[var.name].add_edge(def_node, use_node)

                # Updates the running reaching definitions state locally
                for var in instr.definitions:
                    # We add a DEF node here for the case that the definition is never used
                    def_node = ("DEF", block, idx, var.name)
                    self.graphs[var.name].add_node(def_node)
                        
                    new_def = (block, idx, var.name)
                    current_reaching = {d for d in current_reaching if d[2] != var.name}
                    current_reaching.add(new_def)

        # Count connected components
        for var_name, graph in self.graphs.items():
            components = list(nx.connected_components(graph))
            self.reuse_metrics[var_name] = len(components)

    def get_metrics(self) -> Dict[str, int]:
        """Returns a dict mapping variable names to their web count."""
        return self.reuse_metrics

class LivenessAnalysis:
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
        for block in self.cfg: 
            block_use = set()
            block_def = set()

            for instruction in block.instructions:
                # Process Usages first
                for var in instruction.requirements:
                    if var.name not in block_def:
                        block_use.add(var.name)

                # Process Definitions second
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
            
            for block in self.cfg.iter_postorder(): #type: ignore
                new_out = set()

                for succ_block in self.cfg.get_successors(block):
                    new_out.update(self.live_in[succ_block])
                self.live_out[block] = new_out

                new_in = self.use_block[block].union(new_out - self.def_block[block])
                if new_in != self.live_in[block]:
                    self.live_in[block] = new_in
                    changed = True

    def is_live_out(self, block: BasicBlock, var_name: str) -> bool:
        return var_name in self.live_out[block]
        
    def is_live_in(self, block: BasicBlock, var_name: str) -> bool:
        return var_name in self.live_in[block]


class LivenessRangeAnalyzer:
    def __init__(self, cfg: ControlFlowGraph) -> None:
        self.cfg = cfg
        self.la = LivenessAnalysis(self.cfg)

        self.max_dist = defaultdict(int) 
        self.live_ranges = defaultdict(list)

        self._calculate_live_ranges()

    def _merge_all_live_ranges(self) -> None:
        for var_name, data in self.live_ranges.items():
            if not data:
                continue
                
            # Sort intervals by their start time
            sorted_ranges = sorted(data, key=lambda r: r[0])
            merged = [sorted_ranges[0]]
            
            # Merge touching or overlapping intervals
            for current_start, current_end in sorted_ranges[1:]:
                last_start, last_end = merged[-1]
                
                if current_start <= last_end:
                    merged[-1] = (last_start, max(last_end, current_end))
                else:
                    # Disjoint 
                    merged.append((current_start, current_end))
                    
            # Calculate distances and the absolute max distance
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
                    
            # Save the enriched data structure back to the variable tracker
            self.max_dist[var_name] = max_dist
            self.live_ranges[var_name] = final_ranges


    def _calculate_live_ranges(self) -> None:
        global_instr_idx = 1
        instr_mapping = {}
        block_ranges = {} 

        # Build block range
        for block in self.cfg.iter_preorder():  #type: ignore
            if not block.instructions:
                continue
                
            start_idx = global_instr_idx

            for instr in block.instructions:
                instr_mapping[instr] = global_instr_idx
                global_instr_idx += 1
                
            end_idx = global_instr_idx 
            block_ranges[block] = (start_idx, end_idx)
 
        # Backward Analysis to build [start, end] intervals relative to each block's boundary
        for block in self.cfg.iter_preorder(): #type: ignore
            if not block.instructions:
                continue
                
            block_start, block_end = block_ranges[block]
            
            interval_ends = {}
            currently_live = set(self.la.live_out[block])
            
            # If a variable is LiveOut, its current interval extends to the block's exit boundary
            for var_name in currently_live:
                interval_ends[var_name] = block_end

            # Traverse backwards through the block
            for instr in reversed(block.instructions):
                idx = instr_mapping[instr]
                
                # DEFs: Variable dies going backwards 
                for var in instr.definitions:
                    if var.name in interval_ends:
                        start = idx
                        end = interval_ends[var.name]
                        self.live_ranges[var.name].append((start, end))
                        
                        # Remove from tracking since it's dead above this point
                        del interval_ends[var.name]
                        currently_live.discard(var.name)
                    else:
                        # Dead code: Defined but never used locally or globally.
                        # The range is just the instruction itself.
                        self.live_ranges[var.name].append((idx, idx))

                # USEs: Variable becomes live going backwards (This marks the END of a new Live Range)
                for var in instr.requirements:
                    if var.name not in interval_ends:
                        interval_ends[var.name] = idx
                        currently_live.add(var.name)

            # Any variables STILL tracked at the top of the block are LiveIn.
            # Their interval spans from the start of the block to their first use inside it.
            for var_name, end_idx in interval_ends.items():
                self.live_ranges[var_name].append((block_start, end_idx))

        # Merge adjacent/overlapping intervals across block boundaries
        self._merge_all_live_ranges()

    def get_metrics(self):
        return { 
            var_name: (self.live_ranges[var_name], 
                        self.max_dist[var_name]) for var_name in 
                        self.max_dist.keys()
        }


class EvalHelper:
    """
    Helper-Klasse zur Erfassung von Metriken für Out-of-SSA-Translation
    """

    def __init__(self, decompiler_task: DecompilerTask) -> None:
        self.task: DecompilerTask  = decompiler_task
        self.cfg: ControlFlowGraph = decompiler_task.cfg #type: ignore

        self.num_variables: int = 0
        self.num_copy_assignments: int = 0
        self.total_operands: int = 0
        self.distinct_operands: int = 0
        self.total_operators: int = 0
        self.distinct_operators: int = 0
        self.variables: Dict[str, Dict] = defaultdict(lambda: {
            "definitions": 0,
            "usages": 0,
            "scopes": 0,
            "max_live_distance": 0,
            "live_ranges": [],
        })

        self._collect_metrics()


    def _calculate_live_ranges(self) -> None:
        lr_analyzer = LivenessRangeAnalyzer(self.cfg)
        for var_name, (live_ranges, max_dist) in lr_analyzer.get_metrics().items():
            self.variables[var_name]["live_ranges"] = live_ranges
            self.variables[var_name]["max_live_distance"] = max_dist 


    def _calculate_reuses(self) -> None:
        reuse_analyzer = VariableReuseAnalyzer(self.cfg, self.task.function_parameters)
        reuse_metrics = reuse_analyzer.get_metrics()

        for var_name, web_count in reuse_metrics.items():
            self.variables[var_name]["disjoint_webs"] = web_count

    def _calculate_basic_metrics(self) -> None:
        # count distinct variables
        variables = set()
        for var in self.task.function_parameters:
            variables.add(var.name)

        # count scopes / blocks
        var_to_block = defaultdict(set)

        for block in self.cfg:
            for instr in block:
                # count definitions
                for var in instr.definitions:
                    variables.add(var.name)
                    var_to_block[var.name].add(block)
                    self.variables[var.name]["definitions"] += 1

                # count usages
                for var in instr.requirements:
                    variables.add(var.name)
                    var_to_block[var.name].add(block)
                    self.variables[var.name]["usages"] += 1

                # count copy assignments
                if isinstance(instr, Assignment) and isinstance(instr.destination, Variable) and isinstance(instr.value, Variable):
                    self.num_copy_assignments += 1

        self.num_variables = len(variables)
        for var_name, scopes in var_to_block.items():
            self.variables[var_name]["scopes"] = len(scopes)

    def _calculate_halstead_metrics(self) -> None:
        total_operands =  0
        total_operators = 0
        distinct_operands = set()
        distinct_operators = set()

        for instr in self.cfg.instructions:
            for expr in instr.subexpressions():
                match expr:
                    # operands
                    case Variable():
                        total_operands += 1
                        distinct_operands.add(expr.name)
                    case Constant():
                        total_operands += 1
                        distinct_operands.add((str(expr.value), str(expr.type)))    
                    # operators
                    case Branch():
                        total_operators += 1
                        distinct_operators.add("If")
                    case Return():
                        total_operators += 1
                        distinct_operators.add("Return")
                    case Break():
                        total_operators += 1
                        distinct_operators.add("Break")
                    case Continue():
                        total_operators += 1
                        distinct_operators.add("Continue")
                    case Assignment():
                        total_operators += 1
                        distinct_operators.add("=(Assignment)")
                    case Operation():
                        total_operators += 1
                        distinct_operators.add(expr.operation)

        # Berechne Halstead Metriken
        n1 = len(distinct_operators)
        n2 = len(distinct_operands)
        N1 = total_operators
        N2 = total_operands

        self.total_operands = N2 
        self.total_operators = N1
        self.distinct_operands = n2
        self.distinct_operators = n1
        
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

    def _collect_metrics(self) -> None:
        self._calculate_basic_metrics()
        self._calculate_halstead_metrics()
        self._calculate_live_ranges()
        self._calculate_reuses()


    def to_dict(self) -> Dict:
        """Gibt alle Metriken als Dict zurück"""
        return {
            "num_variables": self.num_variables,
            "num_copy_assignments": self.num_copy_assignments,
            # Halstead Metriken
            "total_operators": self.total_operators,
            "distinct_operators": self.distinct_operators,
            "total_operands": self.total_operands,
            "distinct_operands": self.distinct_operands,
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
                    "disjoint_webs": v.get("disjoint_webs", 1),
                }
                for k, v in self.variables.items()
            },
        }

    def toJSON(self) -> str:
        """Gibt alle Metriken als JSON String zurück"""
        return json.dumps(self.to_dict(), indent=2, default=str)
