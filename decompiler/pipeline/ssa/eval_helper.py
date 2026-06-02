import json
from collections import defaultdict
from typing import Dict, List, Set

from decompiler.structures.graphs.cfg import BasicBlock, ControlFlowGraph
from decompiler.structures.pseudo.expressions import Constant, Variable
from decompiler.structures.pseudo.instructions import Assignment
from decompiler.structures.pseudo.operations import BinaryOperation, Call, UnaryOperation

from collections import defaultdict
from typing import Dict, Set

def _get_variables_from_expr(expr) -> List[Variable]:
    """Extrahiert alle Variablen aus einem Ausdruck"""
    variables = []
    for subexpr in expr.subexpressions():
        if isinstance(subexpr, Variable):
            variables.append(subexpr)
    return variables

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
            for block in self.cfg:
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
        self.num_definitions: int = 0
        self.num_usages: int = 0
        self.num_copy_assignments: int = 0
        self.distinct_operators: Set = set()
        self.distinct_operands: Set = set()
        self.total_operators: int = 0
        self.total_operands: int = 0
        self.variables: Dict[str, Dict] = defaultdict(lambda: {
            "definitions": 0,
            "usages": 0,
            "live_ranges": [],
            "scopes": set(),
        })

        self._collect_metrics()

    def _get_all_variables(self) -> Set[Variable]:
        """Gibt alle Variablen im CFG zurück"""
        variables = set()
        for instruction in self.cfg.instructions:
            for expr in instruction.subexpressions():
                if isinstance(expr, Variable):
                    variables.add(expr)
        return variables

    def _extract_operators(self, expr) -> None:
        """Extrahiert alle Operatoren aus einem Ausdruck für Halstead-Metriken"""
        for subexpr in expr.subexpressions():
            if isinstance(subexpr, (BinaryOperation, UnaryOperation, Call)):
                self.distinct_operators.add(subexpr.operation)
                self.total_operators += 1

    def _extract_operands(self, expr) -> None:
        """Extrahiert alle Operanden (Variablen und Konstanten) für Halstead-Metriken"""
        for subexpr in expr.subexpressions():
            if isinstance(subexpr, Variable):
                self.distinct_operands.add(subexpr.name)
                self.total_operands += 1
            elif isinstance(subexpr, Constant):
                # Track constants by (value, type) to distinguish e.g. int 1 vs float 1.0
                operand_key = (subexpr.value, str(subexpr.type))
                self.distinct_operands.add(operand_key)
                self.total_operands += 1

    def _collect_metrics(self) -> None:
        """Sammelt alle Metriken aus dem CFG"""
        try:
            import traceback
            self._count_variables()
            self._count_definitions_and_usages()
            self._count_copy_assignments()
            self._calculate_halstead_metrics()
            self._calculate_live_ranges()
            self._calculate_scopes()
        except Exception as ex:
            traceback.print_exception(ex)

    def _count_variables(self) -> None:
        """Zählt die Anzahl der eindeutigen Variablen (nach name, ohne ssa_label)"""
        variables = set()
        for instruction in self.cfg.instructions:
            for expr in instruction.subexpressions():
                if isinstance(expr, Variable):
                    variables.add(expr)

        unique_names = {var.name for var in variables}
        self.num_variables = len(unique_names)

    def _count_definitions_and_usages(self) -> None:
        """Zählt Definitionen und Nutzungen pro Variable (nach name)"""
        definitions_per_var: Dict[str, int] = defaultdict(int)
        usages_per_var: Dict[str, int] = defaultdict(int)

        for instruction in self.cfg.instructions:
            if isinstance(instruction, Assignment):
                for var in instruction.definitions:
                    definitions_per_var[var.name] += 1
                for var in _get_variables_from_expr(instruction.value):
                    usages_per_var[var.name] += 1

        self.num_definitions = sum(definitions_per_var.values())
        self.num_usages = sum(usages_per_var.values())

        for name, count in definitions_per_var.items():
            self.variables[name]["definitions"] = count
        for name, count in usages_per_var.items():
            self.variables[name]["usages"] = count

    def _count_copy_assignments(self) -> None:
        """Zählt Copy Assignments (a = b wo beide Seiten variablen sind)"""
        for instruction in self.cfg.instructions:
            if isinstance(instruction, Assignment):
                if isinstance(instruction.value, Variable) and isinstance(instruction.destination, Variable):
                    self.num_copy_assignments += 1

    def _calculate_halstead_metrics(self) -> None:
        """Berechnet Halstead-Metriken für alle Anweisungen im CFG"""
        for instruction in self.cfg.instructions:
            if isinstance(instruction, Assignment):
                # Destination ist ein Operand
                for var in instruction.definitions:
                    if isinstance(var, Variable):
                        self.distinct_operands.add(var.name)
                    self.total_operands += 1

                # Value kann Operatoren und Operanden enthalten
                self._extract_operators(instruction.value)
                self._extract_operands(instruction.value)
        
        # Berechne Halstead Metriken
        import math
        
        n1 = len(self.distinct_operators)
        n2 = len(self.distinct_operands)
        N1 = self.total_operators
        N2 = self.total_operands
        
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
        
        self.halstead_distinct_operators = n1
        self.halstead_distinct_operands = n2
        self.halstead_total_operators = N1
        self.halstead_total_operands = N2
        self.halstead_vocabulary = vocabulary
        self.halstead_length = length
        self.halstead_volume = round(halstead_volume, 2)
        self.halstead_difficulty = round(halstead_difficulty, 2)
        self.halstead_effort = round(halstead_effort, 2)
        self.halstead_bugs = round(halstead_bugs, 6)

    def _calculate_live_ranges(self) -> None:
        """
        Berechnet Live-Ranges pro Variable mittels Dataflow-Analyse.
        Die Distanz ist die Anzahl der Instruktionen zwischen Definition und letzter lokaler Nutzung.
        Überlebt eine Variable den Basic Block (LiveOut), wird das Ende des Blocks als letzte Nutzung gewertet.
        """
        # 1. Führe die Dataflow-Analyse aus (benötigt die LivenessDataflowAnalysis Klasse)
        dataflow = LivenessDataflowAnalysis(self.cfg)
        
        # 2. Erstelle ein Mapping von Instruktion zu globalem Index für die Distanzmessung
        global_instr_idx = 0
        instr_mapping = {} 
        
        for block in self.cfg:
            for instr in block.instructions:
                instr_mapping[instr] = global_instr_idx
                global_instr_idx += 1

        # 3. Rückwärts-Analyse pro Basic Block zur Berechnung der exakten Distanzen
        for block in self.cfg:
            # Variablen, die diesen Block überleben (LiveOut)
            currently_live = set(dataflow.live_out[block])
            last_seen_use_index = {} 
            
            # Wenn eine Variable den Block überlebt, setzen wir als ihre "letzte Nutzung" 
            # den Index der allerletzten Instruktion in diesem Block.
            if block.instructions:
                block_end_index = instr_mapping[block.instructions[-1]]
                for var_name in currently_live:
                    last_seen_use_index[var_name] = block_end_index
            
            # Gehe den Block rückwärts durch
            for instr in reversed(block.instructions):
                idx = instr_mapping[instr]
                
                # Prüfe Definitionen (Hier endet die Live-Range beim Rückwärtsgehen)
                if isinstance(instr, Assignment):
                    for var in instr.definitions:
                        if var.name in last_seen_use_index:
                            # Distanz berechnen
                            distance = last_seen_use_index[var.name] - idx
                            self.variables[var.name]["live_ranges"].append(distance)
                            
                            # Variable "stirbt" hier (beim Rückwärtsgehen), also aus dem Tracking entfernen
                            del last_seen_use_index[var.name]
                            currently_live.discard(var.name)
                        else:
                            # Variable wurde definiert, aber danach NIE genutzt (Dead Code)
                            self.variables[var.name]["live_ranges"].append(0)

                # Prüfe Nutzungen (Hier beginnt die Live-Range beim Rückwärtsgehen)
                for var in  instr.requirements:
                    if var.name not in last_seen_use_index:
                        last_seen_use_index[var.name] = idx
                        currently_live.add(var.name)

    def _calculate_scopes(self) -> None:
        """Zählt die Anzahl der Scopes (Basic Blocks) pro Variable"""
        for var in self._get_all_variables():
            seen_blocks = set()

            for bb in self.cfg:
                for instr in bb.instructions:
                    if var in instr.definitions or var in instr.requirements:
                        block_scope = f"BB_{bb.address}"
                        if block_scope not in seen_blocks:
                            self.variables[var.name]["scopes"].add(block_scope)
                            seen_blocks.add(block_scope)


    def to_dict(self) -> Dict:
        """Gibt alle Metriken als Dict zurück"""
        return {
            "num_variables": self.num_variables,
            "num_definitions": self.num_definitions,
            "num_usages": self.num_usages,
            "num_copy_assignments": self.num_copy_assignments,
            # Halstead Metriken
            "halstead_distinct_operators": self.halstead_distinct_operators,
            "halstead_distinct_operands": self.halstead_distinct_operands,
            "halstead_total_operators": self.halstead_total_operators,
            "halstead_total_operands": self.halstead_total_operands,
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
                    "live_ranges": v["live_ranges"],
                    "scopes": list(v["scopes"]),
                }
                for k, v in self.variables.items()
            },
        }

    def toJSON(self) -> str:
        """Gibt alle Metriken als JSON String zurück"""
        return json.dumps(self.to_dict(), indent=2, default=str)
