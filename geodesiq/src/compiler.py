from typing import Callable, Optional, List

import numpy as np
import qiskit
import qiskit.circuit
import qiskit.quantum_info
from qiskit_qasm3_import import parse

import rustworkx as rx

from . import optimizer 
from . import utils

class Compiler:
    """
    Taking a quantum circuit and by controlling the use of the various compilation engines 
    compiling the circuit.

    Parameters
    ----------
    circuit : circuit.Circuit
        The circuit that needs to be compiled.
    restriction : lie.Restriction
        The hardware available interactions and restrictions
    precision : float, default=0.999
        Precision of compiled gates.
    max_steps : int, default=1000
        Number of steps before an optimization in the compiler is deemed to have failed.
    max_step_size : float, default=2.

    Attributes
    ----------
    circuit : circuit.Circuit
        Stores the target circuit as a Circuit object.
    blocks : circuit.Circuit
        Stores the final circuit blocks to be compiled.
    gates : dictionary
        Dictionary of Hamiltonian objects that generate the desired circuit blocks.
    projected_basis : basis.Basis
        The basis of the restricted Hamiltonian
    init_parameters : np.ndarray
    max_steps : int
    precision : float
    """
    def __init__(self, 
                 circuit: Optional[qiskit.circuit.QuantumCircuit] = None, 
                 qasm: Optional[str] = None):
        if circuit == None and qasm == None:
            raise ValueError("Either a circuit or a qasm file must be provided.")
        elif circuit and qasm:
            raise ValueError("Only one of circuit or qasm file can be provided.")
        if circuit:
            self.circuit = circuit
        elif qasm: 
            self.circuit = parse(qasm)
        
        if not Compiler.is_circuit_only_gates(self.circuit):
            raise ValueError("The circuit must only contain Gate instructions for compilation"
                             " in the current version.")
        self.optimized_gate_parameters = {}

        self.qubits = Compiler.get_qubits(self.circuit)
        self.gates = self.circuit.data
        self.graph = Compiler.build_graph_from_circuit(self.circuit)
    
    def draw_graph(self, filename : str) -> None:
        def node_label(node):
            if self.is_instruction_a_gate(node):
                return node.operation.name
            else:
                return node

        rx.visualization.graphviz_draw(
            self.graph,
            node_attr_fn=lambda node: {"label": str(node_label(node))},
            edge_attr_fn=lambda edge: {"label": str(edge)},
            filename=filename
        )

    def draw_circuit(self, filename : str) -> None:
        self.circuit.draw(output="mpl", filename=filename)

    def compile(self) -> qiskit.circuit.QuantumCircuit:
        """
        Compiles the circuit from the graph representation.
        """
        fusing = True
        node = 2 * len(self.qubits)
        nodes = list(range(2 * len(self.qubits), len(self.graph.nodes())))
        while fusing:
            is_fused = False
            node = nodes[0]
            next_nodes = self.graph.adj_direction(node, False)
            for next_node in next_nodes:
                if Compiler.is_instruction_a_gate(self.graph.get_node_data(next_node)):
                    paths = rx.all_simple_paths(self.graph, node, next_node)
                    if all([len(p) == 2 for p in paths]):
                        circ_instr, is_fused = self._fuse_gates([node, next_node])
                        if is_fused:
                            print(f"Contracting nodes {node} and {next_node} with gates " 
                                  f"{self.graph.get_node_data(node).operation.name} and "
                                  f"{self.graph.get_node_data(next_node).operation.name}.")
                            new_node = self.graph.contract_nodes([node, next_node], circ_instr)
                            nodes.remove(next_node)
                            nodes[0] = new_node
                            break 
            if not is_fused:
                nodes.pop(0)
            if not len(nodes):
                fusing = False

        self.circuit = Compiler.build_circuit_from_graph(self.graph, self.circuit.qubits)
        self.qubits = Compiler.get_qubits(self.circuit)
        self.gates = self.circuit.data
        return self.circuit
    
    def _fuse_gates(self, nodes: List[int]) -> bool:
        circ = qiskit.circuit.QuantumCircuit.from_instructions([self.graph.get_node_data(node) for node in nodes])
        operation = qiskit.quantum_info.Operator.from_circuit(circ)
        opt = optimizer.Optimizer(target_unitary=operation.data, 
                        full_basis=utils.construct_full_pauli_basis(operation.num_qubits),
                        projected_basis=utils.construct_two_body_pauli_basis(operation.num_qubits),
                        commute=False,
                        max_steps=1000,
                        max_step_size=0.6,
                        gram_schmidt_step_size=1.3,
                        )
        if opt.is_succesful:
            gate_num = self._add_optimized_gate_parameter(opt.parameters[-1])
            new_op = qiskit.circuit.library.UnitaryGate(data=operation.data,
                                                        num_qubits=operation.num_qubits)
            new_op.name = f"U_{gate_num}"
            circ_instr = qiskit.circuit.CircuitInstruction(new_op, circ.qubits)
            return circ_instr, True
        else:
            return None, False

    def _add_optimized_gate_parameter(self, parameters : np.array) -> int:
        gate_num = len(self.optimized_gate_parameters) + 1
        self.optimized_gate_parameters[f"U_{gate_num}"] = parameters
        return gate_num

    @staticmethod
    def build_graph_from_circuit(circuit: qiskit.circuit.QuantumCircuit) -> rx.PyDAG:
        if not Compiler.is_circuit_only_gates(circuit):
            raise ValueError("Can only build a graph of circuits that only have Gates currently.")
        graph = rx.PyDAG()

        # Add the qubits as nodes given by strings
        qubit_node_map = {q : i  for i, q in enumerate(Compiler.get_qubits(circuit))}
        qubit_node_map_out = {q : i + len(qubit_node_map) for i, q in enumerate(Compiler.get_qubits(circuit))}
        in_nodes = graph.add_nodes_from(list(qubit_node_map.keys()))
        out_nodes = graph.add_nodes_from(list(qubit_node_map.keys()))

        # Create dictionary of gates applied to each qubit
        qubit_gates = {q : [] for q in Compiler.get_qubits(circuit)}

        # List of gates with qubit indices
        gates = [(g, [Compiler.get_qubit_name(q) for q in g.qubits]) for g in circuit.data]

        # Add the gates with the qubits as edge labels
        for gate, qubits in gates:
            gate_node = graph.add_node(gate)
            for q in qubits:
                if not qubit_gates[q]:
                    graph.add_edge(qubit_node_map[q], gate_node, q)
                else:
                    graph.add_edge(qubit_gates[q][-1], gate_node, q)
                qubit_gates[q].append(gate_node)
        
        # Add the edges to the out qubit nodes
        for q in qubit_gates:
            graph.add_edge(qubit_gates[q][-1], qubit_node_map_out[q], q)

        return graph

    @staticmethod
    def build_circuit_from_graph(graph: rx.PyDAG, qubits: qiskit.circuit.Qubit) -> qiskit.circuit.QuantumCircuit:
        instructions = [node for node in graph.nodes() if issubclass(type(node), qiskit.circuit.CircuitInstruction)]
        return qiskit.circuit.QuantumCircuit.from_instructions(instructions, qubits=qubits)
        
    @staticmethod
    def is_instruction_a_gate(instruction: qiskit.circuit.Instruction | str) -> bool:
        if type(instruction) is str: 
            return False
        else:
            return issubclass(instruction.operation.base_class, qiskit.circuit.Gate)
    
    @staticmethod
    def is_circuit_only_gates(circuit: qiskit.circuit.QuantumCircuit) -> bool:
        if all([Compiler.is_instruction_a_gate(instr) for instr in circuit.data]):
            return True
        else:
            return False

    @staticmethod
    def get_qubits(circuit: qiskit.circuit.QuantumCircuit) -> List[str]:
        qubits = []
        for qreg in circuit.qregs:
            for q in qreg:
                qubits.append(Compiler.get_qubit_name(q))
        return qubits
    
    @staticmethod
    def get_qubit_name(qubit: qiskit.circuit.Qubit) -> str:
        return qubit._register.name + str(qubit._index)