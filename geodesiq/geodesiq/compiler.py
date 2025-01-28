from typing import Callable, Optional, List, Tuple, Dict

import numpy as np
import qiskit
import qiskit.circuit
import qiskit.quantum_info
from qiskit_qasm3_import import parse

import rustworkx as rx

from . import optimizer 
from . import utils
from . import lie


class NodeTree(rx.visit.BFSVisitor):
    def __init__(self):
        self.nodes = [] 

    def discover_vertex(self, v):
        self.nodes.append(v)


class Compiler:
    """
    Compiler class for quantum circuits, designed to handle compilation from a Qiskit 
    QuantumCircuit or QASM input, build circuit graphs, optimize gate parameters, and 
    apply fusion algorithms to reduce circuit complexity.

    Parameters
    ----------
    circuit : qiskit.circuit.QuantumCircuit, optional
        A Qiskit QuantumCircuit object that defines the quantum circuit to compile. 
        Either `circuit` or `qasm` must be provided, but not both.
    qasm : str, optional
        A string containing the QASM (Quantum Assembly Language) representation of 
        the quantum circuit. Either `qasm` or `circuit` must be provided, but not both.

    Raises
    ------
    ValueError
        If neither `circuit` nor `qasm` are provided, or if both are provided. Additionally,
        raises if the circuit contains more than one quantum register or includes non-Gate 
        instructions.

    Attributes
    ----------
    circuit : qiskit.circuit.QuantumCircuit
        The QuantumCircuit object representing the compiled circuit.
    qubits : List[str]
        A list of qubit names in the circuit.
    gates : List[qiskit.circuit.Instruction]
        The list of gate instructions from the QuantumCircuit.
    graph : rx.PyDAG
        A directed acyclic graph (DAG) representing the quantum circuit structure.
    optimized_gate_parameters : dict
        A dictionary storing optimized gate parameters indexed by the gate name.

    Methods
    -------
    draw_graph(filename=None)
        Visualizes and optionally saves the graph representation of the circuit.
    
    draw_circuit(filename=None)
        Draws the circuit diagram and optionally saves it to a file.

    get_unitary()
        Returns the unitary matrix representation of the circuit as a NumPy array.

    compile(restriction=[], fusion_engine='sequential', transpile_gate_set=[], max_qubits=6, max_steps=1000)
        Compiles the quantum circuit based on the graph representation using specified 
        fusion algorithms and optimization strategies.

    build_graph_from_circuit(circuit)
        Converts a QuantumCircuit into a PyDAG graph representation.

    build_circuit_from_graph(graph, qubits)
        Reconstructs a QuantumCircuit from its graph representation.

    is_instruction_a_gate(instruction)
        Checks if the given instruction is a Gate operation.

    is_circuit_only_gates(circuit)
        Verifies if the circuit contains only gate instructions.

    is_edge_fuseable(graph, edge)
        Determines if an edge between two nodes in the graph is fuseable.

    gate_nodes(graph)
        Returns a list of gate nodes in the graph.

    gate_edges(graph, only_fuseable_edges=True)
        Returns a list of gate edges in the graph, optionally filtering to fuseable edges.

    edge_parameter_ratio(graph, restriction, edge)
        Computes the parameter ratio for an edge between two nodes in the graph.

    gate_edge_parameter_ratios(graph, restriction)
        Returns a dictionary mapping edges to their parameter ratios.

    edge_sparsity(graph, restriction, edge)
        Computes the sparsity of a gate operation along an edge in the graph.

    gate_edge_sparsity(graph, restriction)
        Returns a dictionary mapping edges to their sparsity values.

    get_qubits(circuit)
        Retrieves a list of qubit names in the quantum circuit.

    get_qubit_name(qubit)
        Generates a string name for a given qubit.

    circuit_from_nodes(graph, nodes)
        Constructs a QuantumCircuit from a list of graph nodes.
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

        if len(self.circuit.qregs) > 1:
            raise ValueError("The circuit must only contain one quantum register currently.")
        
        if not Compiler.is_circuit_only_gates(self.circuit):
            raise ValueError("The circuit must only contain Gate instructions for compilation"
                             " in the current version.")
        self.optimized_gate_parameters = {}

        self.qubits = Compiler.get_qubits(self.circuit)
        self.gates = self.circuit.data
        self.graph = Compiler.build_graph_from_circuit(self.circuit)
    
    def draw_graph(self, filename : str | None = None) -> None:
        """
        Draw the graph representation of the circuit.

        Parameters
        ----------
        filename : str, optional
            If provided, the graph visualization will be saved to this file.
        """
        def node_label(node):
            if self.is_instruction_a_gate(node):
                return node.operation.name
            else:
                return node
        return rx.visualization.graphviz_draw(
            self.graph,
            node_attr_fn=lambda node: {"label": str(node_label(node))},
            edge_attr_fn=lambda edge: {"label": str(edge)},
            filename=filename
        )

    def draw_circuit(self, filename : str | None = None) -> None:
        """
        Draw the quantum circuit diagram.

        Parameters
        ----------
        filename : str, optional
            If provided, the circuit diagram will be saved to this file.
        """
        return self.circuit.draw(output="mpl", filename=filename)

    def get_unitary(self) -> np.ndarray:
        """
        Compute and return the unitary matrix of the quantum circuit.

        Returns
        -------
        np.ndarray
            The unitary matrix representation of the circuit.
        """
        return qiskit.quantum_info.Operator.from_circuit(self.circuit).data

    def compile(self, 
                restriction: List[str] = [],
                fusion_engine: str = "sequential",
                max_fusion_blocks: int = 2,
                transpile_gate_set: List[str] = [],
                max_qubits: int = 6, 
                max_steps: int = 1000) -> qiskit.circuit.QuantumCircuit:
        """
        Compile the circuit based on the specified options.

        Parameters
        ----------
        restriction : List[str], optional
            A list of restrictions on allowed gate operations.
        fusion_engine : str, optional
            The fusion engine to use for optimization.
        transpile_gate_set : List[str], optional
            The gate set to transpile the circuit into.
        max_qubits : int, optional
            The maximum number of qubits allowed in fused gates.
        max_steps : int, optional
            The maximum number of optimization steps.

        Returns
        -------
        qiskit.circuit.QuantumCircuit
            The compiled QuantumCircuit.
        """
        if restriction:
            self.restriction = restriction
        else: 
            self.restriction = utils.check_2_local_comb

        self.max_fusion_blocks = max_fusion_blocks

        if transpile_gate_set:
            self.circuit = qiskit.compiler.transpile(self.circuit, basis_gates=transpile_gate_set)
            print(f"Transpiled circuit with gate set {transpile_gate_set}")
            self.draw_circuit("circuit_graphs/circuit_transpiled_circuit_test.png")

        if fusion_engine == "sequential":
            self._sequential_fusion(max_qubits, max_steps)
        elif fusion_engine == "minimal_parameter_ratio":
            self._minimal_parameter_ratio_fusion(max_qubits, max_steps)
        elif fusion_engine == "sparsity":
            self._sparsity_fusion(max_qubits, max_steps)
        elif fusion_engine == "deep_blocks":
            self._deep_block_fusion(max_qubits, max_steps)
        else:
            raise ValueError(f"Compiler {fusion_engine} not recognised.")
        
        self.circuit = Compiler.build_circuit_from_graph(self.graph, self.circuit.qubits)
        self.qubits = Compiler.get_qubits(self.circuit)
        self.gates = self.circuit.data
        return self.circuit

    def circuit_distance(self):
        """
        Compute the Frobenius distance of the circuit.
        """
        return Compiler.frobenius_distance_for_circuit(self.circuit)

    def _generate_next_block(self, node: int, max_qubits: int) -> List[int]:
        gates = NodeTree()
        rx.bfs_search(self.graph, [node], gates)
        
        qubit_indices = []
        block_indices = []
        for index in gates.nodes:
            if not Compiler.is_instruction_a_gate(self.graph.get_node_data(index)):
                continue
            qs = [e[-1] for e in self.graph.in_edges(index) if e[-1]]
            qubit_indices = list(set(qs + qubit_indices))
            if len(qubit_indices) <= max_qubits:
                block_indices.append(index)
            else:
                break
            if len(block_indices) == self.max_fusion_blocks:
                break
        return block_indices

    def _fuse_gates(self, 
                    nodes: List[int], 
                    max_qubits: int, 
                    max_steps: int,
                    commute: bool = False,
                    max_step_size: float = 1.1,
                    gram_schmidt_step_size: float = 1.3,
                    init_parameters_spread: float = 0.05,
                    multi_gate=False) -> Tuple[qiskit.circuit.Operation | None, bool, str]:
        """
        Fuse gates for a set of graph nodes.

        Parameters
        ----------
        nodes : List[int]
            The list of graph nodes to fuse.
        max_qubits : int
            The maximum allowed qubits in the fused gate.
        max_steps : int
            The maximum number of optimization steps.
        commute : bool, optional
            Whether to allow commuting gates during fusion.
        max_step_size : float, optional
            The maximum step size for optimization.
        gram_schmidt_step_size : float, optional
            The Gram-Schmidt step size during parameter optimization.
        init_parameters_spread : float, optional
            Initial parameter spread for optimization.

        Returns
        -------
        Tuple[qiskit.circuit.Operation | None, bool, str]
            A tuple with the fused operation, success flag, and new gate name.
        """
        circ = Compiler.circuit_from_nodes(self.graph, nodes)
        operation = qiskit.quantum_info.Operator.from_circuit(circ)
        if operation.num_qubits > max_qubits:
            print(f"Contracting nodes with gates "
                    f"{[self.graph.get_node_data(node).operation.name for node in nodes]} "
                    f"would give a gate with more than " 
                    f"the maximum {max_qubits} qubits.")
            return None, False, "max"

        full_basis = utils.construct_full_pauli_basis(operation.num_qubits)
        projected_basis = utils.construct_restricted_pauli_basis(operation.num_qubits, self.restriction)

        opt = optimizer.Optimizer(target_unitary=operation.data, 
                                    full_basis=full_basis,
                                    projected_basis=projected_basis,
                                    commute=commute,
                                    max_steps=max_steps,
                                    max_step_size=max_step_size,
                                    gram_schmidt_step_size=gram_schmidt_step_size,
                                    init_parameters_spread=init_parameters_spread,
                                    )
        if opt.is_succesful:
            gate_num = self._add_optimized_gate_parameter(opt.parameters[-1])
            new_op = qiskit.circuit.library.UnitaryGate(data=operation.data,
                                                        num_qubits=operation.num_qubits)
            new_op.name = f"U_{gate_num}"
            circ_instr = qiskit.circuit.CircuitInstruction(new_op, circ.qubits)
            print(f"Contracting nodes with gates " 
                    f"{[self.graph.get_node_data(node).operation.name for node in nodes]} "
                    f"to give gate {new_op.name}.")
            new_node = self.graph.contract_nodes(nodes, circ_instr)
            return [new_node], True, [new_op.name]
        elif multi_gate:
            circ_instrs = []
            new_op_names = []
            parameters = utils.prepare_random_parameters(opt.free_indices, 
                                                                opt.commuting_ansatz_matrix, 
                                                                spread=opt.init_parameters_spread)
            target_unitary = operation.data
            while not opt.is_succesful:
                opt = optimizer.Optimizer(target_unitary=target_unitary, 
                                            full_basis=full_basis,
                                            projected_basis=projected_basis,
                                            init_parameters=parameters, 
                                            commute=commute,
                                            max_steps=1,
                                            max_step_size=max_step_size,
                                            gram_schmidt_step_size=gram_schmidt_step_size,
                                            )
                gate_num = self._add_optimized_gate_parameter(opt.parameters[-1])
                unitary = opt.hamiltonian.unitary
                target_unitary = target_unitary @ unitary.conj().T 
                new_op = qiskit.circuit.library.UnitaryGate(data=unitary,
                                                            num_qubits=operation.num_qubits)
                new_op.name = f"U_{gate_num}"
                new_op_names.append(new_op.name)
                circ_instrs.append(qiskit.circuit.CircuitInstruction(new_op, circ.qubits))
                print(f"Contracting nodes with gates " 
                        f"{[self.graph.get_node_data(node).operation.name for node in nodes]} "
                        f"to give gates {new_op_names}.")
                if opt.is_succesful:
                    new_nodes = []
                    for circ_instr in circ_instrs:
                        if not new_nodes:
                            ref_node = self.graph.contract_nodes(nodes, circ_instr)
                        else:
                            ref_node = self.graph.insert_node_on_out_edges(ref_node, circ_instr)
                        new_nodes.append(new_nodes)
                    return new_nodes, True, new_op_names
                parameters = opt.parameters[-1]
        else:
            return None, False, ""

    def _add_optimized_gate_parameter(self, parameters : np.array) -> int:
        """
        Store optimized parameters for a gate and return the gate index.

        Parameters
        ----------
        parameters : np.array
            The optimized parameters.

        Returns
        -------
        int
            The index of the newly added gate parameters.
        """
        gate_num = len(self.optimized_gate_parameters) + 1
        self.optimized_gate_parameters[f"U_{gate_num}"] = parameters
        return gate_num

    def _sequential_fusion(self, max_qubits: int, max_steps: int) -> rx.PyDAG:
        """
        Perform sequential fusion of gates.

        Parameters
        ----------
        max_qubits : int
            The maximum allowed qubits in the fused gate.
        max_steps : int
            The maximum number of optimization steps.

        Returns
        -------
        rx.PyDAG
            The updated graph with fused nodes.
        """
        nodes = Compiler.gate_nodes(self.graph)
        node = nodes[0]
        max_qubit_nodes = []
        fusing = True
        while fusing:
            is_fused = False
            node = nodes[0]
            next_nodes = self.graph.adj_direction(node, False)
            for next_node in next_nodes:
                if next_node in max_qubit_nodes:
                    continue
                if Compiler.is_instruction_a_gate(self.graph.get_node_data(next_node)):
                    # paths = rx.all_simple_paths(self.graph, node, next_node)
                    # if all([len(p) == 2 for p in paths]):
                    if Compiler.is_edge_fuseable(self.graph, (node, next_node)):
                        if Compiler.edge_parameter_ratio(self.graph, self.restriction, (node, next_node)) > 1:
                            continue
                        new_nodes, is_fused, new_gate_name = self._fuse_gates([node, next_node], max_qubits, max_steps)
                        if new_gate_name == "max":
                            max_qubit_nodes.append(node)
                            break
                        elif is_fused:
                            nodes.remove(next_node)
                            nodes[0] = new_nodes[0]
                            break
            if not is_fused:
                nodes.pop(0)
            if not len(nodes):
                fusing = False

    def _minimal_parameter_ratio_fusion(self, max_qubits: int, max_steps: int) -> rx.PyDAG:
        """
        Perform fusion to minimize parameter ratios.

        Parameters
        ----------
        max_qubits : int
            The maximum allowed qubits in the fused gate.
        max_steps : int
            The maximum number of optimization steps.

        Returns
        -------
        rx.PyDAG
            The updated graph with fused nodes.
        """
        fusing = True
        while fusing:
            is_fused = False
            edge_parameter_ratios = dict(sorted(Compiler.gate_edge_parameter_ratios(self.graph, self.restriction).items(), 
                                                key=lambda item: item[1], reverse=False))
            for edge in edge_parameter_ratios:
                if edge_parameter_ratios[edge] > 1:
                    print(f"Parameter ratio {edge_parameter_ratios[edge]} for edge {edge} is greater than 1.")
                    continue
                new_nodes, is_fused, new_gate_name = self._fuse_gates(list(edge), max_qubits, max_steps)
                if new_gate_name == "max":
                    continue
                elif is_fused:
                    break
            if not is_fused:
                fusing = False

    def _sparsity_fusion(self, max_qubits: int, max_steps: int) -> rx.PyDAG:
        """
        Perform fusion to maximize sparsity.

        Parameters
        ----------
        max_qubits : int
            The maximum allowed qubits in the fused gate.
        max_steps : int
            The maximum number of optimization steps.

        Returns
        -------
        rx.PyDAG
            The updated graph with fused nodes.
        """
        fusing = True
        while fusing:
            is_fused = False
            edge_sparsities = dict(sorted(Compiler.gate_edge_sparsity(self.graph, self.restriction).items(), 
                                                key=lambda item: item[1], reverse=True))
            for edge in edge_sparsities:
                new_nodes, is_fused, new_gate_name = self._fuse_gates(list(edge), max_qubits, max_steps)
                if new_gate_name == "max":
                    continue
                elif is_fused:
                    break
            if not is_fused:
                fusing = False

    def _deep_block_fusion(self, max_qubits: int, max_steps: int) -> rx.PyDAG:
        """
        Perform fusion on deep blocks.

        Parameters
        ----------
        max_qubits : int
            The maximum allowed qubits in the fused gate.
        max_steps : int
            The maximum number of optimization steps.

        Returns
        -------
        rx.PyDAG
            The updated graph with fused nodes.
        """
        block_nodes = self._generate_next_block(6, max_qubits)
        new_nodes, is_fused, new_gate_names = self._fuse_gates(block_nodes, max_qubits, max_steps, multi_gate=True)
        #TODO: Implement deep block fusion

    @staticmethod
    def build_graph_from_circuit(circuit: qiskit.circuit.QuantumCircuit) -> rx.PyDAG:
        """
        Build a graph representation of the circuit.

        Parameters
        ----------
        circuit : qiskit.circuit.QuantumCircuit
            The quantum circuit.

        Returns
        -------
        rx.PyDAG
            The graph representation of the circuit.
        """
        if not Compiler.is_circuit_only_gates(circuit):
            raise ValueError("Can build a graph of circuits that only contains Gates currently.")
        graph = rx.PyDAG()

        # Add the qubits as nodes given by strings
        qubit_node_map = {q : i  for i, q in enumerate(Compiler.get_qubits(circuit))}
        qubit_node_map_out = {q : i + len(qubit_node_map) for i, q in enumerate(Compiler.get_qubits(circuit))}
        graph.add_nodes_from(list(qubit_node_map.keys()))
        graph.add_nodes_from(list(qubit_node_map.keys()))

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
        """
        Reconstruct a circuit from a graph.

        Parameters
        ----------
        graph : rx.PyDAG
            The graph representation of the circuit.
        qubits : qiskit.circuit.Qubit
            The qubits to use in the reconstructed circuit.

        Returns
        -------
        qiskit.circuit.QuantumCircuit
            The reconstructed QuantumCircuit.
        """
        sorted_nodes = rx.topological_sort(graph)
        instructions = [
            graph.get_node_data(node_index) for node_index in sorted_nodes 
            if issubclass(type(graph.get_node_data(node_index)),  qiskit.circuit.CircuitInstruction)
        ]
        return qiskit.circuit.QuantumCircuit.from_instructions(instructions, qubits=qubits)
        
    @staticmethod
    def is_instruction_a_gate(instruction: qiskit.circuit.Instruction | str) -> bool:
        """
        Checks if the given instruction is a gate operation.

        Parameters
        ----------
        instruction : qiskit.circuit.Instruction or str
            Instruction to check.

        Returns
        -------
        bool
            True if the instruction is a gate, False otherwise.
        """
        if type(instruction) is str: 
            return False
        else:
            return issubclass(instruction.operation.base_class, qiskit.circuit.Gate)
    
    @staticmethod
    def is_circuit_only_gates(circuit: qiskit.circuit.QuantumCircuit) -> bool:
        """
        Verifies if a circuit contains only gate instructions.

        Parameters
        ----------
        circuit : qiskit.circuit.QuantumCircuit
            The quantum circuit to check.

        Returns
        -------
        bool
            True if the circuit contains only gates, False otherwise.
        """
        if all([Compiler.is_instruction_a_gate(instr) for instr in circuit.data]):
            return True
        else:
            return False

    @staticmethod
    def is_edge_fuseable(graph: rx.PyDAG, edge: Tuple[int, int]) -> bool:
        """
        Determines if an edge between two nodes in the graph is fuseable.

        Parameters
        ----------
        graph : rx.PyDAG
            Graph representation of the circuit.
        edge : Tuple[int, int]
            Edge to check.

        Returns
        -------
        bool
            True if the edge is fuseable, False otherwise.
        """
        paths = rx.all_simple_paths(graph, edge[0], edge[1])
        return all([len(p) == 2 for p in paths])

    @staticmethod
    def gate_nodes(graph: rx.PyDAG) -> List[int]:
        """
        Retrieves a list of gate nodes in the graph.

        Parameters
        ----------
        graph : rx.PyDAG
            Graph representation of the circuit.

        Returns
        -------
        List[int]
            List of gate node indices.
        """
        return [i for i in graph.node_indices() if type(graph.get_node_data(i)) is not str]

    @staticmethod
    def gate_edges(graph: rx.PyDAG, only_fuseable_edges: bool = True) -> List[Tuple[int, int]]:
        """
        Retrieves gate edges in the graph.

        Parameters
        ----------
        graph : rx.PyDAG
            Graph representation of the circuit.
        only_fuseable_edges : bool, optional
            Whether to include only fuseable edges, by default True.

        Returns
        -------
        List[Tuple[int, int]]
            List of gate edges.
        """
        nodes = Compiler.gate_nodes(graph)
        return [edge for edge in graph.edge_list() 
                    if set(edge).issubset(set(nodes)) and 
                        (Compiler.is_edge_fuseable(graph, edge) or not only_fuseable_edges)]

    @staticmethod
    def edge_parameter_ratio(graph: rx.PyDAG, restriction: List[str], edge: Tuple[int, int], tol: float = 1e-10) -> float:
        """
        Computes the parameter ratio for a given edge in the graph.

        Parameters
        ----------
        graph : rx.PyDAG
            Graph representation of the circuit.
        restriction : List[str]
            List of allowed gate restrictions.
        edge : Tuple[int, int]
            Edge to compute the ratio for.

        Returns
        -------
        float
            Parameter ratio of the edge.
        """
        circ = Compiler.circuit_from_nodes(graph, list(edge))
        operation = qiskit.quantum_info.Operator.from_circuit(circ)

        full_basis = utils.construct_full_pauli_basis(operation.num_qubits)
        projected_basis = utils.construct_restricted_pauli_basis(operation.num_qubits, restriction)

        edge_parameter = lie.Unitary.parameters_from_unitary(operation.data, full_basis)
        # qubits_weighting = utils.qubits_weighting(operation.num_qubits)
        # ratio = qubits_weighting * np.count_nonzero(edge_parameter) / projected_basis.lie_algebra_dim
        ratio =  np.sum(np.abs(edge_parameter) > tol)  / projected_basis.lie_algebra_dim
        return ratio

    @staticmethod
    def gate_edge_parameter_ratios(graph: rx.PyDAG, restriction: List[str]) -> Dict[Tuple[int, int], float]:
        """
        Computes parameter ratios for all gate edges in the graph.

        Parameters
        ----------
        graph : rx.PyDAG
            Graph representation of the circuit.
        restriction : List[str]
            List of allowed gate restrictions.

        Returns
        -------
        Dict[Tuple[int, int], float]
            Dictionary of edges and their parameter ratios.
        """
        edges = Compiler.gate_edges(graph)
        edge_parameters = {edge : Compiler.edge_parameter_ratio(graph, restriction, edge) for edge in edges}
        return edge_parameters

    @staticmethod
    def edge_sparsity(graph: rx.PyDAG, restriction: List[str], edge: Tuple[int, int]) -> float:
        """
        Computes the sparsity of a gate operation along an edge.

        Parameters
        ----------
        graph : rx.PyDAG
            Graph representation of the circuit.
        restriction : List[str]
            List of allowed gate restrictions.
        edge : Tuple[int, int]
            Edge to compute sparsity for.

        Returns
        -------
        float
            Sparsity value of the edge.
        """
        circ = Compiler.circuit_from_nodes(graph, list(edge))
        operation = qiskit.quantum_info.Operator.from_circuit(circ)
        sparsity = 1.0 - (np.count_nonzero(operation.data) / float(operation.data.size) )
        return sparsity

    @staticmethod
    def gate_edge_sparsity(graph: rx.PyDAG, restriction: List[str]) -> Dict[Tuple[int, int], float]:
        """
        Computes sparsity values for all gate edges in the graph.

        Parameters
        ----------
        graph : rx.PyDAG
            Graph representation of the circuit.
        restriction : List[str]
            List of allowed gate restrictions.

        Returns
        -------
        Dict[Tuple[int, int], float]
            Dictionary of edges and their sparsity values.
        """
        edges = Compiler.gate_edges(graph)
        edge_parameters = {edge : Compiler.edge_sparsity(graph, restriction, edge) for edge in edges}
        return edge_parameters

    @staticmethod
    def get_qubits(circuit: qiskit.circuit.QuantumCircuit) -> List[str]:
        """
        Retrieves a list of qubit names in the quantum circuit.

        Parameters
        ----------
        circuit : qiskit.circuit.QuantumCircuit
            The quantum circuit.

        Returns
        -------
        List[str]
            List of qubit names.
        """
        qubits = []
        for qreg in circuit.qregs:
            for q in qreg:
                qubits.append(Compiler.get_qubit_name(q))
        return qubits
    
    @staticmethod
    def get_qubit_name(qubit: qiskit.circuit.Qubit) -> str:
        """
        Generates a string representation of a qubit name.

        Parameters
        ----------
        qubit : qiskit.circuit.Qubit
            Qubit object.

        Returns
        -------
        str
            String name of the qubit.
        """
        return qubit._register.name + str(qubit._index)

    @staticmethod
    def circuit_from_nodes(graph: rx.PyDAG, nodes: List[int]) -> qiskit.circuit.Instruction:
        """
        Constructs a circuit instruction from a list of graph nodes.

        Parameters
        ----------
        graph : rx.PyDAG
            Graph representation of the circuit.
        nodes : List[int]
            List of nodes to include in the circuit.

        Returns
        -------
        qiskit.circuit.Instruction
            Circuit instruction composed of the given nodes.
        """
        subgraph = graph.subgraph(nodes)
        sorted_nodes = rx.topological_sort(subgraph)
        instructions = [subgraph.get_node_data(node_index) for node_index in sorted_nodes]
        return qiskit.circuit.QuantumCircuit.from_instructions(instructions)
    
    @staticmethod
    def frobenius_distance(qubits: int, gateA: np.ndarray, gateB: np.ndarray = None) -> float:
        if not gateB:
            gateB = np.eye(2**qubits)
        delta = gateB - gateA
        return np.sqrt(abs(utils.trace_dot_jit(delta,  delta.conj().T)/2**qubits))

    @staticmethod
    def frobenius_distance_for_circuit(circuitA: qiskit.circuit.QuantumCircuit) -> float:
        op = qiskit.quantum_info.Operator.from_circuit(circuitA)
        return Compiler.frobenius_distance(op.num_qubits, op.data)

