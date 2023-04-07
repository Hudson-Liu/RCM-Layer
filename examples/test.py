
def equivalent_dense_nodes(input_nodes: int, hidden_units: int, rcm_outputs: int, final_outputs: int) -> int:
    """Calculates the necessary number of hidden Dense nodes needed to equal the same number of parameters as the RCM layer"""
    return ((input_nodes + hidden_units + rcm_outputs) ** 2  + (rcm_outputs * final_outputs)) // (input_nodes + final_outputs)

print(equivalent_dense_nodes(256, 0, 100, 100))