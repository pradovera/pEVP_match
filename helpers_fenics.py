from scipy.sparse import csr_matrix
import fenics as fen

def fenics2Sparse(expr):
    """
    This function converts a fenics expression to a sparse matrix.

    Parameters:
    expr (fenics.Expression): The fenics expression to be converted.

    Returns:
    scipy.sparse.csr_matrix: The converted sparse matrix.
    """
    emat = fen.as_backend_type(fen.assemble(expr)).mat()
    er, ec, ev = emat.getValuesCSR()
    return csr_matrix((ev, ec, er), shape = emat.size, dtype = complex)
