from scipy.sparse import csr_matrix
import fenics as fen

def fenics2Sparse(expr):
    emat = fen.as_backend_type(fen.assemble(expr)).mat()
    er, ec, ev = emat.getValuesCSR()
    return csr_matrix((ev, ec, er), shape = emat.size, dtype = complex)
