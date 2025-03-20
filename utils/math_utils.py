# utils/math_utils.py
import ast
import operator

# Safe math evaluation setup
operators_safe = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.BitXor: operator.xor,
    ast.USub: operator.neg,
}

def safe_eval(expr):
    """Safely evaluate a mathematical expression."""
    try:
        return eval_(ast.parse(expr, mode='eval').body)
    except Exception:
        return "Error"

def eval_(node):
    """Helper function for safe_eval to evaluate AST nodes."""
    if isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.BinOp):
        return operators_safe[type(node.op)](eval_(node.left), eval_(node.right))
    elif isinstance(node, ast.UnaryOp):
        return operators_safe[type(node.op)](eval_(node.operand))
    else:
        raise TypeError(node)