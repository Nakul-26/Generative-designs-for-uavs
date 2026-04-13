import genetic_optimizer as go

# stub run_xfoil to avoid calling external xfoil
def stub_run_xfoil(naca, reynolds=1000000, alpha=5):
    # return plausible CL, CD depending on simple thickness heuristic
    # thicker airfoils (higher last two digits) -> slightly higher CL and CD
    try:
        digits = naca.replace("NACA ", "")
        thickness = int(digits[2:])
    except Exception:
        thickness = 12
    cl = 0.8 + (thickness - 12) * 0.002
    cd = 0.02 + (thickness - 12) * 0.0003
    return float(cl), float(cd)

# monkeypatch the imported run_xfoil in the optimizer module
go.run_xfoil = stub_run_xfoil

# shorten run for quick test
go.GENERATIONS = 2
go.POPULATION_SIZE = 10

if __name__ == '__main__':
    go.run_ga()
