import os, subprocess

FORKS = {
    "turing-equation": "./turing-equation/equation.py",
    "EquationOfTuring": "./EquationOfTuring/main.py",
    "ETOmega": "./ETOmega/evolve_turing_equation.py"
}

def run(fork, path):
    print(f"\n## Executando {fork}...\n")
    result = subprocess.run(["python3", path, "--maximize", "expected_improvement"],
                            capture_output=True, text=True)
    print(result.stdout.splitlines()[-10:])

if __name__ == "__main__":
    for name, path in FORKS.items():
        if os.path.exists(path):
            run(name, path)
