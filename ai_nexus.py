# AI-Nexus Main

from src.quantum.core import FERN

from src.simulator.core import QENME

from src.validation.validation import ModelValidator

def main():

    print("AI-Nexus: Integrated Quantum-AI Platform")

    # Run quantum sim

    fern = FERN()

    qec_result = fern.run_memory_experiment(d=7)

    print(f"QEC LER: {qec_result['logical_error_rate']}")

    # Run simulator

    qenme = QENME({'distance': 7})

    sim_results = qenme.run_all()

    print(f"Sim results: {list(sim_results.keys())}")

    # Validate

    validator = ModelValidator()

    from src.validation.validation import MathematicalModel

    model = MathematicalModel()

    val_result = validator.validate_conservation(model, None, [0,1,2])

    print(f"Validation: {val_result.message}")

    print("Integration complete. Add API keys for NLP.")

if __name__ == "__main__":

    main()