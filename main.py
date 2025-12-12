import argparse
import sys
from baseline_evaluation import baseline_evaluation
from finetuning_evaluation import finetune_common_voice
from LORA_evaluation import finetune_with_lora

def main():

    parser = argparse.ArgumentParser(description="Accented English Speech Recognition")
    parser.add_argument("experiment", help="Command to execute (baseline, finetuning, LORAFinetuning))")
    
    args = parser.parse_args()
    
    if args.experiment == "b":
        baseline = baseline_evaluation()
        baseline.prepare_data()
        baseline.run_evaluation()
    elif args.experiment == "f":
        finetune_common_voice()
    elif args.experiment == "l":
        finetune_with_lora()
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()