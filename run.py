import sys
import subprocess
import argparse

def run_model(model_name, gpu=0, dataset="basic", lambda_factor=None, date=None):
    print(f"\n[INFO] Running model: {model_name}")

    cmd = [
        "python", "utills\train_test.py",
        "--gpu", str(gpu),
        "--dataset", str(dataset),
        "--model", str(model_name),
    ]

    if lambda_factor is not None:
        cmd += ["--lambda_factor", str(lambda_factor)]

    if date is not None:
        cmd += ["--date", str(date)]

    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)
    # if result.stderr:
    #     print("[ERROR]", result.stderr)

def run_all_models(gpu=0, dataset="basic", lambda_factor=None, date=None):
    model_list = ["SQR", "RQR", "CP", "BNN", "MC", "GQNN", "GQNN_2"]
    for model_name in model_list:
        run_model(model_name, gpu=gpu, dataset=dataset, lambda_factor=lambda_factor, date=date)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--date", type=float, default=0000, help="Dates index")
    parser.add_argument("--dataset", type=str, default="basic", help="Dataset name")
    parser.add_argument("--lambda_factor", type=float, default=None, help="Lambda factor (optional)")

    args = parser.parse_args()

    if args.model:
        run_model(args.model, gpu=args.gpu, date=args.date, dataset=args.dataset, lambda_factor=args.lambda_factor)
    else:
        print("[INFO] No model specified. Running all models...")
        run_all_models(gpu=args.gpu, date=args.date, dataset=args.dataset, lambda_factor=args.lambda_factor)

if __name__ == "__main__":
    main()
    
# how to use
# python run.py --gpu 0 --date 0000 --dataset "basic" --lambda_factor 0.05
# python run.py --model "GQNN" --gpu 0 --date 0000 --dataset "basic" --lambda_factor 0.05