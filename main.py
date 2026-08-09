from src.pipeline import run_pipeline


def main():
    run_pipeline(
        train_test_path='data/train-test.csv',
        validation_path='data/validation.csv',
        template_path='data/validation-predictions-template.csv',
        december_path='data/december_chart_inputs.csv'
    )


if __name__ == "__main__":
    main()