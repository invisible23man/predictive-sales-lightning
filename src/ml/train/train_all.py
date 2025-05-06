from src.ml.train.train_category import train_model_for_category

if __name__ == "__main__":
    categories = ["Beauty", "Clothing", "Electronics"]
    for category in categories:
        train_model_for_category(category)
