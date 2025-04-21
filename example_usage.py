from data_loader import MINDDataLoader

def main():
    # Initialize the data loader with the path to your MIND dataset
    train_loader = MINDDataLoader('MINDsmall_train')
    dev_loader = MINDDataLoader('MINDsmall_dev')

    # Load all data components
    print("Loading training data...")
    train_news, train_behaviors, train_entity_emb, train_rel_emb = train_loader.load_all_data()
    
    print("Loading development data...")
    dev_news, dev_behaviors, dev_entity_emb, dev_rel_emb = dev_loader.load_all_data()

    # Print some basic statistics
    print("\nTraining Data Statistics:")
    print(f"Number of news articles: {len(train_news)}")
    print(f"Number of user behaviors: {len(train_behaviors)}")
    print(f"Number of entity embeddings: {len(train_entity_emb)}")
    print(f"Number of relation embeddings: {len(train_rel_emb)}")

    print("\nDevelopment Data Statistics:")
    print(f"Number of news articles: {len(dev_news)}")
    print(f"Number of user behaviors: {len(dev_behaviors)}")
    print(f"Number of entity embeddings: {len(dev_entity_emb)}")
    print(f"Number of relation embeddings: {len(dev_rel_emb)}")

    # Example of processing user history and impressions
    print("\nProcessing user history and impressions...")
    train_user_history = train_loader.process_user_history()
    train_user_interactions = train_loader.process_impressions()

    # Print some example data
    print("\nExample news article:")
    print(train_news.iloc[0])

    print("\nExample user behavior:")
    print(train_behaviors.iloc[0])

if __name__ == "__main__":
    main() 