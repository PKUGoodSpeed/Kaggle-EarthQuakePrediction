from models.trainer import NNModelTrainer
from models.customizedCNN import CustomizedCNNModel

if __name__ == "__main__":
    model = CustomizedCNNModel()
    trainer = NNModelTrainer(num_channel=model.NUM_CHANNEL)
    trainer.setModelName(model.getModelName())
    trainer.loadModel(model.getModel())
    # trainer.loadWeightsFromFile()
    trainer.train(
        learning_rate=0.005, decaying_rate=0.95,
        epochs_per_decay=5, epochs=200)