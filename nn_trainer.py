import numpy as np 
import matplotlib.pyplot as plt 


class NNTrainer: 

    def __init__(self, name, model, train_loader, test_loader):
        self.name = name
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
 

        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []

    def accuracy_check(self, pred, labels):
        pred = np.round(pred).squeeze()
        return (np.sum(pred == labels))


    def train(self, epochs, batch_size, lr, loss_function, lr_scheduling=True, lr_step_size=.1, lr_step_inc=10): 

        for epoch in range(epochs): 
            train_loss = 0
            train_correct = 0
            test_loss = 0
            test_correct = 0

            for data, labels in self.train_loader.get_batches(batch_size): 
                pred = self.model.forward(data, store_act=True)
                loss = loss_function.forward(pred, np.expand_dims(labels, 1))
                self.model.backward(loss_function.backward(), update_weights=True, lr=lr)
                train_loss += loss
                train_correct += self.accuracy_check(pred, labels)

            self.train_losses.append(train_loss)
            self.train_acc.append(train_correct/len(self.train_loader))

            for data, labels in self.test_loader.get_batches(batch_size): 
                pred = self.model.forward(data)
                loss = loss_function.forward(pred, labels)
                test_loss += loss
                test_correct += self.accuracy_check(pred, labels)
                
            self.test_losses.append(loss)
            self.test_acc.append(test_correct/len(self.test_loader))

            if lr_scheduling and epoch % lr_step_inc == 0: 
                lr *= lr_step_size

            print("Epoch: {}, train_loss: {} test_loss: {}, train_acc: {}, test_acc: {}".format(epoch, self.train_losses[epoch], self.test_losses[epoch], self.train_acc[epoch], self.test_acc[epoch]))


    def plot_loss(self):
        epochs = np.arange(0, len(self.train_losses))
        plt.plot(epochs, self.train_losses)
        plt.plot(epochs, self.test_losses)
        plt.title("Neural Network Loss Per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(["training", "Testing"])
        plt.show()

    def plot_acc(self):
        epochs = np.arange(0, len(self.train_losses))
        plt.plot(epochs, self.train_acc)
        plt.plot(epochs, self.test_acc)
        plt.title("Neural Network Accuracy Per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(["training", "Testing"])
        plt.show()