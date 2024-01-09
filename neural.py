##
 #  Worg
 ##
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import vgg  #  the file containing our network architecture
from torcheval.metrics.functional import binary_f1_score, binary_auroc
import matplotlib.pyplot as plt
from data_formatter import get_test_files
from data_formatter import SIZE_IMAGES

'''  Handle the number of batches used for each step '''

epochs = 40


train_samples, validation_samples, test_samples = np.load('temp/len.npy')

batch_size = 1
train_batches = train_samples // batch_size
validation_batches = validation_samples // batch_size
test_batches = test_samples // batch_size
image_size = SIZE_IMAGES[0]




'''  Function that writes the predicted labels in a csv file '''
def write_predicted(testing_file_names, predictions, file_name):
    fd = open(file_name, 'w')
    fd.write('name,label\n')
    for i in range(test_samples):
        x = predictions[i].item()
        fd.write(testing_file_names[i] + ',' + str(x) + '\n')
    fd.close()


def load_train_data():
    train_images = np.load('temp/train_mels_150/train.npy')
    train_images = torch.from_numpy(train_images).float().cuda()
    
    train_images = train_images.reshape((train_batches, batch_size, 1, image_size, image_size))

    train_labels = np.load('temp/train_mels_150/train_labels.npy')
    train_labels = torch.from_numpy(train_labels).long().cuda()
    
    train_labels = train_labels.reshape((train_batches, batch_size))
    return train_images, train_labels


def load_validation_data():
    validation_images = np.load('temp/validation_mels_150/validation.npy')
    validation_images = torch.from_numpy(validation_images).float().cuda()
    validation_images = validation_images.reshape((validation_batches, batch_size, 1, image_size, image_size))

    validation_labels = np.load('temp/validation_mels_150/validation_labels.npy')
    validation_labels = torch.from_numpy(validation_labels).long().cuda()
    validation_labels = validation_labels.reshape((validation_batches, batch_size))
    return validation_images, validation_labels


def load_test_data():
    test_images = np.load('temp/test_mels_150/test.npy')
    test_images = torch.from_numpy(test_images).float().cuda()
    test_images = test_images.reshape((test_batches, batch_size, 1, image_size, image_size))
    return test_images



def train_and_test():
    #  Load data from files
    train_images, train_labels = load_train_data()
    validation_images, validation_labels = load_validation_data()
    test_images = load_test_data()
    print('Finished loading data')


    #  Initialialize the normalization function
    # because there are significantly more objects
    m = np.load('temp/test_mels_150/test.npy')
    norm = transforms.Normalize(mean = np.mean(m), std=np.std(m))
    del m

    #  Normalizing values accross all images
    for i in range(train_batches):
        for j in range(batch_size):
            train_images[i][j] = norm(train_images[i][j])

    for i in range(validation_batches):
        for j in range(batch_size):
            validation_images[i][j] = norm(validation_images[i][j])

    for i in range(test_batches):
        for j in range(batch_size):
            test_images[i][j] = norm(test_images[i][j])


    #  All the computation is made using the GPU.
    net = vgg.VGG_16().cuda()
    optimizer = optim.SGD(net.parameters(), lr = 8e-4, momentum = 0.9, nesterov = True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.6)
    criterion = nn.CrossEntropyLoss()
    print('Starting training...')


    metrics = {"f1":[], "ROC_AUC":[], "accuracy":[], "loss":[]}

    
    for epoch in range(epochs):
        running_loss = 0.0
        #  Train mode
        print_offset = train_batches // 5

        net.train()
        for i in range(train_batches):
            inputs, labels = train_images[i], train_labels[i]
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            optimizer.zero_grad()
            if i % print_offset == print_offset - 1:
                print('[Epoch %d] %d/%d --- running_loss = %.3f' % (epoch, i + 1, train_batches, running_loss / print_offset))
                running_loss = 0.0

        #  Evaluate on training
        correct = 0
        total = 0
        with torch.no_grad():
            
            net.eval()
            for i in range(train_batches):
                inputs, labels = train_images[i], train_labels[i]
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        training_accuracy = 100 * correct / total
        print('[%d / %d] ---> training accuracy = %.2f --- training loss = %.2f' % (correct, total, training_accuracy, loss))

        #  Evaluate on validation
        correct = 0
        total = 0
        validation_loss = 0
        predictions = torch.zeros(validation_samples, dtype = torch.long)
        index = 0
        net.eval()
        with torch.no_grad():
            for i in range(validation_batches):
                inputs, labels = validation_images[i], validation_labels[i]
                outputs = net(inputs)
                predicted = torch.max(outputs.data, 1).indices
                loss = criterion(outputs, labels)
                validation_loss += loss.item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for j in range(index, index + batch_size):
                    predictions[j] = predicted[j - index]
                index += batch_size
        
        lb = validation_labels.reshape(predictions.shape)
        f1 = binary_f1_score(predictions.cuda(), lb.cuda())
        roc_auc = binary_auroc(predictions.cuda(), lb.cuda())

        validation_accuracy = 100 * correct / total
        validation_loss = validation_loss / validation_batches
        print('[%d / %d] ---> validation accuracy = %.2f --- validation loss = %.3f --- f1 = %.3f --- roc_auc = %.3f' % (
            correct, total, validation_accuracy, validation_loss, f1, roc_auc))
        
        metrics['accuracy'].append(validation_accuracy)
        metrics['f1'].append(f1.cpu())
        metrics['ROC_AUC'].append(roc_auc.cpu())
        metrics['loss'].append(validation_loss)
        
        print()
        scheduler.step()



    print('Saves plots...')

    for i in metrics:
        metrics
        plt.plot(metrics[i])
        plt.title(i)
        plt.ylabel(i)
        plt.xlabel("epochs")
        plt.savefig("metrics/" + i)
        plt.cla()

    print('Computing predictions for test set...')
    best_accuracy = validation_accuracy
    best_loss = validation_loss
    correct = 0
    total = 0
    index = 0
    predictions = torch.zeros(test_samples, dtype = torch.long)
    with torch.no_grad():
        for i in range(test_batches):
            inputs = test_images[i]
            net.eval()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)

            for j in range(index, index + batch_size):
                predictions[j] = predicted[j - index]
            index += batch_size

    testing_file_names = get_test_files()
    write_predicted(testing_file_names, predictions, 'submission.txt')


if __name__ == '__main__':
    train_and_test()
